"""scene_quality review type (#133): a yes/no choice per raised flag, report-only accept, group adds."""

import pytest

import curation_settings
import features
from recommendations_db import RecommendationsDB
from review.quality import QualityReview
from review.service import ReviewService
from review.store import ReviewStore
from review.types import InvalidOption, TargetMissing, get_review_type
from tests.fake_stash import FakeStash

PLUGIN_ID = "stash-sense"
SIZE = 2 * 1024 ** 3
RT = QualityReview()


@pytest.fixture(autouse=True)
def show_unfinished(monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)


@pytest.fixture(autouse=True)
def real_cache(monkeypatch):
    """Undo conftest's empty list: read the settings from the fake Stash."""
    cache = curation_settings.ExclusionCache()
    monkeypatch.setattr(curation_settings, "_cache", cache)
    return cache


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "stash_sense.db")


@pytest.fixture
def store(db):
    return ReviewStore(db.db_path)


@pytest.fixture
def stash():
    return FakeStash()


@pytest.fixture
def svc(db, store, stash):
    return ReviewService(db, store, stash)


async def configure(stash, **values):
    await stash.configure_plugin(PLUGIN_ID, values)
    curation_settings.invalidate()


def quality_scene(stash, oshash="osh1"):
    fid = stash.add_file("/m/a.mp4", SIZE, duration=1800.0, oshash=oshash)
    return stash.add_scene(fid, title="a scene"), fid


def rec_for(db, sid, fid, flags=("broken", "rip"), *, oshash="osh1", control=False):
    details = {
        "file": {"id": fid, "oshash": oshash, "size": SIZE, "basename": "a.mp4", "codec": "h264",
                 "width": 1280, "height": 720, "bit_rate": 2_000_000, "duration": 1800.0, "format": "mp4"},
        "flags": {f: {"why": f} for f in flags},
        "scores": {"rip": 0.7},
        "control": control,
        "quality_version": 1,
    }
    rid, _ = db.upsert_recommendation("scene_quality", "scene", sid, details, confidence=0.5)
    return db.get_recommendation(rid)


def test_registered():
    assert isinstance(get_review_type("scene_quality"), QualityReview)


async def test_preview_choices_per_raised_flag(db, stash):
    sid, fid = quality_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid), None, stash)
    assert [c["field"] for c in p.choices] == ["broken", "rip"]
    assert all(c["kind"] == "pick" and [v["value"] for v in c["values"]] == ["yes", "no"] and c["label"] for c in p.choices)
    assert p.default_option == {"broken": "yes", "rip": "yes"}
    assert p.high_confidence is False
    assert p.is_default_option
    assert p.summary == "Broken; Tube or leak rip"
    ev = p.evidence
    assert ev["quality"] == {"broken": {"why": "broken"}, "rip": {"why": "rip"}}
    assert ev["scores"] == {"rip": 0.7} and ev["control"] is False and ev["report_only"] is True
    assert ev["scene"] == {"id": sid, "title": "a scene"} and ev["file"]["oshash"] == "osh1"
    assert ev["groups"] == {}


async def test_control_card_offers_all_flags_default_no(db, stash):
    sid, fid = quality_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid, flags=(), control=True), None, stash)
    assert [c["field"] for c in p.choices] == ["broken", "rip", "below_studio", "format"]
    assert p.default_option == {f: "no" for f in ("broken", "rip", "below_studio", "format")}
    assert p.summary == "Control sample: is anything wrong with this file?"
    assert p.evidence["control"] is True and p.high_confidence is False


async def test_report_only_accept_changes_no_groups(db, stash, svc, store):
    g12, g13 = stash.add_group("broken q"), stash.add_group("rip q")
    await configure(stash, qualityGroupBroken=g12, qualityGroupRip=g13)
    sid, fid = quality_scene(stash)
    rec = rec_for(db, sid, fid)
    pv = await svc.preview(rec.id)
    assert pv["evidence"]["report_only"] is True and pv["plan"]["field_changes"] == []
    res = await svc.accept(rec.id, option=None, plan_token=pv["plan_token"])
    assert stash.scenes[sid]["groups"] == [] and stash.writes == 0
    assert db.get_recommendation(rec.id).status == "resolved"
    assert store.latest_applied_for_rec(rec.id)["id"] == res["entry"]["id"]


async def test_live_accept_adds_groups_for_yes_flags(db, stash, svc, store):
    g12, g13 = stash.add_group("broken q"), stash.add_group("rip q")
    await configure(stash, qualityReportOnly=False, qualityGroupBroken=g12, qualityGroupRip=g13)
    sid, fid = quality_scene(stash)
    rec = rec_for(db, sid, fid)
    option = {"broken": "yes", "rip": "no"}
    pv = await svc.preview(rec.id, option)
    changes = pv["plan"]["field_changes"]
    assert len(changes) == 1 and changes[0]["field"] == "groups"
    assert changes[0]["before"] == []
    assert changes[0]["after"] == [{"group_id": g12, "scene_index": None}]
    await svc.accept(rec.id, option=option, plan_token=pv["plan_token"])
    assert [g["group_id"] for g in stash.scenes[sid]["groups"]] == [g12]
    assert db.get_recommendation(rec.id).status == "resolved"
    preview = await RT.preview(rec, option, stash)
    assert RT.label(rec, preview) == {"suggested": {"broken": "yes", "rip": "yes"}, "chosen": option}


async def test_flag_without_group_writes_nothing(db, stash):
    g12 = stash.add_group("broken q")
    await configure(stash, qualityReportOnly=False, qualityGroupBroken=g12)
    sid, fid = quality_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid, flags=("rip",)), None, stash)
    assert p.plan.field_changes == [] and p.plan.is_empty


async def test_scene_already_in_group_no_change(db, stash):
    g12 = stash.add_group("broken q")
    await configure(stash, qualityReportOnly=False, qualityGroupBroken=g12)
    sid, fid = quality_scene(stash)
    stash.scenes[sid]["groups"] = [{"group_id": g12, "scene_index": 3}]
    p = await RT.preview(rec_for(db, sid, fid, flags=("broken",)), None, stash)
    assert p.plan.field_changes == []


async def test_adds_to_existing_groups(db, stash):
    g1, g12 = stash.add_group("other"), stash.add_group("broken q")
    await configure(stash, qualityReportOnly=False, qualityGroupBroken=g12)
    sid, fid = quality_scene(stash)
    stash.scenes[sid]["groups"] = [{"group_id": g1, "scene_index": 2}]
    p = await RT.preview(rec_for(db, sid, fid, flags=("broken",)), None, stash)
    (change,) = p.plan.field_changes
    assert change.before == [{"group_id": g1, "scene_index": 2}]
    assert change.after == [{"group_id": g1, "scene_index": 2}, {"group_id": g12, "scene_index": None}]


async def test_invalid_option_raises(db, stash):
    sid, fid = quality_scene(stash)
    rec = rec_for(db, sid, fid)
    with pytest.raises(InvalidOption):
        await RT.preview(rec, {"broken": "maybe"}, stash)
    with pytest.raises(InvalidOption):
        await RT.preview(rec, {"format": "yes"}, stash)  # not a raised flag on this card
    with pytest.raises(InvalidOption):
        await RT.preview(rec, {"bogus": "yes"}, stash)


async def test_partial_option_filled_from_default(db, stash):
    sid, fid = quality_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid), {"rip": "no"}, stash)
    assert p.plan.option == {"broken": "yes", "rip": "no"} and not p.is_default_option


async def test_target_missing(db, stash):
    sid, fid = quality_scene(stash)
    rec = rec_for(db, sid, fid)
    changed = rec_for(db, sid, fid, oshash="other")
    with pytest.raises(TargetMissing):
        await RT.preview(changed, None, stash)
    del stash.scenes[sid]
    with pytest.raises(TargetMissing):
        await RT.preview(rec, None, stash)

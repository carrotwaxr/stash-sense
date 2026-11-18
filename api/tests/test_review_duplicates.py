"""scene_duplicates review type (#130) end to end against FakeStash."""

import os

import pytest

import curation_settings
import features
from content_fingerprint.store import ContentFpStore
from curation_settings import CurationSettings, Exclusion
from recommendations_db import RecommendationsDB
from review.duplicates import SceneDuplicatesReviewType
from review.service import Rejected, ReviewService
from review.store import ReviewStore
from review.types import InvalidOption, TargetMissing, get_review_type
from tests.fake_stash import FakeStash

GB = 1024 ** 3
CODECS = ("h264", "hevc", "wmv3", "mpeg4", "av1", "vp9", "vc1")


@pytest.fixture(autouse=True)
def show_unfinished(monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("DATA_DIR", str(d))
    return d


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


RT = SceneDuplicatesReviewType()


def rec_for(db, cls, scene_ids, *, files=None, moments=None):
    """A scene_duplicates rec; files[i] is the oshash of scene_ids[i]'s file named in the pair."""
    pairs = []
    for i in range(1, len(scene_ids)):
        m = (moments or {}).get(i, {"t_a": 60, "t_b": 60})
        pairs.append({
            "scene_a": scene_ids[0], "scene_b": scene_ids[i],
            "file_a": files[0] if files else None, "file_b": files[i] if files else None,
            "sources": ["phash"], "cls": cls, "evidence": {}, "moment": m,
        })
    rid = db.create_recommendation(
        "scene_duplicates", "scene_set", f"{cls}:{'-'.join(scene_ids)}",
        {"cls": cls, "scene_ids": list(scene_ids), "pairs": pairs}, confidence=1.0,
    )
    return db.get_recommendation(rid)


def pair(stash, *, a=None, b=None, sa=None, sb=None):
    """Two scenes with one file each; a and b are add_file kwargs, sa and sb add_scene kwargs."""
    a = {"path": "/m/a.mp4", "size": 1 * GB, "oshash": "aaaa", "md5": "ma", **(a or {})}
    b = {"path": "/m/b.mp4", "size": 1 * GB, "oshash": "bbbb", "md5": "mb", **(b or {})}
    fa = stash.add_file(a.pop("path"), a.pop("size"), **a)
    fb = stash.add_file(b.pop("path"), b.pop("size"), **b)
    s1 = stash.add_scene(fa, **{"title": "scene one", **(sa or {})})
    s2 = stash.add_scene(fb, **{"title": "scene two", **(sb or {})})
    return s1, s2, fa, fb


async def accept(svc, rec, option=None):
    p = await svc.preview(rec.id, option)
    return await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"],
                            confirm_delete=[f["file_id"] for f in p["plan"]["delete_files"]])


def test_registered():
    rt = get_review_type("scene_duplicates")
    assert isinstance(rt, SceneDuplicatesReviewType)
    assert rt.display_name == "Duplicate Scenes"


# ---------------- keeper pick by file hash ----------------

async def test_pick_applies_exact_file(svc, stash, db):
    # Scene one's file is listed first on the card but sorts last by oshash.
    s1, s2, fa, fb = pair(stash, a={"oshash": "ffff"}, b={"oshash": "1111", "width": 1280, "height": 720})
    rec = rec_for(db, "duplicate", [s1, s2], files=["ffff", "1111"])

    p = await RT.preview(rec, None, stash)
    pick = next(c for c in p.choices if c["field"] == "keep")
    assert [v["value"] for v in pick["values"]] == ["ffff", "1111"]  # card order
    assert p.default_option == {"keep": "ffff", "keep_extra": ""}

    p = await RT.preview(rec, {"keep": pick["values"][1]["value"]}, stash)
    assert [f.file_id for f in p.plan.keep_files] == [fb]
    assert [f.file_id for f in p.plan.delete_files] == [fa]

    out = await accept(svc, rec, {"keep": "1111"})
    assert out["entry"]["status"] == "applied"
    assert fb in stash.files and fa not in stash.files
    assert out["label_id"] is not None  # the correction carries the oshashes


async def test_identical_oshash_copies_resolve_to_one_file(svc, stash, db):
    s1, s2, fa, fb = pair(stash, a={"oshash": "same", "md5": "m"}, b={"oshash": "same", "md5": "m"})
    rec = rec_for(db, "duplicate", [s1, s2])

    p = await RT.preview(rec, None, stash)
    assert len(p.plan.keep_files) == 1 and len(p.plan.delete_files) == 1
    keys = [v["value"] for v in next(c for c in p.choices if c["field"] == "keep")["values"]]
    assert len(set(keys)) == 2
    assert p.high_confidence

    await accept(svc, rec)
    assert len({fa, fb} & set(stash.files)) == 1


async def test_unresolvable_pick_is_held(svc, stash, db):
    s1, s2, fa, fb = pair(stash)
    rec = rec_for(db, "duplicate", [s1, s2])

    with pytest.raises(InvalidOption):
        await RT.preview(rec, {"keep": "gone"}, stash)
    with pytest.raises(InvalidOption):
        await RT.preview(rec, {"keep": "aaaa", "keep_extra": "gone"}, stash)
    with pytest.raises(InvalidOption):
        await RT.preview(rec, {"keep": "aaaa", "keep_extra": "aaaa"}, stash)
    with pytest.raises(Rejected):
        await svc.preview(rec.id, {"keep": "gone"})
    assert stash.writes == 0 and {fa, fb} <= set(stash.files)


# ---------------- merge plan ----------------

async def test_merge_values_union(svc, stash, db):
    fd = stash.add_file("/m/d.mp4", GB, oshash="dddd")
    f1 = stash.add_file("/m/1.mp4", GB, oshash="1111", width=1280, height=720)
    f2 = stash.add_file("/m/2.mp4", GB, oshash="2222", width=1280, height=720)
    dest = stash.add_scene(
        fd, title="Dest", play_count=3, performer_ids=["p1", "p2"], tag_ids=["t1"], urls=["u1"],
        groups=[{"group_id": "g1", "scene_index": 1}, {"group_id": "g2", "scene_index": None}],
        gallery_ids=["x1"])
    src1 = stash.add_scene(
        f1, title="Source one", date="2020-01-01", rating100=80, performer_ids=["p2", "p3"], tag_ids=["t2"],
        urls=["u1", "u2"], groups=[{"group_id": "g1", "scene_index": 5}, {"group_id": "g3", "scene_index": None}],
        gallery_ids=["x2"])
    src2 = stash.add_scene(f2, title="Source two", date="2019-05-05", organized=True, code="C-2",
                           tag_ids=["t3", "t1"])
    rec = rec_for(db, "duplicate", [dest, src1, src2])

    p = await RT.preview(rec, None, stash)
    m = p.plan.merge
    assert m.destination == dest and m.sources == [src1, src2]
    assert m.values == {
        "date": "2020-01-01",
        "code": "C-2",
        "rating100": 80,
        "organized": True,
        "urls": ["u1", "u2"],
        "performer_ids": ["p1", "p2", "p3"],
        "tag_ids": ["t1", "t2", "t3"],
        "groups": [{"group_id": "g1", "scene_index": 1}, {"group_id": "g2", "scene_index": None},
                   {"group_id": "g3", "scene_index": None}],
        "gallery_ids": ["x1", "x2"],
    }
    assert "title" not in m.values
    assert sorted(f.file_id for f in p.plan.delete_files) == sorted([f1, f2])
    assert {f.file_id: f.scene_id for f in p.plan.delete_files} == {f1: src1, f2: src2}
    assert p.plan.primary_file is None

    await accept(svc, rec)
    s = stash.scenes[dest]
    assert src1 not in stash.scenes and src2 not in stash.scenes
    assert s["title"] == "Dest" and s["date"] == "2020-01-01" and s["organized"] is True
    assert s["performer_ids"] == ["p1", "p2", "p3"] and s["file_ids"] == [fd]


async def test_destination_is_history_scene(svc, stash, db):
    s1, s2, fa, fb = pair(stash, a={"width": 1280, "height": 720},
                          sa={"play_count": 2, "rating100": 60}, sb={"title": "no history"})
    rec = rec_for(db, "duplicate", [s1, s2])

    p = await RT.preview(rec, None, stash)
    assert p.default_option["keep"] == "bbbb"  # the history scene's file loses
    assert p.plan.merge.destination == s1 and p.plan.merge.sources == [s2]
    assert [(f.file_id, f.scene_id) for f in p.plan.delete_files] == [(fa, s1)]
    pf = p.plan.primary_file
    assert (pf.scene_id, pf.before, pf.after) == (s1, fa, fb)
    assert p.evidence["playback"]["scene_id"] == s1

    await accept(svc, rec)
    assert list(stash.scenes) == [s1]
    s = stash.scenes[s1]
    assert s["file_ids"] == [fb] and s["title"] == "scene one" and s["play_count"] == 2 and s["rating100"] == 60


async def test_mislabel_never_unions(svc, stash, db):
    s1, s2, fa, fb = pair(stash, sa={"title": "Right", "performer_ids": ["p1"], "play_count": 1},
                          sb={"title": "Wrong", "performer_ids": ["p2"], "tag_ids": ["t2"], "date": "2021-01-01"})
    rec = rec_for(db, "mislabel", [s1, s2])

    p = await RT.preview(rec, None, stash)
    mf = next(c for c in p.choices if c["field"] == "metadata_from")
    assert [v["value"] for v in mf["values"]] == [s1, s2]
    assert p.default_option["metadata_from"] == s1
    assert p.plan.merge.destination == s1 and p.plan.merge.values == {}
    assert not p.high_confidence

    p = await RT.preview(rec, {"metadata_from": s2}, stash)
    assert p.plan.merge.destination == s2 and p.plan.merge.sources == [s1]
    assert p.plan.merge.values == {}
    # s2's file loses the tie, so it is deleted and primary moves to the kept file
    assert [f.file_id for f in p.plan.delete_files] == [fb]
    assert (p.plan.primary_file.scene_id, p.plan.primary_file.after) == (s2, fa)

    with pytest.raises(InvalidOption):
        await RT.preview(rec, {"metadata_from": "999"}, stash)

    await accept(svc, rec, {"metadata_from": s2})
    assert list(stash.scenes) == [s2]
    s = stash.scenes[s2]
    assert (s["title"], s["performer_ids"], s["tag_ids"], s["date"]) == ("Wrong", ["p2"], ["t2"], "2021-01-01")
    assert s["file_ids"] == [fa]


# ---------------- batch eligibility ----------------

async def test_batch_eligible_only_identical_or_same_encode(stash, db, data_dir):
    # same md5, different oshash
    s1, s2, *_ = pair(stash, a={"md5": "same"}, b={"md5": "same"})
    rec = rec_for(db, "duplicate", [s1, s2])
    p = await RT.preview(rec, None, stash)
    assert p.high_confidence
    # the option differs from the default
    other = "bbbb" if p.default_option["keep"] == "aaaa" else "aaaa"
    p = await RT.preview(rec, {"keep": other}, stash)
    assert not p.high_confidence

    assert p.evidence["flags"] == ["tie"]  # a tie is fine when the deleted file is identical

    # same encode: codec, frame size and duration within 1 s; the keeper wins as the smaller file
    s3, s4, f3, f4 = pair(stash, a={"oshash": "c1", "duration": 600.0}, b={"oshash": "c2", "duration": 600.6})
    stash.files[f4]["size"] = GB // 2
    p = await RT.preview(rec_for(db, "duplicate", [s3, s4]), None, stash)
    assert p.default_option["keep"] == "c2" and p.evidence["flags"] == []
    assert p.high_confidence
    # the same encode at a tie: not identical, so reviewed one by one
    s3b, s4b, *_ = pair(stash, a={"oshash": "t1", "duration": 600.0}, b={"oshash": "t2", "duration": 600.6})
    p = await RT.preview(rec_for(db, "duplicate", [s3b, s4b]), None, stash)
    assert p.evidence["flags"] == ["tie"] and not p.high_confidence
    # supercut: the longer file is kept too, so nothing is deleted
    s3c, s4c, *_ = pair(stash, a={"oshash": "sc1", "duration": 600.0, "md5": "z"},
                        b={"oshash": "sc2", "duration": 3000.0, "width": 1280, "height": 720, "md5": "z"})
    p = await RT.preview(rec_for(db, "duplicate", [s3c, s4c]), None, stash)
    assert "supercut_candidate" in p.evidence["flags"] and p.plan.delete_files == []
    assert not p.high_confidence
    # a flagged suggestion that still deletes an identical copy
    f5 = stash.add_file("/m/sc3.mp4", GB, oshash="sc3", md5="z", duration=600.0)
    s5c = stash.add_scene(f5)
    p = await RT.preview(rec_for(db, "duplicate", [s3c, s4c, s5c]), None, stash)
    assert "supercut_candidate" in p.evidence["flags"] and len(p.plan.delete_files) == 1
    assert not p.high_confidence
    # duration 2 s apart
    s5, s6, *_ = pair(stash, a={"oshash": "d1", "duration": 600.0}, b={"oshash": "d2", "duration": 602.0})
    assert not (await RT.preview(rec_for(db, "duplicate", [s5, s6]), None, stash)).high_confidence
    # different resolution
    s7, s8, *_ = pair(stash, a={"oshash": "e1"}, b={"oshash": "e2", "width": 1280, "height": 720})
    assert not (await RT.preview(rec_for(db, "duplicate", [s7, s8]), None, stash)).high_confidence

    # held: every file failed its health check
    fs = ContentFpStore(os.path.join(data_dir, "content_fp.db"))
    for h in ("h1", "h2"):
        fs.put_health(fs.upsert_file(h, 3 * GB), "suspect", ["frozen"], {})
    fs.close()
    s9, s10, *_ = pair(stash, a={"oshash": "h1", "md5": "x", "size": 3 * GB},
                       b={"oshash": "h2", "md5": "x", "size": 3 * GB})
    p = await RT.preview(rec_for(db, "duplicate", [s9, s10]), None, stash)
    assert "held_unhealthy" in p.evidence["flags"] and not p.high_confidence
    assert p.plan.delete_files == []
    assert {f["health"] for s in p.evidence["scenes"] for f in s["files"]} == {"suspect"}

    # identical files under any other class
    for cls in ("mislabel", "language_variant", "contained", "partial", "review"):
        a, b, *_ = pair(stash, a={"oshash": f"{cls}1", "md5": "y"}, b={"oshash": f"{cls}2", "md5": "y"})
        assert not (await RT.preview(rec_for(db, cls, [a, b]), None, stash)).high_confidence, cls


# ---------------- classes that keep files ----------------

@pytest.mark.parametrize("cls", ["partial", "review"])
async def test_partial_card_is_empty_plan(svc, stash, db, cls):
    s1, s2, fa, fb = pair(stash, b={"width": 1280, "height": 720})
    rec = rec_for(db, cls, [s1, s2])

    p = await RT.preview(rec, None, stash)
    assert p.plan.is_empty
    assert p.default_option == {"keep": "aaaa", "keep_extra": ""}

    out = await accept(svc, rec)
    assert out["entry"]["status"] == "applied"
    assert stash.writes == 0 and {s1, s2} <= set(stash.scenes) and {fa, fb} <= set(stash.files)
    assert db.get_recommendation(rec.id).status == "resolved"


async def test_language_variant_keeps_both_files(svc, stash, db):
    s1, s2, fa, fb = pair(stash, b={"width": 1280, "height": 720})
    rec = rec_for(db, "language_variant", [s1, s2])

    p = await RT.preview(rec, None, stash)
    assert p.plan.delete_files == [] and p.plan.primary_file is None
    assert p.plan.merge.destination == s1 and p.plan.merge.sources == [s2]

    await accept(svc, rec)
    assert list(stash.scenes) == [s1] and stash.scenes[s1]["file_ids"] == [fa, fb]


async def test_contained_keeps_long_file(stash, db):
    # the long file is better: the short one goes
    s1, s2, fa, fb = pair(stash, a={"duration": 1000.0}, b={"duration": 800.0, "width": 1280, "height": 720})
    rec = rec_for(db, "contained", [s1, s2])
    p = await RT.preview(rec, None, stash)
    assert [f.file_id for f in p.plan.delete_files] == [fb]
    # picking the short file alone still keeps the long one
    p = await RT.preview(rec, {"keep": "bbbb", "keep_extra": ""}, stash)
    assert fa not in p.plan.delete_file_ids and p.plan.delete_files == []

    # the short file is better: both kept
    s3, s4, fc, fd = pair(stash, a={"oshash": "long", "duration": 1000.0, "width": 1280, "height": 720},
                          b={"oshash": "short", "duration": 800.0})
    p = await RT.preview(rec_for(db, "contained", [s3, s4]), None, stash)
    assert p.plan.delete_files == []
    assert not p.high_confidence


async def test_no_codec_mention_when_lists_empty(stash, db, monkeypatch):
    s1, s2, fa, fb = pair(stash, a={"codec": "h264"}, b={"codec": "hevc", "size": GB // 2})
    stash.files[fb]["bit_rate"] = 3_500_000  # the same effective detail from fewer bits
    rec = rec_for(db, "duplicate", [s1, s2])

    p = await RT.preview(rec, None, stash)
    text = (p.summary + " " + p.evidence["keeper_reason"]).lower()
    assert not any(c in text for c in CODECS), text
    assert p.default_option["keep"] == "bbbb"

    async def disfavor_hevc(stash, **kw):
        return Exclusion(CurationSettings(disfavored_video_codecs=("hevc",)), frozenset(), 0.0)

    monkeypatch.setattr(curation_settings, "current", disfavor_hevc)
    p = await RT.preview(rec, None, stash)
    assert p.default_option["keep"] == "aaaa" and "hevc" in p.evidence["keeper_reason"]


async def test_target_missing(stash, db):
    s1, s2, fa, fb = pair(stash)
    rec = rec_for(db, "duplicate", [s1, s2])
    del stash.scenes[s2]
    with pytest.raises(TargetMissing):
        await RT.preview(rec, None, stash)

    # fewer than two files in the whole set
    s3 = stash.add_scene(stash.add_file("/m/only.mp4", GB, oshash="only"))
    with pytest.raises(TargetMissing):
        await RT.preview(rec_for(db, "duplicate", [s3]), None, stash)


async def test_evidence_shape(stash, db):
    s1, s2, fa, fb = pair(stash, b={"width": 1280, "height": 720})
    rec = rec_for(db, "duplicate", [s1, s2], files=["aaaa", "bbbb"], moments={1: {"t_a": 60, "t_b": 75}})

    p = await RT.preview(rec, None, stash)
    ev = p.evidence
    assert ev["cls"] == "duplicate" and len(ev["pairs"]) == 1
    assert [s["id"] for s in ev["scenes"]] == [s1, s2]
    f = ev["scenes"][0]["files"][0]
    assert f["file_id"] == fa and f["oshash"] == "aaaa" and f["is_primary"] and f["health"] is None
    assert ev["stills"] == [{"scene_id": s1, "file_id": fa, "t": 60}, {"scene_id": s2, "file_id": fb, "t": 75}]
    assert ev["playback"] == {"scene_id": s1, "t": 60}
    assert "merge 2 scenes" in p.summary and "delete 1 file" in p.summary and "1.0 GB" in p.summary
    assert RT.scene_ids(rec) == {s1, s2}


async def test_one_scene_set_deletes_without_merge(svc, stash, db):
    fa = stash.add_file("/m/a.mp4", GB, oshash="aaaa", width=1280, height=720)
    fb = stash.add_file("/m/b.mp4", GB, oshash="bbbb")
    s1 = stash.add_scene(fa, fb)
    rec = rec_for(db, "duplicate", [s1])

    p = await RT.preview(rec, None, stash)
    assert p.plan.merge is None
    assert [f.file_id for f in p.plan.delete_files] == [fa]
    assert (p.plan.primary_file.before, p.plan.primary_file.after) == (fa, fb)
    await accept(svc, rec)
    assert stash.scenes[s1]["file_ids"] == [fb]

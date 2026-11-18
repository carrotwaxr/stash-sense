"""scene_splits review type (#131): cut times, trims, snapping, leave whole, the redundant card."""

import numpy as np
import pytest

import features
from recommendations_db import RecommendationsDB
from review.service import ReviewService
from review.splits import SceneSplitsReviewType, format_time, parse_time, parse_times
from review.store import ReviewStore
from review.types import InvalidOption, TargetMissing, get_review_type
from splits import signals as sg
from splits.store import SplitStore
from tests.fake_stash import FakeStash

DURATION = 5523.0
SIZE = 3 * 1024 ** 3


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


RT = SceneSplitsReviewType()


def long_scene(stash, oshash="osh1"):
    fid = stash.add_file("/m/long.mp4", SIZE, duration=DURATION, oshash=oshash)
    sid = stash.add_scene(fid, title="long one")
    return sid, fid


def cut(t, kind="black"):
    return {"t": t, "kind": kind, "evidence": f"black at {t}"}


def rec_for(db, sid, fid, *, cuts=(1030.0, 1942.0), keep_until=5400.0, confidence="high", kind="split",
            oshash="osh1", covering=None):
    details = {
        "file": {"id": fid, "oshash": oshash, "size": SIZE, "duration": DURATION},
        "cuts": [cut(t) for t in cuts], "keep_until": keep_until, "confidence": confidence,
        "reasons": ["blacks with silence"], "leftovers": [], "series": "s001", "planner_version": 1,
        "signals_version": 1, "coverage": [], "covering": covering or [], "kind": kind, "profile_used": False,
    }
    rid, _ = db.upsert_recommendation("scene_splits", "scene", sid, details, confidence=0.9)
    return db.get_recommendation(rid)


def cache_black(oshash, start):
    """Cached signals with one black run starting at `start`."""
    from jobs.content_fingerprint_job import _data_dir
    import os
    t = 0.4 + np.arange(0, 3000, 1 / 6)  # 1030.4 is on the frame grid
    y = np.full(t.shape, 100)
    y[(t >= start - 1e-9) & (t < start + 1.0)] = 5
    s = sg.Signals(t=t, y=y, sat=np.zeros_like(t), ydif=np.zeros_like(t), scd=np.zeros_like(t),
                   au_t=np.arange(0, 3000, 0.25), rms=np.full(12000, -30.0), sil=np.zeros((0, 2)),
                   duration=DURATION, covered=[[0.0, 3000.0]])
    st = SplitStore(os.path.join(_data_dir(), "splits.db"))
    st.put_signals(oshash, SIZE, sg.SIGNALS_VERSION, sg.encode(s))
    st.close()


def test_registered():
    assert isinstance(get_review_type("scene_splits"), SceneSplitsReviewType)


def test_time_text():
    assert parse_time("1:30:00") == 5400 and parse_time("17:10.5") == 1030.5 and parse_time("1030") == 1030
    assert parse_times("17:10, 22:57  1:30:00") == [1030, 1377, 5400]
    assert format_time(1030) == "17:10" and format_time(1030.4) == "17:10.40" and format_time(5400) == "1:30:00"
    assert format_time(3605.25) == "1:00:05.25" and format_time(59.999) == "1:00"
    for bad in ("abc", "1:2:3:4", "-5", "1:xx", "1.5:30"):
        with pytest.raises(ValueError):
            parse_time(bad)


async def test_default_option_is_the_plan(db, stash):
    sid, fid = long_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid, cuts=(1030.0, 1377.0)), None, stash)
    assert p.default_option == {"cuts": "17:10, 22:57", "keep_from": "", "keep_until": "1:30:00"}
    assert p.is_default_option and p.high_confidence
    ev = p.evidence
    assert [c["t"] for c in ev["cuts"]] == [1030.0, 1377.0] and ev["cuts"][0]["kind"] == "black"
    assert [s["t"] for s in ev["stills"]] == [1027.0, 1033.0, 1374.0, 1380.0]
    assert ev["playback_points"] == [1025.0, 1372.0]

    for conf in ("medium", "low"):
        p = await RT.preview(rec_for(db, sid, fid, confidence=conf), None, stash)
        assert not p.high_confidence
    p = await RT.preview(rec_for(db, sid, fid), {"cuts": "17:10"}, stash)
    assert not p.high_confidence


async def test_accept_builds_ranges(db, stash):
    sid, fid = long_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid), None, stash)
    sp = p.plan.split
    assert sp.ranges == [{"start": 0, "end": 1030}, {"start": 1030, "end": 1942}, {"start": 1942, "end": 5400}]
    assert sp.oshash == "osh1" and sp.file_id == fid and sp.scene_id == sid and sp.cuts == [1030, 1942]
    assert p.summary.startswith("split into 3 scenes")

    p = await RT.preview(rec_for(db, sid, fid), {"keep_from": "1:00"}, stash)
    assert p.plan.split.ranges[0] == {"start": 60, "end": 1030}


async def test_adjusted_time_snaps_to_black(db, stash):
    sid, fid = long_scene(stash)
    rec = rec_for(db, sid, fid)
    p = await RT.preview(rec, {"cuts": "17:12, 32:22"}, stash)
    assert p.plan.split.cuts == [1032, 1942]  # no store: stays

    cache_black("osh1", 1030.4)
    p = await RT.preview(rec, {"cuts": "17:12, 32:22"}, stash)
    assert p.plan.split.cuts == pytest.approx([1030.4, 1942])
    assert p.plan.option["cuts"] == "17:12, 32:22"  # the option is what was typed
    assert "snapped" in p.summary
    # a black more than 4 s before the typed time is not its black
    p = await RT.preview(rec, {"cuts": "17:20, 32:22"}, stash)
    assert p.plan.split.cuts[0] == 1040


async def test_leave_whole(db, stash):
    sid, fid = long_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid), {"cuts": "", "keep_until": ""}, stash)
    assert p.plan.split is None and p.summary == "leave whole"
    assert p.plan.is_empty


async def test_trim_without_cuts_is_a_handoff(db, stash):
    sid, fid = long_scene(stash)
    p = await RT.preview(rec_for(db, sid, fid, cuts=(), keep_until=None), {"keep_from": "0:30"}, stash)
    assert p.plan.split.ranges == [{"start": 30, "end": DURATION}] and p.plan.split.cuts == []
    p = await RT.preview(rec_for(db, sid, fid, cuts=(), keep_until=None), {"keep_until": "1:20:00"}, stash)
    assert p.plan.split.ranges == [{"start": 0, "end": 4800}]
    p = await RT.preview(rec_for(db, sid, fid, cuts=(), keep_until=None), None, stash)
    assert p.plan.split is None


@pytest.mark.parametrize("opt", [
    {"cuts": "32:22, 17:10"},            # decreasing
    {"cuts": "17:10, 17:10"},
    {"cuts": "0:00"},                     # at keep_from
    {"cuts": "1:31:00"},                  # past keep_until (1:30:00)
    {"cuts": "17:10, 17:30"},             # under 30 s apart
    {"cuts": "17:10, nope"},
    {"cuts": "17:10", "keep_from": "20:00"},
    {"keep_until": "2:00:00"},            # past the file
])
async def test_invalid_times(db, stash, opt):
    sid, fid = long_scene(stash)
    with pytest.raises(InvalidOption):
        await RT.preview(rec_for(db, sid, fid), opt, stash)


async def test_file_changed(db, stash):
    sid, fid = long_scene(stash, oshash="other")
    with pytest.raises(TargetMissing, match="changed"):
        await RT.preview(rec_for(db, sid, fid), None, stash)
    with pytest.raises(TargetMissing):
        await RT.preview(rec_for(db, "999", fid), None, stash)


async def test_redundant_card(db, stash):
    sid, fid = long_scene(stash)
    covering = [{"scene_id": "41", "file_id": "9", "start": 0.0, "end": 3000.0},
                {"scene_id": "42", "file_id": "10", "start": 3000.0, "end": 5523.0}]
    p = await RT.preview(rec_for(db, sid, fid, cuts=(), keep_until=None, kind="redundant", covering=covering),
                         None, stash)
    assert p.plan.is_empty and p.plan.split is None and p.choices == [] and p.high_confidence
    assert p.evidence["covering"] == covering and "41" in p.summary and "42" in p.summary


async def test_label_on_correction(svc, db, stash):
    sid, fid = long_scene(stash)
    rec = rec_for(db, sid, fid)
    p = await svc.preview(rec.id, {"cuts": "17:10, 30:00"})
    out = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[])
    assert out["entry"]["status"] == "applied" and out["label_id"] is not None
    done = db.get_recommendation(rec.id)
    assert done.resolution_details["split"]["ranges"][1] == {"start": 1030, "end": 1800}
    assert stash.writes == 0
    label = RT.label(rec, await RT.preview(rec, {"cuts": "17:10, 30:00"}, stash))
    assert label["suggested"]["cuts"] == "17:10, 32:22" and label["chosen"]["cuts"] == "17:10, 30:00"
    assert RT.label(rec, await RT.preview(rec, None, stash)) is None

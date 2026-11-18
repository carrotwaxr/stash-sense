"""The scene_splits analyzer (#131): signals extracted once and cached, one plan per candidate upserted in place."""

import json
import sqlite3
from functools import lru_cache

import pytest

import curation_settings
import media_exec
import media_library
from analyzers import scene_splits
from analyzers.scene_splits import SceneSplitsAnalyzer
from curation_settings import CurationSettings, Exclusion, ExclusionUnavailable
from media_exec import CommandResult
from recommendations_db import RecommendationsDB
from splits import planner
from splits import signals as sg
from splits.store import SplitStore
from tests.fake_stash import FakeStash

TYPE = "scene_splits"
MIN = 60
DURATION = 61 * MIN
BLACKS = (1200.0, 2400.0)      # the starts of two 2 s black runs, each with 3 s of silence


@lru_cache(maxsize=1)
def _ffmpeg_output():
    """What ffmpeg prints for a 61 min file with a black, silent gap at each of BLACKS."""
    video = []
    for i in range(int(DURATION * 6)):
        t = i / 6
        dark = any(b <= t < b + 2 for b in BLACKS)
        video.append(f"frame:{i}    pts:{i}       pts_time:{t:.6f}\nlavfi.scd.score=0.0\n"
                     f"lavfi.signalstats.YDIF={12.0 + i % 5}\nlavfi.signalstats.YAVG={5 if dark else 100}\n"
                     f"lavfi.signalstats.SATAVG=20.0\n")
    audio = []
    for i in range(int(DURATION * 4)):
        t = i * 0.25
        quiet = any(b <= t < b + 3 for b in BLACKS)
        audio.append(f"frame:{i}    pts:{i}       pts_time:{t}\n"
                     f"lavfi.astats.Overall.RMS_level={-120 if quiet else -30}\n")
    silence = "".join(f"[silencedetect @ 0x1] silence_start: {b}\n"
                      f"[silencedetect @ 0x1] silence_end: {b + 3} | silence_duration: 3\n" for b in BLACKS)
    return "".join(video).encode(), "".join(audio).encode(), silence.encode()


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
def stash():
    return FakeStash()


@pytest.fixture
def ffmpeg(monkeypatch):
    """Patches run_command; ``calls`` lists (pass, path) and ``fail`` holds paths whose ffmpeg exits non-zero."""
    video, audio, silence = _ffmpeg_output()
    calls, fail = [], set()

    async def run_command(cmd, timeout=None):
        assert timeout is not None and timeout >= 3 * 3600
        path = cmd[cmd.index("-i") + 1]
        kind = "video" if "-vf" in cmd else "audio"
        calls.append((kind, path))
        if path in fail:
            return CommandResult(cmd, 1, b"", b"Invalid data found")
        return CommandResult(cmd, 0, video if kind == "video" else audio, b"" if kind == "video" else silence)

    class Lib:
        def to_local(self, p):
            return p

    monkeypatch.setattr(media_exec, "run_command", run_command)
    monkeypatch.setattr(media_library, "get_media_library", lambda: Lib())
    return type("FF", (), {"calls": calls, "fail": fail})


@pytest.fixture(autouse=True)
def exclusion(monkeypatch):
    state = {"error": None}

    async def current(stash, **kw):
        if state["error"]:
            raise state["error"]
        return Exclusion(CurationSettings(), frozenset(), 0.0)

    monkeypatch.setattr(curation_settings, "current", current)
    return state


def add_scene(stash, tmp_path, name, minutes=61, title=None, oshash=None):
    path = tmp_path / f"{name}.mp4"
    path.write_bytes(b"x")
    fid = stash.add_file(str(path), 1000 + len(name), duration=minutes * MIN, oshash=oshash or f"h-{name}")
    return stash.add_scene(fid, title=title or name), fid, str(path)


async def run(stash, db, stop=False):
    analyzer = SceneSplitsAnalyzer(stash, db)
    if stop:
        analyzer.request_stop()
    result = await analyzer.run()
    return analyzer, result


def recs(db, status=None):
    return db.get_recommendations(type=TYPE, status=status, limit=1000)


async def test_extracts_once_and_caches(stash, db, ffmpeg, tmp_path):
    _, _, p1 = add_scene(stash, tmp_path, "one")
    _, _, p2 = add_scene(stash, tmp_path, "two")
    analyzer, _ = await run(stash, db)
    assert sorted(ffmpeg.calls) == sorted([("video", p1), ("audio", p1), ("video", p2), ("audio", p2)])
    assert analyzer.stats["extracted"] == 2
    ffmpeg.calls.clear()
    analyzer, _ = await run(stash, db)
    assert ffmpeg.calls == []
    assert analyzer.stats["cached"] == 2


async def test_upserts_one_rec_per_scene(stash, db, ffmpeg, tmp_path):
    sid, fid, _ = add_scene(stash, tmp_path, "one", title="Some Title Vol 2")
    await run(stash, db)
    (rec,) = recs(db)
    assert (rec.target_type, rec.target_id, rec.status) == ("scene", sid, "pending")
    d = rec.details
    assert set(d) == {"file", "cuts", "keep_until", "confidence", "reasons", "leftovers", "series", "profile_used",
                      "planner_version", "signals_version", "coverage", "covering", "kind"}
    assert d["file"] == {"id": fid, "oshash": "h-one", "size": 1003, "duration": DURATION}
    assert d["series"] == "some title"
    assert d["kind"] == "split"
    assert [round(c["t"]) for c in d["cuts"]] == [1200, 2400]
    assert d["planner_version"] == planner.PLANNER_VERSION and d["signals_version"] == sg.SIGNALS_VERSION
    assert d["coverage"] == [] and d["covering"] == []
    assert rec.confidence == {"high": 0.9, "medium": 0.6, "low": 0.3}[d["confidence"]]


async def test_rerun_leaves_plans_unchanged(stash, db, ffmpeg, tmp_path):
    add_scene(stash, tmp_path, "one")
    first, _ = await run(stash, db)
    assert first.stats["created"] == 1
    second, result = await run(stash, db)
    assert second.stats["unchanged"] == 1 and second.stats["updated"] == 0
    assert result.extra["rescored"] == {"confidence_changed": 0, "cuts_moved": 0}


async def test_accepted_plan_not_reproposed(stash, db, ffmpeg, tmp_path, monkeypatch):
    sid, _, _ = add_scene(stash, tmp_path, "one")
    await run(stash, db)
    (rec,) = recs(db)
    db.resolve_recommendation(rec.id, "applied", {"apply_log_id": 1, "option": {"cuts": ""}})
    real = planner.plan_file
    monkeypatch.setattr(scene_splits.planner, "plan_file",
                        lambda *a, **k: planner.SplitPlan([], None, "low", ["changed"], [], False))
    analyzer, _ = await run(stash, db)
    assert analyzer.stats["accepted_skipped"] == 1
    (again,) = recs(db)
    assert again.status == "resolved" and again.details == rec.details
    # a new file for the scene is proposed again
    stash.files[stash.scenes[sid]["file_ids"][0]]["fingerprints"][0]["value"] = "new-hash"
    monkeypatch.setattr(scene_splits.planner, "plan_file", real)
    analyzer, _ = await run(stash, db)
    assert analyzer.stats["reopened"] == 1
    assert recs(db)[0].status == "pending"


async def test_rescore_reports_moves(stash, db, ffmpeg, tmp_path, monkeypatch):
    add_scene(stash, tmp_path, "one")
    add_scene(stash, tmp_path, "two")
    await run(stash, db)
    real = planner.plan_file

    def moved(d, total, **kw):
        plan = real(d, total, **kw)
        plan.confidence = "low" if plan.confidence != "low" else "medium"
        plan.cuts = [dict(c, t=c["t"] + 5) for c in plan.cuts]
        return plan

    monkeypatch.setattr(scene_splits.planner, "plan_file", moved)
    analyzer, result = await run(stash, db)
    assert result.extra["rescored"] == {"confidence_changed": 2, "cuts_moved": 2}
    assert analyzer.stats["updated"] == 2


async def test_stale_when_no_longer_candidate(stash, db, ffmpeg, tmp_path):
    sid, fid, _ = add_scene(stash, tmp_path, "one")
    await run(stash, db)
    stash.files[fid]["duration"] = 59 * MIN
    await run(stash, db, stop=True)
    assert recs(db)[0].status == "pending"      # a stopped run retires nothing
    analyzer, _ = await run(stash, db)
    (rec,) = recs(db)
    assert (rec.status, rec.resolution_action) == ("resolved", "stale")
    assert analyzer.stats["stale"] == 1


async def test_dismissed_is_untouched(stash, db, ffmpeg, tmp_path):
    add_scene(stash, tmp_path, "one")
    await run(stash, db)
    (rec,) = recs(db)
    db.dismiss_recommendation(rec.id, "no")
    ffmpeg.calls.clear()
    analyzer, _ = await run(stash, db)
    assert analyzer.stats["skipped"] == 1 and ffmpeg.calls == []
    assert recs(db)[0].status == "dismissed"


async def test_failed_and_missing_files_are_counted_and_skipped(stash, db, ffmpeg, tmp_path):
    _, _, bad = add_scene(stash, tmp_path, "bad")
    add_scene(stash, tmp_path, "ok")
    stash.add_scene(stash.add_file("/nowhere/gone.mp4", 5, duration=7200, oshash="h-gone"))
    ffmpeg.fail.add(bad)
    analyzer, _ = await run(stash, db)
    assert analyzer.stats["failed"] == 1 and analyzer.stats["missing"] == 1
    assert len(recs(db)) == 1
    # a failure is not cached: it is tried again
    ffmpeg.calls.clear()
    await run(stash, db)
    assert ("video", bad) in ffmpeg.calls


async def test_redundant_candidate_has_no_cuts_and_no_extraction(stash, db, ffmpeg, tmp_path, monkeypatch):
    sid, fid, _ = add_scene(stash, tmp_path, "one")
    cand = scene_splits.cand_mod.SplitCandidate(sid, stash.files[fid], coverage=[(0.0, 3660.0)],
                                                covering=[{"scene_id": "99", "file_id": "98", "start": 0, "end": 3660}],
                                                redundant=True, title="one")

    async def generate(*a, **k):
        return [cand]

    monkeypatch.setattr(scene_splits.cand_mod, "generate", generate)
    await run(stash, db)
    assert ffmpeg.calls == []
    (rec,) = recs(db)
    assert rec.details["kind"] == "redundant" and rec.details["cuts"] == []
    assert rec.details["covering"] == cand.covering


FEAT = {"blk_len": 1.5, "blk_min": 5, "sil_len": 3.0, "dip_ratio": 0.1, "scd": 0.0}


async def test_seed_imported_and_profiles_used(stash, db, ffmpeg, tmp_path, data_dir, monkeypatch):
    labels = [{"series": "foo series", "key": k, "total": 3600, "cuts": [1200, 2400],
               "feats": [FEAT, FEAT], "source": "owner"} for k in ("1", "2")]
    (data_dir / "split_seed.json").write_text(json.dumps({"aliases": [["^foo.*", "foo series"]], "labels": labels}))
    add_scene(stash, tmp_path, "a", title="Foo Collection Vol 3")
    add_scene(stash, tmp_path, "b", title="Other Thing")
    seen = {}
    real = planner.plan_file

    def spy(d, total, **kw):
        seen.setdefault("profiles", []).append(kw.get("profile"))
        return real(d, total, **kw)

    monkeypatch.setattr(scene_splits.planner, "plan_file", spy)
    await run(stash, db)
    assert not (data_dir / "split_seed.json").exists() and (data_dir / "split_seed.json.imported").exists()
    by_title = {r.target_id: r.details for r in recs(db)}
    a, b = [by_title[s] for s in sorted(by_title, key=int)]
    assert (a["series"], a["profile_used"]) == ("foo series", True)
    assert (b["series"], b["profile_used"]) == ("other thing", False)
    assert sum(p is not None for p in seen["profiles"]) == 1


async def test_accepted_plans_become_labels_once(stash, db, ffmpeg, tmp_path, data_dir):
    add_scene(stash, tmp_path, "one", title="Foo Vol 1")
    await run(stash, db)
    (rec,) = recs(db)
    split = {"scene_id": rec.target_id, "file_id": "1", "oshash": "h-one", "size": 1003, "duration": DURATION,
             "ranges": [], "cuts": [1200.0, 2400.0]}
    db.resolve_recommendation(rec.id, "applied", {"apply_log_id": 1, "option": {"cuts": "20:00, 40:00"}, "split": split})
    leave = db.upsert_recommendation(TYPE, "scene", "999", {"series": "x"}, 0.3, None)[0]
    db.resolve_recommendation(leave, "applied", {"apply_log_id": 2, "option": {"cuts": ""}})
    for _ in range(2):
        analyzer, _ = await run(stash, db)
        assert analyzer.stats["accepted_labels"] == 1
    store = SplitStore(data_dir / "splits.db")
    rows = store._conn.execute("SELECT series, key, source, cuts_json, feats_json FROM series_labels").fetchall()
    store.close()
    assert [tuple(r)[:3] for r in rows] == [("foo", f"rec:{rec.id}", "accepted")]
    assert json.loads(rows[0]["cuts_json"]) == [1200.0, 2400.0]
    assert json.loads(rows[0]["feats_json"])[0]["blk_len"] > 0      # the cached signals gave the cut's features


async def test_exclusion_failure_fails_closed(stash, db, ffmpeg, tmp_path, exclusion):
    add_scene(stash, tmp_path, "one")
    exclusion["error"] = ExclusionUnavailable("down")
    with pytest.raises(ExclusionUnavailable):
        await run(stash, db)
    assert ffmpeg.calls == [] and recs(db) == []

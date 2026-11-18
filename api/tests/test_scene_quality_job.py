"""The scene_quality analyzer (#133): per-file evidence cached by content, flags raised per scene."""

import os

import pytest

import curation_settings
import media_library
from analyzers import scene_quality as quality_job
from analyzers.scene_quality import CONTROL_SAMPLE, SceneQualityAnalyzer
from content_fingerprint.store import ContentFpStore
from curation_settings import CurationSettings, Exclusion
from duplicates import health
from quality import probe, sharpness
from quality.store import QualityStore
from recommendations_db import RecommendationsDB
from tests.fake_stash import FakeStash

UNMAPPED = "/elsewhere/"


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    d.mkdir()
    monkeypatch.setenv("DATA_DIR", str(d))
    ContentFpStore(str(d / "content_fp.db")).close()
    return d


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "stash_sense.db")


@pytest.fixture
def stash():
    return FakeStash()


@pytest.fixture(autouse=True)
def library(monkeypatch):
    class Lib:
        def to_local(self, p):
            if p.startswith(UNMAPPED):
                raise ValueError("outside every MEDIA_PATH_MAP entry")
            return p

    monkeypatch.setattr(media_library, "get_media_library", lambda: Lib())


@pytest.fixture(autouse=True)
def exclusion(monkeypatch):
    state = {"settings": CurationSettings(quality_report_only=False), "scene_ids": frozenset()}

    async def current(stash, **kw):
        return Exclusion(state["settings"], state["scene_ids"], 0.0)

    monkeypatch.setattr(curation_settings, "current", current)
    return state


@pytest.fixture(autouse=True)
def checks(monkeypatch):
    """Fake cheap, deep and sharpness checks. ``cheap``/``deep`` map a path to a status (default ok / suspect),
    ``tags`` a path to its container tags; each ``*_calls`` lists the paths checked. A path not on disk goes to the
    real cheap check."""
    s = type("Checks", (), {})()
    s.cheap, s.deep, s.tags = {}, {}, {}
    s.cheap_calls, s.deep_calls, s.sharp_calls = [], [], []
    real_cheap = probe.cheap_check

    async def cheap_check(local, stash_duration, run=None):
        if not os.path.exists(local):
            return await real_cheap(local, stash_duration, run=run)
        s.cheap_calls.append(local)
        status = s.cheap.get(local, "ok")
        return probe._result(status, [] if status == "ok" else ["tail decode failed: fake"], tags=s.tags.get(local))

    async def check_health(local, points=24, run=None):
        s.deep_calls.append(local)
        status = s.deep.get(local, "suspect")
        return {"status": status, "reasons": [] if status == "ok" else ["3 of 24 windows fail"],
                "params": {"points": points}}

    async def measure(local, duration, run=None):
        s.sharp_calls.append(local)
        return {"blur": 7.5, "block": 2.0, "windows": 6, "version": sharpness.SHARPNESS_VERSION}

    monkeypatch.setattr(probe, "cheap_check", cheap_check)
    monkeypatch.setattr(health, "check_health", check_health)
    monkeypatch.setattr(sharpness, "measure", measure)
    return s


def add_scene(stash, tmp_path, name, *, on_disk=True, path=None, codec="h264", width=1920, height=1080,
              bit_rate=5_000_000, **fields):
    path = path or str(tmp_path / f"{name}.mp4")
    if on_disk:
        with open(path, "wb") as fh:
            fh.write(b"x")
    fid = stash.add_file(path, 1000 + len(name), duration=1800.0, oshash=f"h-{name}", codec=codec, width=width,
                         height=height)
    stash.files[fid]["bit_rate"] = bit_rate
    return stash.add_scene(fid, title=name, **fields), path


async def evaluate(stash, db, data_dir, settings=None, excluded=()):
    """collect + assemble on the stores in DATA_DIR: (analyzer, scene id -> details)."""
    settings = settings or CurationSettings()
    analyzer = SceneQualityAnalyzer(stash, db)
    excl = Exclusion(settings, frozenset(excluded), 0.0)
    qstore = QualityStore(str(data_dir / "quality.db"))
    content = ContentFpStore(str(data_dir / "content_fp.db"))
    try:
        rows = await analyzer.collect(excl, qstore, content)
        out = await analyzer.assemble(rows, settings, qstore)
    finally:
        qstore.close()
        content.close()
    return analyzer, out


async def test_broken_scene_flagged_and_deep_checked_once(stash, db, data_dir, checks, tmp_path):
    sid, path = add_scene(stash, tmp_path, "bad")
    checks.cheap[path] = "suspect"
    checks.deep[path] = "suspect"
    _, out = await evaluate(stash, db, data_dir)
    flags = out[sid]["flags"]
    assert set(flags) == {"broken"}
    assert flags["broken"]["status"] == "suspect"
    assert flags["broken"]["tier"] == "deep"
    assert flags["broken"]["reasons"] == ["3 of 24 windows fail"]
    assert out[sid]["scores"]["broken"] == "suspect"
    assert (checks.cheap_calls, checks.deep_calls) == ([path], [path])

    analyzer, out = await evaluate(stash, db, data_dir)
    assert set(out[sid]["flags"]) == {"broken"}
    assert (checks.cheap_calls, checks.deep_calls) == ([path], [path])
    assert analyzer.stats["cached"] == 1


async def test_cheap_ok_runs_no_deep_check(stash, db, data_dir, checks, tmp_path):
    sid, path = add_scene(stash, tmp_path, "fine")
    _, out = await evaluate(stash, db, data_dir)
    assert out[sid]["flags"] == {}
    assert out[sid]["scores"]["broken"] == "ok"
    assert checks.cheap_calls == [path]
    assert checks.deep_calls == []


async def test_below_studio_flag_gets_sharpness(stash, db, data_dir, checks, tmp_path):
    for i in range(9):
        add_scene(stash, tmp_path, f"peer{i}", studio_id="7", date="2020-03-01")
    weak, weak_path = add_scene(stash, tmp_path, "weak", width=640, height=360, bit_rate=400_000, studio_id="7",
                                date="2020-06-01")
    _, out = await evaluate(stash, db, data_dir)
    flags = out[weak]["flags"]
    assert set(flags) == {"below_studio"}
    ev = flags["below_studio"]
    assert ev["group"] == "studio+year" and ev["peers"] == 9
    assert ev["sharpness"]["blur"] == 7.5
    assert out[weak]["scores"]["below_studio"] == ev["score"]
    assert checks.sharp_calls == [weak_path]
    assert all(d["flags"] == {} and d["scores"]["below_studio"] is None for s, d in out.items() if s != weak)

    await evaluate(stash, db, data_dir)
    assert checks.sharp_calls == [weak_path]       # cached in quality.db


async def test_rip_from_tags(stash, db, data_dir, checks, tmp_path):
    rip, rip_path = add_scene(stash, tmp_path, "ripped")
    clean, _ = add_scene(stash, tmp_path, "clean")
    checks.tags[rip_path] = {"comment": "downloaded from spankbang.com"}
    _, out = await evaluate(stash, db, data_dir)
    ev = out[rip]["flags"]["rip"]
    assert [c["signal"] for c in ev["signals"]] == ["tag_tube"]
    assert ev["score"] == pytest.approx(0.9)
    assert set(out[rip]["flags"]) == {"rip"}
    assert out[clean]["flags"] == {}
    assert out[clean]["scores"]["rip"] == 0.0


async def test_format_only_when_nothing_else(stash, db, data_dir, checks, tmp_path):
    settings = CurationSettings(disfavored_video_codecs=("wmv3",))
    clean, _ = add_scene(stash, tmp_path, "clean", codec="wmv3")
    broken, broken_path = add_scene(stash, tmp_path, "broken", codec="wmv3")
    checks.cheap[broken_path] = "suspect"
    _, out = await evaluate(stash, db, data_dir, settings=settings)
    assert set(out[clean]["flags"]) == {"format"}
    assert out[clean]["flags"]["format"]["codec"] == "wmv3"
    assert set(out[broken]["flags"]) == {"broken"}


async def test_excluded_scene_is_skipped(stash, db, data_dir, checks, tmp_path):
    keep, _ = add_scene(stash, tmp_path, "keep")
    gone, gone_path = add_scene(stash, tmp_path, "gone")
    analyzer, out = await evaluate(stash, db, data_dir, excluded={gone})
    assert set(out) == {keep}
    assert gone_path not in checks.cheap_calls
    assert analyzer.stats["excluded"] == 1


async def test_missing_on_mount_is_broken_missing(stash, db, data_dir, checks, tmp_path):
    sid, path = add_scene(stash, tmp_path, "vanished", on_disk=False)
    _, out = await evaluate(stash, db, data_dir)
    assert set(out[sid]["flags"]) == {"broken"}
    assert out[sid]["flags"]["broken"]["status"] == "missing"
    assert checks.deep_calls == []


async def test_unmapped_path_is_counted_not_flagged(stash, db, data_dir, checks, tmp_path):
    sid, _ = add_scene(stash, tmp_path, "far", on_disk=False, path=UNMAPPED + "far.mp4")
    analyzer, out = await evaluate(stash, db, data_dir)
    assert sid not in out
    assert analyzer.stats["unmapped"] == 1
    assert checks.cheap_calls == []


async def test_details_carry_no_path(stash, db, data_dir, checks, tmp_path):
    sid, path = add_scene(stash, tmp_path, "bad")
    checks.cheap[path] = "suspect"
    _, out = await evaluate(stash, db, data_dir)
    d = out[sid]
    assert set(d["file"]) == {"id", "oshash", "size", "basename", "codec", "width", "height", "bit_rate",
                              "duration", "format"}
    assert d["control"] is False and d["quality_version"] == 1
    assert str(tmp_path) not in repr(d)


async def test_run_returns_counts(stash, db, checks, tmp_path):
    _, path = add_scene(stash, tmp_path, "bad")
    add_scene(stash, tmp_path, "fine")
    checks.cheap[path] = "suspect"
    result = await SceneQualityAnalyzer(stash, db).run()
    assert result.items_processed == 2
    assert result.recommendations_created == 1
    assert result.extra["flags"]["broken"] == 1
    assert result.extra["checks"]["probed"] == 2


# ---------------- run(): upserts, stale resolution, control sample, registration ----------------

TYPE = "scene_quality"


def recs(db, status=None):
    return db.get_recommendations(status=status, type=TYPE, target_type="scene", limit=1000)


async def run(stash, db, stop=False):
    analyzer = SceneQualityAnalyzer(stash, db)
    if stop:
        analyzer.request_stop()
    return analyzer, await analyzer.run()


async def test_one_rec_per_flagged_scene_upserted_in_place(stash, db, checks, tmp_path):
    sid, path = add_scene(stash, tmp_path, "bad")
    add_scene(stash, tmp_path, "fine")
    checks.cheap[path] = "suspect"
    first, result = await run(stash, db)
    assert first.stats["created"] == 1 and result.recommendations_created == 1
    (rec,) = recs(db)
    assert rec.target_id == sid and rec.confidence == 1.0
    assert set(rec.details["flags"]) == {"broken"}

    second, _ = await run(stash, db)
    assert second.stats["unchanged"] == 1 and second.stats["updated"] == 0
    assert [r.id for r in recs(db)] == [rec.id]

    stash.files[stash.scenes[sid]["file_ids"][0]]["bit_rate"] = 123
    third, _ = await run(stash, db)
    assert third.stats["updated"] == 1 and third.stats["created"] == 0
    (again,) = recs(db)
    assert again.id == rec.id and again.details["file"]["bit_rate"] == 123


async def test_unflagged_pending_rec_resolved_stale_only_when_finished(stash, db, checks, tmp_path):
    sid, path = add_scene(stash, tmp_path, "bad")
    checks.cheap[path] = "suspect"
    await run(stash, db)
    checks.cheap[path] = "ok"
    stash.files[stash.scenes[sid]["file_ids"][0]]["fingerprints"][0]["value"] = "h-new"    # a new file: re-probed
    await run(stash, db, stop=True)
    assert recs(db)[0].status == "pending"      # a stopped run retires nothing
    analyzer, _ = await run(stash, db)
    (rec,) = recs(db)
    assert (rec.status, rec.resolution_action) == ("resolved", "stale")
    assert rec.resolution_details == {"reason": "no quality flag on the current file"}
    assert analyzer.stats["stale"] == 1


async def test_dismissed_scene_untouched(stash, db, checks, tmp_path):
    _, path = add_scene(stash, tmp_path, "bad")
    checks.cheap[path] = "suspect"
    await run(stash, db)
    (rec,) = recs(db)
    db.dismiss_recommendation(rec.id, "no")
    analyzer, _ = await run(stash, db)
    assert analyzer.stats["dismissed"] == 1 and analyzer.stats["created"] == 0
    assert recs(db)[0].status == "dismissed"


async def test_control_sample_in_report_only(stash, db, exclusion, tmp_path):
    exclusion["settings"] = CurationSettings(quality_report_only=True)
    for i in range(50):
        add_scene(stash, tmp_path, f"clean{i}")
    first, result = await run(stash, db)
    controls = recs(db)
    assert len(controls) == CONTROL_SAMPLE == 20
    assert first.stats["controls"] == 20 and result.extra["controls"] == 20
    for r in controls:
        assert r.details["control"] is True and r.details["flags"] == {} and "scores" in r.details
        assert r.confidence == 0.0
    ids = sorted(r.id for r in controls)

    second, _ = await run(stash, db)
    assert second.stats["controls"] == 0
    assert sorted(r.id for r in recs(db)) == ids
    assert all(r.status == "pending" for r in recs(db))      # never resolved stale

    # a resolved control is not open any more: the next run draws a replacement from the unsampled scenes
    db.resolve_recommendation(controls[0].id, "applied", {})
    third, _ = await run(stash, db)
    assert third.stats["controls"] == 1
    assert len(recs(db)) == 21


async def test_no_control_sample_when_report_only_off(stash, db, exclusion, tmp_path):
    exclusion["settings"] = CurationSettings(quality_report_only=False)
    for i in range(5):
        add_scene(stash, tmp_path, f"clean{i}")
    await run(stash, db)
    assert recs(db) == []


def test_job_registered_hidden():
    from job_models import JOB_REGISTRY
    from recommendations_router import ANALYZERS
    assert JOB_REGISTRY["scene_quality"].production_ready is False
    assert ANALYZERS["scene_quality"] is SceneQualityAnalyzer


async def test_mass_missing_is_a_mount_problem_not_broken_files(stash, db, checks, tmp_path):
    """Most mapped files gone at once means the media mount is down: raise nothing for them and leave their
    open cards alone, rather than filing thousands of broken cards."""
    _, bad = add_scene(stash, tmp_path, "bad")
    checks.cheap[bad] = "suspect"
    gone = [add_scene(stash, tmp_path, f"gone-{i}", on_disk=False)[0] for i in range(quality_job.MISSING_GUARD_MIN + 1)]
    db.upsert_recommendation(type=TYPE, target_type="scene", target_id=gone[0],
                             details={"flags": {"broken": {"status": "suspect"}}, "control": False}, confidence=1.0)
    result = await SceneQualityAnalyzer(stash, db).run()
    assert result.extra["checks"]["missing_suppressed"] == len(gone)
    assert result.extra["flags"]["broken"] == 1           # the decode failure still counts
    assert {r.target_id for r in recs(db, "pending")} >= {gone[0]}
    assert not [r for r in recs(db, "pending") if r.target_id in gone and (r.details or {}).get("control")]


async def test_a_few_missing_files_are_still_broken(stash, db, checks, tmp_path):
    for i in range(30):
        add_scene(stash, tmp_path, f"fine-{i}")
    sid, _ = add_scene(stash, tmp_path, "vanished", on_disk=False)
    result = await SceneQualityAnalyzer(stash, db).run()
    assert result.extra["checks"]["missing_suppressed"] == 0
    assert sid in {r.target_id for r in recs(db, "pending")}

"""The content index job (#129): fixture acceptance with real ffmpeg, resume, exclusions, the CPU knobs, import.

The fixture set is synthetic but built like the library's real cases: notes and chords for audio (Chromaprint
fingerprints pitch content, so noise would not behave like music), shots of distinct pictures for video. Files are
made once per session.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import wave

import numpy as np
import pytest

import curation_settings
import media_exec
from base_job import JobContext
from content_fingerprint import (
    AUDIO_ALGO,
    AUDIO_ALGO_VERSION,
    VIDEO_ALGO,
    VIDEO_ALGO_VERSION,
    audio,
    importer,
    match,
    video,
)
from content_fingerprint.store import ContentFpStore
from jobs.content_fingerprint_job import (
    IMPORT_NAME,
    STORE_NAME,
    ContentFingerprintJob,
    library_files,
)
from media_library import MediaLibrary, PathMapping
from tests.fake_stash import FakeStash


def _has_chromaprint():
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-muxers"], capture_output=True, text=True, check=False).stdout
    return " chromaprint " in out


needs_ffmpeg = pytest.mark.skipif(not _has_chromaprint(), reason="ffmpeg without the chromaprint muxer")

SR = 22050
FPS = 10


# -- synthetic media ----------------------------------------------------------------------------------------------

def song(seed, seconds):
    """Notes of 0.2-0.6 s, each a three-note chord, with an envelope: music as far as Chromaprint can tell."""
    rng = np.random.default_rng(seed)
    out = np.zeros(int(seconds * SR), dtype=np.float32)
    t = 0
    while t < len(out):
        n = min(int(rng.uniform(0.2, 0.6) * SR), len(out) - t)
        tt = np.arange(n) / SR
        freqs = 110 * 2 ** (rng.integers(0, 36, size=3) / 12)
        out[t:t + n] = sum(np.sin(2 * np.pi * f * tt) for f in freqs) * np.hanning(n) / 3
        t += n
    return out


def frames(seed, seconds):
    """Shots of 2-5 s, each a random 6x4 colour grid drifting in brightness, as rgb24 frames at 160x96."""
    rng = np.random.default_rng(seed)
    total = int(seconds * FPS)
    out = np.zeros((total, 96, 160, 3), dtype=np.uint8)
    i = 0
    while i < total:
        n = min(int(rng.uniform(2, 5) * FPS), total - i)
        grid = rng.integers(20, 235, size=(4, 6, 3)).astype(np.float32)
        big = np.kron(grid, np.ones((24, 27, 1)))[:96, :160]
        for k in range(n):
            out[i + k] = np.clip(big * (0.9 + 0.2 * k / max(n, 1)), 0, 255).astype(np.uint8)
        i += n
    return out


def _wav(path, samples):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(samples, -1, 1) * 32000).astype("<i2").tobytes())


def _mux(out, video_frames, samples, vcodec_args=(), acodec_args=("-c:a", "aac", "-b:a", "128k"), tmp=None):
    wav = tmp / (out.name + ".wav")
    _wav(wav, samples)
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "160x96",
           "-r", str(FPS), "-i", "-", "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
           "-preset", "ultrafast", "-pix_fmt", "yuv420p", *vcodec_args, *acodec_args, "-shortest", str(out)]
    subprocess.run(cmd, input=video_frames.tobytes(), check=True)
    wav.unlink()
    return out


@pytest.fixture(scope="session")
def media(tmp_path_factory):
    """{name: path} of the fixture files under one directory."""
    d = tmp_path_factory.mktemp("library")
    a, v = song(1, 90), frames(1, 90)
    f = {}
    f["orig"] = _mux(d / "orig.mp4", v, a, ("-g", "50"), tmp=d)
    f["copy"] = d / "copy.mp4"
    shutil.copy(f["orig"], f["copy"])
    f["remux"] = d / "remux.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-i", str(f["orig"]), "-c", "copy", str(f["remux"])],
                   check=True)
    f["reenc"] = _mux(d / "reenc.mkv", v, a * 0.8, ("-g", "120", "-crf", "35"),
                      ("-c:a", "libmp3lame", "-b:a", "96k"), tmp=d)
    cut = int(17.3 * SR)
    f["trimmed"] = _mux(d / "trimmed.mp4", v[int(17.3 * FPS):], a[cut:], ("-g", "50"), tmp=d)
    f["long"] = _mux(d / "long.mp4", frames(5, 170), np.concatenate([song(2, 40), a, song(3, 40)]), tmp=d)
    f["dub"] = _mux(d / "dub.mp4", v, song(7, 90), ("-g", "80"), tmp=d)
    bed = song(9, 90) * 0.35
    f["bed1"] = _mux(d / "bed1.mp4", frames(11, 90), bed + song(4, 90), tmp=d)
    f["bed2"] = _mux(d / "bed2.mp4", frames(12, 90), bed + song(5, 90), tmp=d)
    f["unrelated"] = _mux(d / "unrelated.mp4", frames(6, 90), song(6, 90), tmp=d)
    return {k: str(p) for k, p in f.items()}


@pytest.fixture(scope="session")
def durations(media):
    return {n: float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
        capture_output=True, text=True, check=True).stdout) for n, p in media.items()}


def _oshash(path):
    """A stand-in for Stash's oshash: equal bytes, equal hash."""
    with open(path, "rb") as fh:
        return hashlib.sha1(fh.read()).hexdigest()[:16]


def _library(stash, paths, root):
    """One scene per file, as Stash sees it at /data/<name>; returns {name: scene id}."""
    scenes = {}
    for name, p in paths.items():
        fid = stash.add_file("/data/" + os.path.relpath(p, root), os.path.getsize(p), duration=None,
                             oshash=_oshash(p))
        scenes[name] = stash.add_scene(fid, title=name)
    return scenes


class _Db:
    """The two RecommendationsDB calls JobContext makes."""

    def __init__(self):
        self.progress, self.result = [], None

    def update_job_progress(self, job_id, items_processed=None, items_total=None, cursor=None):
        self.progress.append((items_processed, items_total))

    def set_job_result(self, job_id, result):
        self.result = result


def _context(stop_after=None):
    db = _Db()
    ctx = JobContext(job_id=1, db=db, queue_manager=None, job_type="content_fingerprint")
    if stop_after is not None:
        real = ctx.report_progress

        async def report(done, total=None):
            await real(done, total)
            if done >= stop_after:
                ctx.request_stop()
        ctx.report_progress = report
    return ctx, db


def _job(stash, root, data, workers=2, **kw):
    lib = MediaLibrary([PathMapping("/data", str(root))])
    return ContentFingerprintJob(stash=stash, media=lib, data_dir=str(data), workers=workers, **kw)


def _pairs(store, ids):
    """{frozenset(names): [overlap rows]} with each row's offset (time in the first-named file minus the second)."""
    names = {v: k for k, v in ids.items() if k != "copy"}   # the copy is orig's content row
    out = {}
    for r in store.overlaps(AUDIO_ALGO_VERSION):
        a, b = names.get(r["file_a"]), names.get(r["file_b"])
        out.setdefault(frozenset((a, b)), []).append(r)
    return out, names


def _ids(store, paths):
    by_key = {(f["oshash"], f["size_bytes"]): fid for fid, f in store.files().items()}
    return {n: by_key[(_oshash(p), os.path.getsize(p))] for n, p in paths.items()}


# -- acceptance -----------------------------------------------------------------------------------------------------

@needs_ffmpeg
async def test_fixture_set_finds_each_kind_of_shared_content(media, durations, tmp_path):
    root = os.path.dirname(media["orig"])
    stash = FakeStash()
    scenes = _library(stash, media, root)
    ctx, db = _context()
    assert await _job(stash, root, tmp_path).run(ctx) is None

    store = ContentFpStore(str(tmp_path / STORE_NAME))
    ids = _ids(store, {k: v for k, v in media.items()})
    # an exact copy is the same content: one row for both files, both scenes on it
    assert ids["copy"] == ids["orig"]
    files = await library_files(stash)
    assert sorted(next(f for f in files if f.path.endswith("/copy.mp4") or f.path.endswith("/orig.mp4")).scene_ids) \
        == sorted([scenes["orig"], scenes["copy"]])

    pairs, _ = _pairs(store, ids)
    dur = durations

    def kind(a, b):
        rows = pairs.get(frozenset((a, b)))
        if not rows:
            return None
        ra = [(r["a_start"], r["a_end"], r["b_start"], r["b_end"], r["ber"]) for r in rows]
        fa = next(n for n in (a, b) if ids[n] == rows[0]["file_a"])
        fb = b if fa == a else a
        return match.classify(ra, dur[fa], dur[fb])["kind"]

    def offset(a, b):
        """Seconds to add to a time in b to get the same moment in a."""
        r = max(pairs[frozenset((a, b))], key=lambda r: r["a_end"] - r["a_start"])
        off = r["a_start"] - r["b_start"]
        return off if ids[a] == r["file_a"] else -off

    assert kind("orig", "remux") == "duplicate"
    assert kind("orig", "reenc") == "duplicate"
    assert kind("orig", "trimmed") in ("duplicate", "contained")
    assert offset("orig", "trimmed") == pytest.approx(17.3, abs=0.3)
    assert kind("long", "orig") == "contained"
    assert offset("long", "orig") == pytest.approx(40.0, abs=0.3)
    # different audio over the same picture, and a music bed under different songs: no audio match
    assert frozenset(("orig", "dub")) not in pairs
    assert frozenset(("bed1", "bed2")) not in pairs
    assert not [p for p in pairs if "unrelated" in p]
    for r in (r for rows in pairs.values() for r in rows):
        assert r["ber"] <= 0.25

    # video: the dub is the same picture at offset 0; unrelated pictures don't vote at all
    sig = {n: store.get_video(ids[n], VIDEO_ALGO, VIDEO_ALGO_VERSION) for n in ("orig", "dub", "reenc", "unrelated")}
    assert all(s is not None and len(s[1]) for s in sig.values())
    dub = video.match(sig["dub"], sig["orig"])
    assert dub["offset_s"] == 0 and dub["fraction"] >= 0.25   # real duplicates 0.25-1.0, same-studio others <= 0.03
    assert video.match(sig["reenc"], sig["orig"])["offset_s"] == 0
    assert video.match(sig["unrelated"], sig["orig"]) is None

    assert db.result["fingerprint"]["missing"] == 0
    assert db.result["audio"] == {"ok": len(set(ids.values()))}
    store.close()


@needs_ffmpeg
async def test_second_run_repeats_nothing(media, tmp_path, monkeypatch):
    root = os.path.dirname(media["orig"])
    stash = FakeStash()
    _library(stash, {k: media[k] for k in ("orig", "trimmed", "unrelated")}, root)
    ctx, _ = _context()
    await _job(stash, root, tmp_path).run(ctx)

    calls = []
    real = media_exec.run_command

    async def counting(cmd, timeout=None):
        calls.append(cmd)
        return await real(cmd, timeout)
    monkeypatch.setattr(media_exec, "run_command", counting)
    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    assert calls == []
    assert db.result["fingerprint"]["to_do"] == 0 and db.result["match"]["to_query"] == 0


# -- resume, exclusions, knobs, mount (canned ffmpeg output) -------------------------------------------------------

def _seed(path):
    return int(hashlib.sha1(path.encode()).hexdigest()[:8], 16)


def _canned_fp(path):
    return np.random.default_rng(_seed(path)).integers(0, 2**32, size=400, dtype=np.uint32)


def _canned(calls):
    """run_command stand-in: a distinct fingerprint and one keyframe per file, recording each command."""
    async def run(cmd, timeout=None):
        calls.append(cmd)
        path = cmd[cmd.index("-i") + 1]
        seed = _seed(path)
        if "chromaprint" in cmd:
            return media_exec.CommandResult(cmd, 0, _canned_fp(path).astype("<u4").tobytes(), b"")
        frame = np.random.default_rng(seed).integers(0, 255, size=72, dtype=np.uint8)
        err = b"[Parsed_showinfo_3 @ 0x1] n:   0 pts:      0 pts_time:0       duration:1\n"
        return media_exec.CommandResult(cmd, 0, frame.tobytes(), err)
    return run


def _fake_library(tmp_path, n, stash=None):
    stash = stash or FakeStash()
    root = tmp_path / "media"
    root.mkdir(exist_ok=True)
    scenes = []
    for i in range(n):
        p = root / f"f{i}.mp4"
        p.write_bytes(b"x" * (i + 1))
        fid = stash.add_file(f"/data/f{i}.mp4", i + 1, oshash=f"h{i}")
        scenes.append(stash.add_scene(fid, title=f"s{i}", created_at=f"2026-01-{i + 1:02d}T00:00:00Z"))
    return stash, root, scenes


def _paths(calls):
    return [c[c.index("-i") + 1] for c in calls]


async def test_stop_and_restart_repeats_no_finished_file(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 10)
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    ctx, _ = _context(stop_after=4)
    await _job(stash, root, tmp_path).run(ctx)
    first = set(_paths(calls))
    store = ContentFpStore(str(tmp_path / STORE_NAME))
    finished = {f["last_path"] for fid, f in store.files().items()
                if fid in store.done_ids("audio_fingerprints", AUDIO_ALGO, AUDIO_ALGO_VERSION)
                and fid in store.done_ids("video_signatures", VIDEO_ALGO, VIDEO_ALGO_VERSION)}
    store.close()
    assert 4 <= len(finished) < 10

    calls.clear()
    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    again = {"/data/" + os.path.basename(p) for p in _paths(calls)}
    assert not again & finished
    assert len(finished) + len(again) == 10
    assert first   # the first run did run ffmpeg
    assert db.result["audio"] == {"ok": 10}


async def test_newest_scenes_first(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 5)
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    ctx, _ = _context()
    await _job(stash, root, tmp_path, workers=1).run(ctx)
    order = [os.path.basename(c[c.index("-i") + 1]) for c in calls if "chromaprint" in c]
    assert order == ["f4.mp4", "f3.mp4", "f2.mp4", "f1.mp4", "f0.mp4"]


def test_a_large_backlog_is_spread_and_a_small_run_is_newest_first():
    from jobs.content_fingerprint_job import LibraryFile, work_order
    # files added in folder batches, as an ingest writes them (on unRAID: one disk per batch)
    todo = [(LibraryFile(f"h{i}", i, f"/data/batch{i // 50}/f{i}.mp4", added=f"2026-01-01T{i // 50:02d}:{i % 50:02d}"),)
            for i in range(300)]
    small = work_order(todo[:40], spread_above=200)
    assert [t[0].oshash for t in small] == [f"h{i}" for i in range(39, -1, -1)]
    big = work_order(todo, spread_above=200)
    assert sorted(t[0].oshash for t in big) == sorted(t[0].oshash for t in todo)
    # any four consecutive picks (four workers) come from several batches, not one
    folders = [os.path.dirname(t[0].path) for t in big]
    windows = [len(set(folders[i:i + 4])) for i in range(len(folders) - 3)]
    assert sum(windows) / len(windows) >= 3


async def test_ffmpeg_runs_under_the_cpu_knobs(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 2)
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    monkeypatch.setattr(media_exec, "_setting", lambda key: {"media_job_nice": 15, "media_job_threads": 2}[key])
    ctx, _ = _context()
    await _job(stash, root, tmp_path).run(ctx)
    assert calls and all(c[:3] == ["nice", "-n", "15"] for c in calls)
    assert all(c[c.index("-i") - 2:c.index("-i")] == ["-threads", "2"] for c in calls)


async def test_excluded_scenes_are_skipped_including_shared_files(tmp_path, monkeypatch):
    stash, root, scenes = _fake_library(tmp_path, 3)
    private = stash.add_tag("private")
    stash.scenes[scenes[0]]["tag_ids"] = [private]
    shared = stash.scenes[scenes[1]]["file_ids"][0]
    stash.add_scene(shared, title="private copy", tag_ids=(private,))   # f1 also belongs to an excluded scene
    stash.plugin_config[curation_settings.PLUGIN_ID] = {curation_settings.KEY_TAGS: private}
    monkeypatch.setattr(curation_settings, "_cache", curation_settings.ExclusionCache())
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    assert {os.path.basename(p) for p in _paths(calls)} == {"f2.mp4"}
    assert db.result["excluded"] == 2


async def test_fails_closed_when_the_exclusion_list_cannot_be_read(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 2)
    stash.fail["get_plugin_configuration"] = RuntimeError("Stash down")
    monkeypatch.setattr(curation_settings, "_cache", curation_settings.ExclusionCache())
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    with pytest.raises(curation_settings.ExclusionUnavailable):
        await _job(stash, root, tmp_path).run(_context()[0])
    assert calls == []


async def test_a_missing_mount_aborts_instead_of_churning(tmp_path, monkeypatch):
    stash = FakeStash()
    for i in range(60):
        stash.add_scene(stash.add_file(f"/data/gone{i}.mp4", 10, oshash=f"g{i}"))
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    with pytest.raises(RuntimeError, match="MEDIA_PATH_MAP"):
        await _job(stash, tmp_path / "nowhere", tmp_path).run(_context()[0])
    assert calls == []


async def test_yields_to_a_higher_priority_job(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 3)
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    ctx, _ = _context()

    async def yes():
        return True
    ctx.should_yield = yes
    assert await _job(stash, root, tmp_path).run(ctx) == "fingerprint"
    assert calls == []


# -- import ----------------------------------------------------------------------------------------------------------

def _source_store(path, files, fp_for):
    src = ContentFpStore(str(path))
    for oshash, size, p in files:
        fid = src.upsert_file(oshash, size, path="/elsewhere/" + os.path.basename(p))
        src.put_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, {"ffmpeg": "6.1.1-test"}, "ok", fp_for(p))
    src.close()


async def test_import_copies_rows_after_the_build_check(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 4)
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    files = [(f"h{i}", i + 1, str(root / f"f{i}.mp4")) for i in range(4)]
    _source_store(tmp_path / IMPORT_NAME, files + [("gone", 99, "/x/gone.mp4")], _canned_fp)

    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    imp = db.result["import"]
    assert imp["status"] == "imported" and imp["imported"] == 4 and imp["in_library"] == 4
    assert imp["source_files"] == 5 and imp["checked"] >= 1
    assert (tmp_path / (IMPORT_NAME + ".imported")).exists() and not (tmp_path / IMPORT_NAME).exists()
    assert not [p for p in os.listdir(tmp_path) if p.startswith(IMPORT_NAME + "-")]   # no orphaned -shm / -wal
    # imported files get no new audio fingerprint, only the video signature and the check's sample
    audio_cmds = [c for c in calls if "chromaprint" in c]
    assert len(audio_cmds) == imp["checked"]
    store = ContentFpStore(str(tmp_path / STORE_NAME))
    assert store.counts(AUDIO_ALGO, AUDIO_ALGO_VERSION) == {"ok": 4}
    assert store.counts(VIDEO_ALGO, VIDEO_ALGO_VERSION, table="video_signatures") == {"ok": 4}
    store.close()


async def test_import_is_refused_when_this_ffmpeg_differs(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 4)
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    files = [(f"h{i}", i + 1, str(root / f"f{i}.mp4")) for i in range(4)]
    _source_store(tmp_path / IMPORT_NAME, files, lambda p: np.arange(400, dtype=np.uint32))   # another decoder

    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    imp = db.result["import"]
    assert imp["status"] == "refused"
    assert "fingerprint differently here" in imp["reason"] and "6.1.1-test" in imp["reason"]
    assert (tmp_path / (IMPORT_NAME + ".refused")).exists()
    # nothing copied: every file was fingerprinted here instead
    assert db.result["fingerprint"]["to_do"] == 4


def test_import_sample_covers_every_extension():
    paths = {(f"a{i}", i): f"/m/a{i}.mp4" for i in range(50)}
    paths.update({(f"w{i}", i): f"/m/w{i}.WMV" for i in range(2)})
    picked = importer.pick_sample(paths, per_ext=3, total=24)
    assert {paths[k][-4:].lower() for k in picked} == {".mp4", ".wmv"}
    assert sum(paths[k].lower().endswith(".wmv") for k in picked) == 2


def test_import_sample_reaches_late_extensions_when_there_are_many():
    exts = [".avi", ".flv", ".m4v", ".mkv", ".mov", ".mp4", ".mpg", ".ts", ".webm", ".wmv"]
    paths = {(f"{e}{i}", i): f"/m/f{i}{e}" for e in exts for i in range(5)}
    picked = importer.pick_sample(paths, per_ext=3, total=24)
    assert {os.path.splitext(paths[k])[1] for k in picked} == set(exts)
    assert len(picked) == 24


async def test_a_corrupt_import_file_is_set_aside_and_the_run_goes_on(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 2)
    (tmp_path / IMPORT_NAME).write_bytes(b"not a database at all" * 100)
    calls = []
    monkeypatch.setattr(media_exec, "run_command", _canned(calls))
    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    assert db.result["import"]["status"] == "refused"
    assert (tmp_path / (IMPORT_NAME + ".refused")).exists()
    assert db.result["audio"] == {"ok": 2}


async def test_a_timeout_is_not_stored_so_the_next_run_retries(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 2)
    calls = []
    canned = _canned(calls)

    async def slow_f0(cmd, timeout=None):
        if cmd[cmd.index("-i") + 1].endswith("f0.mp4"):
            raise TimeoutError
        return await canned(cmd, timeout)
    monkeypatch.setattr(media_exec, "run_command", slow_f0)
    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    assert db.result["fingerprint"]["timed_out"] == 1 and db.result["audio"] == {"ok": 1}

    monkeypatch.setattr(media_exec, "run_command", canned)
    calls.clear()
    ctx, db = _context()
    await _job(stash, root, tmp_path).run(ctx)
    assert {os.path.basename(p) for p in _paths(calls)} == {"f0.mp4"}
    assert db.result["audio"] == {"ok": 2}


async def test_repeated_unexpected_errors_stop_the_run(tmp_path, monkeypatch):
    stash, root, _ = _fake_library(tmp_path, 30)

    async def no_ffmpeg(cmd, timeout=None):
        raise FileNotFoundError("ffmpeg")
    monkeypatch.setattr(media_exec, "run_command", no_ffmpeg)
    with pytest.raises(RuntimeError, match="failed unexpectedly"):
        await _job(stash, root, tmp_path, workers=1).run(_context()[0])

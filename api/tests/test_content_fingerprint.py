"""Content fingerprint store, Chromaprint audio fingerprints, path mapping and the resumable worker."""

import shutil
import subprocess
import threading

import numpy as np
import pytest

from content_fingerprint import AUDIO_ALGO, AUDIO_ALGO_VERSION, audio, stash_source, worker
from content_fingerprint.store import ContentFpStore


def _has_chromaprint():
    if not shutil.which("ffmpeg"):
        return False
    out = subprocess.run(["ffmpeg", "-hide_banner", "-muxers"], capture_output=True, text=True).stdout
    return " chromaprint " in out


needs_chromaprint = pytest.mark.skipif(not _has_chromaprint(), reason="ffmpeg without the chromaprint muxer")


def _make(tmp_path, name, audio_src=None, seconds=20):
    """A small test file: testsrc video, optionally with an audio source (an ffmpeg lavfi expression)."""
    out = tmp_path / name
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", f"testsrc=size=64x48:rate=10:duration={seconds}"]
    if audio_src:
        cmd += ["-f", "lavfi", "-i", f"{audio_src}:duration={seconds}" if "duration" not in audio_src else audio_src]
    cmd += ["-c:v", "libx264", "-preset", "ultrafast"] + (["-c:a", "aac"] if audio_src else []) + ["-shortest", str(out)]
    subprocess.run(cmd, check=True)
    return str(out)


# -- store --------------------------------------------------------------------------------------------------------

def test_store_round_trip_and_reopen(tmp_path):
    db = str(tmp_path / "fp.db")
    s = ContentFpStore(db)
    fid = s.upsert_file("abc", 100, path="/x/a.mp4")
    assert s.upsert_file("abc", 100, md5="m") == fid          # same content -> same row, md5 filled in
    assert not s.has_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION)
    fp = np.array([1, 2, 0xFFFFFFFF], dtype=np.uint32)
    s.put_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, {"ffmpeg": "t"}, "ok", fp)
    s.close()
    s = ContentFpStore(db)                                     # reopen: schema already there
    assert s.has_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION)
    assert s.get_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION).tolist() == fp.tolist()
    row = s._conn.execute("SELECT md5, last_path FROM content_files WHERE id = ?", (fid,)).fetchone()
    assert (row[0], row[1]) == ("m", "/x/a.mp4")


def test_failure_never_replaces_an_ok_fingerprint(tmp_path):
    s = ContentFpStore(str(tmp_path / "fp.db"))
    fid = s.upsert_file("abc", 100)
    s.put_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, {}, "ok", np.array([7], dtype=np.uint32))
    s.put_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, {}, "failed", error="boom")
    assert s.counts(AUDIO_ALGO, AUDIO_ALGO_VERSION) == {"ok": 1}


def test_failed_is_retried_only_when_asked(tmp_path):
    s = ContentFpStore(str(tmp_path / "fp.db"))
    fid = s.upsert_file("abc", 100)
    s.put_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, {}, "failed", error="boom")
    assert s.has_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION)
    assert not s.has_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, retry_failed=True)


# -- audio --------------------------------------------------------------------------------------------------------

@needs_chromaprint
def test_fingerprint_is_deterministic_and_about_8_items_a_second(tmp_path):
    f = _make(tmp_path, "a.mp4", "anoisesrc=color=pink:seed=7:amplitude=0.5")
    s1, fp1, _ = audio.fingerprint(f)
    s2, fp2, _ = audio.fingerprint(f)
    assert s1 == s2 == "ok"
    assert fp1.tolist() == fp2.tolist()
    assert 120 <= len(fp1) <= 200                              # 20 s at ~8.07 items/s


def _ber(a, b):
    n = min(len(a), len(b))
    return np.unpackbits((a[:n] ^ b[:n]).view(np.uint8)).mean()


@needs_chromaprint
def test_same_audio_reencoded_matches_and_different_audio_does_not(tmp_path):
    """The property everything rests on (calibration 2026-10-22: AAC vs MP3 of one source 0.046, unrelated 0.45+)."""
    noise = "anoisesrc=color=pink:seed=7:amplitude=0.5"
    aac = tmp_path / "aac.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", f"{noise}:duration=20",
                    "-c:a", "aac", str(aac)], check=True)
    mp3 = tmp_path / "mp3.mkv"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", f"{noise}:duration=20",
                    "-c:a", "libmp3lame", str(mp3)], check=True)
    chirp = _make(tmp_path, "chirp.mp4", "aevalsrc='0.5*sin(2*PI*(200+400*mod(t,4))*t)':s=44100:d=20")
    _, a, _ = audio.fingerprint(str(aac))
    _, b, _ = audio.fingerprint(str(mp3))
    _, c, _ = audio.fingerprint(chirp)
    assert _ber(a, b) < 0.15
    assert _ber(a, c) > 0.3


@needs_chromaprint
def test_file_without_audio_is_no_audio_not_failed(tmp_path):
    status, fp, _ = audio.fingerprint(_make(tmp_path, "silent.mp4"))
    assert status == "no_audio" and fp is None


def test_missing_file_is_failed(tmp_path):
    if not shutil.which("ffmpeg"):
        pytest.skip("no ffmpeg")
    status, fp, err = audio.fingerprint(str(tmp_path / "nope.mp4"))
    assert status == "failed" and fp is None and err


# -- paths --------------------------------------------------------------------------------------------------------

def test_mount_map_longest_prefix_wins_and_respects_boundaries():
    m = stash_source.parse_mount_map(["/data=/mnt/stash", "/data/tagged=/mnt/tagged"])
    assert stash_source.map_path("/data/tagged/x.mp4", m) == "/mnt/tagged/x.mp4"
    assert stash_source.map_path("/data/unsorted/y.mp4", m) == "/mnt/stash/unsorted/y.mp4"
    assert stash_source.map_path("/database/z.mp4", m) == "/database/z.mp4"   # not under /data


def test_bad_mount_map_is_refused():
    with pytest.raises(ValueError):
        stash_source.parse_mount_map(["/data"])


def test_list_source_maps_paths(tmp_path):
    lst = tmp_path / "files.jsonl"
    lst.write_text('{"oshash": "a", "size": 5, "path": "/data/x.mp4", "scene_ids": ["9"]}\n\n')
    recs = stash_source.from_list(str(lst), stash_source.parse_mount_map(["/data=/mnt/stash"]))
    assert [(r.oshash, r.size, r.path, r.scene_ids) for r in recs] == [("a", 5, "/mnt/stash/x.mp4", ["9"])]


# -- worker -------------------------------------------------------------------------------------------------------

def _recs(n):
    return [stash_source.FileRecord(f"h{i}", 10 + i, f"/f{i}.mp4") for i in range(n)]


def test_worker_fingerprints_then_skips_on_rerun(tmp_path):
    s = ContentFpStore(str(tmp_path / "fp.db"))
    calls = []
    def fake(path):
        calls.append(path)
        return ("ok", np.array([len(path)], dtype=np.uint32), None) if path != "/f2.mp4" else ("no_audio", None, "none")
    st = worker.run(_recs(4), s, workers=2, fingerprint_fn=fake, params={"t": 1})
    assert (st["done"], st["ok"], st["no_audio"], st["skipped"]) == (4, 3, 1, 0)
    st = worker.run(_recs(5), s, workers=2, fingerprint_fn=fake, params={"t": 1})
    assert (st["done"], st["skipped"]) == (1, 4)               # only the new file
    assert sorted(calls) == ["/f0.mp4", "/f1.mp4", "/f2.mp4", "/f3.mp4", "/f4.mp4"]


def test_worker_stop_leaves_the_rest_for_the_next_run(tmp_path):
    s = ContentFpStore(str(tmp_path / "fp.db"))
    stop = threading.Event()
    def fake(path):
        stop.set()                                             # stop after the first file
        return "ok", np.array([1], dtype=np.uint32), None
    st = worker.run(_recs(6), s, workers=1, fingerprint_fn=fake, params={}, stop=stop)
    assert st["done"] == 1
    st = worker.run(_recs(6), s, workers=1, fingerprint_fn=lambda p: ("ok", np.array([1], dtype=np.uint32), None), params={})
    assert (st["done"], st["skipped"]) == (5, 1)


def test_worker_limit(tmp_path):
    s = ContentFpStore(str(tmp_path / "fp.db"))
    st = worker.run(_recs(5), s, fingerprint_fn=lambda p: ("ok", np.array([1], dtype=np.uint32), None), params={}, limit=2)
    assert (st["done"], st["skipped"]) == (2, 0)              # held back by --limit is not "already fingerprinted"
    st = worker.run(_recs(5), s, fingerprint_fn=lambda p: ("ok", np.array([1], dtype=np.uint32), None), params={})
    assert (st["done"], st["skipped"]) == (3, 2)

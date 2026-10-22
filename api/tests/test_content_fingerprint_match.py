"""Finding shared audio: the index, offset voting, run verification, and the end-to-end match on real encodes."""

import shutil
import subprocess

import numpy as np
import pytest

from content_fingerprint import AUDIO_ALGO_VERSION, audio, index as fpindex, match
from content_fingerprint.store import ContentFpStore

rng = np.random.default_rng(7)


def _noisy(fp, ber):
    """``fp`` with each bit flipped at probability ``ber`` (a re-encode)."""
    flips = rng.random((len(fp), 32)) < ber
    mask = np.packbits(flips, axis=1, bitorder="little").view("<u4").ravel()
    return fp ^ mask


# -- synthetic (no ffmpeg) ------------------------------------------------------------------------------------------

def test_clip_is_found_inside_a_longer_file_at_its_offset():
    long = rng.integers(0, 2**32, 2000, dtype=np.uint32)
    clip = _noisy(long[700:1100], 0.05)
    other = rng.integers(0, 2**32, 1500, dtype=np.uint32)
    idx = fpindex.build([(1, long), (2, other)])
    cands = fpindex.candidates(idx, clip, min_votes=10)
    assert cands and cands[0][0] == 1 and abs(cands[0][1] - 700) <= 2
    off, ber = fpindex.best_offset(clip, long, cands[0][1])
    assert off == 700 and ber < 0.1
    runs = fpindex.matched_runs(clip, long, off)
    assert len(runs) == 1
    a0, a1, b0, b1, _ = runs[0]
    assert (a0, b0) == (0, 700) and a1 >= 384                  # whole clip, to window resolution


def test_unrelated_audio_gives_no_candidates():
    a = rng.integers(0, 2**32, 2000, dtype=np.uint32)
    b = rng.integers(0, 2**32, 2000, dtype=np.uint32)
    assert fpindex.candidates(fpindex.build([(1, a)]), b) == []


def test_common_values_are_dropped_from_the_index():
    silence = np.full(400, 12345, dtype=np.uint32)              # the same value everywhere, in every file
    entries = [(i, np.concatenate([silence, rng.integers(0, 2**32, 400, dtype=np.uint32)])) for i in range(60)]
    idx = fpindex.build(entries, max_files_per_value=50)
    assert 12345 not in set(idx.values.tolist())


def test_two_separate_shared_pieces_are_two_runs():
    a = rng.integers(0, 2**32, 3000, dtype=np.uint32)
    b = a.copy()
    b[1000:1600] = rng.integers(0, 2**32, 600, dtype=np.uint32)  # a different middle (a re-cut)
    runs = fpindex.matched_runs(a, b, 0)
    assert len(runs) == 2
    assert runs[0][1] <= 1024 and runs[1][0] >= 1568


def test_a_short_glitch_does_not_split_a_run():
    a = rng.integers(0, 2**32, 2000, dtype=np.uint32)
    b = a.copy()
    b[960:990] = rng.integers(0, 2**32, 30, dtype=np.uint32)      # under one window of garbage
    assert len(fpindex.matched_runs(a, b, 0, gap=1)) == 1


def test_index_round_trips_through_disk(tmp_path):
    idx = fpindex.build([(3, rng.integers(0, 2**32, 500, dtype=np.uint32))])
    idx.save(str(tmp_path / "ix"))
    back = fpindex.Index.load(str(tmp_path / "ix"))
    assert back.values.tolist() == idx.values.tolist() and back.file_ids.tolist() == idx.file_ids.tolist()


# -- real encodes ---------------------------------------------------------------------------------------------------

def _has_chromaprint():
    if not shutil.which("ffmpeg"):
        return False
    return " chromaprint " in subprocess.run(["ffmpeg", "-hide_banner", "-muxers"], capture_output=True, text=True).stdout


@pytest.mark.skipif(not _has_chromaprint(), reason="ffmpeg without the chromaprint muxer")
def test_end_to_end_clip_inside_long_file(tmp_path):
    src = "anoisesrc=color=pink:seed=7:amplitude=0.5:duration=90,volume='0.4+0.35*sin(0.7*t)+0.2*sin(2.3*t)':eval=frame"
    long = tmp_path / "long.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i", src, "-c:a", "aac", str(long)], check=True)
    clip = tmp_path / "clip.mkv"                                 # seconds 30-60, re-encoded to MP3
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-ss", "30", "-t", "30", "-i", str(long),
                    "-c:a", "libmp3lame", str(clip)], check=True)
    other = tmp_path / "other.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                    "anoisesrc=color=brown:seed=99:amplitude=0.5:duration=60", "-c:a", "aac", str(other)], check=True)
    store = ContentFpStore(str(tmp_path / "fp.db"))
    ids = {}
    for name, path in (("long", long), ("clip", clip), ("other", other)):
        ids[name] = store.upsert_file(name, 1, path=str(path))
        status, fp, _ = audio.fingerprint(str(path))
        store.put_audio(ids[name], "chromaprint_raw", AUDIO_ALGO_VERSION, {}, status, fp)
    fps = dict(store.all_audio("chromaprint_raw", AUDIO_ALGO_VERSION))
    idx = fpindex.build(list(fps.items()))
    rows = match.overlaps_for(ids["clip"], fps[ids["clip"]], idx, fps, min_overlap_s=15)
    assert [r[1] for r in rows] == [ids["long"]]
    _, _, a0, a1, b0, b1, ber = rows[0]
    # 4 s windows: the MP3's padded tail can cost the last one
    assert a0 < 4.5 and a1 - a0 > 20 and abs(b0 - (30 + a0)) < 1.0 and ber < 0.2
    assert match.overlaps_for(ids["other"], fps[ids["other"]], idx, fps) == []
    store.put_overlaps(AUDIO_ALGO_VERSION, rows)
    stored = store.overlaps(AUDIO_ALGO_VERSION, ids["long"])
    assert len(stored) == 1 and stored[0]["file_a"] < stored[0]["file_b"]


@pytest.mark.parametrize("rows, da, db, kind", [
    ([(0, 1190, 0, 1190, 0.02)], 1200, 1210, "duplicate"),
    ([(0, 600, 1800, 2400, 0.05)], 610, 4300, "contained"),
    ([(100, 400, 900, 1200, 0.05)], 1500, 3000, "partial"),        # 5 min shared, a fifth of the shorter file
    ([(0, 18, 0, 18, 0.03)], 1500, 1600, "incidental"),           # a studio intro sting
    ([(160, 240, 43, 123, 0.09), (168, 240, 44, 115, 0.09)], 1100, 780, "incidental"),  # a looped music bed
    ([(i * 100, i * 100 + 18.5, i * 90 + 7, i * 90 + 25.5, 0.1) for i in range(12)], 1900, 1500, "incidental"),  # 12 scraps of music
])
def test_classify(rows, da, db, kind):
    assert match.classify(rows, da, db)["kind"] == kind


def test_classify_counts_repeated_pieces_once():
    c = match.classify([(0, 600, 0, 600, 0.0), (0, 600, 10, 610, 0.1)], 600, 610)
    assert c["cover_a"] == 1.0 and c["shared_s"] == 600

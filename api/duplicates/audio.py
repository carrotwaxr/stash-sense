"""Audio bit error rate between two files, for verifying duplicate candidates (#130).

Chromaprint raw items (uint32, one per ITEM_SEC) of two copies of the same recording differ in few bits,
whatever the container or video codec; unrelated audio differs in about half of them. ``audio_ber`` slides one
print over the other and reports the best error rate and where it lines up (a trimmed or re-cut copy is offset).

Prints come from the content index when it has reached the file (``stored_fp``, cut to the first
``AUDIO_SECONDS`` with ``head_items``), otherwise from a 5-minute ffmpeg run (``fingerprint_head``) that uses the
index's chromaprint algorithm so both kinds compare.

Ported from stash-curator's ``mergetool/dupecheck.py`` (thresholds validated 2026-10-13; re-validate before
changing), with the offset search vectorized per shift.
"""
from __future__ import annotations

import numpy as np

from content_fingerprint import AUDIO_ALGO, AUDIO_ALGO_VERSION
from content_fingerprint.audio import CHROMAPRINT_ALGORITHM
from media_exec import run_command

# --- tuned constants (validated 2026-10-13; re-validate before changing) ---
AUDIO_SECONDS = 300      # head of each file that is compared
ITEM_SEC = 0.1238        # seconds of audio per chromaprint item
MAX_SHIFT = 240.0        # largest offset searched, seconds either way
MIN_OVERLAP = 200        # items that must overlap for an offset to count (~25 s)
MIN_DISTINCT = 0.2       # fraction of distinct items below which a print is near-silent

FFMPEG_TIMEOUT = 600

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _bit_errors(x: np.ndarray, y: np.ndarray) -> int:
    """Number of differing bits between two equal-length uint32 arrays."""
    diff = np.bitwise_xor(x, y)
    return int(_POPCOUNT8[diff.view(np.uint8)].sum(dtype=np.int64))


def audio_ber(a: np.ndarray, b: np.ndarray, max_shift_sec: float = MAX_SHIFT,
              min_overlap: int = MIN_OVERLAP) -> tuple[float, float] | None:
    """``(best bit-error rate, offset in seconds)`` over all shifts, or None when it says nothing.

    A positive offset means ``b`` starts that many seconds into ``a`` (``a[off:]`` lines up with ``b``); a negative
    one means ``a`` starts that far into ``b``. None when either print is shorter than ``min_overlap`` items or
    near-silent (few distinct items: a low error rate between two silences proves nothing), or no shift overlaps
    enough.
    """
    a = np.ascontiguousarray(a, dtype=np.uint32)
    b = np.ascontiguousarray(b, dtype=np.uint32)
    if len(a) < min_overlap or len(b) < min_overlap:
        return None
    if min(len(np.unique(a)) / len(a), len(np.unique(b)) / len(b)) < MIN_DISTINCT:
        return None
    m = int(max_shift_sec / ITEM_SEC)
    best, best_off = 1.0, None
    for off in range(-m, m + 1):
        x, y = (a[off:], b) if off >= 0 else (a, b[-off:])
        n = min(len(x), len(y))
        if n < min_overlap:
            continue
        e = _bit_errors(x[:n], y[:n]) / (32 * n)
        if e < best:
            best, best_off = e, off
    if best_off is None:
        return None
    return round(best, 3), round(best_off * ITEM_SEC, 1)


def head_items(fp: np.ndarray, seconds: float = AUDIO_SECONDS) -> np.ndarray:
    """The items covering the first ``seconds`` of a print."""
    return fp[:int(seconds / ITEM_SEC)]


def fast_command(local_path: str, seconds: int = AUDIO_SECONDS) -> list[str]:
    """ffmpeg command for the first ``seconds`` of audio, with the content index's chromaprint algorithm."""
    return ["ffmpeg", "-v", "error", "-nostdin", "-t", str(seconds), "-i", local_path, "-vn", "-map", "0:a:0",
            "-ac", "1", "-f", "chromaprint", "-algorithm", str(CHROMAPRINT_ALGORITHM), "-fp_format", "raw", "-"]


async def fingerprint_head(local_path: str) -> np.ndarray | None:
    """Raw chromaprint items of the first ``AUDIO_SECONDS`` of a file, for files the content index hasn't reached.

    ``local_path`` is a path on the sidecar (``get_media_library().to_local(stash_path)``). None when ffmpeg fails,
    the file has no audio, or the output is empty; a failure is never an empty print.
    """
    try:
        res = await run_command(fast_command(local_path), timeout=FFMPEG_TIMEOUT)
    except (OSError, TimeoutError):
        return None
    if res.returncode != 0 or not res.stdout or len(res.stdout) % 4:
        return None
    return np.frombuffer(res.stdout, dtype="<u4").astype(np.uint32)


def stored_fp(store, oshash: str, size_bytes: int) -> np.ndarray | None:
    """The content index's ok audio print for this content (oshash + size), or None when it has none."""
    row = store._one("SELECT id FROM content_files WHERE oshash = ? AND size_bytes = ?", (oshash, size_bytes))
    if row is None:
        return None
    fp = store.get_audio(row[0], AUDIO_ALGO, AUDIO_ALGO_VERSION)
    return None if fp is None else fp.astype(np.uint32)

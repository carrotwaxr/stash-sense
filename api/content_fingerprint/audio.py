"""Chromaprint raw fingerprints of a whole audio track, through ffmpeg's chromaprint muxer.

Verified identical across ffmpeg 6.1.1 and 8.0.1 on AAC, WMA and MP2 audio (2026-10-22), so stores built with
either build can be merged.
"""

import functools
import subprocess

import numpy as np

CHROMAPRINT_ALGORITHM = 1   # the chromaprint muxer's default (CHROMAPRINT_ALGORITHM_TEST2)
_NO_AUDIO = ("does not contain any stream", "matches no streams", "Output file is empty")


@functools.lru_cache(maxsize=None)
def ffmpeg_version(ffmpeg="ffmpeg"):
    out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True).stdout
    return out.split()[2] if out.startswith("ffmpeg version") else "unknown"


def params(ffmpeg="ffmpeg"):
    """What a fingerprint depends on, stored with every row."""
    return {"ffmpeg": ffmpeg_version(ffmpeg), "chromaprint_algorithm": CHROMAPRINT_ALGORITHM, "channels": 1}


def fingerprint(path, ffmpeg="ffmpeg", timeout=None):
    """(status, fingerprint, error) for the audio track of ``path``.

    status is "ok" (fingerprint is a uint32 array), "no_audio" (the file has no audio track) or "failed".
    """
    cmd = [ffmpeg, "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0", "-ac", "1",
           "-f", "chromaprint", "-algorithm", str(CHROMAPRINT_ALGORITHM), "-fp_format", "raw", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "failed", None, f"timed out after {timeout} s"
    err = r.stderr.decode(errors="replace").strip()
    if any(s in err for s in _NO_AUDIO):
        return "no_audio", None, err[-500:] or None
    if r.returncode != 0:
        return "failed", None, err[-500:] or f"ffmpeg exit {r.returncode}"
    if len(r.stdout) % 4:
        return "failed", None, f"raw output of {len(r.stdout)} bytes is not whole uint32 values"
    fp = np.frombuffer(r.stdout, dtype="<u4")
    if not len(fp):
        return "no_audio", None, err[-500:] or "empty fingerprint"
    return "ok", fp, None

"""Chromaprint raw fingerprints of a whole audio track, through ffmpeg's chromaprint muxer.

Byte-identical between ffmpeg 6.1.1 and 8.0.1 on AAC, WMA and MP2 audio (2026-10-22, rechecked on WMA 2026-11-17).
ffmpeg 4.4.2 is not: its WMA decoder differs (7-11% of bits on every WMA file sampled) and AAC differs by a few bits
per million. Stores built with different builds only merge after the import's build-equality check.

``command`` and ``parse`` are split so the sidecar job can run the command its own way (lower priority, thread cap);
``fingerprint`` runs it directly for the standalone CLI.
"""

import functools
import subprocess

import numpy as np

CHROMAPRINT_ALGORITHM = 1   # the chromaprint muxer's default (CHROMAPRINT_ALGORITHM_TEST2)
_NO_AUDIO = ("does not contain any stream", "matches no streams", "Output file is empty")


@functools.lru_cache(maxsize=None)
def ffmpeg_version(ffmpeg="ffmpeg"):
    try:
        out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return "unknown"
    return out.split()[2] if out.startswith("ffmpeg version") else "unknown"


def params(ffmpeg="ffmpeg"):
    """What a fingerprint depends on, stored with every row."""
    return {"ffmpeg": ffmpeg_version(ffmpeg), "chromaprint_algorithm": CHROMAPRINT_ALGORITHM, "channels": 1}


def command(path, ffmpeg="ffmpeg"):
    return [ffmpeg, "-v", "error", "-nostdin", "-i", path, "-map", "0:a:0", "-ac", "1",
            "-f", "chromaprint", "-algorithm", str(CHROMAPRINT_ALGORITHM), "-fp_format", "raw", "-"]


def parse(returncode, stdout, stderr):
    """(status, fingerprint, error) from a finished ``command``.

    status is "ok" (fingerprint is a uint32 array), "no_audio" (the file has no audio track) or "failed".
    """
    err = stderr.decode(errors="replace").strip() if isinstance(stderr, bytes) else (stderr or "").strip()
    if any(s in err for s in _NO_AUDIO):
        return "no_audio", None, err[-500:] or None
    if returncode != 0:
        return "failed", None, err[-500:] or f"ffmpeg exit {returncode}"
    if len(stdout) % 4:
        return "failed", None, f"raw output of {len(stdout)} bytes is not whole uint32 values"
    fp = np.frombuffer(stdout, dtype="<u4")
    if not len(fp):
        return "no_audio", None, err[-500:] or "empty fingerprint"
    return "ok", fp, None


def fingerprint(path, ffmpeg="ffmpeg", timeout=None):
    """(status, fingerprint, error) for the audio track of ``path``."""
    try:
        r = subprocess.run(command(path, ffmpeg), capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return "failed", None, f"timed out after {timeout} s"
    return parse(r.returncode, r.stdout, r.stderr)

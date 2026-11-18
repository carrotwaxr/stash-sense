"""Per-frame video and audio signals for split-point detection (#131).

The signals come from two ffmpeg passes over a file: a 6 fps video pass
(signalstats plus scdet) and an audio pass (silencedetect plus RMS every
0.25 s).  They are cached per file in a quantized, compressed form.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass

import numpy as np

# Bump when extraction or quantization changes; cached signals of an older
# version are re-extracted.
SIGNALS_VERSION = 1

VIDEO_FILTER = "fps=6,scale=160:-2,scdet=t=100,signalstats,metadata=print:file=-"
AUDIO_FILTER = (
    "silencedetect=n=-35dB:d=0.3,aresample=8000,asetnsamples=n=2000:p=0,"
    "astats=metadata=1:reset=1,"
    "ametadata=print:key=lavfi.astats.Overall.RMS_level:file=-"
)
SILENT_RMS = -120.0


@dataclass
class Signals:
    t: np.ndarray  # frame times, seconds (6 fps)
    y: np.ndarray  # YAVG, int
    sat: np.ndarray  # SATAVG
    ydif: np.ndarray  # YDIF
    scd: np.ndarray  # scdet score
    au_t: np.ndarray  # RMS times, every 0.25 s
    rms: np.ndarray  # RMS dB, -120 for silence
    sil: np.ndarray  # [[start, end]] silencedetect n=-35dB d=0.3
    duration: float
    covered: list  # [[start, end]] stretches the signals cover


def _q(a, scale: int, hi: int = 255) -> np.ndarray:
    """Round to 1/scale and store as uint8, clipped to 0..hi/scale."""
    return np.clip(np.rint(np.asarray(a, dtype=np.float64) * scale), 0, hi).astype(
        np.uint8
    )


def _pack_times(a) -> tuple[np.ndarray, np.ndarray]:
    """Seconds -> (first value in ms, deltas in ms); uint16 deltas when they fit, else int32."""
    ms = np.rint(np.asarray(a, dtype=np.float64) * 1000).astype(np.int64)
    if ms.size == 0:
        return np.zeros(0, dtype=np.uint32), np.zeros(0, dtype=np.uint16)
    d = np.diff(ms)
    fits = d.size == 0 or (d.min() >= 0 and d.max() <= 65535)
    return np.array([ms[0]], dtype=np.uint32), d.astype(np.uint16 if fits else np.int32)


def _unpack_times(first: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    if first.size == 0:
        return np.zeros(0, dtype=np.float64)
    ms = np.concatenate([first.astype(np.int64), deltas.astype(np.int64)]).cumsum()
    return ms.astype(np.float64) / 1000.0


def encode(s: Signals) -> bytes:
    """Quantize and compress.

    t and au_t: first value plus deltas in ms (exact to the ms). y: uint8.
    sat and ydif: uint8 at x4, clipped to 0..63.75. scd: uint8 at x2, clipped
    to 0..127.5. rms: int8 whole dB, clipped to -120..0. sil: float32.
    """
    t0, td = _pack_times(s.t)
    a0, ad = _pack_times(s.au_t)
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        version=np.array([SIGNALS_VERSION], dtype=np.uint16),
        t0=t0,
        td=td,
        y=np.clip(np.asarray(s.y), 0, 255).astype(np.uint8),
        sat=_q(s.sat, 4),
        ydif=_q(s.ydif, 4),
        scd=_q(s.scd, 2),
        au_t0=a0,
        au_td=ad,
        rms=np.clip(np.rint(np.asarray(s.rms, dtype=np.float64)), SILENT_RMS, 0).astype(
            np.int8
        ),
        sil=np.asarray(s.sil, dtype=np.float32).reshape(-1, 2),
        duration=np.array([s.duration], dtype=np.float64),
        covered=np.asarray(s.covered, dtype=np.float64).reshape(-1, 2),
    )
    return buf.getvalue()


def decode(b: bytes) -> Signals:
    with np.load(io.BytesIO(b)) as z:
        return Signals(
            t=_unpack_times(z["t0"], z["td"]),
            y=z["y"].astype(np.int64),
            sat=z["sat"].astype(np.float64) / 4.0,
            ydif=z["ydif"].astype(np.float64) / 4.0,
            scd=z["scd"].astype(np.float64) / 2.0,
            au_t=_unpack_times(z["au_t0"], z["au_td"]),
            rms=z["rms"].astype(np.float64),
            sil=z["sil"].astype(np.float64),
            duration=float(z["duration"][0]),
            covered=[[float(a), float(b_)] for a, b_ in z["covered"]],
        )


def to_dict(s: Signals) -> dict:
    """The curator shape the ported code reads."""
    return {
        "fr": [
            [float(t), int(y), float(sat), float(yd), float(scd)]
            for t, y, sat, yd, scd in zip(s.t, s.y, s.sat, s.ydif, s.scd)
        ],
        "au": [[float(t), float(r)] for t, r in zip(s.au_t, s.rms)],
        "sil": [[float(a), float(b)] for a, b in np.asarray(s.sil).reshape(-1, 2)],
        "covered": [[float(a), float(b)] for a, b in s.covered],
    }


def from_curator(d: dict, duration: float) -> Signals:
    fr = np.asarray(d.get("fr", []), dtype=np.float64).reshape(-1, 5)
    au = np.asarray(d.get("au", []), dtype=np.float64).reshape(-1, 2)
    return Signals(
        t=fr[:, 0],
        y=fr[:, 1].astype(np.int64),
        sat=fr[:, 2],
        ydif=fr[:, 3],
        scd=fr[:, 4],
        au_t=au[:, 0],
        rms=au[:, 1],
        sil=np.asarray(d.get("sil", []), dtype=np.float64).reshape(-1, 2),
        duration=float(duration),
        covered=[[float(a), float(b)] for a, b in d.get("covered", [])],
    )


def video_command(path: str) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "error",
        "-i", path, "-an", "-map", "0:v:0", "-vf", VIDEO_FILTER, "-f", "null", "-",
    ]  # fmt: skip


def audio_command(path: str) -> list[str]:
    return [
        "ffmpeg", "-nostdin", "-hide_banner", "-v", "info",
        "-i", path, "-vn", "-af", AUDIO_FILTER, "-f", "null", "-",
    ]  # fmt: skip


_PTS_TIME = re.compile(r"pts_time:([\d.]+)")


def parse_video(text: str) -> list[list]:
    """Parse `metadata=print` text into [[t, Y, SAT, YDIF, scd]] rows."""
    frames: list[list] = []
    cur = None
    for line in text.splitlines():
        if line.startswith("frame:"):
            m = _PTS_TIME.search(line)
            cur = [round(float(m.group(1)), 2) if m else 0.0, 0, 0.0, 0.0, 0.0]
            frames.append(cur)
        elif "=" in line and cur is not None:
            k, _, v = line.strip().partition("=")
            k = k.split(".")[-1]
            try:
                if k == "YAVG":
                    cur[1] = round(float(v))
                elif k == "SATAVG":
                    cur[2] = round(float(v), 1)
                elif k == "YDIF":
                    cur[3] = round(float(v), 1)
                elif k == "score":
                    cur[4] = round(float(v), 1)
            except ValueError:
                pass
    return frames


def parse_audio(stdout: str, stderr: str, duration: float) -> tuple[list, list]:
    """Return (au, sil): RMS rows [[t, dB]] and silences [[start, end]]."""
    sil: list = []
    start = None
    for line in stderr.splitlines():
        m = re.search(r"silence_start: ([-\d.]+)", line)
        if m:
            start = float(m.group(1))
        m = re.search(r"silence_end: ([-\d.]+)", line)
        if m and start is not None:
            sil.append([round(max(start, 0.0), 2), round(float(m.group(1)), 2)])
            start = None
    if start is not None:
        sil.append([round(max(start, 0.0), 2), round(float(duration), 2)])

    au: list = []
    t = None
    for line in stdout.splitlines():
        m = _PTS_TIME.search(line)
        if m:
            t = float(m.group(1))
        elif "RMS_level=" in line and t is not None:
            v = line.strip().split("=")[1]
            au.append(
                [
                    round(t, 2),
                    SILENT_RMS if v in ("-inf", "nan") else round(float(v), 1),
                ]
            )
    return au, sil

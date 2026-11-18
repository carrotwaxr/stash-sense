"""Keeper health check: stream sanity plus sampled decode windows.

Ported from stash-curator's ``mergetool/dupecheck.py``. Before a duplicate set is merged, the file that would be kept
is probed and decoded in a few short windows spread through it, so a truncated or damaged keeper is caught while the
other copy still exists.

Every command runs under the user's nice and thread settings (``media_exec.media_job_command``), and the windows
decode one after another so a spinning disk is never seeked in parallel.
"""
from __future__ import annotations

import json
import os
from typing import Awaitable, Callable

from media_exec import media_job_command, run_command

# Lines a decoder prints right after a seek lands mid-stream (partial first packet,
# references to frames before the seek point). Not damage.
BENIGN = ('Header missing', 'non-existing PPS', 'no frame!', 'reference picture missing', 'Missing reference picture',
          'co located POCs unavailable', 'mmco: unref short failure', 'number of reference frames',
          'illegal short term buffer', 'Invalid timestamp', 'big_values too big', 'overread', 'invalid new backstep',
          'Could not find ref with POC')
# Timestamp irregularities the null muxer complains about: playback-harmless, reported as a warning.
WARN_ONLY = ('non monotonically increasing dts',)

PROBE_TIMEOUT = 120
WINDOW_TIMEOUT = 180       # a 1 s window, or its 21 s confirm decode
SEQUENTIAL_TIMEOUT = 600   # the 120 s sequential decode

# (exit code, stderr + stdout)
Run = Callable[[list], Awaitable[tuple]]


def real_errors(stderr: str) -> list[str]:
    return [line for line in stderr.strip().splitlines()
            if line.strip() and not any(b in line for b in BENIGN) and not any(w in line for w in WARN_ONLY)]


def _timeout_for(args: list) -> float:
    if args[0] == "ffprobe":
        return PROBE_TIMEOUT
    try:
        length = float(args[args.index("-t") + 1])
    except (ValueError, IndexError):
        return SEQUENTIAL_TIMEOUT
    return WINDOW_TIMEOUT if length <= 21 else SEQUENTIAL_TIMEOUT


async def _default_run(args: list) -> tuple:
    """(exit code, stderr + stdout): ffmpeg reports on stderr, ffprobe prints its JSON on stdout after any warnings."""
    try:
        res = await run_command(media_job_command(args), timeout=_timeout_for(args))
    except TimeoutError:
        return 1, "timed out"
    except OSError as e:
        return 1, f"could not run {args[0]}: {e}"
    return res.returncode, res.stderr.decode("utf-8", "replace") + res.stdout.decode("utf-8", "replace")


async def _ffmpeg_null(path: str, start: float, length: float, run: Run) -> tuple:
    args = ["ffmpeg", "-nostdin", "-v", "error"]
    if start > 0:
        args += ["-ss", f"{start:.1f}"]
    args += ["-i", path, "-t", f"{length:.1f}", "-f", "null", "-"]
    return await run(args)


async def check_health(local_path: str, points: int = 24, run: Run | None = None) -> dict:
    """{'status': ok|warn|suspect|missing, 'reasons': [...], 'params': {'points': points}}.

    Stream sanity, then ``points`` 1 s decode windows across the file. A failing window is re-decoded entering 20 s
    earlier: a seek artifact vanishes, damage repeats. Every window failing is a container that can't be entered
    mid-stream (MPEG-PS mp2): a clean sequential decode of the first 120 s clears it.

    ``local_path`` is a path on the sidecar. ``run`` takes an argv list and returns ``(exit code, text)``; it defaults
    to running the command under the media job nice and thread settings.
    """
    run = run or _default_run
    params = {"points": points}

    def result(status: str, reasons: list) -> dict:
        return {"status": status, "reasons": reasons, "params": params}

    if not os.path.exists(local_path):
        return result("missing", ["file not on disk"])
    code, out = await run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,duration",
                           "-of", "json", local_path])
    if code:
        return result("suspect", ["ffprobe failed: " + out.strip()[:200]])
    try:
        start = out.index("{")
        meta = json.loads(out[start:])
    except ValueError:
        return result("suspect", ["ffprobe printed no metadata"])
    probe_warn = out[:start].strip()
    dur = float(meta.get("format", {}).get("duration") or 0)
    kinds: dict = {}
    for s in meta.get("streams", []):
        kinds.setdefault(s.get("codec_type"), []).append(float(s.get("duration") or dur))
    reasons, hard, notes = [], False, []
    for k in ("video", "audio"):
        if k not in kinds:
            reasons.append(f"no {k} stream")
            hard = True
    if "video" in kinds and "audio" in kinds:
        v, a = max(kinds["video"]), max(kinds["audio"])
        if abs(v - a) > 2:
            reasons.append(f"video/audio durations differ: {v:.1f} vs {a:.1f}")
            if abs(v - a) > max(60, 0.05 * max(v, a)):   # a few s of trailing audio is a muxing quirk
                hard = True
    if probe_warn:
        reasons.append("probe warnings: " + probe_warn[:200])
    if dur <= 0:
        return result("suspect", reasons + ["no duration"])

    bad, warns = [], []
    for k in range(points):   # one at a time: parallel seeks thrash a spinning disk
        t = dur * (k + 0.5) / points
        code, err = await _ffmpeg_null(local_path, t, 1, run)
        errs = real_errors(err)
        if code or errs:
            code2, err2 = await _ffmpeg_null(local_path, max(0.0, t - 20), 21, run)
            if code2 or len(real_errors(err2)) > 2:     # the new entry point may itself print 1-2 artifact lines
                bad.append((t, (errs[0] if errs else f"exit {code}")[:120]))
        elif "non monotonically increasing dts" in err and not warns:
            warns.append("non-monotonic dts (timestamp irregularity; plays fine)")
    if bad and len(bad) == points:
        code, err = await _ffmpeg_null(local_path, 0, 120, run)
        if code == 0 and not real_errors(err):
            notes.append("seek artifacts only (every window fails after a seek; sequential 0-120 s is clean)")
            bad = []
    if bad:
        reasons.append(f"{len(bad)}/{points} sample windows fail to decode: "
                       + " | ".join(f"t={t:.0f}s {m}" for t, m in bad[:3]))
        hard = True
    reasons += warns
    # notes explain a clean result without downgrading it
    return result("suspect" if hard else ("warn" if reasons else "ok"), reasons + notes)

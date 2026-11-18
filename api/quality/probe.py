"""Cheap pass and the tier decision (#133).

Every file gets one ffprobe, the container index check and a 30 s tail decode. Only a suspect pays for the 24-window
health check of #130 (``duplicates.health``). Commands run through the injectable ``run`` callable, which defaults to
the media job nice and thread settings.
"""
from __future__ import annotations

import json
import os

from duplicates import health
from duplicates.health import Run, real_errors
from quality.container import HARD_INDEX_FLAGS, index_status

CHEAP_VERSION = 1
TAIL_S = 30


def _result(status: str, reasons: list, index: dict | None = None, tags: dict | None = None,
            probe: dict | None = None) -> dict:
    return {"status": status, "reasons": reasons, "index": index or {"kind": "other", "flags": []},
            "tags": tags or {}, "probe": probe or {}, "version": CHEAP_VERSION}


async def cheap_check(local_path: str, stash_duration: float, run: Run | None = None) -> dict:
    """Probe, index check and tail decode: {'status': ok|warn|suspect|missing, 'reasons', 'index', 'tags', 'probe',
    'version'}. ``stash_duration`` is Stash's own duration in seconds (0 skips the comparison)."""
    run = run or health._default_run
    if not os.path.exists(local_path):
        return _result("missing", ["file not on disk"])
    try:
        index = index_status(local_path)
    except OSError as e:
        return _result("suspect", [f"could not read the file: {e}"])

    code, out = await run(["ffprobe", "-v", "error", "-show_entries",
                           "format=duration,format_name:format_tags:stream=codec_type,codec_name,duration",
                           "-of", "json", local_path])
    if code:
        return _result("suspect", ["ffprobe failed: " + out.strip()[:200]], index)
    try:
        start = out.index("{")
        meta = json.loads(out[start:])
    except ValueError:
        return _result("suspect", ["ffprobe printed no metadata"], index)

    probe_warn = out[:start].strip()
    fmt = meta.get("format", {})
    dur = float(fmt.get("duration") or 0)
    tags = {str(k).lower(): v for k, v in (fmt.get("tags") or {}).items()}
    kinds: dict = {}
    codecs: dict = {}
    for s in meta.get("streams", []):
        kinds.setdefault(s.get("codec_type"), []).append(float(s.get("duration") or dur))
        codecs.setdefault(s.get("codec_type"), s.get("codec_name"))
    info = {"duration": dur, "format_name": fmt.get("format_name"),
            "video_codec": codecs.get("video"), "has_audio": "audio" in kinds}

    hard: list[str] = []
    soft: list[str] = []
    for k in ("video", "audio"):
        if k not in kinds:
            hard.append(f"no {k} stream")
    if "video" in kinds and "audio" in kinds:
        v, a = max(kinds["video"]), max(kinds["audio"])
        gap = abs(v - a)
        if gap > max(60, 0.05 * max(v, a)):
            hard.append(f"video/audio durations differ: {v:.1f} vs {a:.1f}")
        elif gap > 2:
            soft.append(f"video/audio durations differ: {v:.1f} vs {a:.1f}")
    for flag in index["flags"]:
        (hard if flag in HARD_INDEX_FLAGS else soft).append(f"container: {flag}")
    if probe_warn:
        soft.append("probe warnings: " + probe_warn[:200])
    if stash_duration and abs(dur - stash_duration) > max(0.02 * stash_duration, 5):
        hard.append(f"duration {dur:.0f}s differs from Stash's {stash_duration:.0f}s")

    code, err = await run(["ffmpeg", "-nostdin", "-v", "error", "-sseof", f"-{TAIL_S}", "-i", local_path,
                           "-f", "null", "-"])
    errs = real_errors(err)
    if code or len(errs) > 2:
        hard.append("tail decode failed: " + (errs[0] if errs else f"exit {code}")[:120])

    status = "suspect" if hard else ("warn" if soft else "ok")
    return _result(status, hard + soft, index, tags, info)


async def tiered(local_path: str, stash_duration: float, *, cheap: dict | None = None, deep: dict | None = None,
                 run: Run | None = None) -> dict:
    """{'status', 'reasons', 'tier': 'cheap'|'deep', 'cheap', 'deep'}. Only a cheap suspect gets the deep check, and
    the deep status is then final. Known ``cheap`` / ``deep`` results are reused without running commands."""
    if cheap is None:
        cheap = await cheap_check(local_path, stash_duration, run=run)
    if cheap["status"] != "suspect":
        return {"status": cheap["status"], "reasons": cheap["reasons"], "tier": "cheap", "cheap": cheap,
                "deep": None}
    if deep is None:
        deep = await health.check_health(local_path, run=run)
    return {"status": deep["status"], "reasons": deep["reasons"], "tier": "deep", "cheap": cheap, "deep": deep}


def is_broken(res: dict) -> bool:
    return res["status"] in ("missing", "suspect")

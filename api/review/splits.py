"""scene_splits: review the split points #131 found in a long or multi-scene file.

A recommendation (analyzers/scene_splits.py) holds the planner's cuts for one scene's primary file. The card
shows each cut with stills on both sides, lets the owner edit the cut times and trim the start and end, and
plans the accept as a hand-off: ``Plan.split`` carries the file and the ranges for stash-curator (#111), and
stash-sense writes nothing to Stash.

Choices are text, so the option is canonical: times are parsed and written back as ``h:mm:ss`` / ``m:ss`` (two
decimals only when not whole seconds). A time the owner changed snaps to the START of the nearest black run in
the cached signals (DATA_DIR/splits.db, opened read-only; no store means no snapping), the same rule the
detector uses (the black and the title card after it belong to the next scene). A time equal to a suggested
cut is left exactly as planned.

- empty cuts and no trim: leave whole (no hand-off, accept only resolves)
- cuts, or a trim with zero cuts: a hand-off with one range per output scene
- redundant card: every stretch is held by other scenes' files; no choices, accept only resolves (the delete
  belongs to the duplicate rules)
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import re
import sqlite3

from splits import detect
from splits import signals as sg

from .plan import Plan, Preview, SplitHandoff
from .types import InvalidOption, ReviewType, TargetMissing, register_review_type

logger = logging.getLogger(__name__)

TYPE_ID = "scene_splits"
MIN_SCENE_GAP_S = 30.0    # two cuts nearer than this are not two scenes
SNAP_BEFORE_S = 4.0       # a black run starting this long before an edited cut ...
SNAP_AFTER_S = 2.0        # ... or this long after it is the cut's black
SAME_S = 0.006            # a time within this of a suggested one is that suggestion (text has two decimals)
KIND_MATCH_S = 2.0        # a cut this near a planned cut keeps its kind and evidence on the card
TRIM_EPS_S = 0.5          # a keep_from / keep_until this near the file's ends is no trim
STILL_OFFSET_S = 3.0
PLAYBACK_LEAD_S = 5.0

_SEP = re.compile(r"[,\s]+")
_PLAIN = re.compile(r"^\d+(?:\.\d+)?$")


def parse_time(text: str) -> float:
    """``h:mm:ss(.ss)``, ``mm:ss(.ss)`` or plain seconds -> seconds; ValueError when it is none of them."""
    parts = text.strip().split(":")
    if not 1 <= len(parts) <= 3 or not all(p and _PLAIN.match(p) for p in parts):
        raise ValueError(f"cannot read {text!r} as a time")
    if len(parts) > 1 and not all(p.isdigit() for p in parts[:-1]):
        raise ValueError(f"cannot read {text!r} as a time")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + float(p)
    if not math.isfinite(secs):
        raise ValueError(f"cannot read {text!r} as a time")
    return secs


def parse_times(text: str) -> list[float]:
    """Times separated by commas or whitespace."""
    return [parse_time(tok) for tok in _SEP.split((text or "").strip()) if tok]


def format_time(t: float) -> str:
    """``h:mm:ss`` from an hour on, else ``m:ss``; whole seconds without decimals, others with two."""
    cs = int(round(float(t) * 100))
    whole, frac = divmod(cs, 100)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    sec = f"{s:02d}" if frac == 0 else f"{s:02d}.{frac:02d}"
    return f"{h}:{m:02d}:{sec}" if h else f"{m}:{sec}"


def _oshash(f: dict) -> str | None:
    return next((fp["value"] for fp in f.get("fingerprints") or [] if fp.get("type") == "oshash"), None)


def _read_runs(oshash: str | None, size: int) -> list[list[float]]:
    """Black runs [start, end, len, min_y, mean_y] from the cached signals, read-only; [] without them."""
    from analyzers.scene_splits import STORE_NAME
    from jobs.content_fingerprint_job import _data_dir
    path = os.path.join(_data_dir(), STORE_NAME)
    if not oshash or not os.path.exists(path):
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            row = conn.execute("SELECT version, blob FROM signals WHERE oshash = ? AND size_bytes = ?",
                               (oshash, size)).fetchone()
        finally:
            conn.close()
        if row is None or row[0] < sg.SIGNALS_VERSION:
            return []
        fr = sg.to_dict(sg.decode(row[1]))["fr"]
    except Exception as e:  # a missing or unreadable store only costs the snapping
        logger.warning("Could not read cached split signals from %s: %s", path, e)
        return []
    return [r for r in detect.dark_runs(fr) if r[2] >= detect.MIN_BLACK_S]


def _snap(t: float, runs: list[list[float]]) -> float:
    near = [r[0] for r in runs if t - SNAP_BEFORE_S <= r[0] <= t + SNAP_AFTER_S]
    return min(near, key=lambda s: (abs(s - t), s)) if near else t


def _check(cuts: list[float], keep_from: float, keep_until: float) -> None:
    if keep_from < 0 or keep_until <= keep_from:
        raise InvalidOption(f"keep_until ({format_time(keep_until)}) must be after keep_from ({format_time(keep_from)})")
    prev = None
    for t in cuts:
        if not keep_from < t < keep_until:
            raise InvalidOption(f"The cut at {format_time(t)} is outside {format_time(keep_from)} to "
                                f"{format_time(keep_until)}")
        if prev is not None:
            if t <= prev:
                raise InvalidOption(f"Cut times must increase: {format_time(t)} does not follow {format_time(prev)}")
            if t - prev < MIN_SCENE_GAP_S:
                raise InvalidOption(f"The cuts at {format_time(prev)} and {format_time(t)} are under "
                                    f"{MIN_SCENE_GAP_S:.0f} s apart")
        prev = t


def _times(text: str, what: str) -> list[float]:
    try:
        return parse_times(text)
    except ValueError as e:
        raise InvalidOption(f"{what}: {e}") from e


@register_review_type
class SceneSplitsReviewType(ReviewType):
    type_id = TYPE_ID
    display_name = "Split Multi-Scene Files"
    description = "Files holding several scenes: confirm the cut times, then hand the split to stash-curator"

    async def preview(self, rec, option: dict | None, stash) -> Preview:
        details = rec.details or {}
        kind = details.get("kind") or "split"
        planned_file = details.get("file") or {}
        sid = str(rec.target_id)

        # 1. the scene and its file, as they are now
        scene = await stash.get_scene_for_review(sid)
        if not scene:
            raise TargetMissing(f"Scene {sid} no longer exists")
        files = scene.get("files") or []
        if not files:
            raise TargetMissing(f"Scene {sid} has no file left")
        f = files[0]
        oshash = _oshash(f)
        if oshash != planned_file.get("oshash") or (
                oshash is None and str(f["id"]) != str(planned_file.get("id"))):
            raise TargetMissing(f"The file of scene {sid} changed since this plan was made; "
                                "run the scene_splits job again")
        duration = float(f.get("duration") or planned_file.get("duration") or 0)
        size = int(f.get("size") or 0)
        file_id = str(f["id"])
        confidence = details.get("confidence") or "low"

        if kind == "redundant":
            return self._redundant(rec, details, scene, f, duration, confidence)

        # 2. the default option is the plan
        planned = [float(c["t"]) for c in details.get("cuts") or []]
        planned_until = details.get("keep_until")
        planned_until = None if planned_until is None else float(planned_until)
        default = {
            "cuts": ", ".join(format_time(t) for t in planned),
            "keep_from": "",
            "keep_until": format_time(planned_until) if planned_until is not None else "",
        }

        # 3. the chosen option, canonical
        option = option or {}
        chosen = {k: str(option[k]).strip() if option.get(k) is not None else default[k] for k in default}
        raw_cuts = _times(chosen["cuts"], "cuts")
        raw_from = _times(chosen["keep_from"], "keep_from")
        raw_until = _times(chosen["keep_until"], "keep_until")
        if len(raw_from) > 1 or len(raw_until) > 1:
            raise InvalidOption("keep_from and keep_until take one time each")
        chosen = {
            "cuts": ", ".join(format_time(t) for t in raw_cuts),
            "keep_from": format_time(raw_from[0]) if raw_from else "",
            "keep_until": format_time(raw_until[0]) if raw_until else "",
        }

        def exact(t: float, pool: list[float]) -> float | None:
            return next((p for p in pool if abs(p - t) <= SAME_S), None)

        keep_from = raw_from[0] if raw_from else 0.0
        keep_until = raw_until[0] if raw_until else duration
        if raw_until and planned_until is not None and abs(planned_until - keep_until) <= SAME_S:
            keep_until = planned_until
        if keep_until > duration + TRIM_EPS_S:
            raise InvalidOption(f"keep_until ({format_time(keep_until)}) is past the end of the file "
                                f"({format_time(duration)})")
        _check(raw_cuts, keep_from, keep_until)

        # 4. snap the cuts the owner changed to the start of the black next to them
        cuts: list[float] = []
        snapped: list[dict] = []
        runs: list[list[float]] | None = None
        for t in raw_cuts:
            p = exact(t, planned)
            if p is not None:
                cuts.append(p)
                continue
            if runs is None:
                runs = await asyncio.to_thread(_read_runs, oshash, size)
            s = _snap(t, runs)
            if s != t:
                snapped.append({"from": t, "to": s})
            cuts.append(s)
        if snapped:
            _check(cuts, keep_from, keep_until)

        # 5. the hand-off
        plan = Plan(rec_id=rec.id, rec_type=self.type_id, option=chosen)
        trimmed = keep_from > TRIM_EPS_S or keep_until < duration - TRIM_EPS_S
        ranges: list[dict] = []
        if cuts or trimmed:
            edges = [keep_from, *cuts, keep_until]
            ranges = [{"start": a, "end": b} for a, b in zip(edges, edges[1:])]
            plan.split = SplitHandoff(scene_id=sid, file_id=file_id, oshash=oshash or "", size=size,
                                      duration=duration, ranges=ranges, cuts=list(cuts))

        # 6. evidence
        by_t = [(float(c["t"]), c) for c in details.get("cuts") or []]

        def cut_row(t: float) -> dict:
            near = min(by_t, key=lambda x: abs(x[0] - t), default=None)
            if near is not None and abs(near[0] - t) <= KIND_MATCH_S:
                return {"t": t, "kind": near[1].get("kind"), "evidence": near[1].get("evidence") or ""}
            return {"t": t, "kind": "manual", "evidence": ""}

        def clamp(t: float) -> float:
            return round(min(max(t, 0.0), duration), 2) if duration else round(max(t, 0.0), 2)

        stills = []
        for t in cuts:
            stills.append({"scene_id": sid, "file_id": file_id, "t": clamp(t - STILL_OFFSET_S)})
            stills.append({"scene_id": sid, "file_id": file_id, "t": clamp(t + STILL_OFFSET_S)})
        evidence = {
            "scene": {"id": sid, "title": scene.get("title") or ""},
            "file": {"id": file_id, "oshash": oshash, "size": size, "duration": duration, "path": f.get("path")},
            "duration": duration,
            "confidence": confidence,
            "reasons": list(details.get("reasons") or []),
            "series": details.get("series"),
            "cuts": [cut_row(t) for t in cuts],
            "leftovers": details.get("leftovers") or [],
            "covering": details.get("covering") or [],
            "snapped": snapped,
            "stills": stills,
            "playback_points": [clamp(t - PLAYBACK_LEAD_S) for t in cuts],
        }

        if plan.split is None:
            summary = "leave whole"
        elif len(ranges) == 1:
            summary = f"trim to one scene ({format_time(keep_from)} to {format_time(keep_until)})"
        else:
            summary = f"split into {len(ranges)} scenes"
        if snapped:
            summary += f"; {len(snapped)} cut{'s' if len(snapped) != 1 else ''} snapped to the start of a black"

        choices = [
            {"field": "cuts", "label": "Cut times (empty = leave whole)", "kind": "text"},
            {"field": "keep_from", "label": "Keep from (empty = the start)", "kind": "text"},
            {"field": "keep_until", "label": "Keep until (empty = the end)", "kind": "text"},
        ]
        high = kind == "split" and confidence == "high" and chosen == default
        return Preview(plan=plan, evidence=evidence, choices=choices, default_option=default,
                       high_confidence=high, summary=summary)

    def _redundant(self, rec, details: dict, scene: dict, f: dict, duration: float, confidence: str) -> Preview:
        sid = str(rec.target_id)
        covering = details.get("covering") or []
        evidence = {
            "scene": {"id": sid, "title": scene.get("title") or ""},
            "file": {"id": str(f["id"]), "oshash": _oshash(f), "size": int(f.get("size") or 0),
                     "duration": duration, "path": f.get("path")},
            "duration": duration,
            "confidence": confidence,
            "reasons": list(details.get("reasons") or []),
            "series": details.get("series"),
            "cuts": [],
            "leftovers": [],
            "covering": covering,
            "redundant": True,
            "stills": [],
            "playback_points": [],
        }
        ids = sorted({str(c.get("scene_id")) for c in covering})
        held = f"held by scene{'s' if len(ids) != 1 else ''} {', '.join(ids)}" if ids else "held by other scenes"
        summary = f"Every stretch of this file is {held} at equal or better quality. Accept only resolves."
        plan = Plan(rec_id=rec.id, rec_type=self.type_id, option={})
        return Preview(plan=plan, evidence=evidence, choices=[], default_option={}, high_confidence=True,
                       summary=summary)

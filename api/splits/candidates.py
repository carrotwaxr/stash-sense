"""Scenes worth planning a split for (#131).

A candidate is a scene whose primary file is over MIN_DURATION_S, or whose file the content index shows holding other
scenes' whole files. Members of the curator's split group and the legacy split tag are outputs of earlier splits and
are skipped, as are excluded scenes.

For each candidate the index says which stretches of the file other scenes already hold (``coverage``, in the
candidate file's time). A file whose every stretch outside the first and last GUARD_S is covered by files at equal or
better quality is ``redundant``: there is nothing in it to split out.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from content_fingerprint import AUDIO_ALGO_VERSION
from content_fingerprint.match import classify
from duplicates import keeper

PAGE_SIZE = 500
MIN_DURATION_S = 3600
CONTAINED_COVER = 0.9      # share of the shorter file the index must find inside the long one
CONTAINED_RATIO = 1.5      # a covered file is at most 1/1.5 of the long file's length
GUARD_S = 150              # the long file's first and last seconds (intro and outro) need no cover
SPLIT_GROUP = "[curator] Split Cuts"
LEGACY_SPLIT_TAG = "[curator] split"


@dataclass
class SplitCandidate:
    scene_id: str
    file: dict                                   # the Stash file (primary)
    coverage: list[tuple[float, float]] = field(default_factory=list)   # merged, in the file's time
    covering: list[dict] = field(default_factory=list)                  # [{scene_id, file_id, start, end}]
    redundant: bool = False
    title: str = ""                              # the scene's title, for the series key


def _oshash(f: dict) -> Optional[str]:
    return next((fp["value"] for fp in f.get("fingerprints") or [] if fp.get("type") == "oshash"), None)


async def _load_scenes(stash, min_duration: Optional[float] = None) -> list[dict]:
    scenes, page = [], 1
    while True:
        chunk, count = await stash.get_scenes_with_files(page, PAGE_SIZE, min_duration)
        scenes += chunk
        if not chunk or len(scenes) >= count:
            return scenes
        page += 1


async def _split_outputs(stash) -> set[str]:
    """Ids of scenes in the split group or carrying the legacy split tag; a missing group or tag skips nothing."""
    out: set[str] = set()
    group = await stash.find_group_id_by_name(SPLIT_GROUP)
    if group:
        out |= {str(i) for i in await stash.find_scene_ids({"groups": {"value": [group], "modifier": "INCLUDES"}})}
    tag = await stash.find_tag_id_by_name(LEGACY_SPLIT_TAG)
    if tag:
        out |= {str(i) for i in await stash.find_scene_ids({"tags": {"value": [tag], "modifier": "INCLUDES"}})}
    return out


def _merge(ranges, gap: float = 0.0) -> list[tuple[float, float]]:
    out: list[list[float]] = []
    for s, e in sorted(ranges):
        if out and s <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def _covers(coverage, start: float, end: float) -> bool:
    """Whether the merged ``coverage`` leaves no second of [start, end] uncovered."""
    if end <= start:
        return True
    pos = start
    for s, e in coverage:
        if s > pos:
            return False
        pos = max(pos, e)
        if pos >= end:
            return True
    return pos >= end


def _index_coverage(store, scenes: list[dict], skip: set[str]) -> dict[str, dict]:
    """{long scene id: {"file", "covering", "ranges", "files"}} for every primary file the index shows holding
    other scenes' files."""
    # (oshash, size) -> [(scene id, file)]
    by_content: dict[tuple, list] = {}
    primary = {str(s["id"]): (s.get("files") or [{}])[0].get("id") for s in scenes}
    for s in scenes:
        for f in s.get("files") or []:
            osh = _oshash(f)
            if osh and f.get("size") is not None:
                by_content.setdefault((osh, int(f["size"])), []).append((str(s["id"]), f))
    content = store.files()
    rows_by_pair: dict[tuple, list] = {}
    for o in store.overlaps(AUDIO_ALGO_VERSION):
        rows_by_pair.setdefault((o["file_a"], o["file_b"]), []).append(
            (o["a_start"], o["a_end"], o["b_start"], o["b_end"], o["ber"]))

    out: dict[str, dict] = {}
    for (ida, idb), rows in sorted(rows_by_pair.items()):
        ca, cb = content.get(ida), content.get(idb)
        if not ca or not cb:
            continue
        xs = by_content.get((ca["oshash"], ca["size_bytes"]))
        ys = by_content.get((cb["oshash"], cb["size_bytes"]))
        if not xs or not ys:     # the file is no longer in Stash
            continue
        dur_a = xs[0][1].get("duration") or ca["duration_s"] or 0
        dur_b = ys[0][1].get("duration") or cb["duration_s"] or 0
        cover = classify(rows, dur_a, dur_b)
        if dur_a >= dur_b:
            longs, shorts, short_dur, long_dur, short_cover = xs, ys, dur_b, dur_a, cover["cover_b"]
            ranges = [(r[0], r[1]) for r in rows]
        else:
            longs, shorts, short_dur, long_dur, short_cover = ys, xs, dur_a, dur_b, cover["cover_a"]
            ranges = [(r[2], r[3]) for r in rows]
        if short_cover < CONTAINED_COVER or short_dur * CONTAINED_RATIO > long_dur:
            continue
        for sid, f in longs:
            if sid in skip or primary.get(sid) != f["id"]:
                continue     # candidates are scenes, planned over their primary file
            acc = out.setdefault(sid, {"file": f, "covering": [], "ranges": [], "files": {}})
            held = [(csid, cf) for csid, cf in shorts if csid != sid and csid not in skip]
            if not held:
                continue
            acc["ranges"] += ranges
            merged = _merge(ranges)
            for csid, cf in held:
                acc["covering"].append({"scene_id": csid, "file_id": cf["id"],
                                        "start": merged[0][0], "end": merged[-1][1]})
                acc["files"][cf["id"]] = cf
    return out


def _redundant(file: dict, coverage, covering: list[dict], files: dict[str, dict]) -> bool:
    if not covering or not _covers(coverage, GUARD_S, (file.get("duration") or 0) - GUARD_S):
        return False
    long_features = keeper.FileFeatures.from_stash_file(file)
    return all(
        # kind="contained": a shorter file inside a longer one says nothing about completeness
        keeper.better(keeper.FileFeatures.from_stash_file(files[c["file_id"]]), long_features, kind="contained")[0] != "b"
        for c in covering)


async def generate(stash, store, excluded: set[str]) -> list[SplitCandidate]:
    """Split candidates, none an excluded scene or an output of an earlier split. ``store`` is the content index, or
    None (coverage is then empty)."""
    skip = {str(e) for e in excluded} | await _split_outputs(stash)
    out: dict[str, SplitCandidate] = {}
    for s in await _load_scenes(stash, MIN_DURATION_S):
        sid, files = str(s["id"]), s.get("files") or []
        if sid not in skip and files and (files[0].get("duration") or 0) > MIN_DURATION_S:
            out[sid] = SplitCandidate(sid, files[0], title=s.get("title") or "")
    if store is not None:
        everyone = await _load_scenes(stash)
        titles = {str(s["id"]): s.get("title") or "" for s in everyone}
        for sid, held in _index_coverage(store, everyone, skip).items():
            if not held["covering"]:
                continue
            c = out.setdefault(sid, SplitCandidate(sid, held["file"], title=titles.get(sid, "")))
            c.covering = sorted(held["covering"], key=lambda x: (x["start"], int(x["scene_id"])))
            c.coverage = _merge(held["ranges"])
            c.redundant = _redundant(c.file, c.coverage, c.covering, held["files"])
    return sorted(out.values(), key=lambda c: int(c.scene_id))

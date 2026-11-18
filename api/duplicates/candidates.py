"""Candidate scene pairs for duplicate detection (#130).

Cheap sources only; later steps verify a pair with evidence and group pairs. Every source yields scene pairs:

  stash_id                shared (endpoint, stash_id)
  studio_title            same studio and the same normalized title
  studio_date_performers  same studio, date and (non-empty) performer set
  phash                   Stash's findDuplicateScenes, phash distance <= PHASH_DISTANCE
  url                     a shared URL; weak, on its own it only says "look if nothing else does"
  multi_file              two files in one scene (scene_a == scene_b)
  index                   a content-index audio overlap of kind duplicate, contained or partial

A pair found by several sources is one Candidate with all of them. When the index names the files, the candidate
keeps their oshashes (the strongest overlap if several file pairs join the same two scenes). A pair inside one
scene is a file pair, so one scene can yield several candidates.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from itertools import combinations
from typing import Optional

from content_fingerprint import AUDIO_ALGO_VERSION
from content_fingerprint.match import classify

PAGE_SIZE = 500
PHASH_DISTANCE = 10
MAX_GROUP = 25          # a metadata or URL group bigger than this is a shared label, not a duplicate set
INDEX_KINDS = ("duplicate", "contained", "partial")


@dataclass(frozen=True)
class Candidate:
    scene_a: str
    scene_b: str                      # scene_a <= scene_b numerically; equal for a multi-file scene
    file_a: Optional[str]             # oshash when the source names files
    file_b: Optional[str]
    sources: frozenset
    weak: bool
    overlap: Optional[dict] = None    # classify() output for index pairs


def normalize_title(title) -> str:
    """Casefolded, punctuation and whitespace runs removed."""
    t = unicodedata.normalize("NFKC", title or "").casefold()
    return re.sub(r"[\W_]+", "", t)


def _oshash(f) -> Optional[str]:
    return next((fp["value"] for fp in f.get("fingerprints") or [] if fp.get("type") == "oshash"), None)


async def _load_scenes(stash) -> list[dict]:
    scenes, page = [], 1
    while True:
        chunk, count = await stash.get_scenes_for_duplicates(page, PAGE_SIZE)
        scenes += chunk
        if not chunk or len(scenes) >= count:
            return scenes
        page += 1


def _swap(overlap):
    return {**overlap, "cover_a": overlap["cover_b"], "cover_b": overlap["cover_a"]}


class _Pairs:
    """Pairs found so far, merged by identity."""

    def __init__(self):
        self.items: dict[tuple, dict] = {}

    def _add(self, key, scene_a, scene_b, source, file_a=None, file_b=None, overlap=None):
        it = self.items.setdefault(key, {"a": scene_a, "b": scene_b, "sources": set(),
                                         "files": (None, None), "overlap": None})
        it["sources"].add(source)
        if overlap is not None and (it["overlap"] is None or overlap["shared_s"] > it["overlap"]["shared_s"]):
            it["overlap"], it["files"] = overlap, (file_a, file_b)
        elif overlap is None and it["files"] == (None, None) and file_a:
            it["files"] = (file_a, file_b)

    def scenes(self, x: str, y: str, source: str):
        """A scene-level pair (never a pair within one scene)."""
        if x == y:
            return
        a, b = sorted((x, y), key=int)
        self._add((a, b), a, b, source)

    def files(self, x, y, source, overlap=None):
        """A pair of files, x and y each (scene id, stash file id, oshash)."""
        (xs, xf, xo), (ys, yf, yo) = x, y
        if (int(xs), int(xf)) > (int(ys), int(yf)):
            x, y = y, x
            (xs, xf, xo), (ys, yf, yo) = x, y
            overlap = _swap(overlap) if overlap else None
        if xs == ys:
            if xf == yf:
                return
            self._add((xs, ys, xf, yf), xs, ys, source, xo, yo, overlap)
        else:
            self._add((xs, ys), xs, ys, source, xo, yo, overlap)


def _group_pairs(pairs: _Pairs, groups: dict, source: str, cap: Optional[int] = MAX_GROUP):
    for ids in groups.values():
        ids = sorted(set(ids), key=int)
        if len(ids) < 2 or (cap is not None and len(ids) > cap):
            continue
        for x, y in combinations(ids, 2):
            pairs.scenes(x, y, source)


def _index_pairs(pairs: _Pairs, store, scenes: list[dict]):
    # (oshash, size) -> [(scene id, stash file id, oshash, duration)]
    by_content: dict[tuple, list] = {}
    for s in scenes:
        for f in s.get("files") or []:
            osh = _oshash(f)
            if osh and f.get("size") is not None:
                by_content.setdefault((osh, int(f["size"])), []).append((s["id"], f["id"], osh, f.get("duration")))
    content = store.files()
    rows_by_pair: dict[tuple, list] = {}
    for o in store.overlaps(AUDIO_ALGO_VERSION):
        rows_by_pair.setdefault((o["file_a"], o["file_b"]), []).append(
            (o["a_start"], o["a_end"], o["b_start"], o["b_end"], o["ber"]))
    for (ida, idb), rows in rows_by_pair.items():
        ca, cb = content.get(ida), content.get(idb)
        if not ca or not cb:
            continue
        xs = by_content.get((ca["oshash"], ca["size_bytes"]))
        ys = by_content.get((cb["oshash"], cb["size_bytes"]))
        if not xs or not ys:     # the file is no longer in Stash
            continue
        overlap = classify(rows, xs[0][3] or ca["duration_s"], ys[0][3] or cb["duration_s"])
        if overlap["kind"] not in INDEX_KINDS:
            continue
        for x in xs:
            for y in ys:
                pairs.files(x[:3], y[:3], "index", overlap)


async def generate(stash, store, excluded: set[str]) -> list[Candidate]:
    """Candidate pairs from every source, none touching an excluded scene. ``store`` is the content index, or None."""
    excluded = {str(e) for e in excluded}
    scenes = [s for s in await _load_scenes(stash) if str(s["id"]) not in excluded]
    for s in scenes:
        s["id"] = str(s["id"])
    pairs = _Pairs()

    stash_ids, titles, dpp, urls = {}, {}, {}, {}
    for s in scenes:
        sid = s["id"]
        for e in s.get("stash_ids") or []:
            stash_ids.setdefault((e["endpoint"], e["stash_id"]), []).append(sid)
        studio = (s.get("studio") or {}).get("id")
        if studio:
            title = normalize_title(s.get("title"))
            if title:
                titles.setdefault((studio, title), []).append(sid)
            performers = frozenset(p["id"] for p in s.get("performers") or [])
            if s.get("date") and performers:
                dpp.setdefault((studio, s["date"], performers), []).append(sid)
        for u in set(s.get("urls") or []):
            if u:
                urls.setdefault(u, []).append(sid)
    _group_pairs(pairs, stash_ids, "stash_id", cap=None)
    _group_pairs(pairs, titles, "studio_title")
    _group_pairs(pairs, dpp, "studio_date_performers")
    _group_pairs(pairs, urls, "url")

    known = {s["id"] for s in scenes}
    for group in await stash.find_duplicate_scenes(PHASH_DISTANCE):
        ids = sorted({str(i) for i in group if str(i) in known}, key=int)
        for x, y in combinations(ids, 2):
            pairs.scenes(x, y, "phash")

    for s in scenes:
        files = [(s["id"], f["id"], _oshash(f)) for f in s.get("files") or []]
        for x, y in combinations(files, 2):
            pairs.files(x, y, "multi_file")

    if store is not None:
        _index_pairs(pairs, store, scenes)

    out = [Candidate(it["a"], it["b"], it["files"][0], it["files"][1], frozenset(it["sources"]),
                     it["sources"] == {"url"}, it["overlap"]) for it in pairs.items.values()]
    return sorted(out, key=lambda c: (int(c.scene_a), int(c.scene_b), c.file_a or "", c.file_b or ""))

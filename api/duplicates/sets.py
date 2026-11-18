"""Group verified duplicate pairs into review items with stable keys (#130).

Duplicate pairs join into sets (union-find over scenes), keyed ``duplicate:<lowest scene id>``. Every other
pair class stays a pair, keyed ``<cls>:<low>-<high>``. A set that gains a scene below its old anchor gets a
new key; ``build_items`` names the old keys it replaces so the analyzer can retire them.
"""
from __future__ import annotations

from dataclasses import dataclass

from duplicates.candidates import Candidate
from duplicates.verdict import PairEvidence

PAIR_CLASSES = ("contained", "partial", "language_variant", "mislabel", "review")
MOMENT_START_S = 60       # where stills are taken in scene A; B follows by the matched offset


@dataclass
class Item:
    key: str
    cls: str                      # duplicate | contained | partial | language_variant | mislabel | review
    scene_ids: list[str]          # numeric order
    pairs: list[dict]

    def to_details(self) -> dict:
        return {"cls": self.cls, "scene_ids": list(self.scene_ids), "pairs": list(self.pairs)}


def _num(scene_id: str):
    s = str(scene_id)
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def _moment(ev: PairEvidence) -> dict:
    offset = ev.ber_offset_s if ev.ber_offset_s is not None else ev.keyframe_offset_s
    t_a = MOMENT_START_S
    return {"t_a": t_a, "t_b": max(0, t_a + (offset or 0))}


def _pair(cand: Candidate, ev: PairEvidence, cls: str) -> dict:
    return {
        "scene_a": cand.scene_a, "scene_b": cand.scene_b,
        "file_a": cand.file_a, "file_b": cand.file_b,
        "sources": sorted(cand.sources), "cls": cls,
        "evidence": ev.to_dict(), "moment": _moment(ev),
    }


def build_items(verified: list[tuple[Candidate, PairEvidence, str]],
                previous_keys: dict[str, set[str]] | None = None) -> tuple[list[Item], list[str]]:
    """Items from verified (candidate, evidence, class) triples, and the old duplicate keys they replace."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    items: list[Item] = []
    dup_pairs: list[dict] = []
    for cand, ev, cls in verified:
        if cls == "duplicate":
            parent[find(cand.scene_a)] = find(cand.scene_b)
            dup_pairs.append(_pair(cand, ev, cls))
        elif cls in PAIR_CLASSES and cand.scene_a != cand.scene_b:
            lo, hi = sorted([cand.scene_a, cand.scene_b], key=_num)
            items.append(Item(f"{cls}:{lo}-{hi}", cls, [lo, hi], [_pair(cand, ev, cls)]))

    groups: dict[str, set[str]] = {}
    for scene in list(parent):
        groups.setdefault(find(scene), set()).add(scene)
    pairs_by_root: dict[str, list[dict]] = {}
    for p in dup_pairs:
        pairs_by_root.setdefault(find(p["scene_a"]), []).append(p)

    superseded: set[str] = set()
    for root, scenes in groups.items():
        ids = sorted(scenes, key=_num)
        key = f"duplicate:{ids[0]}"
        items.append(Item(key, "duplicate", ids, pairs_by_root.get(root, [])))
        for old_key, old_scenes in (previous_keys or {}).items():
            if old_key.startswith("duplicate:") and old_key != key and set(old_scenes) & scenes:
                superseded.add(old_key)

    items.sort(key=lambda i: i.key)
    return items, sorted(superseded)

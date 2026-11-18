"""What an accept would do, as data.

A review type builds a Plan from a recommendation, a chosen option and
Stash's current state. The executor is the only code that turns a Plan
into Stash writes, so every write is logged the same way.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

# Scene fields the framework may write, in SceneUpdateInput shape.
SCENE_FIELDS: frozenset[str] = frozenset({
    "title", "details", "date", "code", "director", "urls", "organized",
    "rating100", "studio_id", "performer_ids", "tag_ids", "groups", "gallery_ids",
})
# Id lists Stash treats as sets; compared order-insensitively.
_SET_FIELDS = frozenset({"performer_ids", "tag_ids", "gallery_ids"})


def scene_field_values(scene: dict) -> dict[str, Any]:
    """A Stash scene (review shape) as SceneUpdateInput values."""
    return {
        "title": scene.get("title") or "",
        "details": scene.get("details") or "",
        "date": scene.get("date"),
        "code": scene.get("code") or "",
        "director": scene.get("director") or "",
        "urls": list(scene.get("urls") or []),
        "organized": bool(scene.get("organized")),
        "rating100": scene.get("rating100"),
        "studio_id": (scene.get("studio") or {}).get("id"),
        "performer_ids": [p["id"] for p in scene.get("performers") or []],
        "tag_ids": [t["id"] for t in scene.get("tags") or []],
        "groups": [{"group_id": g["group"]["id"], "scene_index": g.get("scene_index")}
                   for g in scene.get("groups") or []],
        "gallery_ids": [g["id"] for g in scene.get("galleries") or []],
    }


def _group_key(groups: Any) -> list:
    return sorted((g["group_id"], -1 if g.get("scene_index") is None else g["scene_index"])
                  for g in groups or [])


def values_equal(field_name: str, a: Any, b: Any) -> bool:
    if field_name in _SET_FIELDS:
        return sorted(a or []) == sorted(b or [])
    if field_name == "groups":
        return _group_key(a) == _group_key(b)
    return a == b


@dataclass
class FieldChange:
    entity_type: str  # only "scene" so far
    entity_id: str
    field: str        # SceneUpdateInput field name
    before: Any
    after: Any


@dataclass
class FileRef:
    file_id: str
    scene_id: str
    path: str
    size: int
    fingerprints: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_stash(cls, scene_id: str, f: dict) -> FileRef:
        return cls(
            file_id=str(f["id"]),
            scene_id=str(scene_id),
            path=f["path"],
            size=int(f.get("size") or 0),
            fingerprints={fp["type"]: fp["value"] for fp in f.get("fingerprints") or []},
        )


@dataclass
class PrimaryFileChange:
    scene_id: str
    before: str
    after: str


@dataclass
class SceneMerge:
    """Stash's sceneMerge: sources' files move to the destination, sources are destroyed.

    Stash unions nothing itself, so values carries the merged metadata.
    """

    destination: str
    sources: list[str]
    values: dict  # SceneUpdateInput fields, each in SCENE_FIELDS
    source_snapshots: list[dict] = field(default_factory=list)  # review-shape scenes, for the apply log

    @property
    def scene_ids(self) -> list[str]:
        return [str(self.destination), *(str(s) for s in self.sources)]


@dataclass
class SplitHandoff:
    """An accepted split for an external tool (stash-curator #111) to execute; stash-sense writes nothing for it."""

    scene_id: str
    file_id: str
    oshash: str
    size: int
    duration: float
    ranges: list[dict]  # [{"start": s, "end": s}], one per output scene, contiguous except keep_from/keep_until trims
    cuts: list[float]


@dataclass
class Plan:
    rec_id: int
    rec_type: str
    option: dict
    field_changes: list[FieldChange] = field(default_factory=list)
    keep_files: list[FileRef] = field(default_factory=list)
    delete_files: list[FileRef] = field(default_factory=list)
    primary_file: PrimaryFileChange | None = None
    # Runs first; delete FileRefs name the scene a file is on before the merge.
    merge: SceneMerge | None = None
    # Handed to an external tool on accept; the executor records it and writes nothing.
    split: SplitHandoff | None = None

    @property
    def bytes_reclaimed(self) -> int:
        return sum(f.size for f in self.delete_files)

    @property
    def delete_file_ids(self) -> set[str]:
        return {f.file_id for f in self.delete_files}

    @property
    def is_empty(self) -> bool:
        return not (self.field_changes or self.delete_files or self.primary_file or self.merge or self.split)

    def token(self) -> str:
        """Fingerprint of what this plan would do, given the state it was built from.

        Covers the option, every field's before and after, and every file to
        delete. Any library change that alters the plan alters the token.
        """
        payload = {
            "rec_id": self.rec_id,
            "rec_type": self.rec_type,
            "option": self.option,
            "field_changes": [asdict(c) for c in self.field_changes],
            "primary_file": asdict(self.primary_file) if self.primary_file else None,
            "delete": sorted((f.file_id, f.path, f.size, sorted(f.fingerprints.items()))
                             for f in self.delete_files),
            "keep": sorted((f.file_id, f.path, f.size, sorted(f.fingerprints.items()))
                           for f in self.keep_files),
        }
        if self.merge:  # only when present, so tokens of plans without a merge stay valid
            payload["merge"] = {
                "destination": self.merge.destination,
                "sources": self.merge.sources,
                "values": self.merge.values,
            }
        if self.split:  # likewise: plans without a split keep their old tokens
            payload["split"] = asdict(self.split)
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:24]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["bytes_reclaimed"] = self.bytes_reclaimed
        d["token"] = self.token()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> Plan:
        return cls(
            rec_id=d["rec_id"],
            rec_type=d["rec_type"],
            option=d.get("option") or {},
            field_changes=[FieldChange(**c) for c in d.get("field_changes") or []],
            keep_files=[FileRef(**f) for f in d.get("keep_files") or []],
            delete_files=[FileRef(**f) for f in d.get("delete_files") or []],
            primary_file=PrimaryFileChange(**d["primary_file"]) if d.get("primary_file") else None,
            merge=SceneMerge(**d["merge"]) if d.get("merge") else None,
            split=SplitHandoff(**d["split"]) if d.get("split") else None,
        )


@dataclass
class Preview:
    """Everything a card shows: evidence, the exact outcome, and the choices."""

    plan: Plan
    evidence: dict
    # Inputs the user may change: [{"field", "label", "kind": "pick"|"text", "values"?: [...]}]
    choices: list[dict]
    default_option: dict
    # The type is confident enough in the default option to allow batch accept.
    high_confidence: bool
    summary: str = ""

    @property
    def is_default_option(self) -> bool:
        return self.plan.option == self.default_option

    def to_dict(self) -> dict:
        return {
            "rec_id": self.plan.rec_id,
            "rec_type": self.plan.rec_type,
            "summary": self.summary,
            "evidence": self.evidence,
            "choices": self.choices,
            "default_option": self.default_option,
            "option": self.plan.option,
            "high_confidence": self.high_confidence,
            "plan": self.plan.to_dict(),
            "plan_token": self.plan.token(),
        }

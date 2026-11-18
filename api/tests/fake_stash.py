"""An in-memory Stash that enforces the rules the review framework depends on.

Modelled on Stash v0.32 (internal/api/resolver_mutation_file.go, scene update):
- a scene's files are ordered primary first
- sceneUpdate rejects unknown fields and a primary_file_id that isn't on the scene
- deleteFiles is all-or-nothing and refuses any file that is a scene's primary
- with deleteTrashPath set, deleted files move to trash; otherwise they are gone
- configurePlugin replaces the plugin's whole settings map (config.SetPluginConfiguration)
- sceneMerge moves the sources' files to the destination after its own (the destination
  keeps its primary, or takes the first source file when it had none), applies the optional
  values partial, adds play and O history, then destroys the sources but not their files;
  it unions nothing itself (pkg/scene/merge.go)
- findScenes' hierarchical tags/groups/studios criteria with depth -1 match descendants

It implements the StashClientUnified methods the review code calls, with
the same names and return shapes (test_review_framework checks the names).
"""

from __future__ import annotations

import copy
import os
from typing import Optional

from review.plan import SCENE_FIELDS


class FakeStashError(RuntimeError):
    pass


class FakeStash:
    REVIEW_METHODS = (
        "get_scene_for_review", "find_scenes_with_tag_name", "update_scene",
        "delete_files", "get_delete_trash_path", "merge_scenes",
    )

    def __init__(self, trash_path: str = ""):
        self.scenes: dict[str, dict] = {}
        self.files: dict[str, dict] = {}
        self.tags: dict[str, str] = {}
        # child id -> parent ids, per hierarchy ("tags", "groups", "studios")
        self.parents: dict[str, dict[str, set[str]]] = {"tags": {}, "groups": {}, "studios": {}}
        self.groups: dict[str, str] = {}
        self.studios: dict[str, str] = {}
        self.plugin_config: dict[str, dict] = {}
        self.trash_path = trash_path
        self.deleted: list[dict] = []  # {"path", "destination"}
        self.phash_groups: list[list[str]] = []  # what find_duplicate_scenes returns
        self.writes = 0
        self.fail: dict[str, Exception] = {}  # method name -> error raised on next call
        self._next_id = 1

    # ---------------- seeding ----------------

    def _id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def add_tag(self, name: str, parent: Optional[str] = None) -> str:
        tid = self._id()
        self.tags[tid] = name
        if parent:
            self.parents["tags"].setdefault(tid, set()).add(parent)
        return tid

    def add_group(self, name: str, parent: Optional[str] = None) -> str:
        gid = self._id()
        self.groups[gid] = name
        if parent:
            self.parents["groups"].setdefault(gid, set()).add(parent)
        return gid

    def add_studio(self, name: str, parent: Optional[str] = None) -> str:
        sid = self._id()
        self.studios[sid] = name
        if parent:  # a studio has one parent in Stash
            self.parents["studios"][sid] = {parent}
        return sid

    def add_file(self, path: str, size: int, *, duration: float = 600.0, oshash: Optional[str] = None,
                 md5: Optional[str] = None, codec: str = "h264", width: int = 1920, height: int = 1080,
                 frame_rate: float = 30.0) -> str:
        fid = self._id()
        fps = []
        if oshash:
            fps.append({"type": "oshash", "value": oshash})
        if md5:
            fps.append({"type": "md5", "value": md5})
        self.files[fid] = {
            "id": fid, "path": path, "basename": os.path.basename(path), "size": size,
            "duration": duration, "format": "mp4", "video_codec": codec, "width": width,
            "height": height, "bit_rate": 5_000_000, "frame_rate": frame_rate, "fingerprints": fps,
        }
        return fid

    def add_scene(self, *file_ids: str, title: str = "", tag_ids: tuple[str, ...] = (), **fields) -> str:
        sid = self._id()
        self.scenes[sid] = {
            "id": sid, "title": title, "details": "", "date": None, "code": "", "director": "",
            "urls": [], "organized": False, "rating100": None, "studio_id": None,
            "performer_ids": [], "tag_ids": list(tag_ids), "group_ids": [], "groups": [],
            "gallery_ids": [], "stash_ids": [], "play_count": 0, "o_counter": 0,
            "file_ids": list(file_ids), **fields,
        }
        return sid

    # ---------------- reads ----------------

    def _render(self, s: dict) -> dict:
        return copy.deepcopy({
            "id": s["id"], "title": s["title"], "details": s["details"], "date": s["date"],
            "code": s["code"], "director": s["director"], "urls": s["urls"],
            "organized": s["organized"], "rating100": s["rating100"],
            "studio": {"id": s["studio_id"]} if s["studio_id"] else None,
            "performers": [{"id": p} for p in s["performer_ids"]],
            "tags": [{"id": t} for t in s["tag_ids"]],
            "play_count": s["play_count"], "o_counter": s["o_counter"],
            "stash_ids": s["stash_ids"],
            "groups": [{"group": {"id": g["group_id"]}, "scene_index": g.get("scene_index")}
                       for g in s["groups"]],
            "galleries": [{"id": g} for g in s["gallery_ids"]],
            "files": [self.files[f] for f in s["file_ids"]],
        })

    def _maybe_fail(self, name: str) -> None:
        if name in self.fail:
            raise self.fail.pop(name)

    async def get_scene_for_review(self, scene_id: str) -> Optional[dict]:
        self._maybe_fail("get_scene_for_review")
        s = self.scenes.get(str(scene_id))
        return self._render(s) if s else None

    async def find_scenes_with_tag_name(self, tag_name: str) -> list[dict]:
        tid = next((t for t, n in self.tags.items() if n == tag_name), None)
        if tid is None:
            return []
        return [self._render(s) for s in sorted(self.scenes.values(), key=lambda s: int(s["id"]))
                if tid in s["tag_ids"]]

    async def get_delete_trash_path(self) -> str:
        return self.trash_path

    async def get_performer_scenes(self, performer_id: str) -> list[dict]:
        return [self._render(s) for s in sorted(self.scenes.values(), key=lambda s: int(s["id"]))
                if str(performer_id) in s["performer_ids"]]

    async def get_scene_files_page(self, page: int, per_page: int = 500) -> tuple[list[dict], int]:
        """findScenes sorted by id, paged from 1: a scene's files primary first, the count of every scene."""
        self._maybe_fail("get_scene_files_page")
        ordered = sorted(self.scenes.values(), key=lambda s: int(s["id"]))
        chunk = ordered[(page - 1) * per_page: page * per_page]
        return [copy.deepcopy({
            "id": s["id"], "created_at": s.get("created_at") or f"2026-01-01T00:00:{int(s['id']) % 60:02d}Z",
            "files": [{k: self.files[f][k] for k in ("id", "path", "size", "duration", "fingerprints")}
                      for f in s["file_ids"]],
        }) for s in chunk], len(ordered)

    async def get_scenes_for_duplicates(self, page: int, per_page: int = 500) -> tuple[list[dict], int]:
        """findScenes sorted by id, paged from 1, with the fields duplicate candidate generation reads."""
        self._maybe_fail("get_scenes_for_duplicates")
        ordered = sorted(self.scenes.values(), key=lambda s: int(s["id"]))
        chunk = ordered[(page - 1) * per_page: page * per_page]
        out = []
        for s in chunk:
            r = self._render(s)
            out.append({
                **{k: r[k] for k in ("id", "title", "date", "urls", "studio", "performers", "stash_ids")},
                "files": [{k: f[k] for k in ("id", "path", "size", "duration", "fingerprints")} for f in r["files"]],
            })
        return out, len(ordered)

    async def find_duplicate_scenes(self, distance: int = 10) -> list[list[str]]:
        """Scene id groups; the fake returns the groups set on ``phash_groups``."""
        self._maybe_fail("find_duplicate_scenes")
        return [[str(i) for i in g] for g in self.phash_groups]

    async def get_plugin_configuration(self, plugin_id: str) -> dict:
        self._maybe_fail("get_plugin_configuration")
        return copy.deepcopy(self.plugin_config.get(plugin_id) or {})

    async def configure_plugin(self, plugin_id: str, values: dict) -> dict:
        """Stash's configurePlugin: the input replaces the whole map."""
        self.plugin_config[plugin_id] = copy.deepcopy(values)
        return copy.deepcopy(values)

    def _with_descendants(self, kind: str, ids: set[str], depth: int) -> set[str]:
        out, frontier, level = set(ids), set(ids), 0
        while frontier and (depth < 0 or level < depth):
            frontier = {c for c, ps in self.parents[kind].items() if ps & frontier} - out
            out |= frontier
            level += 1
        return out

    async def find_scene_ids(self, scene_filter: dict) -> list[str]:
        self._maybe_fail("find_scene_ids")
        if len(scene_filter) != 1:
            raise FakeStashError(f"fake findScenes models one criterion, got {sorted(scene_filter)}")
        (field, crit), = scene_filter.items()
        if field not in ("tags", "groups", "studios") or crit.get("modifier") != "INCLUDES":
            raise FakeStashError(f"fake findScenes does not model {field} {crit.get('modifier')}")
        wanted = self._with_descendants(field, {str(v) for v in crit["value"]}, crit.get("depth") or 0)

        def scene_values(s):
            if field == "studios":
                return {s["studio_id"]} if s["studio_id"] else set()
            if field == "groups":
                return set(s["group_ids"]) | {g["group_id"] for g in s["groups"]}
            return set(s["tag_ids"])

        return [s["id"] for s in sorted(self.scenes.values(), key=lambda s: int(s["id"]))
                if scene_values(s) & wanted]

    # ---------------- writes ----------------

    async def update_scene(self, scene_id: str, **fields) -> dict:
        self._maybe_fail("update_scene")
        s = self.scenes.get(str(scene_id))
        if s is None:
            raise FakeStashError(f"GraphQL error: scene with id {scene_id} not found")
        unknown = set(fields) - SCENE_FIELDS - {"primary_file_id"}
        if unknown:
            raise FakeStashError(f"GraphQL error: unknown field(s) {sorted(unknown)} in SceneUpdateInput")
        pf = fields.pop("primary_file_id", None)
        if pf is not None:
            if str(pf) not in s["file_ids"]:
                raise FakeStashError(f"GraphQL error: file {pf} is not associated with scene {scene_id}")
            s["file_ids"].remove(str(pf))
            s["file_ids"].insert(0, str(pf))
        for k, v in fields.items():
            s[k] = copy.deepcopy(v)
        self.writes += 1
        return {"id": s["id"]}

    async def merge_scenes(self, source_ids: list[str], destination_id: str,
                           values: dict | None = None, play_history: bool = True,
                           o_history: bool = True) -> dict:
        """Stash's sceneMerge (pkg/scene/merge.go Merge). Unions nothing: the caller sends merged values."""
        self._maybe_fail("merge_scenes")
        sources = list(dict.fromkeys(str(i) for i in source_ids))
        dest = str(destination_id)
        if dest in sources:
            raise FakeStashError("GraphQL error: destination scene cannot be in source list")
        d = self.scenes.get(dest)
        if d is None:
            raise FakeStashError(f"GraphQL error: finding destination scene ID {dest}")
        missing = [i for i in sources if i not in self.scenes]
        if missing:
            raise FakeStashError(f"GraphQL error: finding source scenes: {missing} not found")
        values = copy.deepcopy(values or {})
        moved = [f for i in sources for f in self.scenes[i]["file_ids"]]
        if moved:
            d["file_ids"].extend(f for f in moved if f not in d["file_ids"])
            # The destination keeps its primary, so the input cannot change it.
            values.pop("primary_file_id", None)
        unknown = set(values) - SCENE_FIELDS - {"primary_file_id", "id"}
        if unknown:
            raise FakeStashError(f"GraphQL error: unknown field(s) {sorted(unknown)} in SceneUpdateInput")
        values.pop("id", None)
        pf = values.pop("primary_file_id", None)  # only still present when no file moved
        if pf is not None:
            if str(pf) not in d["file_ids"]:
                raise FakeStashError(f"GraphQL error: file {pf} is not associated with scene {dest}")
            d["file_ids"].remove(str(pf))
            d["file_ids"].insert(0, str(pf))
        for k, v in values.items():
            d[k] = v
        if play_history:
            d["play_count"] += sum(self.scenes[i]["play_count"] for i in sources)
        if o_history:
            d["o_counter"] += sum(self.scenes[i]["o_counter"] for i in sources)
        for i in sources:  # destroyed without deleting their files
            del self.scenes[i]
        self.writes += 1
        return {"id": dest}

    async def delete_files(self, file_ids: list[str]) -> bool:
        self._maybe_fail("delete_files")
        ids = [str(i) for i in file_ids]
        # Validate everything first: Stash rolls back the whole transaction on any error.
        for fid in ids:
            f = self.files.get(fid)
            if f is None:
                raise FakeStashError(f"GraphQL error: file with id {fid} not found")
            if any(s["file_ids"] and s["file_ids"][0] == fid for s in self.scenes.values()):
                raise FakeStashError(f"GraphQL error: cannot delete primary file {f['path']}")
        for fid in ids:
            f = self.files.pop(fid)
            for s in self.scenes.values():
                if fid in s["file_ids"]:
                    s["file_ids"].remove(fid)
            dest = os.path.join(self.trash_path, f["basename"]) if self.trash_path else None
            self.deleted.append({"path": f["path"], "destination": dest})
        self.writes += 1
        return True

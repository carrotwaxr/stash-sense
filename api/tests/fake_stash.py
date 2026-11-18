"""An in-memory Stash that enforces the rules the review framework depends on.

Modelled on Stash v0.32 (internal/api/resolver_mutation_file.go, scene update):
- a scene's files are ordered primary first
- sceneUpdate rejects unknown fields and a primary_file_id that isn't on the scene
- deleteFiles is all-or-nothing and refuses any file that is a scene's primary
- with deleteTrashPath set, deleted files move to trash; otherwise they are gone

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
        "delete_files", "get_delete_trash_path",
    )

    def __init__(self, trash_path: str = ""):
        self.scenes: dict[str, dict] = {}
        self.files: dict[str, dict] = {}
        self.tags: dict[str, str] = {}
        self.trash_path = trash_path
        self.deleted: list[dict] = []  # {"path", "destination"}
        self.writes = 0
        self.fail: dict[str, Exception] = {}  # method name -> error raised on next call
        self._next_id = 1

    # ---------------- seeding ----------------

    def _id(self) -> str:
        self._next_id += 1
        return str(self._next_id)

    def add_tag(self, name: str) -> str:
        tid = self._id()
        self.tags[tid] = name
        return tid

    def add_file(self, path: str, size: int, *, duration: float = 600.0, oshash: Optional[str] = None,
                 md5: Optional[str] = None, codec: str = "h264", width: int = 1920, height: int = 1080) -> str:
        fid = self._id()
        fps = []
        if oshash:
            fps.append({"type": "oshash", "value": oshash})
        if md5:
            fps.append({"type": "md5", "value": md5})
        self.files[fid] = {
            "id": fid, "path": path, "basename": os.path.basename(path), "size": size,
            "duration": duration, "format": "mp4", "video_codec": codec, "width": width,
            "height": height, "bit_rate": 5_000_000, "fingerprints": fps,
        }
        return fid

    def add_scene(self, *file_ids: str, title: str = "", tag_ids: tuple[str, ...] = (), **fields) -> str:
        sid = self._id()
        self.scenes[sid] = {
            "id": sid, "title": title, "details": "", "date": None, "code": "", "director": "",
            "urls": [], "organized": False, "rating100": None, "studio_id": None,
            "performer_ids": [], "tag_ids": list(tag_ids), "file_ids": list(file_ids), **fields,
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

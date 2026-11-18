"""review_demo: a test recommendation type that exercises the whole review flow.

Only scenes carrying the tag DEMO_TAG are considered, so it never touches a
library that didn't opt in. For each such scene it suggests:

- a title taken from the kept file's basename (a metadata change, revertible)
- when the scene has several files: keep the largest, delete the rest

It is high-confidence (batch-eligible) when the suggestion is unchanged and
the scene has one file, or every other file shares an oshash or md5 with the
keeper. Picking another keeper or editing the title records a label.
"""

from __future__ import annotations

import os
from typing import Optional

from analyzers.base import AnalysisResult, BaseAnalyzer

from .plan import (
    FieldChange,
    FileRef,
    Plan,
    Preview,
    PrimaryFileChange,
    scene_field_values,
)
from .types import InvalidOption, ReviewType, TargetMissing, register_review_type

DEMO_TAG = "stash-sense-review-demo"
TYPE_ID = "review_demo"


def _title_from(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _identical(a: dict, b: dict) -> bool:
    fa = {fp["type"]: fp["value"] for fp in a.get("fingerprints") or []}
    fb = {fp["type"]: fp["value"] for fp in b.get("fingerprints") or []}
    return any(fa.get(k) and fa.get(k) == fb.get(k) for k in ("md5", "oshash"))


@register_review_type
class DemoReviewType(ReviewType):
    type_id = TYPE_ID
    display_name = "Review Demo"
    description = f"Test type: renames and dedupes files of scenes tagged '{DEMO_TAG}'"

    async def preview(self, rec, option: dict | None, stash) -> Preview:
        scene = await stash.get_scene_for_review(rec.target_id)
        if not scene:
            raise TargetMissing(f"Scene {rec.target_id} no longer exists")
        files = scene.get("files") or []
        if not files:
            raise TargetMissing(f"Scene {rec.target_id} has no files")
        by_id = {str(f["id"]): f for f in files}
        primary_id = str(files[0]["id"])

        # Largest file wins; the current primary breaks ties.
        suggested_keeper = max(files, key=lambda f: (int(f.get("size") or 0), str(f["id"]) == primary_id))
        default = {"keep_file_id": str(suggested_keeper["id"]), "title": _title_from(suggested_keeper["path"])}
        chosen = {**default, **(option or {})}
        keep_id = str(chosen["keep_file_id"])
        if keep_id not in by_id:
            raise InvalidOption(f"File {keep_id} is not on scene {rec.target_id}")
        title = str(chosen.get("title") or "").strip()
        if not title:
            raise InvalidOption("Title cannot be empty")
        chosen = {"keep_file_id": keep_id, "title": title}

        current = scene_field_values(scene)
        sid = str(scene["id"])
        plan = Plan(rec_id=rec.id, rec_type=self.type_id, option=chosen)
        if current["title"] != title:
            plan.field_changes.append(FieldChange("scene", sid, "title", current["title"], title))
        keeper = by_id[keep_id]
        plan.keep_files = [FileRef.from_stash(sid, keeper)]
        plan.delete_files = [FileRef.from_stash(sid, f) for f in files if str(f["id"]) != keep_id]
        if plan.delete_files and keep_id != primary_id:
            plan.primary_file = PrimaryFileChange(sid, primary_id, keep_id)

        others = [f for f in files if str(f["id"]) != keep_id]
        identical = [str(f["id"]) for f in others if _identical(f, keeper)]
        high = chosen == default and len(identical) == len(others)

        evidence = {
            "scene": {"id": sid, "title": current["title"]},
            "files": [
                {
                    "file_id": str(f["id"]), "path": f["path"], "size": int(f.get("size") or 0),
                    "duration": f.get("duration"), "video_codec": f.get("video_codec"),
                    "width": f.get("width"), "height": f.get("height"), "bit_rate": f.get("bit_rate"),
                    "is_primary": str(f["id"]) == primary_id,
                    "identical_to_keeper": str(f["id"]) in identical,
                }
                for f in files
            ],
            "stills": [
                {"scene_id": sid, "file_id": str(f["id"]), "t": round((f.get("duration") or 0) * 0.2, 1)}
                for f in files
            ],
            "playback": {"scene_id": sid, "t": round((keeper.get("duration") or 0) * 0.2, 1)},
        }
        choices = [{"field": "title", "label": "Title", "kind": "text"}]
        if len(files) > 1:
            choices.insert(0, {
                "field": "keep_file_id", "label": "File to keep", "kind": "pick",
                "values": [{"value": str(f["id"]), "label": os.path.basename(f["path"])} for f in files],
            })
        parts = []
        if plan.field_changes:
            parts.append("rename")
        if plan.delete_files:
            parts.append(f"delete {len(plan.delete_files)} file(s)")
        return Preview(
            plan=plan, evidence=evidence, choices=choices, default_option=default,
            high_confidence=high, summary=", ".join(parts) or "no change",
        )


class DemoReviewAnalyzer(BaseAnalyzer):
    """Creates one review_demo recommendation per scene tagged DEMO_TAG."""

    type = TYPE_ID

    async def run(self, incremental: bool = True) -> AnalysisResult:
        scenes = await self.stash.find_scenes_with_tag_name(DEMO_TAG)
        self.set_items_total(len(scenes))
        created = 0
        for i, scene in enumerate(scenes, 1):
            if self.is_stop_requested():
                break
            rec_id = self.create_recommendation(
                target_type="scene", target_id=str(scene["id"]),
                details={"tag": DEMO_TAG}, confidence=1.0, source_analysis_id=self.run_id,
            )
            created += rec_id is not None
            self.update_progress(i, created)
        return AnalysisResult(items_processed=len(scenes), recommendations_created=created)

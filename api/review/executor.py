"""The one place a reviewed Plan becomes Stash writes, each logged first."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Optional

from .plan import SCENE_FIELDS, Plan
from .store import ReviewStore

logger = logging.getLogger(__name__)


class PlanRejected(ValueError):
    """The plan breaks a safety rule; nothing was written."""


class ApplyFailed(RuntimeError):
    """A write failed part way; the log entry records what ran."""

    def __init__(self, message: str, entry: dict):
        super().__init__(message)
        self.entry = entry


async def _check_files(plan: Plan, stash) -> None:
    """Re-read each scene a deletion touches and refuse anything unsafe."""
    by_scene: dict[str, set[str]] = defaultdict(set)
    for f in plan.delete_files:
        by_scene[f.scene_id].add(f.file_id)
    for scene_id, doomed in by_scene.items():
        scene = await stash.get_scene_for_review(scene_id)
        if not scene:
            raise PlanRejected(f"Scene {scene_id} no longer exists")
        attached = [str(f["id"]) for f in scene.get("files") or []]
        missing = doomed - set(attached)
        if missing:
            raise PlanRejected(f"Files {sorted(missing)} are no longer on scene {scene_id}")
        survivors = [fid for fid in attached if fid not in doomed]
        if not survivors:
            raise PlanRejected(f"Plan would delete every file of scene {scene_id}")
        primary = attached[0]
        if primary in doomed:
            pf = plan.primary_file
            if not pf or pf.scene_id != scene_id or pf.after not in survivors:
                raise PlanRejected(
                    f"Plan deletes the primary file of scene {scene_id} without moving primary to a kept file"
                )


def _check_fields(plan: Plan) -> None:
    for c in plan.field_changes:
        if c.entity_type != "scene":
            raise PlanRejected(f"Entity type {c.entity_type!r} is not supported yet")
        if c.field not in SCENE_FIELDS:
            raise PlanRejected(f"Scene field {c.field!r} is not writable through review")


async def apply_plan(
    plan: Plan,
    stash,
    store: ReviewStore,
    *,
    target_type: str | None,
    target_id: str | None,
    action: str = "apply",
    batch_id: str | None = None,
    reverts_id: int | None = None,
) -> dict:
    """Apply plan and return its apply_log entry. Raises PlanRejected or ApplyFailed."""
    _check_fields(plan)
    if plan.delete_files:
        await _check_files(plan, stash)
    trash_path = await stash.get_delete_trash_path() if plan.delete_files else ""

    prior_state = {
        "fields": [
            {"entity_type": c.entity_type, "entity_id": c.entity_id, "field": c.field, "value": c.before}
            for c in plan.field_changes
        ],
        "primary_file": asdict(plan.primary_file) if plan.primary_file else None,
        "deleted_files": [asdict(f) for f in plan.delete_files],
    }
    entry_id = store.begin(
        action=action, rec_id=plan.rec_id, rec_type=plan.rec_type,
        target_type=target_type, target_id=target_id, option=plan.option,
        plan=plan.to_dict(), plan_token=plan.token(), prior_state=prior_state,
        trash_path=trash_path, batch_id=batch_id, reverts_id=reverts_id,
    )

    steps: list[dict] = []
    try:
        per_scene: dict[str, dict] = defaultdict(dict)
        for c in plan.field_changes:
            per_scene[c.entity_id][c.field] = c.after
        for scene_id, fields in per_scene.items():
            await stash.update_scene(scene_id, **fields)
            steps.append({"step": "update_scene", "scene_id": scene_id, "fields": sorted(fields)})

        if plan.primary_file:
            pf = plan.primary_file
            await stash.update_scene(pf.scene_id, primary_file_id=pf.after)
            steps.append({"step": "set_primary_file", "scene_id": pf.scene_id, "file_id": pf.after})

        if plan.delete_files:
            ids = [f.file_id for f in plan.delete_files]
            await stash.delete_files(ids)
            steps.append({
                "step": "delete_files", "file_ids": ids,
                "destination": f"trash: {trash_path}" if trash_path else "permanently deleted",
            })
    except Exception as e:
        logger.warning("Apply %d (%s rec %s) failed after %d steps: %s",
                       entry_id, plan.rec_type, plan.rec_id, len(steps), e)
        store.finish(entry_id, status="failed", result={"steps": steps}, error=str(e))
        raise ApplyFailed(str(e), store.get(entry_id)) from e

    store.finish(entry_id, status="applied", result={
        "steps": steps,
        "bytes_reclaimed": plan.bytes_reclaimed,
        "files_deleted": len(plan.delete_files),
        "fields_changed": len(plan.field_changes),
    })
    return store.get(entry_id)

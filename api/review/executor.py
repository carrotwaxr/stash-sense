"""The one place a reviewed Plan becomes Stash writes, each logged first."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import asdict
from typing import Optional

from .plan import SCENE_FIELDS, Plan, scene_field_values
from .store import ReviewStore

logger = logging.getLogger(__name__)


class PlanRejected(ValueError):
    """The plan breaks a safety rule; nothing was written."""


class ApplyFailed(RuntimeError):
    """A write failed part way; the log entry records what ran."""

    def __init__(self, message: str, entry: dict):
        super().__init__(message)
        self.entry = entry


def _file_ids(scene: dict) -> list[str]:
    return [str(f["id"]) for f in scene.get("files") or []]


def _check_survivors(plan: Plan, scene_id: str, attached: list[str], doomed: set[str], what: str) -> None:
    """attached is the scene's files, primary first, as they will be when the deletes run."""
    survivors = [fid for fid in attached if fid not in doomed]
    if not survivors:
        raise PlanRejected(f"Plan would delete every file of {what}")
    if attached[0] in doomed:
        pf = plan.primary_file
        if not pf or str(pf.scene_id) != scene_id or pf.after not in survivors:
            raise PlanRejected(
                f"Plan deletes the primary file of {what} without moving primary to a kept file"
            )


async def _check_files(plan: Plan, stash) -> dict[str, dict]:
    """Re-read each scene a merge or deletion touches and refuse anything unsafe.

    Returns the re-read merge scenes by id (empty without a merge).
    """
    by_scene: dict[str, set[str]] = defaultdict(set)
    for f in plan.delete_files:
        by_scene[str(f.scene_id)].add(str(f.file_id))

    merged: dict[str, dict] = {}
    m = plan.merge
    if m:
        for sid in m.scene_ids:
            scene = await stash.get_scene_for_review(sid)
            if not scene:
                raise PlanRejected(f"Scene {sid} no longer exists")
            merged[sid] = scene
        # Deletes name the scene a file is on before the merge.
        for sid in m.scene_ids:
            missing = by_scene.get(sid, set()) - set(_file_ids(merged[sid]))
            if missing:
                raise PlanRejected(f"Files {sorted(missing)} are no longer on scene {sid}")
        # After the merge the destination holds its own files, then each source's, primary first.
        dest = str(m.destination)
        combined = list(dict.fromkeys(fid for sid in m.scene_ids for fid in _file_ids(merged[sid])))
        doomed: set[str] = set().union(*(by_scene.pop(sid, set()) for sid in m.scene_ids))
        if doomed:
            _check_survivors(plan, dest, combined, doomed, f"scene {dest} after the merge")
        pf = plan.primary_file
        if pf and str(pf.scene_id) == dest and (pf.after not in combined or pf.after in doomed):
            raise PlanRejected(f"Primary file {pf.after} is not a kept file of scene {dest} after the merge")

    for scene_id, doomed in by_scene.items():
        scene = await stash.get_scene_for_review(scene_id)
        if not scene:
            raise PlanRejected(f"Scene {scene_id} no longer exists")
        attached = _file_ids(scene)
        missing = doomed - set(attached)
        if missing:
            raise PlanRejected(f"Files {sorted(missing)} are no longer on scene {scene_id}")
        _check_survivors(plan, scene_id, attached, doomed, f"scene {scene_id}")
    return merged


def _check_fields(plan: Plan) -> None:
    for c in plan.field_changes:
        if c.entity_type != "scene":
            raise PlanRejected(f"Entity type {c.entity_type!r} is not supported yet")
        if c.field not in SCENE_FIELDS:
            raise PlanRejected(f"Scene field {c.field!r} is not writable through review")
    m = plan.merge
    if not m:
        return
    if not m.sources:
        raise PlanRejected("Merge has no source scenes")
    sources = {str(s) for s in m.sources}
    if str(m.destination) in sources:
        raise PlanRejected(f"Merge destination {m.destination} is also a source")
    for k in m.values:
        if k not in SCENE_FIELDS:
            raise PlanRejected(f"Scene field {k!r} is not writable through review")
    for c in plan.field_changes:
        if str(c.entity_id) in m.scene_ids:
            raise PlanRejected(
                f"Field change on merged scene {c.entity_id}: merged scene fields belong in the merge values"
            )
    if plan.primary_file and str(plan.primary_file.scene_id) in sources:
        raise PlanRejected(f"Primary file change on source scene {plan.primary_file.scene_id}, which the merge destroys")


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
    merged = await _check_files(plan, stash) if plan.delete_files or plan.merge else {}
    trash_path = await stash.get_delete_trash_path() if plan.delete_files else ""

    prior_state = {
        "fields": [
            {"entity_type": c.entity_type, "entity_id": c.entity_id, "field": c.field, "value": c.before}
            for c in plan.field_changes
        ],
        "primary_file": asdict(plan.primary_file) if plan.primary_file else None,
        "deleted_files": [asdict(f) for f in plan.delete_files],
    }
    if plan.merge:
        m = plan.merge
        dest = merged[str(m.destination)]
        dest_values = scene_field_values(dest)
        prior_state["merged_scenes"] = [merged[str(s)] for s in m.sources]
        prior_state["merge_destination"] = {
            "scene_id": str(m.destination),
            "fields": {k: dest_values[k] for k in m.values},
            "scene": dest,
        }
    entry_id = store.begin(
        action=action, rec_id=plan.rec_id, rec_type=plan.rec_type,
        target_type=target_type, target_id=target_id, option=plan.option,
        plan=plan.to_dict(), plan_token=plan.token(), prior_state=prior_state,
        trash_path=trash_path, batch_id=batch_id, reverts_id=reverts_id,
    )

    steps: list[dict] = []
    try:
        if plan.merge:
            m = plan.merge
            await stash.merge_scenes(m.sources, m.destination, values=m.values)
            steps.append({"step": "merge_scenes", "scene_id": str(m.destination),
                          "source_ids": [str(s) for s in m.sources], "fields": sorted(m.values)})

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

        if plan.split:  # nothing to write: the accept itself is the hand-off
            sp = plan.split
            steps.append({"step": "split_handoff", "scene_id": str(sp.scene_id), "file_id": str(sp.file_id),
                          "cuts": len(sp.cuts), "ranges": len(sp.ranges)})
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
        **({"scenes_merged": len(plan.merge.sources)} if plan.merge else {}),
    })
    return store.get(entry_id)

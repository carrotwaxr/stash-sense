"""Review flow: preview, accept, batch accept, revert, labels.

Every accept rebuilds the plan from Stash's current state and must match
the token of the preview the user saw; every delete must be named in the
request's confirm list. Errors carry an HTTP status and a payload the UI
can render (the fresh preview, the files to confirm, the drifted fields).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Optional

import curation_settings
import features
from curation_settings import SceneExcluded
from recommendations_db import Recommendation, RecommendationsDB

from .executor import ApplyFailed, PlanRejected, apply_plan
from .plan import FieldChange, Plan, Preview, scene_field_values, values_equal
from .store import ReviewStore, new_batch_id
from .types import InvalidOption, ReviewType, TargetMissing, get_review_type


class ReviewError(Exception):
    status = 400
    code = "review_error"

    def __init__(self, message: str, **payload):
        super().__init__(message)
        self.message = message
        self.payload = payload

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, **self.payload}


class NotFound(ReviewError):
    status, code = 404, "not_found"


class HiddenType(ReviewError):
    status, code = 403, "hidden_type"


class NotPending(ReviewError):
    status, code = 409, "not_pending"


class Stale(ReviewError):
    status, code = 409, "stale"


class ConfirmRequired(ReviewError):
    status, code = 409, "confirm_delete"


class BatchIneligible(ReviewError):
    status, code = 400, "not_batch_eligible"


class NotRevertible(ReviewError):
    status, code = 409, "not_revertible"


class Drifted(ReviewError):
    status, code = 409, "drifted"


class Rejected(ReviewError):
    status, code = 422, "rejected"


class Failed(ReviewError):
    status, code = 502, "apply_failed"


class Excluded(ReviewError):
    status, code = 403, "scene_excluded"


logger = logging.getLogger(__name__)


def _plan_scene_ids(plan: Plan) -> set[str]:
    """Every scene a plan writes: field changes, primary file, merged scenes, and the scenes of deleted files."""
    ids = {c.entity_id for c in plan.field_changes}
    ids |= {f.scene_id for f in plan.delete_files}
    if plan.primary_file:
        ids.add(plan.primary_file.scene_id)
    if plan.merge:
        ids.update(plan.merge.scene_ids)
    return {str(i) for i in ids if i is not None}


def _files_payload(preview: Preview) -> list[dict]:
    return [{"file_id": f.file_id, "path": f.path, "size": f.size} for f in preview.plan.delete_files]


class ReviewService:
    def __init__(self, rec_db: RecommendationsDB, store: ReviewStore, stash):
        self.rec_db = rec_db
        self.store = store
        self.stash = stash
        # One apply at a time: a double-click or two tabs can't apply a rec twice.
        self._apply_lock = asyncio.Lock()

    # ---------------- lookups ----------------

    def _type(self, type_id: str) -> ReviewType:
        rtype = get_review_type(type_id)
        if rtype is None:
            raise NotFound(f"'{type_id}' is not a review type")
        if type_id in features.hidden_recommendation_types():
            raise HiddenType(
                f"'{type_id}' is not production-ready yet; turn on "
                "Settings > Features > Show Unfinished Features to use it"
            )
        return rtype

    def _rec(self, rec_id: int) -> tuple[Recommendation, ReviewType]:
        rec = self.rec_db.get_recommendation(rec_id)
        if rec is None:
            raise NotFound(f"Recommendation {rec_id} not found")
        return rec, self._type(rec.type)

    async def _refresh_exclusion(self) -> None:
        """Read the #125 list fresh; writes call this first so later checks in the
        same request use a list no older than the request."""
        await curation_settings.current(self.stash, max_age=0)

    async def _check_exclusion(self, scene_ids) -> None:
        """Refuse scenes on the #125 exclusion list (ExclusionUnavailable propagates: fail closed)."""
        hit = (await curation_settings.current(self.stash)).excluded(scene_ids)
        if hit:
            raise Excluded(str(SceneExcluded(hit)), scene_ids=sorted(hit))

    async def _build(self, rec: Recommendation, rtype: ReviewType, option: dict | None) -> Preview:
        await self._check_exclusion(rtype.scene_ids(rec))
        try:
            preview = await rtype.preview(rec, option, self.stash)
        except TargetMissing as e:
            raise NotFound(str(e)) from e
        except InvalidOption as e:
            raise Rejected(str(e)) from e
        # The plan may write scenes the recommendation doesn't name; check those too.
        await self._check_exclusion(_plan_scene_ids(preview.plan))
        return preview

    # ---------------- single item ----------------

    async def preview(self, rec_id: int, option: dict | None = None) -> dict:
        rec, rtype = self._rec(rec_id)
        preview = await self._build(rec, rtype, option)
        out = preview.to_dict()
        out["status"] = rec.status
        # Where deleted files go, for the confirm dialog ("" = permanently deleted).
        out["trash_path"] = await self.stash.get_delete_trash_path() if preview.plan.delete_files else ""
        return out

    async def accept(
        self,
        rec_id: int,
        *,
        option: dict | None,
        plan_token: str,
        confirm_delete: list[str] | None = None,
    ) -> dict:
        rec, rtype = self._rec(rec_id)
        await self._refresh_exclusion()
        async with self._apply_lock:
            rec = self.rec_db.get_recommendation(rec_id)
            if rec.status != "pending":
                done = self.store.latest_applied_for_rec(rec_id)
                if rec.status == "resolved" and done:
                    return {"already_applied": True, "entry": done}
                raise NotPending(f"Recommendation {rec_id} is {rec.status}")
            preview = await self._build(rec, rtype, option)
            return await self._apply(rec, rtype, preview, plan_token, confirm_delete)

    async def _apply(
        self,
        rec: Recommendation,
        rtype: ReviewType,
        preview: Preview,
        plan_token: str,
        confirm_delete: list[str] | None,
        batch_id: str | None = None,
    ) -> dict:
        plan = preview.plan
        if plan_token != plan.token():
            raise Stale(
                "The library changed since this preview; review the updated plan",
                preview=preview.to_dict(),
            )
        if {str(i) for i in confirm_delete or []} != plan.delete_file_ids:
            raise ConfirmRequired(
                f"Confirm the {len(plan.delete_files)} file(s) this deletes",
                files=_files_payload(preview),
                bytes=plan.bytes_reclaimed,
            )
        try:
            entry = await apply_plan(
                plan, self.stash, self.store,
                target_type=rec.target_type, target_id=rec.target_id, batch_id=batch_id,
            )
        except PlanRejected as e:
            raise Rejected(str(e)) from e
        except ApplyFailed as e:
            raise Failed(f"Apply failed: {e}", entry=e.entry) from e

        # The writes are done and logged; a bookkeeping failure must not hide that.
        warnings = []
        try:
            self.rec_db.resolve_recommendation(
                rec.id, action="applied",
                details={"apply_log_id": entry["id"], "option": plan.option,
                         **({"split": asdict(plan.split)} if plan.split else {})},
            )
        except Exception as e:
            logger.warning("Applied %s (log %d) but could not resolve rec %d: %s",
                           rec.type, entry["id"], rec.id, e)
            warnings.append(f"applied, but the recommendation could not be marked resolved: {e}")
        label_id = None
        try:
            label = rtype.label(rec, preview)
            if label:
                label_id = self.store.add_label(
                    rec_type=rec.type, rec_id=rec.id, target_type=rec.target_type,
                    target_id=rec.target_id, suggested=label["suggested"], chosen=label["chosen"],
                    evidence=preview.evidence, apply_log_id=entry["id"],
                )
        except Exception as e:
            logger.warning("Applied %s (log %d) but could not save its label: %s", rec.type, entry["id"], e)
            warnings.append(f"applied, but the correction could not be saved as a label: {e}")
        return {"already_applied": False, "entry": entry, "label_id": label_id, "warnings": warnings}

    # ---------------- batch ----------------

    async def batch_preview(self, type_id: str, limit: int = 200) -> dict:
        rtype = self._type(type_id)
        excluded = (await curation_settings.current(self.stash)).scene_ids
        recs = self.rec_db.get_recommendations(
            status="pending", type=type_id, limit=limit, exclude_scene_ids=excluded,
        )
        items, skipped = [], []
        for rec in recs:
            try:
                preview = await self._build(rec, rtype, None)
            except ReviewError as e:  # a missing scene shouldn't sink the batch
                skipped.append({"rec_id": rec.id, "reason": f"preview failed: {e}"})
                continue
            if not preview.high_confidence:
                skipped.append({"rec_id": rec.id, "reason": "not high-confidence"})
                continue
            p = preview.plan
            items.append({
                "rec_id": rec.id,
                "summary": preview.summary,
                "plan_token": p.token(),
                "field_changes": [c.__dict__ for c in p.field_changes],
                "delete_files": _files_payload(preview),
                "bytes": p.bytes_reclaimed,
            })
        any_deletes = any(i["delete_files"] for i in items)
        return {
            "type": type_id,
            "trash_path": await self.stash.get_delete_trash_path() if any_deletes else "",
            "items": items,
            "skipped": skipped,
            "summary": {
                "items": len(items),
                "files_deleted": sum(len(i["delete_files"]) for i in items),
                "bytes_reclaimed": sum(i["bytes"] for i in items),
                "field_changes": sum(len(i["field_changes"]) for i in items),
                "skipped": len(skipped),
            },
        }

    async def batch_accept(self, type_id: str, items: list[dict]) -> dict:
        """Apply items with the type's default option. Refuses the whole batch if any item
        is not high-confidence; per-item staleness is reported and the rest proceed."""
        rtype = self._type(type_id)
        await self._refresh_exclusion()
        async with self._apply_lock:
            built: list[tuple[Recommendation, dict]] = []
            ineligible: list[dict] = []
            for item in items:
                if "option" in item:
                    raise BatchIneligible("Batch accept applies the suggested option only")
                rec = self.rec_db.get_recommendation(int(item["rec_id"]))
                if rec is None or rec.type != type_id:
                    ineligible.append({"rec_id": item["rec_id"], "reason": f"not a {type_id} recommendation"})
                    continue
                if rec.status != "pending":
                    ineligible.append({"rec_id": rec.id, "reason": f"is {rec.status}"})
                    continue
                try:
                    preview = await self._build(rec, rtype, None)
                except ReviewError as e:
                    ineligible.append({"rec_id": rec.id, "reason": e.message})
                    continue
                if not preview.high_confidence:
                    ineligible.append({"rec_id": rec.id, "reason": "not high-confidence"})
                    continue
                built.append((rec, item))
            if ineligible:
                raise BatchIneligible(
                    "Batch refused: some items are not eligible for batch accept",
                    items=ineligible,
                )

            batch_id = new_batch_id()
            results = []
            for rec, item in built:
                try:
                    # Rebuild now: an earlier item in this batch may have changed this one's
                    # state, and the plan applied must be the one the caller previewed.
                    preview = await self._build(rec, rtype, None)
                    if not preview.high_confidence:
                        raise BatchIneligible("No longer high-confidence after earlier items applied")
                    out = await self._apply(
                        rec, rtype, preview, item.get("plan_token", ""),
                        item.get("confirm_delete"), batch_id=batch_id,
                    )
                    results.append({"rec_id": rec.id, "status": "applied", "apply_log_id": out["entry"]["id"]})
                except ReviewError as e:
                    results.append({"rec_id": rec.id, "status": e.code, "message": e.message})
                except Exception as e:  # report it and keep the results of items already applied
                    logger.warning("Batch %s item %d failed: %s", batch_id, rec.id, e)
                    results.append({"rec_id": rec.id, "status": "error", "message": str(e)})
        return {
            "batch_id": batch_id,
            "results": results,
            "applied": sum(1 for r in results if r["status"] == "applied"),
        }

    # ---------------- revert ----------------

    async def revert(self, entry_id: int, *, force: bool = False) -> dict:
        await self._refresh_exclusion()
        async with self._apply_lock:
            entry = self.store.get(entry_id)
            if entry is None:
                raise NotFound(f"Apply log entry {entry_id} not found")
            if entry["action"] != "apply" or entry["status"] not in ("applied", "failed"):
                raise NotRevertible(f"Entry {entry_id} is a {entry['status']} {entry['action']}")
            if entry["reverted_by"]:
                raise NotRevertible(f"Entry {entry_id} was already reverted by {entry['reverted_by']}")
            plan = Plan.from_dict(entry["plan"])
            await self._check_exclusion(
                _plan_scene_ids(plan)
                | ({entry["target_id"]} if entry.get("target_type") == "scene" else set())
            )
            steps = (entry.get("result") or {}).get("steps", [])
            merge_ran = bool(plan.merge) and (
                entry["status"] == "applied" or any(st.get("step") == "merge_scenes" for st in steps)
            )
            if entry["status"] == "failed":
                # Only the scenes whose update step ran were written.
                written = {st["scene_id"] for st in steps if st.get("step") == "update_scene"}
                plan.field_changes = [c for c in plan.field_changes if c.entity_id in written]
            merged_sources = [str(s) for s in plan.merge.sources] if merge_ran else []
            if merge_ran:
                # The merge wrote the destination's fields; the log holds what they were.
                md = entry["prior_state"]["merge_destination"]
                plan.field_changes += [
                    FieldChange("scene", md["scene_id"], k, before=v, after=plan.merge.values[k])
                    for k, v in md["fields"].items()
                ]
            if not plan.field_changes:
                raise NotRevertible(
                    "Nothing to revert: this apply only "
                    + ("merged scenes and deleted files" if merge_ran else "deleted files")
                    + ", which the log records but cannot restore"
                )

            current: dict[str, dict] = {}
            for sid in {c.entity_id for c in plan.field_changes}:
                scene = await self.stash.get_scene_for_review(sid)
                if not scene:
                    raise NotRevertible(f"Scene {sid} no longer exists")
                current[sid] = scene_field_values(scene)

            drifted = [
                {"scene_id": c.entity_id, "field": c.field, "applied": c.after,
                 "current": current[c.entity_id][c.field]}
                for c in plan.field_changes
                if not values_equal(c.field, current[c.entity_id][c.field], c.after)
            ]
            if drifted and not force:
                raise Drifted("Fields changed since this apply; revert with force to overwrite them",
                              fields=drifted)

            back = Plan(
                rec_id=plan.rec_id, rec_type=plan.rec_type, option={"revert_of": entry_id},
                field_changes=[
                    FieldChange(c.entity_type, c.entity_id, c.field,
                                before=current[c.entity_id][c.field], after=c.before)
                    for c in plan.field_changes
                ],
            )
            try:
                new_entry = await apply_plan(
                    back, self.stash, self.store, target_type=entry["target_type"],
                    target_id=entry["target_id"], action="revert", reverts_id=entry_id,
                )
            except ApplyFailed as e:
                raise Failed(f"Revert failed: {e}", entry=e.entry) from e
            self.store.mark_reverted(entry_id, new_entry["id"])
            if entry["rec_id"] and self.rec_db.get_recommendation(entry["rec_id"]):
                self.rec_db.set_recommendation_status(entry["rec_id"], "pending")
            return {
                "entry": new_entry,
                "files_not_restored": entry["prior_state"].get("deleted_files") or [],
                "merged_scenes_not_restored": merged_sources,
            }

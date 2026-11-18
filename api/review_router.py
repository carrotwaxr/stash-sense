"""Review framework API (#127). See docs/plans/2026-11-17-review-framework-design.md."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

import curation_settings
import features
from review import REVIEW_TYPES
from review.media import StillError, extract_still
from review.service import NotFound, ReviewError, ReviewService
from review.store import ReviewStore
from review.types import TargetMissing

router = APIRouter(prefix="/review", tags=["review"])

_service: ReviewService | None = None


def set_review_service(service: ReviewService | None) -> None:
    global _service
    _service = service


def get_review_service() -> ReviewService:
    global _service
    if _service is None:
        from recommendations_router import get_rec_db, get_stash_client
        db = get_rec_db()
        _service = ReviewService(db, ReviewStore(db.db_path), get_stash_client())
    return _service


def _err(e: ReviewError) -> JSONResponse:
    return JSONResponse(status_code=e.status, content={"detail": e.to_dict()})


class PreviewRequest(BaseModel):
    option: dict | None = None


class AcceptRequest(BaseModel):
    option: dict | None = None
    plan_token: str
    confirm_delete: list[str] = Field(default_factory=list)


class BatchItem(BaseModel):
    rec_id: int
    plan_token: str
    confirm_delete: list[str] = Field(default_factory=list)


class BatchAcceptRequest(BaseModel):
    items: list[BatchItem]


class RevertRequest(BaseModel):
    force: bool = False


@router.get("/types")
async def list_review_types():
    """Review types visible with the current feature settings, with pending counts."""
    svc = get_review_service()
    hidden = features.hidden_recommendation_types()
    excluded = (await curation_settings.current(svc.stash, allow_stale=True)).scene_ids
    counts = svc.rec_db.get_recommendation_counts(exclude_scene_ids=excluded)
    return {"types": [
        {
            "type_id": t.type_id,
            "display_name": t.display_name,
            "description": t.description,
            "counts": counts.get(t.type_id, {}),
        }
        for t in REVIEW_TYPES.values() if t.type_id not in hidden
    ]}


@router.post("/{rec_id}/preview")
async def preview(rec_id: int, req: PreviewRequest | None = None):
    try:
        return await get_review_service().preview(rec_id, req.option if req else None)
    except ReviewError as e:
        return _err(e)


@router.post("/{rec_id}/accept")
async def accept(rec_id: int, req: AcceptRequest):
    """Apply the plan the caller previewed. 409 `stale` if the library changed since,
    409 `confirm_delete` unless confirm_delete names exactly the files it deletes."""
    try:
        return await get_review_service().accept(
            rec_id, option=req.option, plan_token=req.plan_token, confirm_delete=req.confirm_delete,
        )
    except ReviewError as e:
        return _err(e)


@router.post("/types/{type_id}/batch/preview")
async def batch_preview(type_id: str, limit: int = Query(200, ge=1, le=1000)):
    try:
        return await get_review_service().batch_preview(type_id, limit=limit)
    except ReviewError as e:
        return _err(e)


@router.post("/types/{type_id}/batch/accept")
async def batch_accept(type_id: str, req: BatchAcceptRequest):
    try:
        return await get_review_service().batch_accept(type_id, [i.model_dump() for i in req.items])
    except ReviewError as e:
        return _err(e)


@router.get("/apply-log")
async def apply_log(
    type: str | None = None,
    since_id: int | None = None,
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    store = get_review_service().store
    return {
        "entries": store.list(rec_type=type, since_id=since_id, limit=limit, offset=offset),
        "total": store.count(rec_type=type),
    }


@router.get("/apply-log/{entry_id}")
async def apply_log_entry(entry_id: int):
    entry = get_review_service().store.get(entry_id)
    if entry is None:
        return _err(NotFound(f"Apply log entry {entry_id} not found"))
    return entry


@router.post("/apply-log/{entry_id}/revert")
async def revert(entry_id: int, req: RevertRequest | None = None):
    try:
        return await get_review_service().revert(entry_id, force=bool(req and req.force))
    except ReviewError as e:
        return _err(e)


@router.get("/labels")
async def labels(type: str | None = None, limit: int = Query(1000, ge=1, le=10000)):
    """Labeled corrections, oldest first: the regression set for each type."""
    return {"labels": get_review_service().store.labels(rec_type=type, limit=limit)}


@router.get("/media/still")
async def still(scene_id: str, file_id: str, t: float = 0.0, width: int = 480):
    try:
        jpeg = await extract_still(get_review_service().stash, scene_id, file_id, t, width)
    except TargetMissing as e:
        return _err(NotFound(str(e)))
    except StillError as e:
        return JSONResponse(status_code=502, content={"detail": {"code": "still_failed", "message": str(e)}})
    return Response(content=jpeg, media_type="image/jpeg", headers={"Cache-Control": "max-age=300"})

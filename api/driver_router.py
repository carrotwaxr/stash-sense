"""Driver API v1 (#126): the stable surface external tools (stash-curator) call.

Everything else in the sidecar may change freely; this module's request and
response shapes may not. A breaking change means a new /api/v2 router.
tests/test_driver_api.py pins every shape; docs/driver-api.md documents them.

The serializers below map internal objects field by field on purpose: an
internal rename must fail the contract tests, not silently change the API.
"""

from __future__ import annotations

import base64
import hmac
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime, timezone
from typing import Any, Literal, Optional, Union

from fastapi import APIRouter, Depends, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, ConfigDict, Field

import curation_settings
import features
from curation_settings import ExclusionUnavailable, SceneExcluded
from job_models import JOB_REGISTRY, JobPriority
from media_library import MediaNotWritableError
from recommendations_db import Recommendation
from review import REVIEW_TYPES, get_review_type
from review.plan import Plan, Preview
from review.service import ReviewError

API_VERSION = 1
API_KEY_ENV = "STASH_SENSE_API_KEY"
API_KEY_HEADER = "ApiKey"  # the header Stash itself uses


# ============================================================================
# Errors: every non-2xx body is {"error": {"code", "message", ...}}
# ============================================================================

class DriverError(Exception):
    def __init__(self, status: int, code: str, message: str, **payload):
        super().__init__(message)
        self.status, self.code, self.message, self.payload = status, code, message, payload


def _error(status: int, code: str, message: str, **payload) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message, **payload}})


class DriverRoute(APIRoute):
    """Turns every failure on a driver route into the driver's error shape."""

    def get_route_handler(self) -> Callable:
        handler = super().get_route_handler()

        async def route(request: Request):
            try:
                return await handler(request)
            except DriverError as e:
                return _error(e.status, e.code, e.message, **e.payload)
            except ReviewError as e:
                payload = dict(e.payload)
                if "preview" in payload:  # stale: the fresh plan, in driver shape
                    payload["preview"] = _preview_from_review_dict(payload["preview"])
                if payload.get("entry"):
                    payload["entry"] = _apply_entry(payload["entry"])
                return _error(e.status, e.code, e.message, **payload)
            except RequestValidationError as e:
                return _error(422, "invalid_request", "The request does not match the driver API schema",
                              fields=json.loads(json.dumps(e.errors(), default=str)))
            except SceneExcluded as e:
                return _error(403, "scene_excluded", str(e), scene_ids=e.scene_ids)
            except ExclusionUnavailable as e:
                return _error(503, "exclusion_unavailable", str(e))

        return route


def require_api_key(request: Request) -> None:
    """When STASH_SENSE_API_KEY is set, every driver call must send it in the ApiKey header."""
    expected = os.environ.get(API_KEY_ENV, "")
    if not expected:
        return
    given = request.headers.get(API_KEY_HEADER, "")
    if not hmac.compare_digest(given.encode(), expected.encode()):
        raise DriverError(401, "unauthorized", f"Missing or wrong {API_KEY_HEADER} header")


router = APIRouter(
    prefix="/api/v1/driver",
    tags=["driver-v1"],
    route_class=DriverRoute,
    dependencies=[Depends(require_api_key)],
)


# ============================================================================
# Wiring
# ============================================================================

def _service():
    from review_router import get_review_service
    return get_review_service()


def _queue():
    from queue_router import _mgr
    try:
        return _mgr()
    except RuntimeError as e:
        raise DriverError(503, "queue_unavailable", str(e)) from e


# ============================================================================
# Serializers (the contract)
# ============================================================================

def _ts(value: str | None) -> str | None:
    """SQLite 'YYYY-MM-DD HH:MM:SS' (UTC) -> ISO 8601 'YYYY-MM-DDTHH:MM:SSZ'."""
    if not value:
        return None
    return value.replace(" ", "T") + ("" if value.endswith("Z") else "Z")


def _parse_since(value: str) -> str:
    """ISO 8601 (any offset; naive means UTC) -> SQLite's UTC text form."""
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as e:
        raise DriverError(400, "invalid_since", f"since must be an ISO 8601 timestamp, got {value!r}") from e
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC).replace(tzinfo=None)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def _json(value: Any) -> Any:
    return json.loads(value) if isinstance(value, str) else value


def _job(row: dict) -> dict:
    return {
        "id": row["id"],
        "type": row["type"],
        "status": row["status"],
        "priority": int(row["priority"]),
        "progress": {"processed": row.get("items_processed") or 0, "total": row.get("items_total")},
        "error": row.get("error_message"),
        "result": _json(row.get("result")),
        "created_at": _ts(row.get("created_at")),
        "started_at": _ts(row.get("started_at")),
        "completed_at": _ts(row.get("completed_at")),
    }


def _rec(rec: Recommendation) -> dict:
    resolution = None
    if rec.status == "resolved":
        resolution = {"action": rec.resolution_action, "details": rec.resolution_details,
                      "resolved_at": _ts(rec.resolved_at)}
    return {
        "id": rec.id,
        "type": rec.type,
        "status": rec.status,
        "target_type": rec.target_type,
        "target_id": rec.target_id,
        "confidence": rec.confidence,
        "evidence": rec.details,
        "reviewable": rec.type in REVIEW_TYPES,
        "resolution": resolution,
        "created_at": _ts(rec.created_at),
        "updated_at": _ts(rec.updated_at),
    }


def _file(f: dict) -> dict:
    return {"file_id": f["file_id"], "scene_id": f["scene_id"], "path": f["path"],
            "size": f["size"], "fingerprints": dict(f.get("fingerprints") or {})}


def _plan(p: dict) -> dict:
    """A Plan.to_dict() in driver shape."""
    pf = p.get("primary_file")
    return {
        "field_changes": [
            {"entity_type": c["entity_type"], "entity_id": c["entity_id"], "field": c["field"],
             "before": c["before"], "after": c["after"]}
            for c in p.get("field_changes") or []
        ],
        "keep_files": [_file(f) for f in p.get("keep_files") or []],
        "delete_files": [_file(f) for f in p.get("delete_files") or []],
        "primary_file": {"scene_id": pf["scene_id"], "before": pf["before"], "after": pf["after"]} if pf else None,
        "bytes_reclaimed": sum(int(f["size"]) for f in p.get("delete_files") or []),
    }


def _preview_from_review_dict(d: dict) -> dict:
    """A Preview.to_dict() (as the review service returns it) in driver shape."""
    return {
        "revision": d["plan_token"],
        "summary": d.get("summary", ""),
        "option": d["option"],
        "default_option": d["default_option"],
        "is_default_option": d["option"] == d["default_option"],
        "high_confidence": bool(d["high_confidence"]),
        "choices": d["choices"],
        "evidence": d["evidence"],
        "plan": _plan(d["plan"]),
        "trash_path": d.get("trash_path", ""),
    }


def _apply_entry(e: dict) -> dict:
    return {
        "id": e["id"],
        "action": e["action"],
        "status": e["status"],
        "rec_id": e.get("rec_id"),
        "rec_type": e["rec_type"],
        "target_type": e.get("target_type"),
        "target_id": e.get("target_id"),
        "batch_id": e.get("batch_id"),
        "reverts_id": e.get("reverts_id"),
        "reverted_by": e.get("reverted_by"),
        "option": e.get("option"),
        "revision": e.get("plan_token"),
        "plan": _plan(e["plan"]),
        "result": e.get("result"),
        "error": e.get("error"),
        "trash_path": e.get("trash_path") or "",
        "created_at": _ts(e.get("created_at")),
        "completed_at": _ts(e.get("completed_at")),
    }


# ---- feed cursor: opaque to callers, (updated_at, id) inside ----

def _encode_cursor(rec: Recommendation) -> str:
    return base64.urlsafe_b64encode(f"{rec.updated_at}|{rec.id}".encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> tuple[str, int]:
    try:
        raw = base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode()
        ts, rec_id = raw.rsplit("|", 1)
        return ts, int(rec_id)
    except Exception as e:
        raise DriverError(400, "invalid_cursor", "cursor is not one this API returned") from e


# ============================================================================
# Request models
# ============================================================================

class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


PriorityName = Literal["critical", "high", "normal", "low"]


class StartJobRequest(_Strict):
    type: str
    params: dict = Field(default_factory=dict)
    priority: PriorityName | int | None = None


class PreviewRequest(_Strict):
    option: dict | None = None


class AcceptRequest(_Strict):
    option: dict | None = None
    revision: str
    confirm_delete: list[str] = Field(default_factory=list)


class DismissRequest(_Strict):
    reason: str | None = None
    permanent: bool = False


# ============================================================================
# Endpoints
# ============================================================================

@router.get("")
async def info(request: Request):
    """API version, the sidecar version, and the job and review types a caller may use."""
    hidden = features.hidden_job_types()
    return {
        "api_version": API_VERSION,
        "sidecar_version": request.app.version,
        "auth_required": bool(os.environ.get(API_KEY_ENV)),
        "job_types": [
            {"type": d.type_id, "name": d.display_name, "description": d.description,
             "default_priority": int(d.default_priority), "params": {}}
            for d in JOB_REGISTRY.values() if d.type_id not in hidden
        ],
        "review_types": [
            {"type": t.type_id, "name": t.display_name, "description": t.description}
            for t in REVIEW_TYPES.values() if t.type_id not in features.hidden_recommendation_types()
        ],
    }


# ---------------- jobs ----------------

@router.post("/jobs")
async def start_job(req: StartJobRequest):
    """Queue a job. 201 with the new job; 200 with the existing one when that type is
    already queued or running (one job per type at a time)."""
    if req.type not in JOB_REGISTRY:
        raise DriverError(404, "unknown_job_type", f"'{req.type}' is not a job type")
    if not features.is_job_type_visible(req.type):
        raise DriverError(403, "hidden_type",
                          f"'{req.type}' is not production-ready yet; turn on "
                          "Settings > Features > Show Unfinished Features to use it")
    if req.params:
        # No job type takes params yet; a type that does will declare them in GET /driver.
        raise DriverError(400, "invalid_params", f"Job type '{req.type}' takes no params",
                          params=sorted(req.params))
    if isinstance(req.priority, str):
        priority = int(JobPriority[req.priority.upper()])
    else:
        priority = req.priority

    mgr = _queue()
    try:
        job_id = mgr.submit(type_id=req.type, triggered_by="driver", priority=priority)
    except MediaNotWritableError as e:
        raise DriverError(409, "media_not_writable", str(e)) from e
    except ValueError as e:  # includes HiddenFeatureError
        raise DriverError(400, "invalid_job", str(e)) from e

    if job_id is not None:
        return JSONResponse(status_code=201, content={"created": True, "job": _job(mgr.get_job(job_id))})
    active = [j for j in mgr.get_jobs(type=req.type, limit=20)
              if j["status"] in ("queued", "running", "stopping")]
    if not active:  # finished between the submit and this read; submit again
        job_id = mgr.submit(type_id=req.type, triggered_by="driver", priority=priority)
        if job_id is not None:
            return JSONResponse(status_code=201, content={"created": True, "job": _job(mgr.get_job(job_id))})
        raise DriverError(409, "job_conflict", f"Could not queue '{req.type}'; try again")
    return {"created": False, "job": _job(active[0])}


@router.get("/jobs/{job_id}")
async def get_job(job_id: int):
    job = _queue().get_job(job_id)
    if job is None:
        raise DriverError(404, "not_found", f"Job {job_id} not found")
    return {"job": _job(job)}


# ---------------- recommendations ----------------

def _visible_rec(rec_id: int) -> Recommendation:
    rec = _service().rec_db.get_recommendation(rec_id)
    if rec is None:
        raise DriverError(404, "not_found", f"Recommendation {rec_id} not found")
    if rec.type in features.hidden_recommendation_types():
        raise DriverError(403, "hidden_type",
                          f"'{rec.type}' is not production-ready yet; turn on "
                          "Settings > Features > Show Unfinished Features to use it")
    return rec


@router.get("/recommendations")
async def list_recommendations(
    type: str | None = None,
    status: Literal["pending", "dismissed", "resolved"] | None = None,
    since: str | None = Query(None, description="ISO 8601; recommendations updated at or after it"),
    cursor: str | None = Query(None, description="next_cursor from the previous page"),
    limit: int = Query(100, ge=1, le=500),
):
    """Recommendations in the order they last changed, oldest first.

    Page with next_cursor. Keep the last cursor and call again later to read only
    what changed since (a change feed). Leaves out hidden types and recommendations
    naming a scene on the exclusion list.
    """
    svc = _service()
    excluded = (await curation_settings.current(svc.stash, allow_stale=True)).scene_ids
    recs = svc.rec_db.get_recommendations_feed(
        type=type, status=status,
        since=_parse_since(since) if since else None,
        after=_decode_cursor(cursor) if cursor else None,
        limit=limit,
        exclude_types=features.hidden_recommendation_types(),
        exclude_scene_ids=excluded,
    )
    # The cursor always points at the last row returned, so the caller can resume
    # from here even when the page wasn't full.
    if recs:
        next_cursor = _encode_cursor(recs[-1])
    else:
        next_cursor = cursor
    return {"recommendations": [_rec(r) for r in recs], "next_cursor": next_cursor,
            "has_more": len(recs) == limit}


async def _review(rec: Recommendation, option: dict | None) -> dict | None:
    if get_review_type(rec.type) is None:
        return None
    return _preview_from_review_dict(await _service().preview(rec.id, option))


async def _check_rec_exclusion(rec: Recommendation) -> None:
    hit = (await curation_settings.current(_service().stash)).excluded(
        curation_settings.rec_scene_ids(rec.target_type, rec.target_id, rec.details))
    if hit:
        raise SceneExcluded(hit)


@router.get("/recommendations/{rec_id}")
async def get_recommendation(rec_id: int):
    """One recommendation; for a reviewable type, also the plan accept would apply now
    (with the suggested option) and its revision."""
    rec = _visible_rec(rec_id)
    await _check_rec_exclusion(rec)
    review = await _review(rec, None) if rec.status == "pending" else None
    return {"recommendation": _rec(rec), "preview": review}


def _require_reviewable(rec: Recommendation) -> None:
    if get_review_type(rec.type) is None:
        raise DriverError(409, "not_reviewable",
                          f"'{rec.type}' recommendations can't be previewed or accepted through the driver API")


@router.post("/recommendations/{rec_id}/preview")
async def preview_recommendation(rec_id: int, req: PreviewRequest | None = None):
    """The plan for a chosen option (partial options are filled from the suggestion)."""
    rec = _visible_rec(rec_id)
    _require_reviewable(rec)
    return {"preview": await _review(rec, req.option if req else None)}


@router.post("/recommendations/{rec_id}/accept")
async def accept_recommendation(rec_id: int, req: AcceptRequest):
    """Apply the plan the caller previewed.

    Refused with 409 `stale` when the plan for this option no longer matches
    revision, and 409 `confirm_delete` unless confirm_delete names exactly the
    files it deletes. Accepting an already-applied recommendation returns the
    apply log entry of the first accept (already_applied: true) and writes nothing.
    """
    rec = _visible_rec(rec_id)
    _require_reviewable(rec)
    out = await _service().accept(rec_id, option=req.option, plan_token=req.revision,
                                  confirm_delete=req.confirm_delete)
    return {
        "already_applied": bool(out.get("already_applied")),
        "entry": _apply_entry(out["entry"]),
        "warnings": out.get("warnings", []),
    }


@router.post("/recommendations/{rec_id}/dismiss")
async def dismiss_recommendation(rec_id: int, req: DismissRequest | None = None):
    """Dismiss a pending recommendation. Dismissing a dismissed one is a no-op;
    a resolved one is refused with 409 `not_pending`."""
    rec = _visible_rec(rec_id)
    db = _service().rec_db
    if rec.status == "resolved":
        raise DriverError(409, "not_pending", f"Recommendation {rec_id} is resolved")
    if rec.status == "pending":
        db.dismiss_recommendation(rec_id, reason=req.reason if req else None,
                                  permanent=bool(req and req.permanent))
    return {"recommendation": _rec(db.get_recommendation(rec_id))}


# ---------------- apply log ----------------

@router.get("/apply-log")
async def list_apply_log(
    since: int | None = Query(None, ge=0, description="Entries with an id greater than this"),
    type: str | None = None,
    limit: int = Query(100, ge=1, le=500),
):
    """Apply log entries oldest first. Pass the last id you saw as since to read only newer ones.
    An entry with status `applying` is still running; re-read it with GET /apply-log/{id}."""
    entries = _service().store.list(rec_type=type, since_id=since, limit=limit, ascending=True)
    return {
        "entries": [_apply_entry(e) for e in entries],
        "next_since": entries[-1]["id"] if entries else since,
        "has_more": len(entries) == limit,
    }


@router.get("/apply-log/{entry_id}")
async def get_apply_log_entry(entry_id: int):
    entry = _service().store.get(entry_id)
    if entry is None:
        raise DriverError(404, "not_found", f"Apply log entry {entry_id} not found")
    return {"entry": _apply_entry(entry)}

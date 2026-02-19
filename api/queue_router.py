"""Queue API router — job submission, monitoring, schedule management."""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from job_models import JOB_REGISTRY, JobPriority

router = APIRouter(prefix="/queue", tags=["queue"])

_queue_manager = None


def init_queue_router(queue_manager):
    """Set the queue manager reference. Called during startup."""
    global _queue_manager
    _queue_manager = queue_manager


def _mgr():
    if _queue_manager is None:
        raise RuntimeError("Queue manager not initialized")
    return _queue_manager


# ============================================================================
# Request/Response models
# ============================================================================

class SubmitJobRequest(BaseModel):
    type: str
    triggered_by: str = "user"
    priority: Optional[int] = None
    cursor: Optional[str] = None

class SubmitJobResponse(BaseModel):
    job_id: int
    message: str

class JobListResponse(BaseModel):
    jobs: list[dict]

class QueueStatusResponse(BaseModel):
    queued: int
    running: int
    running_jobs: list[dict]

class UpdateScheduleRequest(BaseModel):
    enabled: bool
    interval_hours: float
    priority: Optional[int] = None

class ScheduleListResponse(BaseModel):
    schedules: list[dict]

class JobTypesResponse(BaseModel):
    types: list[dict]

class MessageResponse(BaseModel):
    message: str


# ============================================================================
# Endpoints — specific paths MUST come before /{job_id} to avoid conflicts
# ============================================================================

@router.get("/status", response_model=QueueStatusResponse)
async def get_queue_status():
    return _mgr().get_status()

@router.get("/types", response_model=JobTypesResponse)
async def get_job_types():
    types = []
    for defn in JOB_REGISTRY.values():
        types.append({
            "type_id": defn.type_id,
            "display_name": defn.display_name,
            "description": defn.description,
            "resource": defn.resource.value,
            "default_priority": int(defn.default_priority),
            "supports_incremental": defn.supports_incremental,
            "schedulable": defn.schedulable,
            "default_interval_hours": defn.default_interval_hours,
            "allowed_intervals": [
                {"hours": hours, "label": label}
                for hours, label in defn.allowed_intervals
            ],
        })
    return {"types": types}

@router.get("/schedules", response_model=ScheduleListResponse)
async def list_schedules():
    return {"schedules": _mgr()._db.get_all_job_schedules()}

@router.put("/schedules/{type_id}", response_model=MessageResponse)
async def update_schedule(type_id: str, request: UpdateScheduleRequest):
    defn = JOB_REGISTRY.get(type_id)
    if not defn:
        raise HTTPException(status_code=404, detail=f"Unknown job type: {type_id}")
    priority = request.priority if request.priority is not None else int(defn.default_priority)
    _mgr()._db.upsert_job_schedule(
        type=type_id,
        enabled=request.enabled,
        interval_hours=request.interval_hours,
        priority=priority,
    )
    return {"message": f"Schedule updated for {type_id}"}

# ============================================================================
# Generic CRUD — /{job_id} routes come AFTER specific paths
# ============================================================================

@router.get("", response_model=JobListResponse)
async def list_jobs(status: Optional[str] = None, type: Optional[str] = None, limit: int = 50):
    jobs = _mgr().get_jobs(status=status, type=type, limit=limit)
    return {"jobs": jobs}

@router.post("", response_model=SubmitJobResponse)
async def submit_job(request: SubmitJobRequest):
    try:
        priority = JobPriority(request.priority) if request.priority is not None else None
    except ValueError:
        priority = request.priority

    try:
        job_id = _mgr().submit(
            type_id=request.type,
            triggered_by=request.triggered_by,
            priority=priority,
            cursor=request.cursor,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if job_id is None:
        raise HTTPException(status_code=409, detail=f"Job of type '{request.type}' is already queued or running")

    return {"job_id": job_id, "message": f"Job queued: {request.type}"}

@router.get("/{job_id}")
async def get_job(job_id: int):
    job = _mgr().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job

@router.delete("/history", response_model=MessageResponse)
async def clear_history():
    count = _mgr().clear_history()
    return {"message": f"Cleared {count} job(s) from history"}

@router.delete("/{job_id}", response_model=MessageResponse)
async def cancel_job(job_id: int):
    _mgr().cancel(job_id)
    return {"message": "Job cancelled"}

@router.post("/{job_id}/stop", response_model=MessageResponse)
async def stop_job(job_id: int):
    _mgr().cancel(job_id)
    return {"message": "Stop requested"}

@router.post("/{job_id}/retry", response_model=SubmitJobResponse)
async def retry_job(job_id: int):
    job = _mgr().get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job["status"] not in ("failed", "cancelled"):
        raise HTTPException(status_code=400, detail="Can only retry failed or cancelled jobs")
    defn = JOB_REGISTRY.get(job["type"])
    if not defn:
        raise HTTPException(status_code=400, detail=f"Unknown job type: {job['type']}")
    new_id = _mgr().submit(
        type_id=job["type"],
        triggered_by="user",
        cursor=job.get("cursor"),
    )
    if new_id is None:
        raise HTTPException(status_code=409, detail="Job already queued")
    return {"job_id": new_id, "message": "Job re-queued"}

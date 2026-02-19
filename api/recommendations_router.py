"""
Recommendations API Router

Endpoints for managing recommendations, running analysis, and configuration.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import face_config
from recommendations_db import RecommendationsDB, Recommendation, AnalysisRun
from stash_client_unified import StashClientUnified
from analyzers import DuplicatePerformerAnalyzer, DuplicateSceneFilesAnalyzer, DuplicateScenesAnalyzer, UpstreamPerformerAnalyzer

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

# Global instances (initialized by main app)
rec_db: Optional[RecommendationsDB] = None
stash_client: Optional[StashClientUnified] = None


def init_recommendations(db_path: str, stash_url: str, stash_api_key: str):
    """Initialize recommendations database and stash client."""
    global rec_db, stash_client
    rec_db = RecommendationsDB(db_path)
    # Clean up any analysis runs left as 'running' from a previous sidecar session
    stale = rec_db.fail_stale_analysis_runs()
    if stale:
        print(f"Marked {stale} stale analysis run(s) as failed")
    if stash_url:
        stash_client = StashClientUnified(stash_url, stash_api_key)


def get_rec_db() -> RecommendationsDB:
    if rec_db is None:
        raise HTTPException(status_code=503, detail="Recommendations database not initialized")
    return rec_db


def get_stash_client() -> StashClientUnified:
    if stash_client is None:
        raise HTTPException(status_code=503, detail="Stash connection not configured. Set STASH_URL env var.")
    return stash_client


def save_scene_fingerprint(
    scene_id: int,
    frames_analyzed: int,
    performer_data: list[dict],
    db_version: Optional[str] = None,
) -> tuple[Optional[int], Optional[str]]:
    """
    Persist a scene fingerprint to the database.

    Args:
        scene_id: Stash scene ID
        frames_analyzed: Number of frames analyzed
        performer_data: List of dicts with keys:
            - performer_id: stashdb universal ID
            - face_count: number of frames this performer appeared in
            - avg_confidence: average match confidence (0-1)
        db_version: Face recognition DB version

    Returns:
        Tuple of (fingerprint_id, error_message). On success, error is None.
        On failure, fingerprint_id is None and error contains the message.
    """
    if rec_db is None:
        return None, "Recommendations database not initialized"

    try:
        total_faces = sum(p.get("face_count", 0) for p in performer_data)

        # Create or update the fingerprint
        fingerprint_id = rec_db.create_scene_fingerprint(
            stash_scene_id=scene_id,
            total_faces=total_faces,
            frames_analyzed=frames_analyzed,
            fingerprint_status="complete",
            db_version=db_version,
        )

        # Clear existing faces and add new ones
        rec_db.delete_fingerprint_faces(fingerprint_id)

        # Calculate proportions
        total_frames = sum(p.get("face_count", 0) for p in performer_data) or 1

        for performer in performer_data:
            face_count = performer.get("face_count", 0)
            rec_db.add_fingerprint_face(
                fingerprint_id=fingerprint_id,
                performer_id=performer.get("performer_id", ""),
                face_count=face_count,
                avg_confidence=performer.get("avg_confidence"),
                proportion=face_count / total_frames if total_frames > 0 else 0,
            )

        return fingerprint_id, None
    except Exception as e:
        error_msg = str(e)
        print(f"[save_scene_fingerprint] Error: {error_msg}")
        return None, error_msg


def save_image_fingerprint(
    image_id: str,
    gallery_id: Optional[str],
    faces: list,
    image_shape: tuple[int, int],
    db_version: Optional[str] = None,
) -> tuple[Optional[int], Optional[str]]:
    """Save image identification results as a fingerprint."""
    if rec_db is None:
        return None, "Recommendations database not initialized"

    try:
        img_h, img_w = image_shape

        fp_id = rec_db.create_image_fingerprint(
            stash_image_id=image_id,
            gallery_id=gallery_id,
            faces_detected=len(faces),
            db_version=db_version,
        )

        # Clear old face data
        rec_db.delete_image_fingerprint_faces(image_id)

        # Save each detected face's best match
        for result in faces:
            if result.matches:
                best = result.matches[0]
                bbox = result.face.bbox  # dict with x, y, w, h in pixels
                rec_db.add_image_fingerprint_face(
                    stash_image_id=image_id,
                    performer_id=best.stashdb_id,
                    confidence=max(0.0, min(1.0, 1.0 - best.combined_score)),
                    distance=best.combined_score,
                    bbox_x=bbox["x"] / img_w if img_w > 0 else 0,
                    bbox_y=bbox["y"] / img_h if img_h > 0 else 0,
                    bbox_w=bbox["w"] / img_w if img_w > 0 else 0,
                    bbox_h=bbox["h"] / img_h if img_h > 0 else 0,
                )

        return fp_id, None
    except Exception as e:
        error_msg = str(e)
        print(f"[save_image_fingerprint] Error: {error_msg}")
        return None, error_msg


# ==================== Pydantic Models ====================

class RecommendationResponse(BaseModel):
    """A single recommendation."""
    id: int
    type: str
    status: str
    target_type: str
    target_id: str
    details: dict
    confidence: Optional[float]
    created_at: str
    updated_at: str


class RecommendationListResponse(BaseModel):
    """List of recommendations."""
    recommendations: list[RecommendationResponse]
    total: int


class RecommendationCountsResponse(BaseModel):
    """Counts by type and status."""
    counts: dict[str, dict[str, int]]
    total_pending: int


class ResolveRequest(BaseModel):
    """Request to resolve a recommendation."""
    action: str = Field(description="Action taken: 'merged', 'deleted', 'linked', etc.")
    details: Optional[dict] = Field(None, description="Action-specific details")


class DismissRequest(BaseModel):
    """Request to dismiss a recommendation."""
    reason: Optional[str] = Field(None, description="Why this was dismissed")


class AnalysisRunResponse(BaseModel):
    """An analysis run."""
    id: int
    type: str
    status: str
    started_at: str
    completed_at: Optional[str]
    items_total: Optional[int]
    items_processed: Optional[int]
    recommendations_created: int
    error_message: Optional[str]


class RunAnalysisResponse(BaseModel):
    """Response when starting an analysis run."""
    run_id: int
    message: str


class StashStatusResponse(BaseModel):
    """Stash connection status."""
    connected: bool
    url: Optional[str]
    error: Optional[str]


# ==================== Recommendation Endpoints ====================

@router.get("", response_model=RecommendationListResponse)
async def list_recommendations(
    status: Optional[str] = None,
    type: Optional[str] = None,
    target_type: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    """List recommendations with optional filtering."""
    db = get_rec_db()
    recs = db.get_recommendations(
        status=status,
        type=type,
        target_type=target_type,
        limit=limit,
        offset=offset,
    )
    return RecommendationListResponse(
        recommendations=[
            RecommendationResponse(
                id=r.id,
                type=r.type,
                status=r.status,
                target_type=r.target_type,
                target_id=r.target_id,
                details=r.details,
                confidence=r.confidence,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in recs
        ],
        total=db.count_recommendations(status=status, type=type, target_type=target_type),
    )


@router.get("/counts", response_model=RecommendationCountsResponse)
async def get_recommendation_counts():
    """Get recommendation counts by type and status."""
    db = get_rec_db()
    counts = db.get_recommendation_counts()
    total_pending = sum(
        type_counts.get("pending", 0)
        for type_counts in counts.values()
    )
    return RecommendationCountsResponse(counts=counts, total_pending=total_pending)


@router.get("/{rec_id}", response_model=RecommendationResponse)
async def get_recommendation(rec_id: int):
    """Get a single recommendation."""
    db = get_rec_db()
    rec = db.get_recommendation(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return RecommendationResponse(
        id=rec.id,
        type=rec.type,
        status=rec.status,
        target_type=rec.target_type,
        target_id=rec.target_id,
        details=rec.details,
        confidence=rec.confidence,
        created_at=rec.created_at,
        updated_at=rec.updated_at,
    )


@router.post("/{rec_id}/resolve")
async def resolve_recommendation(rec_id: int, request: ResolveRequest):
    """Mark a recommendation as resolved."""
    db = get_rec_db()
    success = db.resolve_recommendation(rec_id, request.action, request.details)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True}


@router.post("/{rec_id}/dismiss")
async def dismiss_recommendation(rec_id: int, request: DismissRequest = None):
    """Dismiss a recommendation (won't be re-created)."""
    db = get_rec_db()
    reason = request.reason if request else None
    success = db.dismiss_recommendation(rec_id, reason)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True}


# ==================== Analysis Endpoints ====================

ANALYZERS = {
    "duplicate_performer": DuplicatePerformerAnalyzer,
    "duplicate_scene_files": DuplicateSceneFilesAnalyzer,
    "duplicate_scenes": DuplicateScenesAnalyzer,
    "upstream_performer_changes": UpstreamPerformerAnalyzer,
}


@router.get("/analysis/types")
async def list_analysis_types():
    """List available analysis types."""
    db = get_rec_db()
    types = []
    for type_name in ANALYZERS.keys():
        settings = db.get_settings(type_name)
        types.append({
            "type": type_name,
            "enabled": settings.enabled if settings else True,
            "description": ANALYZERS[type_name].__doc__.strip().split("\n")[0] if ANALYZERS[type_name].__doc__ else None,
        })
    return {"types": types}


@router.post("/analysis/{type}/run", response_model=RunAnalysisResponse)
async def run_analysis(type: str, full: bool = False):
    """Start an analysis. Now delegates to the job queue."""
    if type not in ANALYZERS:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {type}")

    from queue_router import _queue_manager
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")

    from job_models import JobPriority
    job_id = _queue_manager.submit(
        type_id=type,
        triggered_by="user",
        priority=JobPriority.HIGH,
    )
    if job_id is None:
        raise HTTPException(status_code=409, detail=f"Analysis '{type}' is already running or queued")

    return RunAnalysisResponse(run_id=job_id, message=f"Analysis '{type}' queued (job #{job_id})")


@router.get("/analysis/runs", response_model=list[AnalysisRunResponse])
async def list_analysis_runs(type: Optional[str] = None, limit: int = 20):
    """List recent analysis runs."""
    db = get_rec_db()
    runs = db.get_recent_analysis_runs(type=type, limit=limit)
    return [
        AnalysisRunResponse(
            id=r.id,
            type=r.type,
            status=r.status,
            started_at=r.started_at,
            completed_at=r.completed_at,
            items_total=r.items_total,
            items_processed=r.items_processed,
            recommendations_created=r.recommendations_created,
            error_message=r.error_message,
        )
        for r in runs
    ]


@router.get("/analysis/runs/{run_id}", response_model=AnalysisRunResponse)
async def get_analysis_run(run_id: int):
    """Get status of an analysis run."""
    db = get_rec_db()
    run = db.get_analysis_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return AnalysisRunResponse(
        id=run.id,
        type=run.type,
        status=run.status,
        started_at=run.started_at,
        completed_at=run.completed_at,
        items_total=run.items_total,
        items_processed=run.items_processed,
        recommendations_created=run.recommendations_created,
        error_message=run.error_message,
    )


# ==================== Stash Connection ====================

@router.get("/stash/status", response_model=StashStatusResponse)
async def stash_status():
    """Check Stash connection status."""
    if stash_client is None:
        return StashStatusResponse(
            connected=False,
            url=None,
            error="STASH_URL environment variable not set",
        )

    try:
        await stash_client.test_connection()
        return StashStatusResponse(
            connected=True,
            url=stash_client.base_url,
            error=None,
        )
    except Exception as e:
        return StashStatusResponse(
            connected=False,
            url=stash_client.base_url,
            error=str(e),
        )


# ==================== Actions ====================

class MergePerformersRequest(BaseModel):
    """Request to merge duplicate performers."""
    destination_id: str
    source_ids: list[str]


@router.post("/actions/merge-performers")
async def merge_performers(request: MergePerformersRequest):
    """Execute a performer merge."""
    stash = get_stash_client()
    try:
        result = await stash.merge_performers(request.source_ids, request.destination_id)
        return {"success": True, "merged_into": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DeleteSceneFilesRequest(BaseModel):
    """Request to delete duplicate scene files."""
    scene_id: str
    file_ids_to_delete: list[str]
    keep_file_id: str
    all_file_ids: list[str]


@router.post("/actions/delete-scene-files")
async def delete_scene_files(request: DeleteSceneFilesRequest):
    """Delete files from a scene, keeping the specified file."""
    stash = get_stash_client()
    try:
        result = await stash.delete_scene_files(
            scene_id=request.scene_id,
            file_ids_to_delete=request.file_ids_to_delete,
            keep_file_id=request.keep_file_id,
            all_file_ids=request.all_file_ids,
        )
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Fingerprints ====================

_current_db_version: Optional[str] = None


def set_db_version(version: str):
    """Set the current face recognition DB version (called from main.py startup)."""
    global _current_db_version
    _current_db_version = version


class FingerprintStatusResponse(BaseModel):
    """Response for fingerprint status endpoint."""
    total_fingerprints: int
    complete_fingerprints: int
    pending_fingerprints: int
    error_fingerprints: int
    current_db_version: Optional[str]
    current_version_count: Optional[int] = None
    needs_refresh_count: Optional[int] = None
    generation_running: bool = False
    generation_progress: Optional[dict] = None


class FingerprintGenerateRequest(BaseModel):
    """Request to start fingerprint generation. Defaults from face_config.py."""
    refresh_outdated: bool = True
    num_frames: int = face_config.NUM_FRAMES
    min_face_size: int = face_config.MIN_FACE_SIZE
    max_distance: float = face_config.MAX_DISTANCE


@router.get("/fingerprints/status", response_model=FingerprintStatusResponse)
async def get_fingerprint_status():
    """Get fingerprint coverage statistics."""
    db = get_rec_db()

    stats = db.get_fingerprint_stats(_current_db_version)

    # Check if generation is running
    from fingerprint_generator import get_generator
    generator = get_generator()
    generation_running = generator is not None and generator.status.value == "running"
    progress = generator.progress.to_dict() if generator else None

    return FingerprintStatusResponse(
        total_fingerprints=stats.get("total_fingerprints", 0),
        complete_fingerprints=stats.get("complete_fingerprints", 0),
        pending_fingerprints=stats.get("pending_fingerprints", 0),
        error_fingerprints=stats.get("error_fingerprints", 0),
        current_db_version=_current_db_version,
        current_version_count=stats.get("current_version_count"),
        needs_refresh_count=stats.get("needs_refresh_count"),
        generation_running=generation_running,
        generation_progress=progress,
    )


@router.post("/fingerprints/generate")
async def start_fingerprint_generation(request: FingerprintGenerateRequest):
    """Start fingerprint generation. Now delegates to the job queue."""
    from queue_router import _queue_manager
    from job_models import JobPriority
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")

    job_id = _queue_manager.submit(
        type_id="fingerprint_generation",
        triggered_by="user",
        priority=JobPriority.HIGH,
    )
    if job_id is None:
        raise HTTPException(status_code=409, detail="Fingerprint generation already running or queued")

    return {"job_id": job_id, "message": "Fingerprint generation queued"}


@router.post("/fingerprints/stop")
async def stop_fingerprint_generation():
    """Stop fingerprint generation via queue."""
    from queue_router import _queue_manager
    if _queue_manager is None:
        raise HTTPException(status_code=503, detail="Queue manager not initialized")

    running = _queue_manager.get_jobs(status="running", type="fingerprint_generation")
    if not running:
        return {"message": "No fingerprint generation running"}
    _queue_manager.cancel(running[0]["id"])
    return {"message": "Stop requested"}


@router.get("/fingerprints/progress")
async def get_fingerprint_progress():
    """Get current fingerprint generation progress."""
    from fingerprint_generator import get_generator

    generator = get_generator()
    if not generator:
        return {
            "status": "idle",
            "message": "No fingerprint generation has been started",
        }

    return generator.progress.to_dict()


@router.post("/fingerprints/refresh")
async def mark_fingerprints_for_refresh(scene_ids: Optional[list[int]] = None):
    """
    Mark fingerprints for refresh by clearing their db_version.
    If scene_ids is provided, only those scenes are marked.
    If scene_ids is None, ALL fingerprints are marked for refresh.
    """
    db = get_rec_db()

    if scene_ids is None:
        # Require explicit confirmation to refresh all
        raise HTTPException(
            status_code=400,
            detail="Must provide scene_ids list. To refresh all, use /fingerprints/refresh-all",
        )

    count = db.mark_fingerprints_for_refresh(scene_ids)
    return {
        "marked_for_refresh": count,
        "scene_ids": scene_ids,
    }


@router.post("/fingerprints/refresh-all")
async def mark_all_fingerprints_for_refresh(confirm: bool = False):
    """Mark ALL fingerprints for refresh. Requires confirm=true."""
    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Must set confirm=true to refresh all fingerprints",
        )

    db = get_rec_db()
    count = db.mark_fingerprints_for_refresh(None)

    return {
        "marked_for_refresh": count,
        "message": f"All {count} fingerprints marked for refresh",
    }


# ==================== Upstream Sync Actions ====================

class UpdatePerformerRequest(BaseModel):
    """Request to apply upstream changes to a performer."""
    performer_id: str
    fields: dict


@router.post("/actions/update-performer")
async def update_performer_fields(request: UpdatePerformerRequest):
    """Apply selected upstream changes to a performer.

    Translates diff-engine field names (StashBox-style) to Stash PerformerUpdateInput names.
    Compound fields (measurements, career_length) are smart-merged with existing values.
    """
    from upstream_field_mapper import parse_measurements, parse_career_length

    stash = get_stash_client()
    fields = dict(request.fields)
    performer_id = request.performer_id

    # Lazy-fetch current performer data (only when needed for smart merge)
    current_performer = None

    async def get_current():
        nonlocal current_performer
        if current_performer is None:
            current_performer = await stash.get_performer(performer_id) or {}
        return current_performer

    # --- Simple field renames ---
    FIELD_RENAME = {
        "aliases": "alias_list",
        "height": "height_cm",
        "breast_type": "fake_tits",
    }
    for old_name, new_name in FIELD_RENAME.items():
        if old_name in fields:
            fields[new_name] = fields.pop(old_name)

    # --- Career years → career_length (smart merge) ---
    career_start = fields.pop("career_start_year", None)
    career_end = fields.pop("career_end_year", None)
    if career_start is not None or career_end is not None:
        current = await get_current()
        existing = parse_career_length(current.get("career_length"))
        start_val = str(career_start) if career_start is not None else (
            str(existing["career_start_year"]) if existing["career_start_year"] else ""
        )
        end_val = str(career_end) if career_end is not None else (
            str(existing["career_end_year"]) if existing["career_end_year"] else ""
        )
        if start_val and end_val:
            fields["career_length"] = f"{start_val}-{end_val}"
        elif start_val:
            fields["career_length"] = f"{start_val}-"
        elif end_val:
            fields["career_length"] = f"-{end_val}"

    # --- Measurement fields → measurements string (smart merge) ---
    cup = fields.pop("cup_size", None)
    band = fields.pop("band_size", None)
    waist = fields.pop("waist_size", None)
    hip = fields.pop("hip_size", None)
    if any(v is not None for v in [cup, band, waist, hip]):
        current = await get_current()
        existing = parse_measurements(current.get("measurements"))
        # Overlay: accepted value wins, else keep existing
        final_band = str(band) if band is not None else (
            str(existing["band_size"]) if existing["band_size"] else ""
        )
        final_cup = str(cup) if cup is not None else (
            existing["cup_size"] or ""
        )
        final_waist = str(waist) if waist is not None else (
            str(existing["waist_size"]) if existing["waist_size"] else ""
        )
        final_hip = str(hip) if hip is not None else (
            str(existing["hip_size"]) if existing["hip_size"] else ""
        )
        bust = f"{final_band}{final_cup}"
        measurements = "-".join([bust, final_waist, final_hip])
        fields["measurements"] = measurements.rstrip("-") or None

    # --- Integer coercion ---
    if "height_cm" in fields and isinstance(fields["height_cm"], str):
        try:
            fields["height_cm"] = int(fields["height_cm"])
        except ValueError:
            pass

    # --- Alias merge (_alias_add meta-key) ---
    alias_additions = fields.pop("_alias_add", None)
    if alias_additions:
        current = await get_current()
        existing_aliases = current.get("alias_list", [])
        seen = {a.lower() for a in existing_aliases}
        merged = list(existing_aliases)
        for alias in alias_additions:
            if alias.lower() not in seen:
                merged.append(alias)
                seen.add(alias.lower())
        fields["alias_list"] = merged

    try:
        result = await stash.update_performer(performer_id, **fields)
        return {"success": True, "performer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpstreamDismissRequest(BaseModel):
    """Request to dismiss an upstream recommendation."""
    reason: Optional[str] = Field(None)
    permanent: bool = Field(False, description="If true, never show updates for this entity again")


@router.post("/{rec_id}/dismiss-upstream")
async def dismiss_upstream_recommendation(rec_id: int, request: UpstreamDismissRequest = None):
    """Dismiss an upstream recommendation with permanent option."""
    db = get_rec_db()
    permanent = request.permanent if request else False
    reason = request.reason if request else None
    success = db.dismiss_recommendation(rec_id, reason=reason, permanent=permanent)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True, "permanent": permanent}


@router.get("/upstream/field-config/{endpoint_b64}")
async def get_field_config(endpoint_b64: str):
    """Get field monitoring config for an endpoint. Endpoint is base64-encoded."""
    import base64
    endpoint = base64.b64decode(endpoint_b64).decode()
    db = get_rec_db()
    fields = db.get_enabled_fields(endpoint, "performer")
    from upstream_field_mapper import DEFAULT_PERFORMER_FIELDS, FIELD_LABELS
    if fields is None:
        return {
            "endpoint": endpoint,
            "fields": {f: {"enabled": True, "label": FIELD_LABELS.get(f, f)} for f in DEFAULT_PERFORMER_FIELDS},
        }
    return {
        "endpoint": endpoint,
        "fields": {f: {"enabled": f in fields, "label": FIELD_LABELS.get(f, f)} for f in DEFAULT_PERFORMER_FIELDS},
    }


@router.post("/upstream/field-config/{endpoint_b64}")
async def set_field_config(endpoint_b64: str, field_configs: dict[str, bool]):
    """Set field monitoring config for an endpoint."""
    import base64
    endpoint = base64.b64decode(endpoint_b64).decode()
    db = get_rec_db()
    db.set_field_config(endpoint, "performer", field_configs)
    return {"success": True}


# ==================== User Settings ====================

@router.get("/settings")
async def get_all_settings():
    """Get all user settings."""
    db = get_rec_db()
    settings = db.get_all_user_settings()
    return {"settings": settings}


@router.get("/settings/{key}")
async def get_setting(key: str):
    """Get a single user setting."""
    db = get_rec_db()
    value = db.get_user_setting(key)
    return {"key": key, "value": value}


class SetSettingRequest(BaseModel):
    """Request to set a user setting."""
    value: object


@router.post("/settings/{key}")
async def set_setting(key: str, request: SetSettingRequest):
    """Set a user setting."""
    db = get_rec_db()
    db.set_user_setting(key, request.value)
    return {"success": True, "key": key}

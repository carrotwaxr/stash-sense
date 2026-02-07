"""
Recommendations API Router

Endpoints for managing recommendations, running analysis, and configuration.
"""

import asyncio
import os
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

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
        total=len(recs),
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


async def run_analysis_task(type: str, run_id: int):
    """Background task to run analysis."""
    db = get_rec_db()
    stash = get_stash_client()

    analyzer_class = ANALYZERS.get(type)
    if not analyzer_class:
        db.fail_analysis_run(run_id, f"Unknown analysis type: {type}")
        return

    analyzer = analyzer_class(stash, db)

    try:
        result = await analyzer.run(incremental=True)
        db.complete_analysis_run(run_id, result.recommendations_created)
    except Exception as e:
        db.fail_analysis_run(run_id, str(e))


@router.post("/analysis/{type}/run", response_model=RunAnalysisResponse)
async def run_analysis(type: str, background_tasks: BackgroundTasks):
    """Start an analysis run (async)."""
    if type not in ANALYZERS:
        raise HTTPException(status_code=400, detail=f"Unknown analysis type: {type}")

    # Check Stash connection
    get_stash_client()

    db = get_rec_db()
    run_id = db.start_analysis_run(type)

    # Run in background
    background_tasks.add_task(run_analysis_task, type, run_id)

    return RunAnalysisResponse(
        run_id=run_id,
        message=f"Analysis '{type}' started. Check /recommendations/analysis/runs/{run_id} for status.",
    )


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

@router.post("/actions/merge-performers")
async def merge_performers(destination_id: str, source_ids: list[str]):
    """Execute a performer merge."""
    stash = get_stash_client()
    try:
        result = await stash.merge_performers(source_ids, destination_id)
        return {"success": True, "merged_into": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/actions/delete-scene-files")
async def delete_scene_files(
    scene_id: str,
    file_ids_to_delete: list[str],
    keep_file_id: str,
    all_file_ids: list[str],
):
    """Delete files from a scene, keeping the specified file."""
    stash = get_stash_client()
    try:
        result = await stash.delete_scene_files(
            scene_id=scene_id,
            file_ids_to_delete=file_ids_to_delete,
            keep_file_id=keep_file_id,
            all_file_ids=all_file_ids,
        )
        return {"success": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Fingerprints ====================

# Background task for fingerprint generation
_fingerprint_task: Optional[asyncio.Task] = None
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
    """Request to start fingerprint generation."""
    refresh_outdated: bool = True
    num_frames: int = 12
    min_face_size: int = 50
    max_distance: float = 0.6


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
async def start_fingerprint_generation(
    request: FingerprintGenerateRequest,
    background_tasks: BackgroundTasks,
):
    """Start background fingerprint generation for all scenes."""
    global _fingerprint_task

    from fingerprint_generator import SceneFingerprintGenerator, set_generator, get_generator

    # Check if already running
    existing = get_generator()
    if existing and existing.status.value == "running":
        raise HTTPException(
            status_code=409,
            detail="Fingerprint generation already running. Stop it first or wait for completion.",
        )

    db = get_rec_db()
    stash = get_stash_client()

    if not _current_db_version:
        raise HTTPException(
            status_code=503,
            detail="Face recognition database not loaded. DB version unknown.",
        )

    # Create generator
    generator = SceneFingerprintGenerator(
        stash_client=stash,
        rec_db=db,
        db_version=_current_db_version,
        num_frames=request.num_frames,
        min_face_size=request.min_face_size,
        max_distance=request.max_distance,
    )
    set_generator(generator)

    # Run in background
    async def run_generation():
        async for progress in generator.generate_all(refresh_outdated=request.refresh_outdated):
            pass  # Progress is tracked in generator.progress

    _fingerprint_task = asyncio.create_task(run_generation())

    return {
        "status": "started",
        "message": "Fingerprint generation started in background",
        "config": {
            "refresh_outdated": request.refresh_outdated,
            "num_frames": request.num_frames,
            "min_face_size": request.min_face_size,
            "max_distance": request.max_distance,
            "db_version": _current_db_version,
        },
    }


@router.post("/fingerprints/stop")
async def stop_fingerprint_generation():
    """Stop the running fingerprint generation gracefully."""
    from fingerprint_generator import get_generator

    generator = get_generator()
    if not generator:
        raise HTTPException(status_code=404, detail="No fingerprint generation running")

    if generator.status.value not in ("running", "stopping"):
        return {
            "status": generator.status.value,
            "message": "Generation is not running",
        }

    generator.request_stop()

    return {
        "status": "stopping",
        "message": "Stop requested. Will finish current scene then stop.",
        "progress": generator.progress.to_dict(),
    }


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
    """Apply selected upstream changes to a performer."""
    stash = get_stash_client()
    fields = dict(request.fields)
    performer_id = request.performer_id

    # Process _alias_add meta-key: merge new aliases into existing alias list
    alias_additions = fields.pop("_alias_add", None)
    if alias_additions:
        # Fetch current performer to get existing aliases
        current = await stash.get_performer(performer_id)
        existing_aliases = current.get("alias_list", []) if current else []
        # Merge: existing + new additions (deduplicated, case-insensitive)
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

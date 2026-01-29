"""
Recommendations API Router

Endpoints for managing recommendations, running analysis, and configuration.
"""

import os
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from recommendations_db import RecommendationsDB, Recommendation, AnalysisRun
from stash_client_unified import StashClientUnified
from analyzers import DuplicatePerformerAnalyzer, DuplicateSceneFilesAnalyzer

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

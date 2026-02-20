"""Model management API router.

Endpoints for checking model installation status, downloading models
from GitHub Releases, and monitoring download progress.
"""

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException

from model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# Set at startup
_manager: Optional[ModelManager] = None


def init_model_router(manager: ModelManager) -> None:
    """Store the ModelManager reference. Called once during lifespan."""
    global _manager
    _manager = manager


def _get_manager() -> ModelManager:
    """Get the model manager, raising if not initialized."""
    if _manager is None:
        raise RuntimeError("Model router not initialized")
    return _manager


@router.get("/models/status")
async def get_model_status():
    """Get installation status of all models."""
    mgr = _get_manager()
    return {"models": mgr.get_status()}


@router.post("/models/download/{model_name}")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """Start downloading a model in the background.

    Returns immediately with a confirmation. Use GET /models/download-progress
    to monitor the download.
    """
    mgr = _get_manager()

    # Validate model name exists in manifest
    status = mgr.get_status()
    if model_name not in status:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown model: {model_name}",
        )

    async def _do_download():
        try:
            await mgr.download_model(model_name)
        except Exception as e:
            logger.warning(f"Background download failed for {model_name}: {e}")

    background_tasks.add_task(_do_download)
    return {"status": "download_started", "model": model_name}


@router.post("/models/download-all")
async def download_all_models(background_tasks: BackgroundTasks):
    """Start downloading all missing models in the background.

    Returns immediately with a confirmation. Use GET /models/download-progress
    to monitor progress.
    """
    mgr = _get_manager()

    async def _do_download_all():
        try:
            await mgr.download_all()
        except Exception as e:
            logger.warning(f"Background download-all failed: {e}")

    background_tasks.add_task(_do_download_all)
    return {"status": "download_started"}


@router.get("/models/download-progress")
async def get_download_progress():
    """Get progress of active and recent downloads."""
    mgr = _get_manager()
    return {"progress": mgr.get_progress()}

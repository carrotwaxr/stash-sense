"""Curation settings endpoints (#125).

The plugin edits the settings in Stash directly; these endpoints report
what the sidecar sees and drop its cache after a save.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import curation_settings
from curation_settings import ExclusionUnavailable, SceneExcluded

router = APIRouter(prefix="/curation", tags=["curation"])


def _stash():
    from recommendations_router import get_stash_client
    return get_stash_client()


@router.get("/settings")
async def get_curation_settings():
    try:
        ex = await curation_settings.current(_stash())
    except ExclusionUnavailable as e:
        return {"settings": None, "excluded_scene_count": None, "error": str(e)}
    return {"settings": ex.settings.to_dict(), "excluded_scene_count": len(ex.scene_ids), "error": None}


@router.post("/refresh")
async def refresh_curation_settings():
    curation_settings.invalidate()
    return await get_curation_settings()


async def scene_excluded_handler(request: Request, exc: SceneExcluded):
    return JSONResponse(
        status_code=403,
        content={"detail": str(exc), "error": "scene_excluded", "scene_ids": exc.scene_ids},
    )


async def exclusion_unavailable_handler(request: Request, exc: ExclusionUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc), "error": "exclusion_unavailable"})


def install(app) -> None:
    app.include_router(router)
    app.add_exception_handler(SceneExcluded, scene_excluded_handler)
    app.add_exception_handler(ExclusionUnavailable, exclusion_unavailable_handler)

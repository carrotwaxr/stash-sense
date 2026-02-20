"""Settings API Router.

Endpoints for reading, updating, and resetting sidecar settings.
Also provides system info (hardware profile, version, uptime).
"""

import time
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hardware import get_hardware_profile
from settings import get_settings_manager, SETTING_DEFS
from stashbox_connection_manager import get_connection_manager

router = APIRouter(tags=["settings"])

# Set at startup
_start_time: Optional[float] = None
_version: str = "0.1.0-beta.1"


def init_settings_router():
    """Record startup time. Called once during lifespan."""
    global _start_time
    _start_time = time.monotonic()


# ==================== Request/Response models ====================

class UpdateSettingRequest(BaseModel):
    value: Any


class BulkUpdateRequest(BaseModel):
    settings: dict[str, Any]


# ==================== Settings endpoints ====================

@router.get("/settings")
async def get_all_settings():
    """Get all settings grouped by category with metadata for UI rendering."""
    mgr = get_settings_manager()
    return mgr.get_all_with_metadata()


@router.get("/settings/{key}")
async def get_setting(key: str):
    """Get a single setting with metadata."""
    if key not in SETTING_DEFS:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")

    mgr = get_settings_manager()
    defn = SETTING_DEFS[key]
    default = mgr.get_default(key)
    is_override = mgr.has_override(key)
    value = mgr.get(key)

    result = {
        "key": key,
        "value": value,
        "default": default,
        "is_override": is_override,
        "type": defn.type.value,
        "label": defn.label,
        "description": defn.description,
    }
    if defn.min_val is not None:
        result["min"] = defn.min_val
    if defn.max_val is not None:
        result["max"] = defn.max_val

    return result


@router.put("/settings/{key}")
async def update_setting(key: str, request: UpdateSettingRequest):
    """Set a single setting override."""
    if key not in SETTING_DEFS:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")

    mgr = get_settings_manager()
    try:
        stored = mgr.set(key, request.value)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {"key": key, "value": stored, "is_override": True}


@router.put("/settings")
async def bulk_update_settings(request: BulkUpdateRequest):
    """Update multiple settings at once. Only keys present are updated."""
    mgr = get_settings_manager()
    errors = {}
    stored = {}

    for key, value in request.settings.items():
        if key not in SETTING_DEFS:
            errors[key] = f"Unknown setting: {key}"
            continue
        try:
            stored[key] = mgr.set(key, value)
        except ValueError as e:
            errors[key] = str(e)

    if errors:
        raise HTTPException(status_code=422, detail={"errors": errors, "stored": stored})

    return {"stored": stored}


@router.delete("/settings/{key}")
async def reset_setting(key: str):
    """Reset a setting to its tier default."""
    if key not in SETTING_DEFS:
        raise HTTPException(status_code=404, detail=f"Unknown setting: {key}")

    mgr = get_settings_manager()
    mgr.delete(key)
    default = mgr.get_default(key)

    return {"key": key, "value": default, "is_override": False}


# ==================== System info ====================

@router.get("/system/info")
async def get_system_info():
    """Get hardware profile, version, and uptime."""
    profile = get_hardware_profile()
    uptime_seconds = time.monotonic() - _start_time if _start_time else 0

    return {
        "version": _version,
        "uptime_seconds": round(uptime_seconds),
        "hardware": {
            "gpu_available": profile.gpu_available,
            "gpu_name": profile.gpu_name,
            "gpu_vram_mb": profile.gpu_vram_mb,
            "cpu_cores": profile.cpu_cores,
            "memory_total_mb": profile.memory_total_mb,
            "memory_available_mb": profile.memory_available_mb,
            "storage_free_mb": profile.storage_free_mb,
            "tier": profile.tier,
            "summary": profile.summary(),
        },
    }


# ==================== StashBox connections ====================

@router.get("/system/stashbox-connections")
async def get_stashbox_connections():
    """List all stash-box endpoints discovered from Stash's configuration."""
    mgr = get_connection_manager()
    return {"connections": mgr.get_connections()}


@router.post("/system/refresh-stashbox-config")
async def refresh_stashbox_config():
    """Re-read stash-box endpoint config from Stash without restarting."""
    mgr = get_connection_manager()
    count = await mgr.refresh()
    return {"endpoints_loaded": count, "connections": mgr.get_connections()}

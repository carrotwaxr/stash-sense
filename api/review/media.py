"""Stills for review cards: one ffmpeg frame from a scene's file on the media mount."""

from __future__ import annotations

import asyncio
import os

from media_exec import run_command
from media_library import MediaPathError, get_media_library

from .types import TargetMissing

STILL_TIMEOUT_S = 30
# A page of cards asks for many stills at once; don't start that many ffmpegs.
_still_slots = asyncio.Semaphore(2)


class StillError(RuntimeError):
    pass


def still_command(local_path: str, t: float, width: int) -> list[str]:
    # -ss before -i seeks by keyframe: fast on multi-GB files, close enough for a still.
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-ss", f"{max(t, 0):.2f}", "-i", local_path,
        "-frames:v", "1", "-vf", f"scale={width}:-2",
        "-f", "image2", "-c:v", "mjpeg", "-q:v", "4", "pipe:1",
    ]


async def extract_still(stash, scene_id: str, file_id: str, t: float, width: int = 480) -> bytes:
    """JPEG bytes of file_id (which must belong to scene_id) at t seconds."""
    scene = await stash.get_scene_for_review(scene_id)
    if not scene:
        raise TargetMissing(f"Scene {scene_id} not found")
    f = next((f for f in scene.get("files") or [] if str(f["id"]) == str(file_id)), None)
    if f is None:
        raise TargetMissing(f"File {file_id} is not on scene {scene_id}")
    try:
        local = get_media_library().to_local(f["path"])
    except MediaPathError as e:
        raise StillError(str(e)) from e
    if not os.path.exists(local):
        raise StillError(f"{local} is not visible in the sidecar's media mount")
    width = max(64, min(int(width), 1280))
    if not (t >= 0):  # also rejects NaN
        t = 0.0
    async with _still_slots:
        res = await run_command(still_command(local, t, width), timeout=STILL_TIMEOUT_S)
    if res.returncode != 0 or not res.stdout:
        raise StillError(f"ffmpeg failed: {res.stderr.decode(errors='replace')[-300:]}")
    return res.stdout

"""Where benchmark data lives and where frames are read from.

The benchmark set and results are personal, so they live under the data
volume (``$DATA_DIR/benchmark``), never in the repo. Frames come from the
media mount when Stash's path maps onto it, otherwise from Stash's stream.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def benchmark_dir() -> Path:
    """Root for the benchmark set, runs and the detection test set."""
    override = os.environ.get("BENCHMARK_DIR")
    if override:
        return Path(override)
    return Path(os.environ.get("DATA_DIR", "./data")) / "benchmark"


def default_set_path() -> Path:
    return benchmark_dir() / "scenes.json"


def default_runs_dir() -> Path:
    return benchmark_dir() / "runs"


def default_detection_dir() -> Path:
    return benchmark_dir() / "detection"


@dataclass
class VideoSource:
    """An ffmpeg input plus the API key it needs (None for a local file)."""
    input: str
    api_key: Optional[str]
    local: bool


def resolve_video_source(
    stash_path: Optional[str],
    scene_id: str,
    stash_url: str,
    api_key: str,
    library=None,
    exists=os.path.exists,
) -> VideoSource:
    """Prefer the file on the media mount; fall back to Stash's stream URL."""
    if stash_path:
        if library is None:
            from media_library import get_media_library
            library = get_media_library()
        try:
            local = library.to_local(stash_path)
        except ValueError:
            local = None
        if local and exists(local):
            return VideoSource(input=local, api_key=None, local=True)
    from frame_extractor import build_stash_stream_url
    return VideoSource(
        input=build_stash_stream_url(stash_url, scene_id),
        api_key=api_key or None,
        local=False,
    )

"""Extract frames from video streams using ffmpeg.

Extracts high-resolution frames from Stash video streams via HTTP,
without downloading the entire video file.
"""
import asyncio
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image
import io


# =============================================================================
# CONFIGURATION - Tune these values as needed
# =============================================================================

@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction. All values are easily tunable."""

    # Number of frames to extract
    num_frames: int = 40  # Target number of frames (30-60 recommended)

    # Time offsets (skip intros/outros)
    start_offset_pct: float = 0.05  # Skip first 5% (logos, intros)
    end_offset_pct: float = 0.95    # Skip last 5% (credits, outros)

    # Alternatively, use absolute offsets (these override percentages if set)
    start_offset_sec: Optional[float] = None  # e.g., 30.0 to skip first 30s
    end_offset_sec: Optional[float] = None    # e.g., 30.0 to skip last 30s

    # Burst mode: grab multiple frames around each sample point
    # Instead of 40 single frames, grab e.g. 10 points × 4 frames each
    burst_mode: bool = False             # Enable burst sampling
    frames_per_burst: int = 4            # Number of frames per sample point
    burst_spread_sec: float = 0.5        # Spread frames ±N seconds around point

    # Frame quality
    output_width: Optional[int] = None   # None = original width
    output_height: Optional[int] = None  # None = original height
    jpeg_quality: int = 95               # JPEG quality (1-100)

    # Face detection filters (applied after extraction)
    min_face_size: int = 40              # Minimum face dimension in pixels
    min_face_confidence: float = 0.5     # Minimum detection confidence

    # Performance
    max_concurrent_extractions: int = 4  # Parallel ffmpeg processes
    extraction_timeout_sec: float = 30.0 # Timeout per frame

    # ffmpeg settings
    ffmpeg_path: str = "ffmpeg"          # Path to ffmpeg binary


# Default config instance
DEFAULT_CONFIG = FrameExtractionConfig()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ExtractedFrame:
    """A single frame extracted from video."""
    image: np.ndarray      # RGB image as numpy array
    timestamp_sec: float   # Timestamp in seconds
    frame_index: int       # Index in extraction sequence
    width: int
    height: int


@dataclass
class ExtractionResult:
    """Result of frame extraction operation."""
    frames: list[ExtractedFrame]
    scene_duration_sec: float
    extraction_times_sec: list[float]  # Actual timestamps extracted
    errors: list[str] = field(default_factory=list)


# =============================================================================
# FRAME EXTRACTION
# =============================================================================

def calculate_extraction_timestamps(
    duration_sec: float,
    config: FrameExtractionConfig = DEFAULT_CONFIG,
) -> list[float]:
    """
    Calculate timestamps for frame extraction.

    In normal mode: evenly-spaced single frames across the video.
    In burst mode: multiple frames around fewer sample points.

    Args:
        duration_sec: Total video duration in seconds
        config: Extraction configuration

    Returns:
        List of timestamps (in seconds) to extract frames at
    """
    # Determine start/end bounds
    if config.start_offset_sec is not None:
        start_sec = config.start_offset_sec
    else:
        start_sec = duration_sec * config.start_offset_pct

    if config.end_offset_sec is not None:
        end_sec = duration_sec - config.end_offset_sec
    else:
        end_sec = duration_sec * config.end_offset_pct

    # Ensure valid range
    start_sec = max(0, start_sec)
    end_sec = min(duration_sec, end_sec)

    if end_sec <= start_sec:
        # Video too short, just sample middle
        return [duration_sec / 2]

    # Use burst mode if enabled
    if config.burst_mode:
        return _calculate_burst_timestamps(
            start_sec, end_sec, duration_sec, config
        )

    # Normal mode: evenly-spaced single frames
    if config.num_frames <= 1:
        return [(start_sec + end_sec) / 2]

    interval = (end_sec - start_sec) / (config.num_frames - 1)
    timestamps = [start_sec + i * interval for i in range(config.num_frames)]

    return timestamps


def _calculate_burst_timestamps(
    start_sec: float,
    end_sec: float,
    duration_sec: float,
    config: FrameExtractionConfig,
) -> list[float]:
    """
    Calculate burst-mode timestamps: multiple frames around each sample point.

    Instead of 40 single frames at 40 points, we grab e.g. 10 points × 4 frames each.
    Frames within a burst are spread ±burst_spread_sec around the sample point.

    Benefits:
    - Nearby frames have similar lighting → better clustering
    - More likely to catch at least one good face angle per region
    - Same total frame count, different distribution

    Args:
        start_sec: Start of usable video range
        end_sec: End of usable video range
        duration_sec: Total video duration
        config: Extraction configuration

    Returns:
        List of timestamps for burst sampling
    """
    # Calculate number of sample points (total frames / frames per burst)
    num_sample_points = max(1, config.num_frames // config.frames_per_burst)

    # Calculate sample point positions (evenly spaced)
    if num_sample_points <= 1:
        sample_points = [(start_sec + end_sec) / 2]
    else:
        interval = (end_sec - start_sec) / (num_sample_points - 1)
        sample_points = [start_sec + i * interval for i in range(num_sample_points)]

    # Generate burst timestamps around each sample point
    timestamps = []
    for point in sample_points:
        # Spread frames evenly within ±burst_spread_sec
        if config.frames_per_burst <= 1:
            offsets = [0.0]
        else:
            # e.g., for 4 frames with spread=0.5: [-0.5, -0.17, +0.17, +0.5]
            offsets = [
                -config.burst_spread_sec + (2 * config.burst_spread_sec * i / (config.frames_per_burst - 1))
                for i in range(config.frames_per_burst)
            ]

        for offset in offsets:
            ts = point + offset
            # Clamp to valid range
            ts = max(0, min(duration_sec, ts))
            timestamps.append(ts)

    return timestamps


def extract_frame_sync(
    stream_url: str,
    timestamp_sec: float,
    api_key: Optional[str] = None,
    config: FrameExtractionConfig = DEFAULT_CONFIG,
) -> Optional[np.ndarray]:
    """
    Extract a single frame from video stream using ffmpeg (synchronous).

    Args:
        stream_url: HTTP URL to video stream
        timestamp_sec: Timestamp to extract (in seconds)
        api_key: Optional API key for authentication
        config: Extraction configuration

    Returns:
        RGB image as numpy array, or None on failure
    """
    # Build ffmpeg command
    # Key: -ss before -i enables fast seeking without decoding everything
    cmd = [
        config.ffmpeg_path,
        "-ss", str(timestamp_sec),        # Seek to timestamp (before -i for fast seek)
        "-headers", f"ApiKey: {api_key}\r\n" if api_key else "",
        "-i", stream_url,                 # Input from HTTP stream
        "-frames:v", "1",                 # Extract exactly 1 frame
        "-f", "image2pipe",               # Output to pipe
        "-vcodec", "mjpeg",               # Output as JPEG
        "-q:v", str(max(1, min(31, 32 - config.jpeg_quality // 3))),  # Quality (1=best, 31=worst)
        "-"                               # Output to stdout
    ]

    # Add scaling if requested
    if config.output_width or config.output_height:
        w = config.output_width or -1
        h = config.output_height or -1
        cmd.insert(-1, "-vf")
        cmd.insert(-1, f"scale={w}:{h}")

    # Remove empty headers arg if no API key
    if not api_key:
        cmd = [c for c in cmd if c != "-headers" and c != ""]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=config.extraction_timeout_sec,
        )

        if result.returncode != 0:
            # ffmpeg failed - this can happen for seeks past end, etc.
            return None

        if not result.stdout:
            return None

        # Decode JPEG to numpy array
        image = Image.open(io.BytesIO(result.stdout))
        if image.mode != "RGB":
            image = image.convert("RGB")

        return np.array(image)

    except subprocess.TimeoutExpired:
        return None
    except Exception:
        return None


async def extract_frame_async(
    stream_url: str,
    timestamp_sec: float,
    frame_index: int,
    api_key: Optional[str] = None,
    config: FrameExtractionConfig = DEFAULT_CONFIG,
) -> Optional[ExtractedFrame]:
    """
    Extract a single frame asynchronously.

    Runs ffmpeg in a thread pool to avoid blocking the event loop.
    """
    loop = asyncio.get_event_loop()

    image = await loop.run_in_executor(
        None,
        extract_frame_sync,
        stream_url,
        timestamp_sec,
        api_key,
        config,
    )

    if image is None:
        return None

    return ExtractedFrame(
        image=image,
        timestamp_sec=timestamp_sec,
        frame_index=frame_index,
        width=image.shape[1],
        height=image.shape[0],
    )


async def extract_frames(
    stream_url: str,
    duration_sec: float,
    api_key: Optional[str] = None,
    config: FrameExtractionConfig = DEFAULT_CONFIG,
) -> ExtractionResult:
    """
    Extract multiple frames from a video stream.

    Args:
        stream_url: HTTP URL to video stream
        duration_sec: Video duration in seconds
        api_key: Optional API key for authentication
        config: Extraction configuration

    Returns:
        ExtractionResult with extracted frames
    """
    timestamps = calculate_extraction_timestamps(duration_sec, config)

    # Create semaphore to limit concurrent extractions
    semaphore = asyncio.Semaphore(config.max_concurrent_extractions)

    async def extract_with_semaphore(ts: float, idx: int) -> Optional[ExtractedFrame]:
        async with semaphore:
            return await extract_frame_async(stream_url, ts, idx, api_key, config)

    # Extract all frames concurrently (limited by semaphore)
    tasks = [
        extract_with_semaphore(ts, idx)
        for idx, ts in enumerate(timestamps)
    ]

    results = await asyncio.gather(*tasks)

    # Filter successful extractions
    frames = [f for f in results if f is not None]
    errors = [
        f"Failed to extract frame at {ts:.1f}s"
        for ts, f in zip(timestamps, results)
        if f is None
    ]

    return ExtractionResult(
        frames=frames,
        scene_duration_sec=duration_sec,
        extraction_times_sec=[f.timestamp_sec for f in frames],
        errors=errors,
    )


# =============================================================================
# STASH INTEGRATION
# =============================================================================

def build_stash_stream_url(stash_url: str, scene_id: str) -> str:
    """Build the streaming URL for a Stash scene."""
    base = stash_url.rstrip("/")
    return f"{base}/scene/{scene_id}/stream"


async def extract_frames_from_stash_scene(
    stash_url: str,
    scene_id: str,
    duration_sec: float,
    api_key: Optional[str] = None,
    config: FrameExtractionConfig = DEFAULT_CONFIG,
) -> ExtractionResult:
    """
    Extract frames from a Stash scene.

    Args:
        stash_url: Base Stash URL (e.g., http://localhost:9999)
        scene_id: Stash scene ID
        duration_sec: Scene duration in seconds
        api_key: Stash API key
        config: Extraction configuration

    Returns:
        ExtractionResult with extracted frames
    """
    stream_url = build_stash_stream_url(stash_url, scene_id)
    return await extract_frames(stream_url, duration_sec, api_key, config)


# =============================================================================
# UTILITIES
# =============================================================================

def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm for ffmpeg."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"


def check_ffmpeg_available(ffmpeg_path: str = "ffmpeg") -> bool:
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    import sys
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def test_extraction():
        stash_url = os.environ.get("STASH_URL", "http://10.0.0.4:6969")
        api_key = os.environ.get("STASH_API_KEY", "")
        scene_id = sys.argv[1] if len(sys.argv) > 1 else "33326"
        duration = float(sys.argv[2]) if len(sys.argv) > 2 else 1800  # Default 30 min

        print(f"Testing frame extraction for scene {scene_id}")
        print(f"Stash URL: {stash_url}")
        print(f"Duration: {duration}s")
        print()

        # Check ffmpeg
        if not check_ffmpeg_available():
            print("ERROR: ffmpeg not found!")
            return

        # Configure for test (fewer frames)
        config = FrameExtractionConfig(
            num_frames=5,  # Just 5 for testing
            start_offset_pct=0.1,
            end_offset_pct=0.9,
        )

        print(f"Extracting {config.num_frames} frames...")
        timestamps = calculate_extraction_timestamps(duration, config)
        print(f"Timestamps: {[f'{t:.1f}s' for t in timestamps]}")
        print()

        result = await extract_frames_from_stash_scene(
            stash_url, scene_id, duration, api_key, config
        )

        print(f"Extracted {len(result.frames)} frames:")
        for frame in result.frames:
            print(f"  Frame {frame.frame_index}: {frame.timestamp_sec:.1f}s, {frame.width}x{frame.height}")

        if result.errors:
            print(f"\nErrors:")
            for err in result.errors:
                print(f"  {err}")

        # Save first frame for inspection
        if result.frames:
            from PIL import Image
            img = Image.fromarray(result.frames[0].image)
            img.save("/tmp/test_frame.jpg")
            print(f"\nFirst frame saved to /tmp/test_frame.jpg")

    asyncio.run(test_extraction())

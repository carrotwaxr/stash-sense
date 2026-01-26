"""Parse Stash sprite sheets and VTT files for multi-frame extraction.

Stash generates sprite sheets (grid of thumbnails) and VTT files (timestamps)
for scene previews. This module extracts individual frames for face detection.
"""
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image


@dataclass
class SpriteFrame:
    """A single frame extracted from a sprite sheet."""
    image: np.ndarray  # RGB image
    timestamp: float  # Seconds into the scene
    index: int  # Frame index in the sprite sheet


@dataclass
class VTTCue:
    """A single cue from a WebVTT file."""
    start_time: float  # Seconds
    end_time: float  # Seconds
    x: int  # X offset in sprite sheet
    y: int  # Y offset in sprite sheet
    width: int
    height: int


def parse_vtt_timestamp(timestamp: str) -> float:
    """Parse VTT timestamp (HH:MM:SS.mmm) to seconds."""
    parts = timestamp.strip().split(":")
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    elif len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    else:
        return float(parts[0])


def parse_vtt_file(vtt_content: str) -> list[VTTCue]:
    """
    Parse a Stash sprite VTT file.

    VTT format example:
    ```
    WEBVTT

    00:00:00.000 --> 00:00:10.000
    sprite.jpg#xywh=0,0,160,90

    00:00:10.000 --> 00:00:20.000
    sprite.jpg#xywh=160,0,160,90
    ```
    """
    cues = []
    lines = vtt_content.strip().split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Skip WEBVTT header and empty lines
        if not line or line == "WEBVTT" or line.startswith("NOTE"):
            i += 1
            continue

        # Look for timestamp line (contains " --> ")
        if " --> " in line:
            timestamps = line.split(" --> ")
            start_time = parse_vtt_timestamp(timestamps[0])
            end_time = parse_vtt_timestamp(timestamps[1].split()[0])  # Remove any trailing info

            # Next line should have the sprite coordinates
            i += 1
            if i < len(lines):
                sprite_line = lines[i].strip()
                # Parse #xywh=x,y,w,h
                match = re.search(r"#xywh=(\d+),(\d+),(\d+),(\d+)", sprite_line)
                if match:
                    x, y, w, h = map(int, match.groups())
                    cues.append(VTTCue(
                        start_time=start_time,
                        end_time=end_time,
                        x=x,
                        y=y,
                        width=w,
                        height=h,
                    ))

        i += 1

    return cues


def extract_frames_from_sprite(
    sprite_image: Image.Image,
    vtt_cues: list[VTTCue],
    max_frames: Optional[int] = None,
    sample_interval: Optional[int] = None,
) -> list[SpriteFrame]:
    """
    Extract individual frames from a sprite sheet.

    Args:
        sprite_image: PIL Image of the sprite sheet
        vtt_cues: List of VTT cues with frame positions
        max_frames: Maximum number of frames to extract
        sample_interval: Only extract every Nth frame (for performance)

    Returns:
        List of SpriteFrame objects
    """
    frames = []

    indices = range(len(vtt_cues))
    if sample_interval:
        indices = indices[::sample_interval]
    if max_frames:
        indices = list(indices)[:max_frames]

    for idx in indices:
        cue = vtt_cues[idx]

        # Crop the frame from the sprite sheet
        frame_img = sprite_image.crop((
            cue.x,
            cue.y,
            cue.x + cue.width,
            cue.y + cue.height,
        ))

        # Convert to RGB numpy array
        frame_array = np.array(frame_img.convert("RGB"))

        frames.append(SpriteFrame(
            image=frame_array,
            timestamp=(cue.start_time + cue.end_time) / 2,  # Midpoint
            index=idx,
        ))

    return frames


def load_sprite_sheet(
    sprite_path: str | Path,
    vtt_path: str | Path,
    max_frames: Optional[int] = 20,
    sample_interval: Optional[int] = None,
) -> list[SpriteFrame]:
    """
    Load and parse a sprite sheet with its VTT file.

    Args:
        sprite_path: Path to sprite sheet image
        vtt_path: Path to VTT file
        max_frames: Maximum frames to extract
        sample_interval: Sample every Nth frame

    Returns:
        List of extracted frames
    """
    with open(vtt_path) as f:
        vtt_content = f.read()

    cues = parse_vtt_file(vtt_content)

    sprite_image = Image.open(sprite_path)

    return extract_frames_from_sprite(
        sprite_image,
        cues,
        max_frames=max_frames,
        sample_interval=sample_interval,
    )


async def fetch_sprite_from_stash(
    stash_url: str,
    scene_id: str,
    api_key: str,
    max_frames: int = 20,
) -> list[SpriteFrame]:
    """
    Fetch sprite sheet and VTT from a Stash instance.

    Args:
        stash_url: Base URL of Stash instance
        scene_id: Scene ID
        api_key: Stash API key
        max_frames: Maximum frames to extract

    Returns:
        List of extracted frames
    """
    import httpx
    from io import BytesIO

    headers = {"ApiKey": api_key}
    base_url = stash_url.rstrip("/")

    async with httpx.AsyncClient(timeout=60.0) as client:
        # Fetch sprite image
        sprite_url = f"{base_url}/scene/{scene_id}/sprite"
        sprite_response = await client.get(sprite_url, headers=headers)
        sprite_response.raise_for_status()
        sprite_image = Image.open(BytesIO(sprite_response.content))

        # Fetch VTT file
        vtt_url = f"{base_url}/scene/{scene_id}/vtt/sprite"
        vtt_response = await client.get(vtt_url, headers=headers)
        vtt_response.raise_for_status()
        vtt_content = vtt_response.text

    cues = parse_vtt_file(vtt_content)

    # Calculate sample interval to get approximately max_frames
    sample_interval = max(1, len(cues) // max_frames) if len(cues) > max_frames else None

    return extract_frames_from_sprite(
        sprite_image,
        cues,
        max_frames=max_frames,
        sample_interval=sample_interval,
    )


if __name__ == "__main__":
    # Test with a local sprite sheet
    import sys

    if len(sys.argv) >= 3:
        sprite_path = sys.argv[1]
        vtt_path = sys.argv[2]

        frames = load_sprite_sheet(sprite_path, vtt_path, max_frames=10)
        print(f"Extracted {len(frames)} frames:")
        for frame in frames:
            print(f"  Frame {frame.index}: {frame.timestamp:.1f}s, shape={frame.image.shape}")
    else:
        print("Usage: python sprite_parser.py <sprite.jpg> <sprite.vtt>")

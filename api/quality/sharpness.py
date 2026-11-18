"""Blur and block sharpness of a file, from ffmpeg's blurdetect and blockdetect over six 3 s windows.

Lower blur is sharper and lower block is less blocky. Windows run one at a time (parallel seeks thrash a spinning
disk) and the result is the median over the windows that parsed.
"""
from __future__ import annotations

import re
import statistics

from duplicates.health import Run, _default_run

SHARPNESS_VERSION = 1
FRACTIONS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)
WINDOW_S = 3

_BLUR = re.compile(r"blur mean: ([\d.]+)")
_BLOCK = re.compile(r"block mean: ([\d.]+)")


def parse(stderr: str) -> tuple[float | None, float | None]:
    """(blur, block): the last mean of each, None when absent."""
    blur = _BLUR.findall(stderr)
    block = _BLOCK.findall(stderr)
    return (float(blur[-1]) if blur else None, float(block[-1]) if block else None)


async def measure(local_path: str, duration: float, run: Run | None = None) -> dict | None:
    """{'blur', 'block', 'windows', 'version'} as medians over the windows that parsed, or None if none did."""
    run = run or _default_run
    blurs: list[float] = []
    blocks: list[float] = []
    windows = 0
    for frac in FRACTIONS:
        args = ["ffmpeg", "-nostdin", "-v", "info", "-ss", f"{duration * frac:.1f}", "-i", local_path,
                "-t", str(WINDOW_S), "-an", "-vf", "blurdetect=block_width=32:block_height=32,blockdetect",
                "-f", "null", "-"]
        code, text = await run(args)
        if code:
            continue
        blur, block = parse(text)
        if blur is None and block is None:
            continue
        windows += 1
        if blur is not None:
            blurs.append(blur)
        if block is not None:
            blocks.append(block)
    if not windows:
        return None
    return {"blur": statistics.median(blurs) if blurs else None,
            "block": statistics.median(blocks) if blocks else None,
            "windows": windows, "version": SHARPNESS_VERSION}

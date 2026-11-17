"""Run ffmpeg for CPU-heavy jobs at a lower priority and with a thread cap.

Both knobs are settings (media_job_nice, media_job_threads) so a long
library pass does not crowd Stash or other containers on the host. Pinning
the whole container to cores is a template choice (--cpuset-cpus).
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from typing import Optional

FFMPEG_BINARIES = {"ffmpeg"}


def _setting(key: str) -> int:
    from settings import SETTING_DEFS, get_setting
    try:
        return int(get_setting(key))
    except RuntimeError:  # settings not initialized
        return int(SETTING_DEFS[key].fallback)


def cpu_heavy_command(cmd: list[str], nice: int, threads: int) -> list[str]:
    """Prefix cmd with `nice -n` and, for ffmpeg, cap decode and filter threads.

    ffmpeg's -threads is per input when placed before -i, so it is added
    before every input. -filter_threads is global. threads=0 leaves ffmpeg's
    own choice (one per core).
    """
    if not cmd:
        raise ValueError("empty command")
    out = list(cmd)
    if threads > 0 and os.path.basename(out[0]) in FFMPEG_BINARIES:
        capped = [out[0], "-filter_threads", str(threads)]
        for arg in out[1:]:
            if arg == "-i":
                capped += ["-threads", str(threads)]
            capped.append(arg)
        out = capped
    if nice > 0:
        out = ["nice", "-n", str(nice)] + out
    return out


def media_job_command(cmd: list[str]) -> list[str]:
    """cpu_heavy_command with the current settings."""
    return cpu_heavy_command(cmd, _setting("media_job_nice"), _setting("media_job_threads"))


@dataclass
class CommandResult:
    args: list[str]
    returncode: int
    stdout: bytes
    stderr: bytes


async def run_command(cmd: list[str], timeout: Optional[float] = None) -> CommandResult:
    """Run cmd without blocking the event loop; kill it on timeout or cancellation."""
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except BaseException:
        # Timeout, or the job task was cancelled: don't leave ffmpeg running.
        if proc.returncode is None:
            proc.kill()
            await asyncio.shield(proc.wait())
        raise
    return CommandResult(cmd, proc.returncode, stdout, stderr)

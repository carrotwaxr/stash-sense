"""Blur and block sharpness: parsing, the window plan and the median, with a fake runner and real ffmpeg."""
import asyncio
import math
import shutil
import subprocess

import pytest

from quality import sharpness

DURATION = 1000.0


def out(blur, block):
    return (f"[Parsed_blurdetect_0 @ 0x1] blur mean: {blur}\n"
            f"[Parsed_blockdetect_1 @ 0x2] block mean: {block}\n")


def test_parse_takes_last_means():
    text = ("blur mean: 5.0\nsome noise\nblur mean: 7.25\nblock mean: 1.5\n")
    assert sharpness.parse(text) == (7.25, 1.5)
    assert sharpness.parse("nothing here") == (None, None)


def test_measure_takes_median_of_windows():
    blurs = [1.0, 2.0, 3.0, 4.0, 5.0, 100.0]
    blocks = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    calls = []

    async def run(args):
        calls.append(list(args))
        i = len(calls) - 1
        return 0, out(blurs[i], blocks[i])

    res = asyncio.run(sharpness.measure("/x/v.mp4", DURATION, run=run))
    assert res == {"blur": 3.5, "block": 35.0, "windows": 6, "version": 1}
    starts = [float(c[c.index("-ss") + 1]) for c in calls]
    assert starts == [round(DURATION * f, 1) for f in (0.1, 0.25, 0.4, 0.55, 0.7, 0.85)]
    assert all(c[c.index("-t") + 1] == "3" for c in calls)


def test_failed_windows_are_skipped():
    async def fail(args):
        return 1, "boom"

    assert asyncio.run(sharpness.measure("/x/v.mp4", DURATION, run=fail)) is None

    n = {"i": 0}

    async def some(args):
        n["i"] += 1
        return (0, out(2.0, 3.0)) if n["i"] % 2 else (1, "boom")

    res = asyncio.run(sharpness.measure("/x/v.mp4", DURATION, run=some))
    assert res["windows"] == 3 and res["blur"] == 2.0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_real_ffmpeg(tmp_path):
    f = tmp_path / "v.mp4"
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
                    "-t", "12", "-pix_fmt", "yuv420p", str(f)], check=True)
    res = asyncio.run(sharpness.measure(str(f), 12.0))
    assert res is not None and res["windows"] >= 1
    assert math.isfinite(res["blur"]) and math.isfinite(res["block"])

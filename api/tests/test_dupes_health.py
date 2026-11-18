"""Keeper health check with a fake command runner."""
import asyncio
import json

from duplicates import health

DUR = 600.0


def probe_out(video=DUR, audio=DUR, dur=DUR, prefix=""):
    streams = []
    if video is not None:
        streams.append({"codec_type": "video", "duration": str(video)})
    if audio is not None:
        streams.append({"codec_type": "audio", "duration": str(audio)})
    return prefix + json.dumps({"format": {"duration": str(dur)}, "streams": streams})


def make_run(probe=None, window=None, confirm=None, sequential=None):
    """Fake runner returning (exit code, text). Each ffmpeg kind is a (code, text) or a callable on its args."""
    calls = []

    async def run(args):
        calls.append(list(args))
        if args[0] == "ffprobe":
            return probe or (0, probe_out())
        length = float(args[args.index("-t") + 1])
        seeks = "-ss" in args
        kind = sequential if (not seeks and length > 100) else (window if length <= 1 else confirm)
        kind = kind or (0, "")
        return kind(args) if callable(kind) else kind

    run.calls = calls
    return run


def check(tmp_path, run, **kw):
    f = tmp_path / "v.mp4"
    f.write_bytes(b"x")
    return asyncio.run(health.check_health(str(f), run=run, **kw))


def test_ok(tmp_path):
    run = make_run()
    r = check(tmp_path, run)
    assert r == {"status": "ok", "reasons": [], "params": {"points": 24}}
    assert sum(c[0] == "ffmpeg" for c in run.calls) == 24


def test_failed_window_confirmed_is_suspect(tmp_path):
    first = {"n": 0}

    def window(args):
        first["n"] += 1
        return (1, "decode error: corrupt") if first["n"] == 3 else (0, "")

    r = check(tmp_path, make_run(window=window, confirm=(1, "still broken")))
    assert r["status"] == "suspect"
    assert "1/24 sample windows fail" in r["reasons"][0]


def test_failed_window_not_confirmed_is_ok(tmp_path):
    first = {"n": 0}

    def window(args):
        first["n"] += 1
        return (1, "oops") if first["n"] == 3 else (0, "")

    assert check(tmp_path, make_run(window=window))["status"] == "ok"


def test_nonmonotonic_dts_warns(tmp_path):
    r = check(tmp_path, make_run(window=(0, "Application provided invalid, non monotonically increasing dts to muxer")))
    assert r["status"] == "warn"
    assert len(r["reasons"]) == 1 and "dts" in r["reasons"][0]


def test_benign_only_is_clean(tmp_path):
    r = check(tmp_path, make_run(window=(0, "[mp3] Header missing\n[h264] non-existing PPS 0 referenced\n")))
    assert r == {"status": "ok", "reasons": [], "params": {"points": 24}}


def test_every_window_failing_but_sequential_clean(tmp_path):
    r = check(tmp_path, make_run(window=(1, "bad"), confirm=(1, "bad"), sequential=(0, "")))
    assert r["status"] == "ok"
    assert any("seek artifacts only" in x for x in r["reasons"])


def test_every_window_failing_and_sequential_failing(tmp_path):
    r = check(tmp_path, make_run(window=(1, "bad"), confirm=(1, "bad"), sequential=(1, "bad")))
    assert r["status"] == "suspect"
    assert "24/24" in r["reasons"][0]


def test_no_audio_stream_is_suspect(tmp_path):
    r = check(tmp_path, make_run(probe=(0, probe_out(audio=None))))
    assert r["status"] == "suspect"
    assert "no audio stream" in r["reasons"]


def test_duration_gap_is_suspect(tmp_path):
    r = check(tmp_path, make_run(probe=(0, probe_out(video=600, audio=510))))
    assert r["status"] == "suspect"
    assert any("durations differ" in x for x in r["reasons"])


def test_small_duration_gap_warns(tmp_path):
    assert check(tmp_path, make_run(probe=(0, probe_out(video=600, audio=596))))["status"] == "warn"


def test_probe_failure_is_suspect(tmp_path):
    r = check(tmp_path, make_run(probe=(1, "moov atom not found")))
    assert r["status"] == "suspect"


def test_missing_file(tmp_path):
    r = asyncio.run(health.check_health(str(tmp_path / "nope.mp4"), run=make_run()))
    assert r["status"] == "missing"
    assert r["params"] == {"points": 24}


def test_windows_run_sequentially(tmp_path):
    active = {"n": 0, "max": 0}

    async def run(args):
        if args[0] == "ffprobe":
            return 0, probe_out()
        active["n"] += 1
        active["max"] = max(active["max"], active["n"])
        await asyncio.sleep(0)
        active["n"] -= 1
        return 0, ""

    check(tmp_path, run)
    assert active["max"] == 1

"""Cheap probe pass and the tier decision (#133)."""

import asyncio
import json
import shutil
import subprocess

import pytest

from quality import probe

DUR = 600.0


@pytest.fixture(scope="module")
def mp4(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    p = tmp_path_factory.mktemp("probe") / "a.mp4"
    subprocess.run(
        [
            "ffmpeg", "-nostdin", "-v", "error", "-y",
            "-f", "lavfi", "-i", "testsrc=d=2:s=160x120",
            "-f", "lavfi", "-i", "sine=d=2",
            "-shortest", "-c:v", "mpeg4", "-c:a", "aac",
            "-movflags", "+faststart", str(p),
        ],
        check=True,
        capture_output=True,
    )
    return str(p)


def probe_out(video=DUR, audio=DUR, dur=DUR, tags=None, prefix=""):
    streams = []
    if video is not None:
        streams.append({"codec_type": "video", "codec_name": "h264", "duration": str(video)})
    if audio is not None:
        streams.append({"codec_type": "audio", "codec_name": "aac", "duration": str(audio)})
    fmt = {"duration": str(dur), "format_name": "mov,mp4,m4a,3gp,3g2,mj2"}
    if tags is not None:
        fmt["tags"] = tags
    return prefix + json.dumps({"format": fmt, "streams": streams})


def make_run(ffprobe=None, tail=None, window=None):
    """Fake runner keyed on argv: ffprobe, tail decode (-sseof), health windows (-ss)."""
    calls = []

    async def run(args):
        calls.append(list(args))
        if args[0] == "ffprobe":
            return ffprobe or (0, probe_out())
        if "-sseof" in args:
            return tail or (0, "")
        return window or (0, "")

    run.calls = calls
    return run


def cheap(path, run, stash=DUR):
    return asyncio.run(probe.cheap_check(path, stash, run=run))


def tiered(path, run, stash=DUR, **kw):
    return asyncio.run(probe.tiered(path, stash, run=run, **kw))


def seeks(run):
    return [c for c in run.calls if "-ss" in c]


def test_clean_file_is_ok(mp4):
    run = make_run()
    res = cheap(mp4, run)
    assert res["status"] == "ok" and res["reasons"] == []
    assert res["version"] == probe.CHEAP_VERSION
    assert res["index"] == {"kind": "mp4", "flags": []}
    assert res["probe"] == {"duration": DUR, "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
                            "video_codec": "h264", "has_audio": True}
    assert any("-sseof" in c and "-30" in c for c in run.calls)
    assert not probe.is_broken(res)


def test_missing_file(tmp_path):
    run = make_run()
    res = cheap(str(tmp_path / "nope.mp4"), run)
    assert res["status"] == "missing"
    assert run.calls == []
    assert probe.is_broken(res)


def test_ffprobe_failure_is_suspect(mp4):
    res = cheap(mp4, make_run(ffprobe=(1, "moov atom not found")))
    assert res["status"] == "suspect"
    assert "ffprobe failed" in res["reasons"][0]
    assert probe.is_broken(res)


def test_ffprobe_without_json_is_suspect(mp4):
    res = cheap(mp4, make_run(ffprobe=(0, "nothing here")))
    assert res["status"] == "suspect"


def test_no_video_stream_is_suspect(mp4):
    res = cheap(mp4, make_run(ffprobe=(0, probe_out(video=None))))
    assert res["status"] == "suspect"
    assert any("no video" in r for r in res["reasons"])


def test_tail_decode_errors_are_suspect(mp4):
    res = cheap(mp4, make_run(tail=(0, "err one\nerr two\nerr three")))
    assert res["status"] == "suspect"
    assert any("tail" in r for r in res["reasons"])


def test_tail_benign_lines_are_ignored(mp4):
    res = cheap(mp4, make_run(tail=(0, "[mp3] Header missing\n" * 5)))
    assert res["status"] == "ok"


def test_tail_nonzero_exit_is_suspect(mp4):
    assert cheap(mp4, make_run(tail=(1, "")))["status"] == "suspect"


def test_duration_mismatch_vs_stash_is_suspect(mp4):
    res = cheap(mp4, make_run(ffprobe=(0, probe_out(dur=600, video=600, audio=600))), stash=1200)
    assert res["status"] == "suspect"
    assert any("stash" in r.lower() for r in res["reasons"])


def test_zero_stash_duration_skips_the_comparison(mp4):
    assert cheap(mp4, make_run(), stash=0)["status"] == "ok"


def test_small_av_gap_is_warn(mp4):
    res = cheap(mp4, make_run(ffprobe=(0, probe_out(video=600, audio=604))))
    assert res["status"] == "warn"
    assert not probe.is_broken(res)


def test_big_av_gap_is_suspect(mp4):
    res = cheap(mp4, make_run(ffprobe=(0, probe_out(video=600, audio=700, dur=700))), stash=700)
    assert res["status"] == "suspect"


def test_probe_warnings_warn(mp4):
    res = cheap(mp4, make_run(ffprobe=(0, probe_out(prefix="[mov] odd atom\n"))))
    assert res["status"] == "warn"


def test_format_tags_are_returned_lowercased(mp4):
    run = make_run(ffprobe=(0, probe_out(tags={"TITLE": "x", "comment": "y"})))
    assert cheap(mp4, run)["tags"] == {"title": "x", "comment": "y"}


def test_hard_index_flag_is_suspect(tmp_path):
    p = tmp_path / "cut.mp4"
    # an ftyp box and an mdat box but no moov
    p.write_bytes(b"\x00\x00\x00\x10ftypisom\x00\x00\x00\x00" + b"\x00\x00\x00\x10mdat" + b"\x00" * 8)
    res = cheap(str(p), make_run())
    assert res["status"] == "suspect"
    assert res["index"]["flags"] == ["mp4_no_moov"]


def test_tiered_runs_deep_only_for_suspects(mp4):
    run = make_run()
    res = tiered(mp4, run)
    assert res["status"] == "ok" and res["tier"] == "cheap" and res["deep"] is None
    assert seeks(run) == []

    run = make_run(tail=(1, "boom"))
    res = tiered(mp4, run)
    assert res["tier"] == "deep"
    assert len(seeks(run)) == 24
    # the deep check came back clean, so it is final; the cheap reasons stay under "cheap"
    assert res["status"] == "ok"
    assert res["deep"]["status"] == "ok"
    assert res["cheap"]["status"] == "suspect" and res["cheap"]["reasons"]


def test_tiered_deep_suspect_is_final(mp4):
    run = make_run(tail=(1, "boom"), window=(1, "decode error: corrupt"))
    res = tiered(mp4, run)
    assert res["status"] == "suspect" and res["tier"] == "deep"
    assert res["reasons"] == res["deep"]["reasons"]
    assert probe.is_broken(res)


def test_tiered_missing_is_final(tmp_path):
    run = make_run()
    res = tiered(str(tmp_path / "nope.mp4"), run)
    assert res["status"] == "missing" and res["tier"] == "cheap"
    assert run.calls == []


def test_tiered_reuses_known_results(mp4):
    run = make_run()
    known_cheap = {"status": "suspect", "reasons": ["x"], "index": {"kind": "mp4", "flags": []},
                   "tags": {}, "probe": {}, "version": probe.CHEAP_VERSION}
    known_deep = {"status": "suspect", "reasons": ["y"], "params": {"points": 24}}
    res = tiered(mp4, run, cheap=known_cheap, deep=known_deep)
    assert run.calls == []
    assert res["status"] == "suspect" and res["reasons"] == ["y"] and res["tier"] == "deep"

    ok_cheap = dict(known_cheap, status="ok", reasons=[])
    res = tiered(mp4, run, cheap=ok_cheap)
    assert run.calls == [] and res["status"] == "ok" and res["tier"] == "cheap"



def test_na_durations_do_not_raise(mp4):
    meta = json.dumps({"format": {"duration": "N/A", "format_name": "mov"},
                       "streams": [{"codec_type": "video", "duration": "N/A"}, {"codec_type": "audio"}]})
    res = cheap(mp4, make_run(ffprobe=(0, meta)), stash=0)
    assert res["status"] in ("ok", "warn", "suspect")

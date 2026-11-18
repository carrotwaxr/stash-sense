"""Container index check (#133)."""

import shutil
import struct
import subprocess

import pytest

from quality.container import HARD_INDEX_FLAGS, index_status

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _build(path, *extra, audio="aac"):
    cmd = [
        "ffmpeg", "-nostdin", "-v", "error", "-y",
        "-f", "lavfi", "-i", "testsrc=d=2:s=160x120",
        "-f", "lavfi", "-i", "sine=d=2",
        "-shortest", "-c:v", "mpeg4", "-c:a", audio,
        *extra, str(path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    return str(path)


def test_mp4_faststart_has_no_flags(tmp_path):
    p = _build(tmp_path / "a.mp4", "-movflags", "+faststart")
    assert index_status(p) == {"kind": "mp4", "flags": []}


def test_mp4_moov_last(tmp_path):
    p = _build(tmp_path / "a.mp4")
    assert index_status(p) == {"kind": "mp4", "flags": ["mp4_moov_last"]}


def test_truncated_mp4_has_no_moov(tmp_path):
    p = _build(tmp_path / "a.mp4")
    data = (tmp_path / "a.mp4").read_bytes()
    (tmp_path / "cut.mp4").write_bytes(data[: len(data) // 2])
    assert index_status(str(tmp_path / "cut.mp4")) == {"kind": "mp4", "flags": ["mp4_no_moov"]}
    assert "mp4_no_moov" in HARD_INDEX_FLAGS
    assert "mp4_moov_last" not in HARD_INDEX_FLAGS


def test_mkv_with_cues(tmp_path):
    p = _build(tmp_path / "a.mkv")
    assert index_status(p) == {"kind": "mkv", "flags": []}


def test_avi_ok_and_without_index(tmp_path):
    p = _build(tmp_path / "a.avi", audio="mp3")
    assert index_status(p) == {"kind": "avi", "flags": []}
    body = b"AVI " + b"LIST" + struct.pack("<I", 4) + b"hdrl"
    raw = b"RIFF" + struct.pack("<I", len(body)) + body
    q = tmp_path / "noidx.avi"
    q.write_bytes(raw)
    assert index_status(str(q)) == {"kind": "avi", "flags": ["avi_no_index"]}


def test_unknown_kind(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world, not a video\n" * 10)
    assert index_status(str(p)) == {"kind": "other", "flags": []}


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        index_status(str(tmp_path / "nope.mp4"))

import pytest

from curation_settings import CurationSettings
from quality.rip import file_clues, format_flag, rip_score


def _by_signal(clues):
    return {c["signal"]: c for c in clues}


def test_tube_domain_in_tags():
    clues = _by_signal(
        file_clues("a.mp4", {"comment": "Downloaded from SpankBang.com"})
    )
    assert clues["tag_tube"]["weight"] == 0.9
    assert "tag_url" not in clues


def test_other_url_in_tags_is_weak():
    clues = _by_signal(file_clues("a.mp4", {"comment": "www.somestudio.com"}))
    assert clues["tag_url"]["weight"] == 0.3
    assert "tag_tube" not in clues


def test_tube_name():
    clues = _by_signal(file_clues("scene [xvideos.com].mp4", {}))
    assert clues["tube_name"]["weight"] == 0.6
    assert file_clues("scene_1080p.mp4", {}) == []


def test_text_is_trimmed():
    clues = _by_signal(file_clues("a.mp4", {"comment": "xvideos " + "x" * 300}))
    assert len(clues["tag_tube"]["text"]) == 120


def test_rip_score_combines():
    assert rip_score([0.6, 0.7]) == pytest.approx(0.88)
    assert rip_score([]) == 0


def test_format_flag_needs_lists():
    assert (
        format_flag({"video_codec": "wmv3", "format": "asf"}, CurationSettings())
        is None
    )


def test_format_flag_codec_and_container():
    s = CurationSettings(disfavored_video_codecs=("wmv3",))
    flag = format_flag({"video_codec": "wmv3", "format": "asf"}, s)
    assert flag["codec"] == "wmv3"
    assert flag["container"] == "asf"
    assert flag["reason"]
    assert format_flag({"video_codec": "h264", "format": "mp4"}, s) is None


def test_format_flag_container_only():
    s = CurationSettings(disfavored_containers=("asf",))
    flag = format_flag({"video_codec": "h264", "format": "asf"}, s)
    assert flag["container"] == "asf"
    assert "asf" in flag["reason"]

"""Pairwise keeper rules (#130): one test per rule, each checks the winner and that the reason names the rule."""

import itertools

from duplicates.keeper import FileFeatures, better, is_upscale, suggest

CODEC_NAMES = ("h264", "hevc", "av1", "vp9", "mpeg4", "wmv", "vc1", "mpeg2")


def mk(key, **kw):
    base = dict(codec="h264", width=1920, height=1080, bit_rate=8_000_000, fps=30.0,
                duration=1200.0, size=1_000_000_000, ext="mp4")
    base.update(kw)
    return FileFeatures(key=key, **base)


def test_original_beats_named_upscale():
    up = mk("up", codec="av1", width=3840, height=2160, bit_rate=20_000_000, upscale_name=True)
    orig = mk("orig", codec="h264", width=1920, height=1080)
    assert better(up, orig)[0] == "b"
    who, why = better(orig, up)
    assert who == "a"
    assert "upscale" in why and "original" in why


def test_upscale_detected_by_file():
    h264 = mk("h", codec="h264", width=1920, height=1080, size=1_000_000_000)
    av1_big = mk("x", codec="av1", width=3840, height=2160, size=600_000_000)
    assert is_upscale(av1_big, [h264])
    assert not is_upscale(h264, [av1_big])
    # same pixels, smaller file: a space re-encode
    av1_same = mk("y", codec="av1", size=400_000_000)
    assert not is_upscale(av1_same, [h264])
    # a studio 4K h264 is never flagged by pixels alone
    studio = mk("s", codec="h264", width=3840, height=2160)
    assert not is_upscale(studio, [h264])
    # an av1 with no non-av1 sibling has nothing to be an upscale of
    assert not is_upscale(av1_big, [mk("z", codec="av1")])
    # flags and encoder tag
    assert is_upscale(mk("n", upscale_name=True), [h264])
    assert is_upscale(mk("p", upscale_path=True), [h264])
    assert is_upscale(mk("e", encoder="Lavf; Topaz Video AI 3.4"), [h264])
    assert not is_upscale(mk("e2", encoder="Lavf58.76"), [h264])
    # the file itself in the sibling list is ignored
    assert not is_upscale(h264, [h264, av1_same])


def test_upscale_by_file_decides_a_pair():
    h264 = mk("h")
    av1_big = mk("x", codec="av1", width=3840, height=2160, bit_rate=30_000_000, size=900_000_000)
    who, why = better(av1_big, h264)
    assert who == "b" and "upscale" in why


def test_effective_detail_beats_resolution():
    starved = mk("s", width=1920, height=1080, bit_rate=1_500_000)
    solid = mk("p", width=1280, height=720, bit_rate=4_000_000)
    who, why = better(starved, solid)
    assert who == "b"
    assert "effective detail" in why
    assert "1280x720 at 4.0 Mbps" in why and "1920x1080 at 1.5 Mbps" in why
    assert better(solid, starved)[0] == "a"


def test_bpp_then_fps_then_container():
    # bits per pixel: the same effective detail, but one file spends far fewer bits per pixel on more pixels
    lean = mk("l", width=1280, height=720, bit_rate=1_100_000)
    rich = mk("r", width=854, height=480, bit_rate=1_000_000)
    who, why = better(lean, rich)
    assert who == "b" and "bits per pixel" in why

    # fps, with bpp inside its band; fps beats a longer duration and a better container
    hi = mk("hi", width=1280, height=720, bit_rate=7_000_000, fps=60.0, duration=1000.0, ext="avi")
    lo = mk("lo", width=1280, height=720, bit_rate=4_000_000, fps=30.0, duration=1200.0)
    who, why = better(lo, hi)
    assert who == "b" and "frame rate" in why

    # duration, with fps equal; beats the container
    short = mk("sh", duration=1000.0, ext="mp4")
    long_ = mk("lg", duration=1200.0, ext="avi")
    who, why = better(short, long_)
    assert who == "b" and "longer" in why
    # not for a contained file, and not within 60 s
    assert better(short, long_, kind="contained")[0] == "a"
    near = mk("nr", duration=1030.0, ext="avi")
    assert better(short, near)[0] == "a"

    # container
    mp4 = mk("m")
    wmv = mk("w", ext="wmv")
    who, why = better(wmv, mp4)
    assert who == "b" and "container" in why

    # nothing left
    assert better(mk("t1"), mk("t2"))[0] == "tie"


def test_disfavored_codec_loses_near_tie_only():
    hevc = mk("hv", codec="hevc", bit_rate=4_000_000)
    h264 = mk("h2", codec="h264", bit_rate=5_000_000)
    who, why = better(hevc, h264, disfavored_codecs=frozenset({"hevc"}))
    assert who == "b" and "hevc" in why
    assert better(h264, hevc, disfavored_codecs=frozenset({"hevc"}))[0] == "a"
    # without the setting the near tie falls to bits per pixel (here h264 has more bits)
    assert "hevc" not in better(hevc, h264)[1]
    # over the 1.15 band on effective detail, hevc still wins
    thin = mk("th", codec="h264", bit_rate=2_500_000)
    who, why = better(hevc, thin, disfavored_codecs=frozenset({"hevc"}))
    assert who == "a" and "effective detail" in why
    # a disfavored container is the same kind of near-tie loss
    avi = mk("av", ext="avi")
    mkv = mk("mk", ext="mkv")
    who, why = better(avi, mkv, disfavored_containers=frozenset({"avi"}))
    assert who == "b" and "avi" in why


def test_no_codec_mention_when_lists_empty():
    pairs = [
        (mk("a", codec="hevc", bit_rate=4_000_000), mk("b", bit_rate=5_000_000)),
        (mk("a", codec="av1", upscale_name=True), mk("b")),
        (mk("a", codec="hevc", bit_rate=1_000_000), mk("b", codec="mpeg4", bit_rate=1_000_000)),
        (mk("a", codec="vp9", bit_rate=3_000_000, fps=60.0), mk("b", codec="h264", bit_rate=3_000_000)),
        (mk("a", codec="h264", rip=True), mk("b", codec="wmv2", width=640, height=480)),
        (mk("a", codec="hevc", duration=900.0), mk("b", codec="h264", duration=1200.0)),
        (mk("a", codec="hevc", ext="avi"), mk("b", codec="h264", ext="mkv")),
        (mk("a", codec="hevc"), mk("b", codec="h264")),
        (mk("a", codec="av1", width=3840, height=2160), mk("b", codec="h264")),
    ]
    for a, b in pairs:
        for x, y in ((a, b), (b, a)):
            why = better(x, y)[1].lower()
            assert why
            assert not any(c in why for c in CODEC_NAMES), why


def test_rip_loses():
    rip = mk("r", rip=True)  # 1080p at 8 Mbps
    clean = mk("c", width=640, height=480, bit_rate=1_000_000, ext="avi")
    who, why = better(rip, clean)
    assert who == "b" and "rip" in why
    assert better(clean, rip)[0] == "a"
    # both rips: the rule does not apply
    assert better(rip, mk("r2", rip=True, width=640, height=480, bit_rate=1_000_000))[0] == "a"


def test_from_stash_file():
    f = FileFeatures.from_stash_file({
        "video_codec": "av1", "width": 3840, "height": 2160, "bit_rate": 9_000_000, "frame_rate": 29.97,
        "duration": 600.5, "size": 123, "basename": "Scene_upscaled.MKV", "path": "/data/x/Scene_upscaled.MKV",
        "fingerprints": [{"type": "phash", "value": "p"}, {"type": "oshash", "value": "abc"}],
    })
    assert (f.key, f.codec, f.ext, f.upscale_name, f.upscale_path) == ("abc", "av1", "mkv", True, False)
    g = FileFeatures.from_stash_file({"path": "/data/upscaled/a/b.mp4", "video_codec": None, "bit_rate": None,
                                      "fingerprints": []})
    assert g.upscale_path and not g.upscale_name and g.bit_rate == 0 and g.key == ""
    h = FileFeatures.from_stash_file({"path": "/data/upscaled/originals/b.mp4"})
    assert not h.upscale_path


# ---- set-level suggestion ----

def _720(key, **kw):
    return mk(key, width=1280, height=720, bit_rate=3_000_000, **kw)


def _480(key, **kw):
    return mk(key, width=854, height=480, bit_rate=1_000_000, **kw)


def test_set_keeps_one_file():
    s = suggest([_480("c"), mk("a"), _720("b")])
    assert s.keep == ["a"]
    assert s.delete == ["b", "c"]
    assert s.reason and not s.held and s.flags == []


def test_upscale_set_keeps_two():
    up = mk("up", codec="av1", width=3840, height=2160, bit_rate=20_000_000, upscale_name=True)
    s = suggest([up, _720("o2"), mk("o1")])
    assert set(s.keep) == {"up", "o1"}
    assert s.delete == ["o2"]
    assert "upscale_pair" in s.flags


def test_longer_but_worse_keeps_both():
    s = suggest([mk("best", duration=900.0), _720("long", duration=1200.0)])
    assert set(s.keep) == {"best", "long"}
    assert s.keep[0] == "best"
    assert s.delete == []
    assert "supercut_candidate" in s.flags
    # a short gap is not a supercut
    s = suggest([mk("best", duration=1180.0), _720("long", duration=1200.0)])
    assert s.keep == ["best"] and "supercut_candidate" not in s.flags


def test_broken_keeper_falls_through():
    files = [mk("best"), _720("mid"), _480("low")]
    s = suggest(files, health={"best": "suspect"})
    assert s.keep == ["mid"]
    assert set(s.delete) == {"best", "low"}
    assert "health check" in s.reason
    s = suggest(files, health={"best": "missing", "mid": "ok", "low": "warn"})
    assert s.keep == ["mid"]
    # warn still counts as healthy
    s = suggest(files, health={"best": "warn"})
    assert s.keep == ["best"] and "health check" not in s.reason
    # nothing healthy: hold everything
    s = suggest(files, health={"best": "suspect", "mid": "missing", "low": "suspect"})
    assert s.held and s.delete == []
    assert set(s.keep) == {"best", "mid", "low"}
    assert "held_unhealthy" in s.flags


def test_contained_recut():
    # a re-cut: the long file is under 1.5x the short one
    long_ = mk("long", duration=1000.0)
    short = mk("short", duration=800.0)
    s = suggest([_720("short", duration=800.0), long_], kind="contained")
    assert s.keep == ["long"] and s.delete == ["short"] and "split_first" not in s.flags
    assert "longer cut" in s.reason
    # equal quality is not strictly better: both kept
    equal = mk("long", duration=1000.0, size=3_000_000_000)
    s = suggest([short, equal], kind="contained")
    assert set(s.keep) == {"long", "short"} and s.delete == [] and "split_first" not in s.flags
    # a better short file keeps both
    long_worse = _720("long", duration=1000.0)
    s = suggest([short, long_worse], kind="contained")
    assert set(s.keep) == {"long", "short"} and s.delete == [] and "split_first" not in s.flags
    # the long file is never deleted, even when it failed its health check
    s = suggest([short, long_worse], kind="contained", health={"long": "suspect"})
    assert "long" in s.keep


def test_contained_split_first():
    short = mk("short", duration=600.0)
    for long_ in (mk("long", duration=1200.0), _720("long", duration=1200.0),
                  mk("long", duration=1200.0, width=3840, height=2160, bit_rate=30_000_000)):
        s = suggest([short, long_], kind="contained")
        assert set(s.keep) == {"long", "short"} and s.delete == []
        assert s.flags == ["split_first"]
        assert "split the longer file first" in s.reason.lower()
    # just under 1.5x is a re-cut
    s = suggest([mk("short", duration=800.0), mk("long", duration=1199.0)], kind="contained")
    assert "split_first" not in s.flags
    # exactly 1.5x splits first
    s = suggest([mk("short", duration=800.0), mk("long", duration=1200.0)], kind="contained")
    assert s.flags == ["split_first"]


def test_suggestion_is_order_independent():
    files = [mk("a"), mk("b"), _720("c"), _480("d")]  # a and b tie
    results = {(tuple(s.keep), tuple(s.delete), s.reason, tuple(s.flags))
               for s in (suggest(list(p)) for p in itertools.permutations(files))}
    assert len(results) == 1
    assert results.pop()[0] == ("a",)


def test_tie_flagged():
    s = suggest([mk("t2"), mk("t1")])
    assert s.keep == ["t1"] and s.delete == ["t2"]
    assert "tie" in s.flags


def test_hevc_not_upscale_by_size():
    small = mk("s", codec="h264", width=720, height=406, size=300_000_000)
    hevc = mk("h", codec="hevc", width=1920, height=1080, size=900_000_000)
    assert not is_upscale(hevc, [small])
    vp9 = mk("v", codec="vp9", width=1920, height=1080, size=900_000_000)
    assert not is_upscale(vp9, [small])
    half = mk("m", codec="h264", width=960, height=540, size=300_000_000)
    av1 = mk("a", codec="av1", width=1920, height=1080, size=900_000_000)
    assert is_upscale(av1, [half])


def test_bpp_capped():
    # far above the cap: 12.2 Mbps buys no visible detail over 4.4 Mbps, so the smaller file wins
    fat = mk("fat", bit_rate=12_200_000, size=1_800_000_000)
    lean = mk("lean", bit_rate=4_400_000, size=650_000_000)
    who, why = better(fat, lean)
    assert who == "b" and "smaller file" in why
    assert better(lean, fat)[0] == "a"
    # below the cap the rule still applies
    hi = mk("hi", width=640, height=480, bit_rate=1_000_000, size=150_000_000)
    lo = mk("lo", width=640, height=480, bit_rate=500_000, size=75_000_000)
    assert better(lo, hi)[0] == "b"
    assert better(hi, lo)[0] == "a"


def test_smaller_file_breaks_ties():
    big = mk("big", size=1_100_000_000)
    small = mk("small", size=1_000_000_000)
    who, why = better(big, small)
    assert who == "b" and "smaller file" in why
    assert better(small, big)[0] == "a"
    # within 5% is still a tie, and so is the same size
    assert better(mk("x", size=1_040_000_000), small)[0] == "tie"
    assert better(mk("y"), mk("z"))[0] == "tie"
    # an earlier rule outranks size: a better container wins over a smaller file
    assert better(mk("a", size=500_000_000, ext="wmv"), mk("b", size=900_000_000))[0] == "b"

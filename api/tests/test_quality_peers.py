"""Below-the-studio's-release flag from studio+year peer percentiles (#133)."""

from duplicates import keeper
from quality.peers import PeerFile, below_studio


def pf(sid, eff, studio="s1", year=2015):
    return PeerFile(scene_id=sid, studio_id=studio, year=year, eff=float(eff))


def peers(n, studio="s1", year=2015, base=2_000_000, prefix="p"):
    # a small spread around `base`
    return [pf(f"{prefix}{studio}{year}_{i}", base * (0.9 + 0.02 * i), studio, year) for i in range(n)]


def test_effective_detail_matches_private():
    f = keeper.FileFeatures(key="k", codec="h264", width=1920, height=1080, bit_rate=8_000_000, fps=30.0,
                            duration=600.0, size=600_000_000, ext="mp4")
    assert keeper.effective_detail(f) == keeper._detail(f)


def test_flags_low_copy_among_good_peers():
    files = peers(10) + [pf("low", 300_000)]
    out = below_studio(files)
    assert set(out) == {"low"}
    ev = out["low"]
    assert ev["group"] == "studio+year"
    assert ev["peers"] == 10
    assert ev["score"] > 0.8


def test_no_flag_inside_normal_spread():
    base = peers(10)
    effs = sorted(p.eff for p in base)
    mid = effs[2] + 0.5 * (effs[3] - effs[2])  # about the 30th percentile
    assert below_studio(base + [pf("mid", mid)]) == {}


def test_falls_back_to_year_window_then_studio():
    files = peers(3, year=2015) + peers(3, year=2014) + peers(3, year=2017) + [pf("low", 300_000)]
    out = below_studio(files)
    assert out["low"]["group"] == "studio+years±2"
    assert out["low"]["peers"] == 9

    files = peers(9, year=2005) + [pf("low", 300_000)]
    out = below_studio(files)
    assert out["low"]["group"] == "studio"
    assert out["low"]["peers"] == 9


def test_too_few_peers_never_flags():
    assert below_studio(peers(7) + [pf("low", 300_000)]) == {}


def test_no_studio_never_flags():
    files = peers(10) + [PeerFile("low", None, 2015, 300_000.0)]
    assert below_studio(files) == {}


def test_file_is_not_its_own_peer():
    # 7 peers plus the file itself is 8 files in the group, but the file does not count as its own peer
    assert below_studio(peers(7) + [pf("low", 300_000)]) == {}
    # and the stats come from the other files only
    out = below_studio(peers(8) + [pf("low", 300_000)])
    assert out["low"]["peers"] == 8


def test_no_year_uses_studio_only():
    files = peers(9, year=2015) + [PeerFile("low", "s1", None, 300_000.0)]
    out = below_studio(files)
    assert out["low"]["group"] == "studio"


def test_percentile_setting_changes_threshold():
    base = peers(10)
    import statistics
    med = statistics.median(p.eff for p in base)
    f = pf("near", med * 0.98)
    assert below_studio(base + [f]) == {}
    out = below_studio(base + [f], percentile=50, ratio=1.0)
    assert "near" in out

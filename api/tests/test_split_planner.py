"""Series profiles and the full split planner (#131): series keys, profile building, confidence rules, static
pairs, coverage edges and menu tails, on synthetic signals. All titles here are invented."""

import pytest

from splits import detect, planner, profiles

FPS = 6
MIN = 60.0


def _frames(total, level=100, blacks=(), snow=(), ydif=8.0):
    """6 fps frames at `level`. blacks=[(start, length, y)] overwrite stretches; snow=[start] puts 5 frames of
    analog snow (SAT 1, YDIF 40) at each start. YDIF 8 keeps the picture "moving" so no menu tail is found."""
    snow_idx = set()
    for s in snow:
        first = round(s * FPS)
        snow_idx.update(range(first, first + 5))
    out = []
    for i in range(int(total * FPS)):
        t = round(i / FPS, 3)
        if i in snow_idx:
            out.append([t, 120, 1.0, 40.0, 0.0])
            continue
        y = level
        for start, length, by in blacks:
            if start <= t < start + length:
                y = by
        out.append([t, y, 20.0, ydif, 0.0])
    return out


def _d(fr, sil=()):
    return {"fr": fr, "au": [], "sil": [list(s) for s in sil]}


def _file(total, cuts, black_len=1.0, sil_len=1.0, level=100, y=5):
    """A file with a black of `black_len` s (and silence of `sil_len` s, if any) starting at each cut time."""
    fr = _frames(total, level=level, blacks=[(t, black_len, y) for t in cuts])
    sil = [(t, t + sil_len) for t in cuts] if sil_len else []
    return _d(fr, sil)


# --- series key ---

# Expected keys are what the October `_series_key` returns for the same input (run against the original).
ORIGINAL_KEYS = {
    "Harbor Lights 4": "harbor lights",
    "The Foo, Vol. 3 - Disc 2 (2004)": "foo,",
    "Harbor Lights #12": "harbor lights",
    "Copper Kettle Vol. 7": "copper kettle",
    "Moonlit Garden, The 3": "moonlit garden",
    "The Glass Orchard 2 (1999)": "glass orchard",
    "Velvet Ladder - Part 2": "velvet ladder",
    "Velvet Ladder Part 3 1080p": "velvet ladder",
    "Quiet Meadow 5: Second Wind": "quiet meadow",
    "Silver Canyon Volume 11 - Disc 1": "silver canyon",
    "Night Ferry 2-3": "night ferry",
    "Paper Lanterns 720p.mp4": "paper lanterns",
    "Amber Tides (Upscaled)": "amber tides",
    "Rusty Gate DVD 2": "rusty gate",
    "Blue Orchard 14 - The Return": "blue orchard",
    "Hollow Pines: Disc 2": "hollow pines",
    "Starlight Express": "starlight express",
    "Crimson Hall (2D Version) 6": "crimson hall",
    "The Lantern Coast, Vol. 9 (2011).mkv": "lantern coast,",
    "Iron Willow #3 - Night Shift": "iron willow",
    "Foo Bar 2 Disc 1": "foo bar",
}


def test_series_key():
    assert profiles.series_key("Harbor Lights 4") == "harbor lights"
    assert profiles.series_key("The Glass Orchard 2 (1999)") == "glass orchard"
    assert profiles.series_key("Silver Canyon Volume 11 - Disc 1") == "silver canyon"
    assert profiles.series_key("Foo Bar 2 Disc 1", aliases=[("^foo.*", "foo series")]) == "foo series"
    assert profiles.series_key("Harbor Lights 4", aliases=[("^foo.*", "foo series")]) == "harbor lights"
    assert profiles.series_key(None) == ""


@pytest.mark.parametrize("title,want", sorted(ORIGINAL_KEYS.items()))
def test_series_key_matches_october_code(title, want):
    assert profiles.series_key(title) == want


# --- profiles ---

SERIES = [[20, 25, 22, 23], [24, 21, 26, 19], [22, 22, 23, 23]]  # scene minutes of three titles


def _title(key, mins):
    cuts = []
    t = 0.0
    for m in mins[:-1]:
        t += m * MIN
        cuts.append(t)
    total = sum(mins) * MIN
    d = _file(total, cuts)
    return {"key": key, "total": total, "cuts": cuts, "feat": [detect.features_at(d, c) for c in cuts]}


@pytest.fixture(scope="module")
def profile():
    return profiles.build_profile([_title(f"k{i}", m) for i, m in enumerate(SERIES)])


def test_profile_from_titles(profile):
    assert profile["transition"] == "black"
    assert profile["n_titles"] == 3 and profile["n_boundaries"] == 9
    assert profile["scene_count"] == {"min": 4, "max": 4, "med": 4}
    assert profile["scene_len_min"]["min"] == 19.0 and profile["scene_len_min"]["max"] == 26.0
    assert profile["summary"]["black_rate"] == 1.0
    assert profile["detect"]["kind"] == "black"

    ok, why = profiles.judge([{"t": t * MIN} for t in (22, 45, 68)], profile, 90 * MIN)
    assert ok and why == []
    ok, why = profiles.judge([{"t": t * MIN} for t in (20, 40, 60)], profile, 112 * MIN)  # last scene 52 min = 2x max
    assert not ok and "scene lengths" in why[0]


def test_summarize_counts_blacks_and_silences():
    rows = [
        {"blk_len": 1.0, "dip_ratio": 0.1, "sil_len": 0.5, "scd": 3},
        {"blk_len": 0.0, "dip_ratio": None, "sil_len": 0.0, "scd": 1},
    ]
    s = profiles.summarize(rows)
    assert s["n_boundaries"] == 2 and s["black_rate"] == 0.5 and s["sil_rate"] == 0.5
    assert s["black_len"]["min"] == 1.0 and s["dip_ratio_med"] == 0.1 and s["scd_med"] == 2


# --- confidence ---


def test_confidence_profile_judged_is_high(profile):  # (a)
    total = 90 * MIN
    plan = planner.plan_file(_file(total, [22 * MIN, 45 * MIN, 68 * MIN]), total, profile=profile)
    assert [c["t"] for c in plan.cuts] == [22 * MIN, 45 * MIN, 68 * MIN]
    assert plan.profile_used
    assert plan.confidence == "high"
    assert plan.cuts[0]["kind"] == "black"
    assert plan.cuts[0]["evidence"].startswith("black 1.0 s (min Y 5, from ~100), silence 1.0 s")


def test_confidence_unmistakable_without_profile_is_medium():  # (b)
    total = 60 * MIN
    plan = planner.plan_file(_file(total, [20 * MIN, 40 * MIN], black_len=3.0, sil_len=1.5), total)
    assert len(plan.cuts) == 2 and not plan.profile_used
    # No profile and no coverage: unmistakable cuts are not enough for high (regression set, 2026-11-18).
    assert plan.confidence == "medium"


def test_mixed_series_not_high(profile):
    mixed = {**profile, "transition": "mixed", "detect": {**profile["detect"], "kind": "mixed"}}
    total = 90 * MIN
    d = _file(total, [22 * MIN, 45 * MIN, 68 * MIN])
    ok, _ = profiles.judge([{"t": t * MIN} for t in (22, 45, 68)], mixed, total)
    assert ok
    plan = planner.plan_file(d, total, profile=mixed)
    assert plan.profile_used and [c["t"] for c in plan.cuts] == [22 * MIN, 45 * MIN, 68 * MIN]
    assert plan.confidence == "medium"
    assert any("mixes fades and hard cuts" in r for r in plan.reasons)


def test_confidence_long_scene_is_not_high():  # (c)
    total = 100 * MIN
    plan = planner.plan_file(_file(total, [20 * MIN, 30 * MIN], black_len=3.0, sil_len=1.5), total)
    assert len(plan.cuts) == 2
    # A 70 min scene fails both the unmistakable rule and the 5-60 min medium rule (plan_new.py).
    assert plan.confidence == "low"


def test_confidence_strong_cuts_without_profile_are_medium():
    total = 60 * MIN
    plan = planner.plan_file(_file(total, [20 * MIN, 40 * MIN]), total)
    assert len(plan.cuts) == 2
    assert plan.confidence == "medium"


def test_confidence_no_cuts_is_low():  # (d)
    total = 60 * MIN
    plan = planner.plan_file(_file(total, []), total)
    assert plan.cuts == [] and plan.confidence == "low"
    assert plan.leave_whole_suggested


def test_unconfirmed_black_in_overlong_scene_drops_to_medium(profile):  # (f)
    total = 74 * MIN
    # Level 45: the 60 min black at Y 25 is a dark run (dip 0.56, no silence) but not strong.
    cuts = [22 * MIN, 44 * MIN]
    fr = _frames(total, level=45, blacks=[(t, 1.0, 5) for t in cuts] + [(60 * MIN, 1.0, 25)])
    plan = planner.plan_file(_d(fr, [(t, t + 1.0) for t in cuts]), total, profile=profile)
    assert [c["t"] for c in plan.cuts] == cuts
    assert plan.confidence == "medium"
    assert any("longer than any in the series" in r for r in plan.reasons)

    fr = _frames(total, level=45, blacks=[(t, 1.0, 5) for t in cuts])
    plan = planner.plan_file(_d(fr, [(t, t + 1.0) for t in cuts]), total, profile=profile)
    assert plan.confidence == "high"


# --- static, coverage, menu ---


def test_static_cuts_merge():
    total = 60 * MIN
    fr = _frames(total, blacks=[(1213.0, 1.0, 5)], snow=[1200.0, 1210.0, 2400.0, 2410.0])
    d = _d(fr, [(1199.0, 1201.0), (1213.0, 1214.0)])  # the 2400 s pair has no silence: weak
    plan = planner.plan_file(d, total)
    assert [(c["t"], c["kind"]) for c in plan.cuts] == [(1200.0, "static")]
    assert plan.cuts[0]["evidence"].startswith("static pair")
    assert any(c["t"] == 2400.0 for c in plan.leftovers)


def test_coverage_cuts():
    total = 60 * MIN
    # A dark-scene stretch (Y 30 in Y 45, not strong) starts at 597 s; nothing near 1800 s.
    d = _d(_frames(total, level=45, blacks=[(597.0, 2.0, 30)]))
    plan = planner.plan_file(d, total, coverage=[(600.0, 1800.0)])
    assert [(c["t"], c["kind"]) for c in plan.cuts] == [(597.0, "coverage"), (1800.0, "coverage")]
    assert plan.confidence == "medium"


def test_coverage_confirmed_by_detector_is_high():
    total = 60 * MIN
    d = _file(total, [1795.0])
    plan = planner.plan_file(d, total, coverage=[(0.0, 1800.0), (1800.0, total)])
    assert [(c["t"], c["kind"]) for c in plan.cuts] == [(1795.0, "coverage")]
    assert plan.confidence == "high"


def test_menu_tail_trims():
    total_content = 1800.0
    fr = _frames(total_content, blacks=[(900.0, 1.0, 5)])
    for i in range(int(total_content * FPS), int((total_content + 243) * FPS)):
        t = round(i / FPS, 3)
        if t < total_content + 3:
            y = 5
        elif 1900.0 <= t < 1901.0:
            y = 5  # a black inside the menu loop
        elif 1810.0 <= t < 1810.2:
            y = 70  # a highlighted frame
        else:
            y = 40
        fr.append([t, y, 20.0, 0.5, 0.0])
    total = fr[-1][0]
    d = _d(fr, [(900.0, 901.0), (1900.0, 1901.0)])
    plan = planner.plan_file(d, total)
    assert plan.keep_until == pytest.approx(1800.0, abs=0.2)
    assert [c["t"] for c in plan.cuts] == [900.0]
    assert all(c["t"] < plan.keep_until for c in plan.leftovers)

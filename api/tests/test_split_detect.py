"""The dark-run split planner (#131): the October v3 rules, checked on synthetic frames and on the regression set."""

import json
from pathlib import Path

import pytest

from splits import detect, signals as sg

FIXTURE = Path(__file__).parent / "fixtures" / "split_regression"
FPS = 6


def _records():
    return [
        json.loads(line)
        for line in (FIXTURE / "index.jsonl").read_text().splitlines()
        if line
    ]


def _frames(total, level=100, blacks=(), dips=()):
    """6 fps frames at `level`; blacks=[(start, length, y)] and dips=[(start, length, y)] overwrite stretches."""
    out = []
    for i in range(int(total * FPS)):
        t = i / FPS
        y = level
        for start, length, by in (*blacks, *dips):
            if start <= t < start + length:
                y = by
        out.append([round(t, 3), y, 20.0, 5.0, 0.0])
    return out


def _d(fr, sil=()):
    return {"fr": fr, "au": [], "sil": [list(s) for s in sil]}


def _plan(d, total):
    cands = detect.refine(d, detect.candidates(d))
    return detect.plan_cuts(cands, None, total)


@pytest.mark.parametrize("rec", _records(), ids=lambda r: r["id"])
def test_plan_cuts_matches_october_code(rec):
    s = sg.decode((FIXTURE / f"{rec['id']}.npz").read_bytes())
    d = sg.to_dict(s)
    cands = detect.refine(d, detect.candidates(d))
    cuts, _ = detect.plan_cuts(cands, None, rec["duration"])
    got = [c["t"] for c in cuts]
    assert len(got) == len(rec["ref_cuts"]), (got, rec["ref_cuts"])
    for g, want in zip(got, rec["ref_cuts"]):
        assert abs(g - want) <= 0.05


def test_two_second_black_with_silence_is_strong_but_not_unmistakable():
    d = _d(_frames(600, blacks=[(300, 2, 5)]), sil=[(300, 302)])
    (c,) = detect.refine(d, detect.candidates(d))
    assert c["kind"] == "black"
    assert detect.is_strong(c)
    assert not detect.unmistakable(c)


def test_long_black_with_long_silence_is_unmistakable():
    d = _d(_frames(600, blacks=[(300, 3, 5)]), sil=[(299.5, 303)])
    (c,) = detect.refine(d, detect.candidates(d))
    assert detect.is_strong(c)
    assert detect.unmistakable(c)


def test_dark_scene_is_not_a_black():
    # 24566: a 7 s stretch at Y 30 between Y 45 stretches is a dark scene, not a transition.
    d = _d(_frames(600, level=45, blacks=[(300, 7, 30)]), sil=[(300, 307)])
    (c,) = detect.refine(d, detect.candidates(d))
    assert c["min_y"] >= detect.DARK_SCENE_MIN_Y
    assert c["dip_ratio"] >= detect.DARK_SCENE_RATIO
    assert not detect.is_strong(c)
    cuts, left = detect.plan_cuts([c], None, 600)
    assert cuts == [] and left == [c]


def test_short_black_needs_silence_and_a_deep_dip():
    fr = _frames(600, blacks=[(300, 0.3, 3)])
    both = detect.refine(
        _d(fr, sil=[(299.9, 300.5)]), detect.candidates(_d(fr, sil=[(299.9, 300.5)]))
    )
    no_sil = detect.refine(_d(fr), detect.candidates(_d(fr)))
    assert both[0]["len"] < detect.SHORT_BLACK_S
    assert detect.is_strong(both[0])
    assert not detect.is_strong(no_sil[0])
    # silence but a shallow dip (Y 60 against 100 is not deep, and not <= BLACK_Y either)
    shallow = _frames(600, blacks=[(300, 0.3, 30)], level=34)
    dd = _d(shallow, sil=[(299.9, 300.5)])
    (c,) = detect.refine(dd, detect.candidates(dd))
    assert not detect.is_strong(c)


def test_two_strong_blacks_close_together_keep_the_longer():
    d = _d(
        _frames(900, blacks=[(300, 1, 5), (360, 3, 5)]),
        sil=[(300, 301), (360, 363)],
    )
    cuts, _ = _plan(d, 900)
    assert [c["t"] for c in cuts] == [360.0]


def test_black_inside_the_intro_guard_is_dropped():
    d = _d(
        _frames(600, blacks=[(100, 3, 5), (300, 3, 5)]), sil=[(100, 103), (300, 303)]
    )
    cuts, _ = _plan(d, 600)
    assert [c["t"] for c in cuts] == [300.0]


def test_cut_time_is_the_run_start_not_the_middle():
    d = _d(_frames(600, blacks=[(300, 4, 5)]), sil=[(300, 304)])
    cuts, _ = _plan(d, 600)
    assert [c["t"] for c in cuts] == [300.0]
    assert cuts[0]["mid"] > cuts[0]["t"]


def test_black_too_close_to_the_end_is_dropped():
    d = _d(_frames(600, blacks=[(560, 3, 5)]), sil=[(560, 563)])
    cuts, _ = _plan(d, 600)
    assert cuts == []


def test_features_at_reports_black_silence_and_dip():
    d = _d(_frames(600, blacks=[(300, 3, 5)]), sil=[(300, 303)])
    f = detect.features_at(d, 301.5)
    assert f["blk_len"] == pytest.approx(3.0)
    assert f["blk_min"] == 5
    assert f["sil_len"] == 3.0
    assert f["dip_ratio"] == pytest.approx(0.05)


# --- DVD menu tails (task 4) ---

MENU_BASELINE = (
    4  # 4 of the 8 fixture records with keep_until match within 5 s on the first run
)


def _menu_file(menu_y, hard_cut):
    """300 s of moving picture, 3 s black, then 240 s of a static picture at `menu_y`."""
    fr = []
    black_start, black_end, end = 300.0, 303.0, 543.0
    for i in range(int(end * FPS)):
        t = round(i / FPS, 3)
        if t < black_start:
            fr.append([t, 100, 20.0, 5.0, 0.0])
        elif t < black_end:
            fr.append([t, 5, 0.0, 0.0, 0.0])
        else:
            y = menu_y
            if menu_y < 80 and 310 <= t < 310.2:
                y = 70  # one highlighted frame so the dark menu has a first bright frame
            scd = 60.0 if hard_cut and t < black_end + 0.7 else 0.0
            fr.append([t, y, 20.0, 0.5, scd])
    return fr


def test_menu_tail_dark():
    got = detect.menu_keep_until(_menu_file(40, hard_cut=False))
    assert got["keep_until"] == pytest.approx(300.0, abs=0.2)
    assert got["black_start"] == pytest.approx(300.0, abs=0.2)
    assert "dark" in got["kind"]


def test_menu_tail_bright():
    fr = _menu_file(120, hard_cut=True)
    first = 303.5  # the detector skips the 0.5 s after the black's last frame
    got = detect.menu_keep_until(fr)
    assert got["kind"] == "bright menu"
    assert got["menu_start"] == pytest.approx(first, abs=0.2)
    assert got["keep_until"] == pytest.approx(first - 3.0, abs=0.2)


def test_no_menu_tail():
    fr = _frames(600, blacks=[(300, 3, 5)])
    for f in fr:
        f[3] = 20.0  # moving picture throughout
    assert detect.menu_keep_until(fr) is None


def _menu_matches():
    have = [r for r in _records() if r["keep_until"] is not None]
    hit = 0
    for r in have:
        s = sg.decode((FIXTURE / f"{r['id']}.npz").read_bytes())
        got = detect.menu_keep_until(sg.to_dict(s)["fr"])
        if got and abs(got["keep_until"] - r["keep_until"]) <= 5:
            hit += 1
    return len(have), hit


def test_fixture_keep_until():
    total, hit = _menu_matches()
    print("keep_until records", total, "matches", hit)
    assert hit >= MENU_BASELINE

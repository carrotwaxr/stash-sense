"""Static (analog snow) scene transitions (#131): bursts, pairs, and the regression set's static titles."""

import json
from pathlib import Path

from splits import signals as sg, static

FIXTURE = Path(__file__).parent / "fixtures" / "split_regression"
FPS = 6


def _frames(total, burst_starts=(), burst_len=5):
    """6 fps frames: ordinary picture, with `burst_len` frames of snow (SAT 1, YDIF 40) at each start."""
    snow = set()
    for s in burst_starts:
        first = round(s * FPS)
        snow.update(range(first, first + burst_len))
    out = []
    for i in range(int(total * FPS)):
        if i in snow:
            out.append([round(i / FPS, 3), 120, 1.0, 40.0, 0.0])
        else:
            out.append([round(i / FPS, 3), 100, 20.0, 5.0, 0.0])
    return out


def _d(fr, sil=()):
    return {"fr": fr, "au": [], "sil": [list(s) for s in sil]}


def test_bursts():
    d = _d(_frames(200, burst_starts=[30, 100], burst_len=5))
    got = static.bursts(d)
    assert [b[0] for b in got] == [30.0, 100.0]
    assert got[0][1] == round(4 / FPS, 1)
    assert static.bursts(_d(_frames(200, burst_starts=[30], burst_len=4))) == []


def test_pairs_cut_at_first_burst():
    starts = [30, 48, 1200, 1218, 2400, 2419, 3500]
    sil = [(s - 0.5, s + 1.0) for s in starts]
    d = _d(_frames(3600, burst_starts=starts), sil=sil)
    cands = static.static_candidates(d, 3600.0)
    assert [c["t"] for c in cands] == [1200.0, 2400.0]
    assert all(c["kind"] == "static" and c["strong"] for c in cands)
    assert cands[0]["pair"] == [1200.0, 1218.0]
    assert cands[0]["len"] == 18.0


def test_pair_needs_silence_for_strong():
    d = _d(_frames(3600, burst_starts=[1200, 1218]), sil=[(1500, 1502)])
    (c,) = static.static_candidates(d, 3600.0)
    assert c["t"] == 1200.0
    assert c["strong"] is False
    assert c["sil"] == 0.0


def test_pair_outside_ten_seconds_of_silence_is_weak():
    d = _d(_frames(3600, burst_starts=[1200, 1218]), sil=[(1215, 1217)])
    (c,) = static.static_candidates(d, 3600.0)
    assert c["strong"] is False


def test_silence_length_is_the_longest_nearby():
    d = _d(_frames(3600, burst_starts=[1200, 1218]), sil=[(1199, 1199.5), (1201, 1203.5)])
    (c,) = static.static_candidates(d, 3600.0)
    assert c["strong"] is True
    assert c["sil"] == 2.5


def _fixture_static():
    out = []
    for line in (FIXTURE / "index.jsonl").read_text().splitlines():
        if not line:
            continue
        rec = json.loads(line)
        s = sg.decode((FIXTURE / f"{rec['id']}.npz").read_bytes())
        d = sg.to_dict(s)
        cuts = [c for c in static.static_candidates(d, rec["duration"]) if c["strong"]]
        out.append((rec, cuts))
    return out


# The fixture labels of these titles are the owner's raw scrub times: a scrub lands on the title card, a few
# seconds after the first burst of the pair. The owner's rule is to cut at the first burst, so a strong static
# cut matches when a label lies from 1 s before it to 7 s after it. A pair with no label near it is tolerated.
def test_fixture_static_titles():
    titles = [(rec, cuts) for rec, cuts in _fixture_static() if len(cuts) >= 3]
    assert titles, "the regression set holds static-transition titles"
    for rec, cuts in titles:
        matched = [c for c in cuts if any(-1.0 <= lab - c["t"] <= 7.0 for lab in rec["cuts"])]
        assert len(matched) >= 0.8 * len(cuts), (
            rec["id"],
            [c["t"] for c in cuts if c not in matched],
        )

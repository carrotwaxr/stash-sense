"""The checked-in #131 regression set: shape, size and signal coverage."""

import json
import re
from pathlib import Path

from splits import signals as sg

FIXTURE = Path(__file__).parent / "fixtures" / "split_regression"
KEYS = {
    "id",
    "series",
    "duration",
    "cuts",
    "leave_whole",
    "keep_from",
    "keep_until",
    "ref_cuts",
}


def _records():
    return [
        json.loads(line)
        for line in (FIXTURE / "index.jsonl").read_text().splitlines()
        if line
    ]


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def test_fixture_is_anonymous():
    for r in _records():
        assert set(r) == KEYS
        assert re.fullmatch(r"[0-9a-f]{12}", r["id"])
        assert re.fullmatch(r"s\d{3}", r["series"])
        for k, v in r.items():
            if k in ("id", "series"):
                continue
            if isinstance(v, list):
                assert all(_is_number(x) for x in v), k
            else:
                assert v is None or isinstance(v, bool) or _is_number(v), k


def test_fixture_size():
    recs = _records()
    assert len(recs) >= 110
    assert sum(len(r["cuts"]) for r in recs) >= 500
    assert sum(1 for r in recs if r["leave_whole"]) >= 8
    total = sum(p.stat().st_size for p in FIXTURE.iterdir())
    assert total < 20 * 1024 * 1024


def test_every_record_has_signals():
    for r in _records():
        s = sg.decode((FIXTURE / f"{r['id']}.npz").read_bytes())
        spans = sum(b - a for a, b in s.covered)
        assert spans >= 0.9 * r["duration"], r["id"]

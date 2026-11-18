"""The anonymized #130 keeper regression set must hold nothing identifying (the repo is public)."""

import json
from pathlib import Path

FIXTURE = Path(__file__).parent / "fixtures" / "dupes_keeper_regression.jsonl"
SIDE_KEYS = {
    "codec",
    "width",
    "height",
    "bit_rate",
    "fps",
    "duration",
    "size",
    "ext",
    "upscale_name",
    "upscale_path",
}
RECORD_KEYS = {"id", "kind", "pick", "a", "b", "cover_a", "cover_b"}


def _records():
    return [
        json.loads(line) for line in FIXTURE.read_text().splitlines() if line.strip()
    ]


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from _strings(v)


def test_fixture_is_anonymous():
    recs = _records()
    assert recs
    for rec in recs:
        assert RECORD_KEYS <= set(rec)
        for side in ("a", "b"):
            assert SIDE_KEYS <= set(rec[side])
        assert rec["pick"] in ("a", "b")
        for s in _strings(rec):
            assert "/" not in s and "\\" not in s and " " not in s, s


def test_fixture_size():
    assert len(_records()) >= 500

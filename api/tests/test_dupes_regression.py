"""The #130 keeper against the owner's recorded picks, and the checked-in mismatch report."""

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 614  # agree + superset on the first run (of 696 records, 16 ties)
MAX_DISAGREE = 66  # disagreements on the first run


def _report_module():
    spec = importlib.util.spec_from_file_location("dupes_keeper_report", ROOT / "scripts" / "dupes_keeper_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_keeper_matches_recorded_picks():
    mod = _report_module()
    c = Counter(r["outcome"] for r in mod.score(mod.load()))
    assert c["agree"] + c["superset"] >= BASELINE
    assert c["disagree"] <= MAX_DISAGREE


def test_report_is_current():
    mod = _report_module()
    assert mod.render(mod.score(mod.load())) == mod.REPORT.read_text()

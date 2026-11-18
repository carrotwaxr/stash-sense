"""The #130 keeper against the owner's recorded picks, and the checked-in mismatch report."""

import importlib.util
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASELINE = 526  # agree + superset (of 696 records: 16 ties, 113 deferred to splitting)
MAX_DISAGREE = 41  # disagreements (66 before the 2026-11-18 rules)


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

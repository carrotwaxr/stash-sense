"""The #131 split planner against the owner's labelled cuts, and the checked-in report."""

import importlib.util
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HIGH_FAILURES = 0  # high plans whose cuts differ from the label (after the 2026-11-18 rule change; the acceptance is 0)
HIGH_BASELINE = 1  # high plans (after the 2026-11-18 rule change; was 12 with 9 failing)


def _report_module():
    spec = importlib.util.spec_from_file_location("split_regression_report", ROOT / "scripts" / "split_regression_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@lru_cache(maxsize=1)
def _scored():
    mod = _report_module()
    return mod, mod.score(mod.load())


def test_high_plans_match_labels():
    _, scored = _scored()
    high = [r for r in scored if r["confidence"] == "high" and not r["leave_whole"]]
    failing = [r["id"] for r in high if not r["exact"]]
    assert len(high) >= HIGH_BASELINE
    assert len(failing) <= HIGH_FAILURES, failing


def test_leave_whole_titles():
    _, scored = _scored()
    assert [r["id"] for r in scored if r["leave_whole"] and r["confidence"] == "high"] == []


def test_report_is_current():
    mod, scored = _scored()
    assert mod.render(scored) == mod.REPORT.read_text()

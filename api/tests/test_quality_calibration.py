import importlib.util
from pathlib import Path

import pytest

from recommendations_db import RecommendationsDB

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "quality_calibration.py"
FLAGS = ("broken", "rip", "below_studio", "format")


def load_script():
    spec = importlib.util.spec_from_file_location("quality_calibration", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_rec(db, sid, flags, control=False):
    return db.create_recommendation(
        type="scene_quality", target_type="scene", target_id=str(sid),
        details={"file": {}, "flags": {f: {} for f in flags}, "scores": {},
                 "control": control, "quality_version": 1},
    )


def accept(db, rec_id, option):
    # The review service resolves with action "applied" and details {"option": plan.option, ...}.
    assert db.resolve_recommendation(
        rec_id, action="applied", details={"apply_log_id": rec_id, "option": option})


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "recs.db")


def build(db):
    for i in range(10):
        rid = add_rec(db, i, ["broken"])
        if i < 8:
            accept(db, rid, {"broken": "yes"})
        elif i == 8:
            accept(db, rid, {"broken": "no"})
        else:
            db.dismiss_recommendation(rid)
    for i in range(100, 120):
        rid = add_rec(db, i, [], control=True)
        opt = {f: "no" for f in FLAGS}
        if i == 100:
            opt["broken"] = "yes"
        accept(db, rid, opt)


def test_precision_and_recall_estimate(db, tmp_path):
    build(db)
    out = load_script().report(tmp_path / "recs.db", unflagged_total=1000)
    b = out["broken"]
    assert b["flagged_reviewed"] == 10
    assert b["yes"] == 8
    assert b["no"] == 2
    assert b["precision"] == pytest.approx(0.8)
    assert b["control_reviewed"] == 20
    assert b["control_yes"] == 1
    p, q, f, u = 0.8, 1 / 20, 10, 1000
    assert b["recall_estimate"] == pytest.approx(p * f / (p * f + q * u))
    # Flags nobody raised have no precision and no reviewed flagged recs.
    assert out["rip"]["flagged_reviewed"] == 0
    assert out["rip"]["precision"] is None
    assert out["rip"]["control_reviewed"] == 20
    assert out["rip"]["control_yes"] == 0


def test_option_read_from_resolution_details(db, tmp_path):
    rid = add_rec(db, 1, ["rip"])
    accept(db, rid, {"rip": "yes"})
    rec = db.get_recommendation(rid)
    assert rec.resolution_details["option"] == {"rip": "yes"}
    out = load_script().report(tmp_path / "recs.db", unflagged_total=10)
    assert out["rip"]["yes"] == 1


def test_pending_and_other_types_ignored_and_nothing_written(db, tmp_path):
    add_rec(db, 1, ["broken"])  # pending
    other = db.create_recommendation(
        type="scene_duplicates", target_type="scene", target_id="9", details={"flags": {"broken": {}}})
    accept(db, other, {"broken": "yes"})
    before = (tmp_path / "recs.db").read_bytes()
    out = load_script().report(tmp_path / "recs.db", unflagged_total=10)
    assert out["broken"]["flagged_reviewed"] == 0
    assert (tmp_path / "recs.db").read_bytes() == before

"""A split hand-off in review plans (#131): accepted splits are data for an external tool."""

import hashlib
import json
from dataclasses import asdict

import pytest

import features
from recommendations_db import RecommendationsDB
from review.executor import apply_plan
from review.plan import Plan, Preview, SplitHandoff
from review.service import ReviewService
from review.store import ReviewStore
from review.types import ReviewType
from tests.fake_stash import FakeStash


@pytest.fixture(autouse=True)
def show_unfinished(monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "stash_sense.db")


@pytest.fixture
def store(db):
    return ReviewStore(db.db_path)


class RecordingStash(FakeStash):
    """FakeStash that fails the test on any write the split hand-off must not make."""

    async def update_scene(self, *a, **kw):
        raise AssertionError("update_scene called")

    async def delete_files(self, *a, **kw):
        raise AssertionError("delete_files called")

    async def merge_scenes(self, *a, **kw):
        raise AssertionError("merge_scenes called")


@pytest.fixture
def stash():
    return RecordingStash()


def handoff(**kw) -> SplitHandoff:
    base = dict(
        scene_id="7", file_id="70", oshash="abcd", size=123456, duration=7200.0,
        ranges=[{"start": 0.0, "end": 3600.0}, {"start": 3600.0, "end": 7200.0}],
        cuts=[3600.0],
    )
    base.update(kw)
    return SplitHandoff(**base)


def plan_with(split, rec_id=1) -> Plan:
    return Plan(rec_id=rec_id, rec_type="test_split", option={"x": 1}, split=split)


def test_split_in_token():
    a = plan_with(handoff())
    b = plan_with(handoff(ranges=[{"start": 0.0, "end": 3000.0}, {"start": 3000.0, "end": 7200.0}]))
    assert a.token() != b.token()
    assert a.token() == plan_with(handoff()).token()

    plain = plan_with(None)
    payload = {
        "rec_id": 1, "rec_type": "test_split", "option": {"x": 1},
        "field_changes": [], "primary_file": None, "delete": [], "keep": [],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    assert plain.token() == hashlib.sha256(blob.encode()).hexdigest()[:24]
    assert plain.token() != a.token()


def test_split_round_trips_and_is_not_empty():
    p = plan_with(handoff())
    assert not p.is_empty
    assert plan_with(None).is_empty
    again = Plan.from_dict(p.to_dict())
    assert again.split == p.split
    assert again.token() == p.token()
    assert Plan.from_dict(plan_with(None).to_dict()).split is None


async def test_apply_split_writes_nothing(stash, store):
    plan = plan_with(handoff())
    entry = await apply_plan(plan, stash, store, target_type="scene", target_id="7")
    assert entry["status"] == "applied"
    [step] = entry["result"]["steps"]
    assert step["step"] == "split_handoff"
    assert step["scene_id"] == "7"
    assert entry["plan"]["split"] == asdict(plan.split)


class _SplitType(ReviewType):
    type_id = "test_split"
    display_name = "Split"
    description = ""

    def __init__(self, split):
        self.split = split

    async def preview(self, rec, option, stash):
        plan = Plan(rec_id=rec.id, rec_type=self.type_id, option={}, split=self.split)
        return Preview(plan=plan, evidence={}, choices=[], default_option={}, high_confidence=True)

    def label(self, rec, preview):
        return None


def _service(db, store, stash, monkeypatch, split):
    from review import types as review_types
    monkeypatch.setitem(review_types.REVIEW_TYPES, "test_split", _SplitType(split))
    return ReviewService(db, store, stash)


async def test_accept_puts_split_in_resolution(db, store, stash, monkeypatch):
    split = handoff()
    svc = _service(db, store, stash, monkeypatch, split)
    rec_id = db.create_recommendation("test_split", "scene", "7", {"scene_id": "7"})
    token = (await svc.preview(rec_id))["plan_token"]

    out = await svc.accept(rec_id, option=None, plan_token=token)

    rec = db.get_recommendation(rec_id)
    assert rec.status == "resolved"
    assert rec.resolution_details == {
        "apply_log_id": out["entry"]["id"], "option": {}, "split": asdict(split),
    }


async def test_accept_without_split(db, store, stash, monkeypatch):
    svc = _service(db, store, stash, monkeypatch, None)
    rec_id = db.create_recommendation("test_split", "scene", "7", {"scene_id": "7"})
    token = (await svc.preview(rec_id))["plan_token"]

    await svc.accept(rec_id, option=None, plan_token=token)

    rec = db.get_recommendation(rec_id)
    assert rec.status == "resolved"
    assert "split" not in rec.resolution_details

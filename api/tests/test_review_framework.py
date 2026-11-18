"""Review framework (#127) end to end against FakeStash, through the review_demo type."""

import inspect
import sqlite3

import pytest

import features
from recommendations_db import RecommendationsDB
from review.demo import DEMO_TAG, DemoReviewAnalyzer
from review.executor import PlanRejected, apply_plan
from review.plan import FieldChange, FileRef, Plan, PrimaryFileChange
from review.service import (
    BatchIneligible,
    ConfirmRequired,
    Drifted,
    Failed,
    HiddenType,
    NotRevertible,
    ReviewService,
    Stale,
)
from review.store import ReviewStore
from stash_client_unified import StashClientUnified
from tests.fake_stash import FakeStash, FakeStashError

GB = 1024 ** 3


@pytest.fixture(autouse=True)
def show_unfinished(monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "stash_sense.db")


@pytest.fixture
def store(db):
    return ReviewStore(db.db_path)


@pytest.fixture
def stash():
    return FakeStash()


@pytest.fixture
def svc(db, store, stash):
    return ReviewService(db, store, stash)


@pytest.fixture
def tag(stash):
    return stash.add_tag(DEMO_TAG)


def dup_scene(stash, tag, *, identical=True, title="old title"):
    """A tagged scene: a small primary file and a larger second copy."""
    small = stash.add_file("/media/a/clip.small.mp4", 1 * GB, oshash="aa", md5="m1")
    big = stash.add_file("/media/a/clip.big.mp4", 3 * GB,
                         oshash="aa" if identical else "bb", md5="m1" if identical else "m2")
    sid = stash.add_scene(small, big, title=title, tag_ids=(tag,))
    return sid, small, big


async def run_analyzer(stash, db):
    return await DemoReviewAnalyzer(stash, db).run()


def rec_for(db, scene_id):
    return db.get_recommendation_by_target("review_demo", "scene", scene_id)


# ---------------- fake fidelity ----------------

def test_fake_implements_real_client_methods():
    for name in FakeStash.REVIEW_METHODS:
        real = getattr(StashClientUnified, name)
        fake = getattr(FakeStash, name)
        assert inspect.iscoroutinefunction(real) and inspect.iscoroutinefunction(fake)
        assert list(inspect.signature(real).parameters)[1:] == list(inspect.signature(fake).parameters)[1:]


async def test_fake_refuses_deleting_primary_like_stash(stash, tag):
    sid, small, big = dup_scene(stash, tag)
    with pytest.raises(FakeStashError, match="cannot delete primary file"):
        await stash.delete_files([small])
    assert small in stash.files  # all-or-nothing


# ---------------- preview ----------------

async def test_preview_shows_exact_outcome(svc, stash, db, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)

    p = await svc.preview(rec.id)

    plan = p["plan"]
    assert p["default_option"] == {"keep_file_id": big, "title": "clip.big"}
    assert [c["field"] for c in plan["field_changes"]] == ["title"]
    assert plan["field_changes"][0]["before"] == "old title"
    assert plan["field_changes"][0]["after"] == "clip.big"
    assert [f["file_id"] for f in plan["keep_files"]] == [big]
    assert [(f["file_id"], f["path"], f["size"]) for f in plan["delete_files"]] == [
        (small, "/media/a/clip.small.mp4", GB)]
    assert plan["bytes_reclaimed"] == GB
    assert plan["primary_file"] == {"scene_id": sid, "before": small, "after": big}
    assert p["high_confidence"] is True
    assert p["plan_token"] == (await svc.preview(rec.id))["plan_token"]  # stable


async def test_hidden_type_refused_when_unfinished_hidden(svc, stash, db, tag, monkeypatch):
    sid, *_ = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    monkeypatch.setattr(features, "show_unfinished", lambda: False)
    with pytest.raises(HiddenType):
        await svc.preview(rec_for(db, sid).id)


# ---------------- accept ----------------

async def test_accept_without_naming_files_is_refused(svc, stash, db, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)

    with pytest.raises(ConfirmRequired) as e:
        await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[])
    assert e.value.payload["files"] == [{"file_id": small, "path": "/media/a/clip.small.mp4", "size": GB}]
    with pytest.raises(ConfirmRequired):  # naming the wrong file is no confirm either
        await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[big])
    assert stash.writes == 0 and small in stash.files


async def test_accept_applies_logs_and_resolves(svc, stash, db, store, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)

    out = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])

    scene = stash.scenes[sid]
    assert scene["title"] == "clip.big" and scene["file_ids"] == [big]
    assert stash.deleted == [{"path": "/media/a/clip.small.mp4", "destination": None}]
    entry = out["entry"]
    assert entry["status"] == "applied" and entry["action"] == "apply"
    assert entry["prior_state"]["fields"] == [
        {"entity_type": "scene", "entity_id": sid, "field": "title", "value": "old title"}]
    deleted = entry["prior_state"]["deleted_files"]
    assert deleted == [{"file_id": small, "scene_id": sid, "path": "/media/a/clip.small.mp4",
                        "size": GB, "fingerprints": {"oshash": "aa", "md5": "m1"}}]
    assert [s["step"] for s in entry["result"]["steps"]] == ["update_scene", "set_primary_file", "delete_files"]
    assert entry["result"]["steps"][-1]["destination"] == "permanently deleted"
    assert db.get_recommendation(rec.id).status == "resolved"
    assert out["label_id"] is None  # suggestion accepted as is


async def test_trash_destination_is_logged(svc, stash, db, tag):
    stash.trash_path = "/trash"
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    out = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])
    assert out["entry"]["trash_path"] == "/trash"
    assert out["entry"]["result"]["steps"][-1]["destination"] == "trash: /trash"


async def test_accept_on_changed_library_is_stale(svc, stash, db, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    stash.scenes[sid]["title"] = "someone edited this"

    with pytest.raises(Stale) as e:
        await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])
    fresh = e.value.payload["preview"]
    assert fresh["plan"]["field_changes"][0]["before"] == "someone edited this"
    assert stash.writes == 0


async def test_double_accept_applies_once(svc, stash, db, store, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    first = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])
    writes = stash.writes

    again = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])

    assert again["already_applied"] is True and again["entry"]["id"] == first["entry"]["id"]
    assert stash.writes == writes and store.count() == 1


async def test_user_correction_is_stored_as_label(svc, stash, db, store, tag):
    sid, small, big = dup_scene(stash, tag, identical=False)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id, {"keep_file_id": small, "title": "My Title"})
    assert p["high_confidence"] is False  # a user pick is never batch material
    assert [f["file_id"] for f in p["plan"]["delete_files"]] == [big]
    assert p["plan"]["primary_file"] is None  # keeper is already primary

    out = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[big])

    [label] = store.labels(rec_type="review_demo")
    assert label["id"] == out["label_id"]
    assert label["suggested"] == {"keep_file_id": big, "title": "clip.big"}
    assert label["chosen"] == {"keep_file_id": small, "title": "My Title"}
    assert {f["file_id"] for f in label["evidence"]["files"]} == {small, big}
    assert label["apply_log_id"] == out["entry"]["id"]
    assert stash.scenes[sid]["title"] == "My Title"


async def test_failed_write_is_logged_and_rec_stays_pending(svc, stash, db, store, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    stash.fail["delete_files"] = FakeStashError("GraphQL error: disk on fire")

    with pytest.raises(Failed) as e:
        await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])

    entry = e.value.payload["entry"]
    assert entry["status"] == "failed" and "disk on fire" in entry["error"]
    assert [s["step"] for s in entry["result"]["steps"]] == ["update_scene", "set_primary_file"]
    assert entry["prior_state"]["fields"][0]["value"] == "old title"  # written before the first write
    assert db.get_recommendation(rec.id).status == "pending"


# ---------------- batch ----------------

async def _batch_world(stash, db, tag):
    same, s_small, _ = dup_scene(stash, tag, identical=True)
    differ, d_small, _ = dup_scene(stash, tag, identical=False)
    single = stash.add_scene(stash.add_file("/media/b/solo.mp4", GB), title="x", tag_ids=(tag,))
    await run_analyzer(stash, db)
    return {"same": (rec_for(db, same), s_small), "differ": (rec_for(db, differ), d_small),
            "single": (rec_for(db, single), None)}


async def test_batch_preview_lists_only_high_confidence(svc, stash, db, tag):
    w = await _batch_world(stash, db, tag)
    bp = await svc.batch_preview("review_demo")
    assert {i["rec_id"] for i in bp["items"]} == {w["same"][0].id, w["single"][0].id}
    assert bp["skipped"] == [{"rec_id": w["differ"][0].id, "reason": "not high-confidence"}]
    assert bp["summary"]["files_deleted"] == 1 and bp["summary"]["bytes_reclaimed"] == GB


async def test_batch_refuses_whole_request_with_low_confidence_item(svc, stash, db, store, tag):
    w = await _batch_world(stash, db, tag)
    items = []
    for key in ("same", "differ"):
        rec, small = w[key]
        p = await svc.preview(rec.id)
        items.append({"rec_id": rec.id, "plan_token": p["plan_token"], "confirm_delete": [small]})

    with pytest.raises(BatchIneligible) as e:
        await svc.batch_accept("review_demo", items)

    assert e.value.payload["items"] == [{"rec_id": w["differ"][0].id, "reason": "not high-confidence"}]
    assert stash.writes == 0 and store.count() == 0


async def test_batch_refuses_an_option(svc, stash, db, tag):
    w = await _batch_world(stash, db, tag)
    rec, small = w["same"]
    with pytest.raises(BatchIneligible):
        await svc.batch_accept("review_demo", [{"rec_id": rec.id, "plan_token": "x",
                                                "option": {"keep_file_id": small}}])


async def test_batch_accept_applies_eligible_items(svc, stash, db, store, tag):
    w = await _batch_world(stash, db, tag)
    bp = await svc.batch_preview("review_demo")
    items = [{"rec_id": i["rec_id"], "plan_token": i["plan_token"],
              "confirm_delete": [f["file_id"] for f in i["delete_files"]]} for i in bp["items"]]

    out = await svc.batch_accept("review_demo", items)

    assert out["applied"] == 2
    entries = store.list()
    assert {e["batch_id"] for e in entries} == {out["batch_id"]}
    assert db.get_recommendation(w["differ"][0].id).status == "pending"


# ---------------- revert ----------------

async def _applied(svc, stash, db, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    out = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])
    return sid, rec, out["entry"]


async def test_revert_restores_metadata_and_reopens(svc, stash, db, store, tag):
    sid, rec, entry = await _applied(svc, stash, db, tag)

    out = await svc.revert(entry["id"])

    assert stash.scenes[sid]["title"] == "old title"
    assert out["entry"]["action"] == "revert" and out["entry"]["reverts_id"] == entry["id"]
    assert [f["path"] for f in out["files_not_restored"]] == ["/media/a/clip.small.mp4"]
    assert store.get(entry["id"])["reverted_by"] == out["entry"]["id"]
    assert db.get_recommendation(rec.id).status == "pending"
    with pytest.raises(NotRevertible):
        await svc.revert(entry["id"])


async def test_revert_refuses_drift_unless_forced(svc, stash, db, tag):
    sid, rec, entry = await _applied(svc, stash, db, tag)
    stash.scenes[sid]["title"] = "edited after apply"

    with pytest.raises(Drifted) as e:
        await svc.revert(entry["id"])
    assert e.value.payload["fields"][0]["current"] == "edited after apply"
    assert stash.scenes[sid]["title"] == "edited after apply"

    await svc.revert(entry["id"], force=True)
    assert stash.scenes[sid]["title"] == "old title"


async def test_delete_only_apply_is_not_revertible(svc, stash, db, tag):
    small = stash.add_file("/m/clip.mp4", GB, oshash="z")
    big = stash.add_file("/m/clip-big.mp4", 2 * GB, oshash="z")
    sid = stash.add_scene(big, small, title="clip-big", tag_ids=(tag,))
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    assert p["plan"]["field_changes"] == []
    out = await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])
    with pytest.raises(NotRevertible, match="cannot restore"):
        await svc.revert(out["entry"]["id"])


# ---------------- executor safety ----------------

async def test_executor_refuses_unsafe_plans(stash, store, tag):
    a = stash.add_file("/m/a.mp4", 1)
    b = stash.add_file("/m/b.mp4", 2)
    sid = stash.add_scene(a, b, tag_ids=(tag,))

    def plan(**kw):
        return Plan(rec_id=1, rec_type="review_demo", option={}, **kw)

    every = [FileRef(a, sid, "/m/a.mp4", 1), FileRef(b, sid, "/m/b.mp4", 2)]
    with pytest.raises(PlanRejected, match="every file"):
        await apply_plan(plan(delete_files=every), stash, store, target_type="scene", target_id=sid)
    with pytest.raises(PlanRejected, match="primary file"):
        await apply_plan(plan(delete_files=every[:1]), stash, store, target_type="scene", target_id=sid)
    with pytest.raises(PlanRejected, match="not writable"):
        await apply_plan(plan(field_changes=[FieldChange("scene", sid, "files", None, [])]),
                         stash, store, target_type="scene", target_id=sid)
    other = stash.add_scene(stash.add_file("/m/c.mp4", 1), tag_ids=(tag,))
    with pytest.raises(PlanRejected, match="no longer on scene"):
        await apply_plan(plan(delete_files=[FileRef(b, other, "/m/b.mp4", 2)],
                              primary_file=PrimaryFileChange(other, "x", "y")),
                         stash, store, target_type="scene", target_id=other)
    assert stash.writes == 0 and store.count() == 0


# ---------------- schema ----------------

def test_v9_database_migrates_to_review_tables(tmp_path):
    path = tmp_path / "old.db"
    RecommendationsDB(path)
    with sqlite3.connect(path) as conn:
        conn.executescript("DROP TABLE apply_log; DROP TABLE review_labels; UPDATE schema_version SET version = 9;")
    RecommendationsDB(path)
    with sqlite3.connect(path) as conn:
        names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"apply_log", "review_labels"} <= names
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 10


# ---------------- review findings ----------------

class _RetitleType:
    """Test type: rec details name a scene and a title; two recs may share a scene."""
    type_id = "test_retitle"
    display_name = "Retitle"
    description = ""

    async def preview(self, rec, option, stash):
        from review.plan import Preview, scene_field_values
        sid = rec.details["scene_id"]
        scene = await stash.get_scene_for_review(sid)
        before = scene_field_values(scene)["title"]
        plan = Plan(rec_id=rec.id, rec_type=self.type_id, option={},
                    field_changes=[FieldChange("scene", sid, "title", before, rec.details["title"])])
        return Preview(plan=plan, evidence={}, choices=[], default_option={}, high_confidence=True)

    def label(self, rec, preview):
        return None


@pytest.fixture
def retitle(monkeypatch):
    from review import types as review_types
    monkeypatch.setitem(review_types.REVIEW_TYPES, "test_retitle", _RetitleType())


async def test_batch_rebuilds_each_item_before_applying(svc, stash, db, store, retitle):
    sid = stash.add_scene(stash.add_file("/m/a.mp4", 1), title="orig")
    r1 = db.create_recommendation("test_retitle", "scene", "a", {"scene_id": sid, "title": "first"})
    r2 = db.create_recommendation("test_retitle", "scene", "b", {"scene_id": sid, "title": "second"})
    bp = await svc.batch_preview("test_retitle")
    items = [{"rec_id": i["rec_id"], "plan_token": i["plan_token"], "confirm_delete": []} for i in bp["items"]]

    out = await svc.batch_accept("test_retitle", items)

    by_rec = {r["rec_id"]: r for r in out["results"]}
    assert by_rec[r1]["status"] == "applied"
    # r2's preview said "orig -> second"; after r1 the scene says "first", so r2 is stale, not applied.
    assert by_rec[r2]["status"] == "stale"
    assert stash.scenes[sid]["title"] == "first"
    [entry] = store.list()
    assert entry["prior_state"]["fields"][0]["value"] == "orig"


async def test_batch_with_vanished_scene_is_refused_not_404(svc, stash, db, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    del stash.scenes[sid]
    with pytest.raises(BatchIneligible) as e:
        await svc.batch_accept("review_demo", [{"rec_id": rec.id, "plan_token": p["plan_token"],
                                                "confirm_delete": [small]}])
    assert "no longer exists" in e.value.payload["items"][0]["reason"]


async def test_batch_keeps_going_after_an_unexpected_error(svc, stash, db, store, tag):
    w = await _batch_world(stash, db, tag)
    bp = await svc.batch_preview("review_demo")
    items = [{"rec_id": i["rec_id"], "plan_token": i["plan_token"],
              "confirm_delete": [f["file_id"] for f in i["delete_files"]]} for i in bp["items"]]
    stash.fail["delete_files"] = ConnectionError("Stash went away")

    out = await svc.batch_accept("review_demo", items)

    statuses = {r["rec_id"]: r["status"] for r in out["results"]}
    assert statuses[w["same"][0].id] == "apply_failed"
    assert statuses[w["single"][0].id] == "applied"
    assert out["batch_id"]


async def test_failed_apply_reverts_the_fields_it_wrote(svc, stash, db, store, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    stash.fail["delete_files"] = FakeStashError("GraphQL error: disk on fire")
    with pytest.raises(Failed) as e:
        await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])
    assert stash.scenes[sid]["title"] == "clip.big"  # written before the failure

    out = await svc.revert(e.value.payload["entry"]["id"])

    assert stash.scenes[sid]["title"] == "old title"
    assert out["entry"]["action"] == "revert"


async def test_token_covers_file_fingerprints(svc, stash, db, tag):
    sid, small, big = dup_scene(stash, tag)
    await run_analyzer(stash, db)
    rec = rec_for(db, sid)
    p = await svc.preview(rec.id)
    stash.files[small]["fingerprints"] = [{"type": "oshash", "value": "changed"}]
    with pytest.raises(Stale):
        await svc.accept(rec.id, option=p["option"], plan_token=p["plan_token"], confirm_delete=[small])

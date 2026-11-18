"""Driver API v1 contract (#126): every endpoint's request and response shape.

These tests pin the shapes external tools rely on. If one fails because a
key changed, that change breaks callers: put it in /api/v2 instead.
"""

import sqlite3

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import driver_router
import features
import queue_router
import review_router
from job_models import JOB_REGISTRY
from queue_manager import QueueManager
from recommendations_db import RecommendationsDB
from review.demo import DEMO_TAG, DemoReviewAnalyzer
from review.service import ReviewService
from review.store import ReviewStore
from tests.fake_stash import FakeStash

JOB_KEYS = {"id", "type", "status", "priority", "progress", "error", "result",
            "created_at", "started_at", "completed_at"}
REC_KEYS = {"id", "type", "status", "target_type", "target_id", "confidence", "evidence",
            "reviewable", "resolution", "created_at", "updated_at"}
PREVIEW_KEYS = {"revision", "summary", "option", "default_option", "is_default_option",
                "high_confidence", "choices", "evidence", "plan", "trash_path"}
PLAN_KEYS = {"field_changes", "keep_files", "delete_files", "primary_file", "bytes_reclaimed"}
FILE_KEYS = {"file_id", "scene_id", "path", "size", "fingerprints"}
CHANGE_KEYS = {"entity_type", "entity_id", "field", "before", "after"}
ENTRY_KEYS = {"id", "action", "status", "rec_id", "rec_type", "target_type", "target_id",
              "batch_id", "reverts_id", "reverted_by", "option", "revision", "plan", "result",
              "error", "trash_path", "created_at", "completed_at"}
BASE = "/api/v1/driver"


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(features, "show_unfinished", lambda: True)
    monkeypatch.delenv(driver_router.API_KEY_ENV, raising=False)
    db = RecommendationsDB(tmp_path / "s.db")
    stash = FakeStash()
    review_router.set_review_service(ReviewService(db, ReviewStore(db.db_path), stash))
    mgr = QueueManager(db)
    queue_router.init_queue_router(mgr)
    import jobs.analysis_jobs as aj
    monkeypatch.setattr(aj, "get_rec_db", lambda: db)
    monkeypatch.setattr(aj, "get_stash_client", lambda: stash)
    app = FastAPI(version="9.9.9")
    app.include_router(driver_router.router)
    yield TestClient(app), db, stash, mgr
    review_router.set_review_service(None)
    queue_router.init_queue_router(None)


def _age_recs(db, seconds=5):
    """Move every recommendation's updated_at into a finished second (the feed holds back the current one)."""
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(f"UPDATE recommendations SET updated_at = datetime(updated_at, '-{seconds} seconds')")


async def _seed(stash, db, *, identical=True, title="t"):
    tag = next((t for t, name in stash.tags.items() if name == DEMO_TAG), None) or stash.add_tag(DEMO_TAG)
    a = stash.add_file(f"/media/x/{title}-a.mp4", 100, oshash=f"{title}o")
    b = stash.add_file(f"/media/x/{title}-b.mp4", 200, oshash=f"{title}o" if identical else f"{title}p")
    sid = stash.add_scene(a, b, title=title, tag_ids=(tag,))
    await DemoReviewAnalyzer(stash, db).run()
    rec = db.get_recommendation_by_target("review_demo", "scene", sid)
    _age_recs(db)
    return rec, a, b


async def _run_queued(mgr, job_id):
    row = mgr.get_job(job_id)
    mgr._start_job(row, JOB_REGISTRY[row["type"]])
    await mgr._running_tasks[job_id]


# ---------------- info and auth ----------------

def test_info_lists_visible_types(world, monkeypatch):
    client, *_ = world
    body = client.get(BASE).json()
    assert set(body) == {"api_version", "sidecar_version", "auth_required", "job_types", "review_types"}
    assert body["api_version"] == 1 and body["sidecar_version"] == "9.9.9" and body["auth_required"] is False
    job = next(j for j in body["job_types"] if j["type"] == "review_demo")
    assert set(job) == {"type", "name", "description", "default_priority", "params"}
    assert body["review_types"] == [
        {"type": "review_demo", "name": "Review Demo", "description": body["review_types"][0]["description"]},
        {"type": "scene_duplicates", "name": "Duplicate Scenes",
         "description": "Scenes holding the same content: merge them and keep the best file"},
        {"type": "scene_splits", "name": "Split Multi-Scene Files",
         "description": "Files holding several scenes: confirm the cut times, then hand the split to stash-curator"},
    ]
    monkeypatch.setattr(features, "show_unfinished", lambda: False)
    body = client.get(BASE).json()
    assert not {"review_demo", "scene_duplicates"} & {j["type"] for j in body["job_types"]}
    assert body["review_types"] == []


def test_api_key_required_when_configured(world, monkeypatch):
    client, *_ = world
    monkeypatch.setenv(driver_router.API_KEY_ENV, "s3cret")
    r = client.get(BASE)
    assert r.status_code == 401
    assert r.json() == {"error": {"code": "unauthorized", "message": "Missing or wrong ApiKey header"}}
    assert client.get(BASE, headers={"ApiKey": "wrong"}).status_code == 401
    r = client.get(BASE, headers={"ApiKey": "s3cret"})
    assert r.status_code == 200 and r.json()["auth_required"] is True
    # Every route is behind the key, not just the index.
    assert client.get(f"{BASE}/jobs/1").status_code == 401
    assert client.get(f"{BASE}/recommendations").status_code == 401


def test_validation_errors_use_the_driver_error_shape(world):
    client, *_ = world
    r = client.post(f"{BASE}/jobs", json={"type": "review_demo", "bogus": 1})
    assert r.status_code == 422
    err = r.json()["error"]
    assert err["code"] == "invalid_request" and err["fields"]
    r = client.post(f"{BASE}/jobs", json={"type": "review_demo", "priority": "urgent"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


# ---------------- jobs ----------------

async def test_enqueue_poll_read_result(world):
    client, db, stash, mgr = world
    tag = stash.add_tag(DEMO_TAG)
    stash.add_scene(stash.add_file("/media/x/one.mp4", 10), tag_ids=(tag,))

    r = client.post(f"{BASE}/jobs", json={"type": "review_demo", "priority": "high"})
    assert r.status_code == 201
    body = r.json()
    assert set(body) == {"created", "job"} and body["created"] is True
    job = body["job"]
    assert set(job) == JOB_KEYS
    assert job["status"] == "queued" and job["priority"] == 10 and job["result"] is None
    assert job["progress"] == {"processed": 0, "total": None}
    assert job["created_at"].endswith("Z") and "T" in job["created_at"]

    # Same type again while queued: the existing job, not a second one.
    again = client.post(f"{BASE}/jobs", json={"type": "review_demo"})
    assert again.status_code == 200
    assert again.json() == {"created": False, "job": job}

    await _run_queued(mgr, job["id"])
    done = client.get(f"{BASE}/jobs/{job['id']}").json()["job"]
    assert set(done) == JOB_KEYS
    assert done["status"] == "completed" and done["error"] is None
    assert done["progress"] == {"processed": 1, "total": 1}
    assert done["result"]["recommendations_created"] == 1
    assert set(done["result"]) == {"analysis_run_id", "items_processed", "recommendations_created",
                                   "excluded_skipped"}

    _age_recs(db)
    recs = client.get(f"{BASE}/recommendations", params={"type": "review_demo"}).json()["recommendations"]
    assert len(recs) == 1 and recs[0]["status"] == "pending"


async def test_failed_job_reports_error(world, monkeypatch):
    client, _db, _stash, mgr = world

    async def boom(self, incremental=True):
        raise RuntimeError("stash went away")
    monkeypatch.setattr(DemoReviewAnalyzer, "run", boom)
    job_id = client.post(f"{BASE}/jobs", json={"type": "review_demo"}).json()["job"]["id"]
    await _run_queued(mgr, job_id)
    job = client.get(f"{BASE}/jobs/{job_id}").json()["job"]
    assert job["status"] == "failed" and "stash went away" in job["error"] and job["result"] is None


def test_job_errors(world, monkeypatch):
    client, *_ = world
    r = client.post(f"{BASE}/jobs", json={"type": "nope"})
    assert r.status_code == 404 and r.json()["error"]["code"] == "unknown_job_type"
    r = client.post(f"{BASE}/jobs", json={"type": "review_demo", "params": {"x": 1}})
    assert r.status_code == 400 and r.json()["error"] == {
        "code": "invalid_params", "message": "Job type 'review_demo' takes no params", "params": ["x"]}
    r = client.get(f"{BASE}/jobs/424242")
    assert r.status_code == 404 and r.json()["error"]["code"] == "not_found"
    monkeypatch.setattr(features, "show_unfinished", lambda: False)
    r = client.post(f"{BASE}/jobs", json={"type": "review_demo"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "hidden_type"


# ---------------- recommendations ----------------

async def test_get_recommendation_shape(world):
    client, db, stash, _ = world
    rec, a, b = await _seed(stash, db)
    body = client.get(f"{BASE}/recommendations/{rec.id}").json()
    assert set(body) == {"recommendation", "preview", "preview_error"} and body["preview_error"] is None
    r = body["recommendation"]
    assert set(r) == REC_KEYS
    assert r["reviewable"] is True and r["resolution"] is None and r["evidence"] == {"tag": DEMO_TAG}
    p = body["preview"]
    assert set(p) == PREVIEW_KEYS
    assert p["is_default_option"] is True and p["option"] == {"keep_file_id": b, "title": "t-b"}
    assert set(p["plan"]) == PLAN_KEYS
    assert [set(f) for f in p["plan"]["delete_files"]] == [FILE_KEYS]
    assert p["plan"]["delete_files"][0]["file_id"] == a and p["plan"]["bytes_reclaimed"] == 100
    assert set(p["plan"]["field_changes"][0]) == CHANGE_KEYS
    assert p["plan"]["primary_file"] == {"scene_id": rec.target_id, "before": a, "after": b}


async def test_preview_with_option_changes_revision(world):
    client, db, stash, _ = world
    rec, a, _b = await _seed(stash, db)
    default = client.get(f"{BASE}/recommendations/{rec.id}").json()["preview"]
    r = client.post(f"{BASE}/recommendations/{rec.id}/preview", json={"option": {"keep_file_id": a}})
    assert r.status_code == 200 and set(r.json()) == {"preview"}
    chosen = r.json()["preview"]
    assert chosen["option"]["keep_file_id"] == a and chosen["is_default_option"] is False
    assert chosen["revision"] != default["revision"]
    r = client.post(f"{BASE}/recommendations/{rec.id}/preview", json={"option": {"keep_file_id": "999"}})
    assert r.status_code == 422 and r.json()["error"]["code"] == "rejected"


async def test_accept_applies_once(world):
    client, db, stash, _ = world
    rec, a, _b = await _seed(stash, db)
    p = client.get(f"{BASE}/recommendations/{rec.id}").json()["preview"]
    body = {"option": p["option"], "revision": p["revision"], "confirm_delete": [a]}

    first = client.post(f"{BASE}/recommendations/{rec.id}/accept", json=body)
    assert first.status_code == 200
    out = first.json()
    assert set(out) == {"already_applied", "entry", "warnings"}
    assert out["already_applied"] is False and out["warnings"] == []
    entry = out["entry"]
    assert set(entry) == ENTRY_KEYS
    assert entry["status"] == "applied" and entry["revision"] == p["revision"]
    assert entry["plan"] == p["plan"] and entry["result"]["files_deleted"] == 1
    assert a not in stash.files

    second = client.post(f"{BASE}/recommendations/{rec.id}/accept", json=body)
    assert second.status_code == 200
    assert second.json()["already_applied"] is True and second.json()["entry"]["id"] == entry["id"]
    assert len(client.get(f"{BASE}/apply-log").json()["entries"]) == 1

    rec_now = client.get(f"{BASE}/recommendations/{rec.id}").json()
    assert rec_now["preview"] is None
    assert rec_now["recommendation"]["status"] == "resolved"
    assert rec_now["recommendation"]["resolution"]["action"] == "applied"
    assert rec_now["recommendation"]["resolution"]["details"]["apply_log_id"] == entry["id"]


async def test_accept_refuses_changed_revision(world):
    client, db, stash, _ = world
    rec, a, b = await _seed(stash, db)
    p = client.get(f"{BASE}/recommendations/{rec.id}").json()["preview"]
    # The library changes after the caller read the plan: the scene gets renamed.
    stash.scenes[rec.target_id]["title"] = "renamed elsewhere"

    r = client.post(f"{BASE}/recommendations/{rec.id}/accept",
                    json={"option": p["option"], "revision": p["revision"], "confirm_delete": [a]})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "stale"
    assert err["message"] == "The library changed since this preview; review the updated plan"
    assert set(err["preview"]) == PREVIEW_KEYS and err["preview"]["revision"] != p["revision"]
    assert a in stash.files and client.get(f"{BASE}/apply-log").json()["entries"] == []

    # A revision read for a different option is refused the same way.
    r = client.post(f"{BASE}/recommendations/{rec.id}/accept",
                    json={"option": {"keep_file_id": a}, "revision": err["preview"]["revision"],
                          "confirm_delete": [b]})
    assert r.status_code == 409 and r.json()["error"]["code"] == "stale"


async def test_accept_requires_naming_deleted_files(world):
    client, db, stash, _ = world
    rec, a, _b = await _seed(stash, db)
    p = client.get(f"{BASE}/recommendations/{rec.id}").json()["preview"]
    r = client.post(f"{BASE}/recommendations/{rec.id}/accept",
                    json={"option": p["option"], "revision": p["revision"]})
    assert r.status_code == 409
    err = r.json()["error"]
    assert err["code"] == "confirm_delete"
    assert err["files"] == [{"file_id": a, "path": "/media/x/t-a.mp4", "size": 100}] and err["bytes"] == 100


async def test_dismiss(world):
    client, db, stash, _ = world
    rec, *_ = await _seed(stash, db)
    r = client.post(f"{BASE}/recommendations/{rec.id}/dismiss", json={"reason": "not a dupe"})
    assert r.status_code == 200 and set(r.json()) == {"recommendation"}
    assert r.json()["recommendation"]["status"] == "dismissed"
    # Idempotent.
    assert client.post(f"{BASE}/recommendations/{rec.id}/dismiss").json()["recommendation"]["status"] == "dismissed"
    r = client.post(f"{BASE}/recommendations/{rec.id}/accept", json={"revision": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_pending"


async def test_dismiss_resolved_is_refused(world):
    client, db, stash, _ = world
    rec, *_ = await _seed(stash, db)
    db.resolve_recommendation(rec.id, action="applied")
    r = client.post(f"{BASE}/recommendations/{rec.id}/dismiss")
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_pending"


def test_unreviewable_type(world):
    client, db, *_ = world
    rec_id = db.create_recommendation(type="duplicate_performer", target_type="performer",
                                      target_id="7", details={"why": "same face"}, confidence=0.9)
    body = client.get(f"{BASE}/recommendations/{rec_id}").json()
    assert body["preview"] is None and body["recommendation"]["reviewable"] is False
    assert body["recommendation"]["evidence"] == {"why": "same face"}
    r = client.post(f"{BASE}/recommendations/{rec_id}/accept", json={"revision": "x"})
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_reviewable"
    assert client.post(f"{BASE}/recommendations/{rec_id}/preview").json()["error"]["code"] == "not_reviewable"
    assert client.get(f"{BASE}/recommendations/424242").status_code == 404


def test_hidden_type_is_invisible(world, monkeypatch):
    client, db, *_ = world
    rec_id = db.create_recommendation(type="review_demo", target_type="scene", target_id="1", details={})
    _age_recs(db)
    monkeypatch.setattr(features, "show_unfinished", lambda: False)
    r = client.get(f"{BASE}/recommendations/{rec_id}")
    assert r.status_code == 403 and r.json()["error"]["code"] == "hidden_type"
    assert client.get(f"{BASE}/recommendations").json()["recommendations"] == []


# ---------------- feed ----------------

def _make(db, n, type_="duplicate_performer"):
    return [db.create_recommendation(type=type_, target_type="performer", target_id=str(i), details={})
            for i in range(n)]


def test_feed_pages_and_resumes(world):
    client, db, *_ = world
    ids = _make(db, 5)
    _age_recs(db, 10)

    page1 = client.get(f"{BASE}/recommendations", params={"limit": 3}).json()
    assert set(page1) == {"recommendations", "next_cursor", "has_more"}
    assert [r["id"] for r in page1["recommendations"]] == ids[:3] and page1["has_more"] is True
    page2 = client.get(f"{BASE}/recommendations", params={"limit": 3, "cursor": page1["next_cursor"]}).json()
    assert [r["id"] for r in page2["recommendations"]] == ids[3:] and page2["has_more"] is False
    empty = client.get(f"{BASE}/recommendations", params={"cursor": page2["next_cursor"]}).json()
    assert empty["recommendations"] == [] and empty["next_cursor"] == page2["next_cursor"]

    # A change to an old rec shows up after the saved cursor, once.
    db.dismiss_recommendation(ids[0])
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("UPDATE recommendations SET updated_at = datetime('now', '-3 seconds') WHERE id = ?",
                     (ids[0],))
    changed = client.get(f"{BASE}/recommendations", params={"cursor": page2["next_cursor"]}).json()
    assert [(r["id"], r["status"]) for r in changed["recommendations"]] == [(ids[0], "dismissed")]


def test_feed_holds_back_the_last_two_seconds(world):
    client, db, *_ = world
    _make(db, 2)  # updated_at = now
    assert client.get(f"{BASE}/recommendations").json()["recommendations"] == []
    with sqlite3.connect(db.db_path) as conn:
        conn.execute("UPDATE recommendations SET updated_at = datetime('now', '-1 seconds')")
    assert client.get(f"{BASE}/recommendations").json()["recommendations"] == []
    _age_recs(db, 2)
    assert len(client.get(f"{BASE}/recommendations").json()["recommendations"]) == 2


def test_feed_filters_and_since(world):
    client, db, *_ = world
    old = _make(db, 2)
    new = _make(db, 1, type_="duplicate_scenes")
    _age_recs(db, 5)
    with sqlite3.connect(db.db_path) as conn:
        conn.execute(f"UPDATE recommendations SET updated_at = '2026-01-01 00:00:00' WHERE id IN {tuple(old)}")

    got = client.get(f"{BASE}/recommendations", params={"since": "2026-06-01T00:00:00Z"}).json()
    assert [r["id"] for r in got["recommendations"]] == new
    # An offset is converted to UTC; a naive time is taken as UTC.
    got = client.get(f"{BASE}/recommendations", params={"since": "2026-01-01T02:00:00+02:00"}).json()
    assert [r["id"] for r in got["recommendations"]] == old + new
    got = client.get(f"{BASE}/recommendations", params={"since": "2026-01-01T00:00:01"}).json()
    assert [r["id"] for r in got["recommendations"]] == new
    got = client.get(f"{BASE}/recommendations", params={"type": "duplicate_scenes"}).json()
    assert [r["id"] for r in got["recommendations"]] == new
    got = client.get(f"{BASE}/recommendations", params={"status": "dismissed"}).json()
    assert got["recommendations"] == []

    assert client.get(f"{BASE}/recommendations", params={"since": "yesterday"}).json()["error"]["code"] == "invalid_since"
    assert client.get(f"{BASE}/recommendations", params={"cursor": "%%%"}).json()["error"]["code"] == "invalid_cursor"
    r = client.get(f"{BASE}/recommendations", params={"status": "weird"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_feed_leaves_out_excluded_scenes(world, monkeypatch):
    client, db, *_ = world
    import curation_settings
    a = db.create_recommendation(type="duplicate_scenes", target_type="scene", target_id="1", details={})
    b = db.create_recommendation(type="duplicate_scenes", target_type="scene", target_id="2", details={})
    _age_recs(db)

    class Excl:
        scene_ids = frozenset({"1"})

        def excluded(self, ids):
            return {i for i in ids if i in self.scene_ids}

    async def current(stash, **kw):
        return Excl()
    monkeypatch.setattr(curation_settings, "current", current)
    assert [r["id"] for r in client.get(f"{BASE}/recommendations").json()["recommendations"]] == [b]
    r = client.get(f"{BASE}/recommendations/{a}")
    assert r.status_code == 403 and r.json()["error"]["code"] == "scene_excluded"
    assert r.json()["error"]["scene_ids"] == ["1"]


# ---------------- apply log ----------------

async def test_apply_log_since(world):
    client, db, stash, _ = world
    entries = []
    for title in ("one", "two"):
        rec, a, _b = await _seed(stash, db, title=title)
        p = client.get(f"{BASE}/recommendations/{rec.id}").json()["preview"]
        out = client.post(f"{BASE}/recommendations/{rec.id}/accept",
                          json={"option": p["option"], "revision": p["revision"], "confirm_delete": [a]}).json()
        entries.append(out["entry"]["id"])

    log = client.get(f"{BASE}/apply-log").json()
    assert set(log) == {"entries", "next_since", "has_more"}
    assert [e["id"] for e in log["entries"]] == entries and log["next_since"] == entries[-1]
    assert all(set(e) == ENTRY_KEYS for e in log["entries"])
    log = client.get(f"{BASE}/apply-log", params={"since": entries[0]}).json()
    assert [e["id"] for e in log["entries"]] == entries[1:]
    log = client.get(f"{BASE}/apply-log", params={"since": entries[-1]}).json()
    assert log == {"entries": [], "next_since": entries[-1], "has_more": False}

    one = client.get(f"{BASE}/apply-log/{entries[0]}").json()
    assert set(one) == {"entry"} and one["entry"]["id"] == entries[0]
    assert client.get(f"{BASE}/apply-log/999").json()["error"]["code"] == "not_found"


def test_v10_database_migrates_to_v11(tmp_path):
    path = tmp_path / "old.db"
    RecommendationsDB(path)
    with sqlite3.connect(path) as conn:  # roll back to the v10 shape
        conn.execute("DROP INDEX idx_rec_updated")
        conn.execute("ALTER TABLE job_queue DROP COLUMN result")
        conn.execute("UPDATE schema_version SET version = 10")
    db = RecommendationsDB(path)
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT version FROM schema_version").fetchone()[0] == 11
        assert "result" in [r[1] for r in conn.execute("PRAGMA table_info(job_queue)")]
        assert conn.execute("SELECT 1 FROM sqlite_master WHERE name = 'idx_rec_updated'").fetchone()
    job_id = db.submit_job(type="review_demo", priority=50, triggered_by="t")
    db.set_job_result(job_id, {"n": 1})
    assert db.get_job(job_id)["result"] == '{"n": 1}'


async def test_finished_job_progress_is_final(world, monkeypatch):
    """Progress the analyzer reported mid-run must not land after the final count."""
    from analyzers.base import AnalysisResult
    client, _db, _stash, mgr = world

    async def run(self, incremental=True):
        self.set_items_total(4)
        self.update_progress(2, 0)
        return AnalysisResult(items_processed=7, recommendations_created=0)
    monkeypatch.setattr(DemoReviewAnalyzer, "run", run)
    job_id = client.post(f"{BASE}/jobs", json={"type": "review_demo"}).json()["job"]["id"]
    await _run_queued(mgr, job_id)
    import asyncio
    await asyncio.sleep(0)  # let any stray scheduled progress write run
    job = client.get(f"{BASE}/jobs/{job_id}").json()["job"]
    assert job["progress"] == {"processed": 7, "total": 7}


def test_unauthenticated_bad_body_is_401_not_422(world, monkeypatch):
    client, *_ = world
    monkeypatch.setenv(driver_router.API_KEY_ENV, "s3cret")
    r = client.post(f"{BASE}/recommendations/1/accept", content=b"{not json",
                    headers={"Content-Type": "application/json"})
    assert r.status_code == 401
    r = client.post(f"{BASE}/jobs", json={"bogus": 1})
    assert r.status_code == 401


@pytest.mark.parametrize("priority", [-1000, 7, True, "5"])
def test_priority_must_be_a_level(world, priority):
    client, *_ = world
    r = client.post(f"{BASE}/jobs", json={"type": "review_demo", "priority": priority})
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"


def test_priority_number_accepted(world):
    client, *_ = world
    r = client.post(f"{BASE}/jobs", json={"type": "review_demo", "priority": 0})
    assert r.status_code == 201 and r.json()["job"]["priority"] == 0


async def test_rec_readable_when_its_preview_fails(world):
    client, db, stash, _ = world
    rec, *_ = await _seed(stash, db)
    del stash.scenes[rec.target_id]
    r = client.get(f"{BASE}/recommendations/{rec.id}")
    assert r.status_code == 200
    body = r.json()
    assert body["recommendation"]["id"] == rec.id and body["preview"] is None
    assert body["preview_error"]["code"] == "not_found"
    assert client.post(f"{BASE}/recommendations/{rec.id}/dismiss").status_code == 200


def test_dismiss_refuses_excluded_scene(world, monkeypatch):
    client, db, *_ = world
    import curation_settings
    rec_id = db.create_recommendation(type="duplicate_scenes", target_type="scene", target_id="1", details={})

    class Excl:
        scene_ids = frozenset({"1"})

        def excluded(self, ids):
            return {i for i in ids if i in self.scene_ids}

    async def current(stash, **kw):
        return Excl()
    monkeypatch.setattr(curation_settings, "current", current)
    r = client.post(f"{BASE}/recommendations/{rec_id}/dismiss")
    assert r.status_code == 403 and r.json()["error"]["code"] == "scene_excluded"
    assert db.get_recommendation(rec_id).status == "pending"


# ---------------- split result (#131) ----------------

SPLIT = {"scene_id": "7", "file_id": "70", "oshash": "abc", "size": 1000, "duration": 3000.0,
         "ranges": [{"start": 0.0, "end": 1500.0}, {"start": 1500.0, "end": 3000.0}], "cuts": [1500.0]}
RESULT = {"status": "applied", "cut_scene_ids": ["501", "502"], "strategy": "reencode", "message": ""}


def _accepted_split(db, store, *, split=True):
    """A recommendation resolved the way an accepted split plan resolves it (task 8)."""
    rec_id = db.create_recommendation(type="scene_duplicates", target_type="scene", target_id="7", details={})
    accept_id = store.begin(action="apply", rec_id=rec_id, rec_type="scene_duplicates", target_type="scene",
                            target_id="7", option={}, plan={}, plan_token="t", prior_state={})
    store.finish(accept_id, status="applied", result={"steps": []})
    db.resolve_recommendation(rec_id, action="applied", details={
        "apply_log_id": accept_id, "option": {}, **({"split": SPLIT} if split else {})})
    _age_recs(db)
    return rec_id, accept_id


def _split_entries(store):
    return [e for e in store.list(ascending=True) if e["action"] == "split_result"]


def test_split_result_records_and_bumps_feed(world):
    client, db, *_ = world
    store = ReviewStore(db.db_path)
    rec_id, accept_id = _accepted_split(db, store)
    before = db.get_recommendation(rec_id)
    r = client.post(f"{BASE}/recommendations/{rec_id}/split-result", json=RESULT)
    assert r.status_code == 200 and set(r.json()) == {"recommendation"}
    rec = r.json()["recommendation"]
    assert set(rec) == REC_KEYS and rec["status"] == "resolved"
    assert rec["resolution"]["action"] == "applied"
    details = rec["resolution"]["details"]
    assert details["split"] == SPLIT and details["apply_log_id"] == accept_id
    assert {k: v for k, v in details["split_result"].items() if k != "reported_at"} == RESULT
    assert details["split_result"]["reported_at"]
    assert rec["updated_at"] > driver_router._ts(before.updated_at)
    assert db.get_recommendation(rec_id).resolved_at == before.resolved_at
    (entry,) = _split_entries(store)
    assert entry["status"] == "applied" and entry["rec_id"] == rec_id
    assert entry["result"]["accept_log_id"] == accept_id


def test_split_result_refused_unless_accepted_split(world):
    client, db, *_ = world
    store = ReviewStore(db.db_path)
    pending = db.create_recommendation(type="scene_duplicates", target_type="scene", target_id="8", details={})
    r = client.post(f"{BASE}/recommendations/{pending}/split-result", json=RESULT)
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_accepted"
    whole, _ = _accepted_split(db, store, split=False)
    r = client.post(f"{BASE}/recommendations/{whole}/split-result", json=RESULT)
    assert r.status_code == 409 and r.json()["error"]["code"] == "no_split"
    assert client.post(f"{BASE}/recommendations/999/split-result", json=RESULT).status_code == 404
    assert _split_entries(store) == []


def test_split_result_is_idempotent(world):
    client, db, *_ = world
    store = ReviewStore(db.db_path)
    rec_id, _ = _accepted_split(db, store)
    url = f"{BASE}/recommendations/{rec_id}/split-result"
    assert client.post(url, json=RESULT).status_code == 200
    assert client.post(url, json=RESULT).status_code == 200
    assert len(_split_entries(store)) == 1
    other = {**RESULT, "status": "failed", "cut_scene_ids": [], "message": "ffmpeg exited 1"}
    r = client.post(url, json=other)
    assert r.status_code == 200
    stored = r.json()["recommendation"]["resolution"]["details"]["split_result"]
    assert {k: v for k, v in stored.items() if k != "reported_at"} == other
    assert len(_split_entries(store)) == 2


@pytest.mark.parametrize("body", [
    {"status": "done"},
    {},
    {"status": "applied", "bogus": 1},
    {"status": "applied", "cut_scene_ids": "501"},
])
def test_split_result_validation(world, body):
    client, db, *_ = world
    store = ReviewStore(db.db_path)
    rec_id, _ = _accepted_split(db, store)
    r = client.post(f"{BASE}/recommendations/{rec_id}/split-result", json=body)
    assert r.status_code == 422 and r.json()["error"]["code"] == "invalid_request"
    assert _split_entries(store) == []

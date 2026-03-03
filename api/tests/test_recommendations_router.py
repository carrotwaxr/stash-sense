"""Tests for the recommendations API router."""

import pytest
from unittest.mock import Mock
from fastapi import FastAPI
from fastapi.testclient import TestClient

from recommendations_db import RecommendationsDB
import recommendations_router as rec_mod


@pytest.fixture
def db(tmp_path):
    """Create a fresh RecommendationsDB."""
    return RecommendationsDB(str(tmp_path / "test.db"))


@pytest.fixture
def client(db):
    """Create a test client with real DB and mocked stash_client."""
    original_db = rec_mod.rec_db
    original_stash = rec_mod.stash_client

    rec_mod.rec_db = db
    rec_mod.stash_client = Mock()

    app = FastAPI()
    app.include_router(rec_mod.router)
    test_client = TestClient(app)

    yield test_client

    rec_mod.rec_db = original_db
    rec_mod.stash_client = original_stash


def _seed_recommendations(db, count=3, rec_type="duplicate_performer", target_type="performer", status="pending"):
    """Helper to seed recommendations into the database."""
    ids = []
    for i in range(count):
        rec_id = db.create_recommendation(
            type=rec_type,
            target_type=target_type,
            target_id=str(100 + i),
            details={"name": f"Test Performer {i}"},
            confidence=0.9 - i * 0.1,
        )
        ids.append(rec_id)
    return ids


def _seed_analysis_run(db, run_type="duplicate_performer", items_total=10, complete=True):
    """Helper to seed an analysis run."""
    run_id = db.start_analysis_run(run_type, items_total=items_total)
    if complete:
        db.complete_analysis_run(run_id, recommendations_created=3)
    return run_id


# ==================== GET /recommendations ====================


class TestListRecommendations:
    """Test GET /recommendations."""

    def test_returns_empty_list(self, client):
        resp = client.get("/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert data["recommendations"] == []
        assert data["total"] == 0

    def test_returns_recommendations(self, client, db):
        _seed_recommendations(db, count=3)
        resp = client.get("/recommendations")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["recommendations"]) == 3
        assert data["total"] == 3

    def test_filter_by_status(self, client, db):
        ids = _seed_recommendations(db, count=3)
        # Dismiss one
        db.dismiss_recommendation(ids[0], reason="test")
        resp = client.get("/recommendations", params={"status": "pending"})
        data = resp.json()
        assert len(data["recommendations"]) == 2
        assert data["total"] == 2

    def test_filter_by_type(self, client, db):
        _seed_recommendations(db, count=2, rec_type="duplicate_performer")
        _seed_recommendations(db, count=1, rec_type="upstream_performer_changes")
        resp = client.get("/recommendations", params={"type": "upstream_performer_changes"})
        data = resp.json()
        assert len(data["recommendations"]) == 1
        assert data["total"] == 1
        assert data["recommendations"][0]["type"] == "upstream_performer_changes"

    def test_filter_by_target_type(self, client, db):
        _seed_recommendations(db, count=2, target_type="performer")
        _seed_recommendations(db, count=1, target_type="scene")
        resp = client.get("/recommendations", params={"target_type": "scene"})
        data = resp.json()
        assert len(data["recommendations"]) == 1
        assert data["total"] == 1

    def test_pagination_limit(self, client, db):
        _seed_recommendations(db, count=5)
        resp = client.get("/recommendations", params={"limit": 2})
        data = resp.json()
        assert len(data["recommendations"]) == 2
        assert data["total"] == 5

    def test_pagination_offset(self, client, db):
        _seed_recommendations(db, count=5)
        resp = client.get("/recommendations", params={"limit": 2, "offset": 3})
        data = resp.json()
        assert len(data["recommendations"]) == 2
        assert data["total"] == 5

    def test_recommendation_response_shape(self, client, db):
        _seed_recommendations(db, count=1)
        resp = client.get("/recommendations")
        rec = resp.json()["recommendations"][0]
        assert "id" in rec
        assert "type" in rec
        assert "status" in rec
        assert "target_type" in rec
        assert "target_id" in rec
        assert "details" in rec
        assert "confidence" in rec
        assert "created_at" in rec
        assert "updated_at" in rec


# ==================== GET /recommendations/counts ====================


class TestRecommendationCounts:
    """Test GET /recommendations/counts."""

    def test_empty_counts(self, client):
        resp = client.get("/recommendations/counts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["counts"] == {}
        assert data["total_pending"] == 0

    def test_counts_by_type_and_status(self, client, db):
        _seed_recommendations(db, count=3, rec_type="duplicate_performer")
        _seed_recommendations(db, count=2, rec_type="upstream_performer_changes")
        resp = client.get("/recommendations/counts")
        data = resp.json()
        assert data["counts"]["duplicate_performer"]["pending"] == 3
        assert data["counts"]["upstream_performer_changes"]["pending"] == 2
        assert data["total_pending"] == 5

    def test_counts_include_dismissed(self, client, db):
        ids = _seed_recommendations(db, count=3, rec_type="duplicate_performer")
        db.dismiss_recommendation(ids[0], reason="test")
        resp = client.get("/recommendations/counts")
        data = resp.json()
        assert data["counts"]["duplicate_performer"]["pending"] == 2
        assert data["counts"]["duplicate_performer"]["dismissed"] == 1
        assert data["total_pending"] == 2


# ==================== GET /recommendations/{rec_id} ====================


class TestGetRecommendation:
    """Test GET /recommendations/{rec_id}."""

    def test_returns_recommendation(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.get(f"/recommendations/{ids[0]}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == ids[0]
        assert data["type"] == "duplicate_performer"
        assert data["status"] == "pending"

    def test_404_for_missing(self, client):
        resp = client.get("/recommendations/99999")
        assert resp.status_code == 404


# ==================== POST /recommendations/{rec_id}/resolve ====================


class TestResolveRecommendation:
    """Test POST /recommendations/{rec_id}/resolve."""

    def test_resolve_success(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.post(
            f"/recommendations/{ids[0]}/resolve",
            json={"action": "merged", "details": {"merged_into": "42"}},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        # Verify resolved in DB
        rec = db.get_recommendation(ids[0])
        assert rec.status == "resolved"
        assert rec.resolution_action == "merged"

    def test_resolve_404(self, client):
        resp = client.post(
            "/recommendations/99999/resolve",
            json={"action": "merged"},
        )
        assert resp.status_code == 404


# ==================== POST /recommendations/{rec_id}/dismiss ====================


class TestDismissRecommendation:
    """Test POST /recommendations/{rec_id}/dismiss."""

    def test_dismiss_success(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.post(
            f"/recommendations/{ids[0]}/dismiss",
            json={"reason": "Not relevant"},
        )
        assert resp.status_code == 200
        assert resp.json()["success"] is True

        rec = db.get_recommendation(ids[0])
        assert rec.status == "dismissed"

    def test_dismiss_without_reason(self, client, db):
        ids = _seed_recommendations(db, count=1)
        resp = client.post(f"/recommendations/{ids[0]}/dismiss", json={})
        assert resp.status_code == 200
        assert resp.json()["success"] is True

    def test_dismiss_404(self, client):
        resp = client.post(
            "/recommendations/99999/dismiss",
            json={"reason": "test"},
        )
        assert resp.status_code == 404


# ==================== GET /recommendations/analysis/runs ====================


class TestListAnalysisRuns:
    """Test GET /recommendations/analysis/runs."""

    def test_empty_runs(self, client):
        resp = client.get("/recommendations/analysis/runs")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_returns_runs(self, client, db):
        _seed_analysis_run(db, run_type="duplicate_performer")
        _seed_analysis_run(db, run_type="upstream_performer_changes")
        resp = client.get("/recommendations/analysis/runs")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 2

    def test_filter_by_type(self, client, db):
        _seed_analysis_run(db, run_type="duplicate_performer")
        _seed_analysis_run(db, run_type="upstream_performer_changes")
        resp = client.get("/recommendations/analysis/runs", params={"type": "duplicate_performer"})
        data = resp.json()
        assert len(data) == 1
        assert data[0]["type"] == "duplicate_performer"

    def test_run_response_shape(self, client, db):
        _seed_analysis_run(db, run_type="duplicate_performer")
        resp = client.get("/recommendations/analysis/runs")
        run = resp.json()[0]
        assert "id" in run
        assert "type" in run
        assert "status" in run
        assert "started_at" in run
        assert "recommendations_created" in run


# ==================== GET /recommendations/analysis/runs/{run_id} ====================


class TestGetAnalysisRun:
    """Test GET /recommendations/analysis/runs/{run_id}."""

    def test_returns_run(self, client, db):
        run_id = _seed_analysis_run(db, run_type="duplicate_performer")
        resp = client.get(f"/recommendations/analysis/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == run_id
        assert data["type"] == "duplicate_performer"
        assert data["status"] == "completed"

    def test_404_for_missing(self, client):
        resp = client.get("/recommendations/analysis/runs/99999")
        assert resp.status_code == 404


# ==================== POST /recommendations/actions/batch-dismiss ====================


class TestBatchDismiss:
    """Test POST /recommendations/actions/batch-dismiss."""

    def test_batch_dismiss_by_type(self, client, db):
        _seed_recommendations(db, count=3, rec_type="duplicate_performer")
        _seed_recommendations(db, count=2, rec_type="upstream_performer_changes")

        resp = client.post(
            "/recommendations/actions/batch-dismiss",
            json={"type": "duplicate_performer"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["dismissed_count"] == 3

        # Verify only the correct type was dismissed
        remaining = db.get_recommendations(status="pending")
        assert len(remaining) == 2
        assert all(r.type == "upstream_performer_changes" for r in remaining)

    def test_batch_dismiss_returns_zero_for_none(self, client):
        resp = client.post(
            "/recommendations/actions/batch-dismiss",
            json={"type": "nonexistent_type"},
        )
        assert resp.status_code == 200
        assert resp.json()["dismissed_count"] == 0

    def test_batch_dismiss_permanent_flag(self, client, db):
        _seed_recommendations(db, count=2, rec_type="duplicate_performer")
        resp = client.post(
            "/recommendations/actions/batch-dismiss",
            json={"type": "duplicate_performer", "permanent": True},
        )
        assert resp.status_code == 200
        assert resp.json()["dismissed_count"] == 2

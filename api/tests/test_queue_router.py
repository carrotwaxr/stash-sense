"""Tests for queue API router."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from queue_router import router, init_queue_router


class TestQueueRouter:
    @pytest.fixture
    def db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(str(tmp_path / "test.db"))

    @pytest.fixture
    def client(self, db):
        from queue_manager import QueueManager
        mgr = QueueManager(db)
        init_queue_router(mgr)
        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_list_jobs_empty(self, client):
        resp = client.get("/queue")
        assert resp.status_code == 200
        assert resp.json()["jobs"] == []

    def test_submit_job(self, client):
        resp = client.post("/queue", json={"type": "duplicate_performer", "triggered_by": "user"})
        assert resp.status_code == 200
        assert resp.json()["job_id"] is not None

    def test_submit_unknown_type(self, client):
        resp = client.post("/queue", json={"type": "nonexistent", "triggered_by": "user"})
        assert resp.status_code == 400

    def test_submit_duplicate(self, client):
        client.post("/queue", json={"type": "duplicate_performer", "triggered_by": "user"})
        resp = client.post("/queue", json={"type": "duplicate_performer", "triggered_by": "user"})
        assert resp.status_code == 409

    def test_get_single_job(self, client):
        resp = client.post("/queue", json={"type": "duplicate_performer", "triggered_by": "user"})
        job_id = resp.json()["job_id"]
        resp = client.get(f"/queue/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["type"] == "duplicate_performer"

    def test_cancel_job(self, client):
        resp = client.post("/queue", json={"type": "duplicate_performer", "triggered_by": "user"})
        job_id = resp.json()["job_id"]
        resp = client.delete(f"/queue/{job_id}")
        assert resp.status_code == 200
        resp = client.get(f"/queue/{job_id}")
        assert resp.json()["status"] == "cancelled"

    def test_get_queue_status(self, client):
        client.post("/queue", json={"type": "duplicate_performer", "triggered_by": "user"})
        resp = client.get("/queue/status")
        assert resp.status_code == 200
        assert resp.json()["queued"] == 1

    def test_list_schedules(self, client, db):
        db.upsert_job_schedule("test_type", True, 24.0, 50)
        resp = client.get("/queue/schedules")
        assert resp.status_code == 200
        assert len(resp.json()["schedules"]) == 1

    def test_update_schedule(self, client):
        resp = client.put("/queue/schedules/duplicate_performer", json={
            "enabled": True, "interval_hours": 48.0,
        })
        assert resp.status_code == 200
        resp = client.get("/queue/schedules")
        schedules = resp.json()["schedules"]
        dup = next(s for s in schedules if s["type"] == "duplicate_performer")
        assert dup["interval_hours"] == 48.0

    def test_get_job_types(self, client):
        resp = client.get("/queue/types")
        assert resp.status_code == 200
        types = resp.json()["types"]
        assert any(t["type_id"] == "duplicate_performer" for t in types)

    def test_job_types_include_description(self, client):
        resp = client.get("/queue/types")
        types = resp.json()["types"]
        for t in types:
            assert "description" in t
            assert isinstance(t["description"], str)

    def test_job_types_include_allowed_intervals(self, client):
        resp = client.get("/queue/types")
        types = resp.json()["types"]
        db_update = next(t for t in types if t["type_id"] == "database_update")
        assert "allowed_intervals" in db_update
        intervals = db_update["allowed_intervals"]
        assert len(intervals) > 0
        # Each interval has hours (int) and label (str)
        for interval in intervals:
            assert "hours" in interval
            assert "label" in interval
            assert isinstance(interval["hours"], int)
            assert isinstance(interval["label"], str)

    def test_job_types_interval_format(self, client):
        """Verify the serialized interval format matches what the frontend expects."""
        resp = client.get("/queue/types")
        types = resp.json()["types"]
        dup = next(t for t in types if t["type_id"] == "duplicate_performer")
        # Infrequent tier starts at 24h
        hours_list = [i["hours"] for i in dup["allowed_intervals"]]
        assert min(hours_list) >= 24

"""Tests for job_queue and job_schedules database tables and CRUD methods."""

import pytest
from datetime import datetime
from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    """Create a fresh RecommendationsDB for testing."""
    return RecommendationsDB(str(tmp_path / "test.db"))


class TestSubmitJob:
    def test_submit_job_returns_valid_id(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        assert job_id is not None
        assert isinstance(job_id, int)
        assert job_id > 0

    def test_submit_job_with_optional_fields(self, db):
        job_id = db.submit_job(
            "face_scan", priority=5, triggered_by="schedule",
            cursor="scene:100", items_total=500
        )
        assert job_id is not None
        job = db.get_job(job_id)
        assert job["cursor"] == "scene:100"
        assert job["items_total"] == 500

    def test_submit_duplicate_type_returns_none(self, db):
        job_id1 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        job_id2 = db.submit_job("upstream_sync", priority=5, triggered_by="schedule")
        assert job_id1 is not None
        assert job_id2 is None

    def test_submit_same_type_after_completed_allowed(self, db):
        job_id1 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.complete_job(job_id1)
        job_id2 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        assert job_id2 is not None
        assert job_id2 != job_id1

    def test_submit_same_type_after_failed_allowed(self, db):
        job_id1 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.fail_job(job_id1, "connection error")
        job_id2 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        assert job_id2 is not None

    def test_submit_same_type_after_cancelled_allowed(self, db):
        job_id1 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.cancel_job(job_id1)
        job_id2 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        assert job_id2 is not None

    def test_submit_duplicate_blocked_when_running(self, db):
        job_id1 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.start_job(job_id1)
        job_id2 = db.submit_job("upstream_sync", priority=5, triggered_by="schedule")
        assert job_id2 is None

    def test_submit_duplicate_blocked_when_stopping(self, db):
        job_id1 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.set_job_status(job_id1, "stopping")
        job_id2 = db.submit_job("upstream_sync", priority=5, triggered_by="schedule")
        assert job_id2 is None

    def test_submit_different_types_allowed(self, db):
        job_id1 = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        job_id2 = db.submit_job("face_scan", priority=5, triggered_by="schedule")
        assert job_id1 is not None
        assert job_id2 is not None


class TestGetJob:
    def test_get_job_returns_correct_fields(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        job = db.get_job(job_id)
        assert job is not None
        assert job["id"] == job_id
        assert job["type"] == "upstream_sync"
        assert job["status"] == "queued"
        assert job["priority"] == 10
        assert job["triggered_by"] == "manual"
        assert job["items_processed"] == 0
        assert job["created_at"] is not None

    def test_get_job_nonexistent_returns_none(self, db):
        assert db.get_job(999) is None


class TestGetQueuedJobs:
    def test_get_queued_jobs_ordered_by_priority(self, db):
        db.submit_job("low_priority", priority=100, triggered_by="schedule")
        db.submit_job("high_priority", priority=1, triggered_by="manual")
        db.submit_job("mid_priority", priority=50, triggered_by="schedule")

        jobs = db.get_queued_jobs()
        assert len(jobs) == 3
        assert jobs[0]["type"] == "high_priority"
        assert jobs[1]["type"] == "mid_priority"
        assert jobs[2]["type"] == "low_priority"

    def test_get_queued_jobs_excludes_non_queued(self, db):
        job_id1 = db.submit_job("running_job", priority=1, triggered_by="manual")
        db.start_job(job_id1)
        db.submit_job("queued_job", priority=10, triggered_by="manual")

        jobs = db.get_queued_jobs()
        assert len(jobs) == 1
        assert jobs[0]["type"] == "queued_job"


class TestStartJob:
    def test_start_job_sets_status_and_started_at(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.start_job(job_id)
        job = db.get_job(job_id)
        assert job["status"] == "running"
        assert job["started_at"] is not None


class TestCompleteJob:
    def test_complete_job_sets_status_and_completed_at(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.start_job(job_id)
        db.complete_job(job_id)
        job = db.get_job(job_id)
        assert job["status"] == "completed"
        assert job["completed_at"] is not None


class TestFailJob:
    def test_fail_job_sets_error_message(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.start_job(job_id)
        db.fail_job(job_id, "connection timeout")
        job = db.get_job(job_id)
        assert job["status"] == "failed"
        assert job["error_message"] == "connection timeout"
        assert job["completed_at"] is not None


class TestCancelJob:
    def test_cancel_job_sets_cancelled_status(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.cancel_job(job_id)
        job = db.get_job(job_id)
        assert job["status"] == "cancelled"
        assert job["completed_at"] is not None


class TestUpdateJobProgress:
    def test_update_items_processed(self, db):
        job_id = db.submit_job("face_scan", priority=10, triggered_by="manual", items_total=100)
        db.start_job(job_id)
        db.update_job_progress(job_id, items_processed=42)
        job = db.get_job(job_id)
        assert job["items_processed"] == 42
        assert job["items_total"] == 100

    def test_update_items_total(self, db):
        job_id = db.submit_job("face_scan", priority=10, triggered_by="manual")
        db.update_job_progress(job_id, items_total=200)
        job = db.get_job(job_id)
        assert job["items_total"] == 200

    def test_update_cursor(self, db):
        job_id = db.submit_job("face_scan", priority=10, triggered_by="manual")
        db.update_job_progress(job_id, cursor="scene:55")
        job = db.get_job(job_id)
        assert job["cursor"] == "scene:55"

    def test_update_multiple_fields(self, db):
        job_id = db.submit_job("face_scan", priority=10, triggered_by="manual")
        db.update_job_progress(job_id, items_processed=10, items_total=50, cursor="scene:10")
        job = db.get_job(job_id)
        assert job["items_processed"] == 10
        assert job["items_total"] == 50
        assert job["cursor"] == "scene:10"

    def test_update_no_fields_is_noop(self, db):
        job_id = db.submit_job("face_scan", priority=10, triggered_by="manual")
        db.update_job_progress(job_id)  # no fields
        job = db.get_job(job_id)
        assert job["items_processed"] == 0


class TestRequeueInterruptedJobs:
    def test_requeue_running_jobs(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.start_job(job_id)
        assert db.get_job(job_id)["status"] == "running"

        count = db.requeue_interrupted_jobs()
        assert count == 1
        job = db.get_job(job_id)
        assert job["status"] == "queued"
        assert job["started_at"] is None

    def test_requeue_stopping_jobs(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.set_job_status(job_id, "stopping")

        count = db.requeue_interrupted_jobs()
        assert count == 1
        assert db.get_job(job_id)["status"] == "queued"

    def test_requeue_does_not_affect_completed(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.start_job(job_id)
        db.complete_job(job_id)

        count = db.requeue_interrupted_jobs()
        assert count == 0
        assert db.get_job(job_id)["status"] == "completed"

    def test_requeue_returns_zero_when_none_interrupted(self, db):
        count = db.requeue_interrupted_jobs()
        assert count == 0

    def test_requeue_clears_stale_fields(self, db):
        """Re-queued jobs should have stale progress/timing fields cleared."""
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.start_job(job_id)
        db.update_job_progress(job_id, items_processed=271, items_total=271)
        # Simulate completed_at being set (e.g. from partial state)
        with db._connection() as conn:
            conn.execute(
                "UPDATE job_queue SET completed_at = datetime('now') WHERE id = ?",
                (job_id,),
            )

        db.requeue_interrupted_jobs()
        job = db.get_job(job_id)
        assert job["status"] == "queued"
        assert job["started_at"] is None
        assert job["completed_at"] is None
        assert job["items_processed"] == 0
        assert job["items_total"] is None


class TestGetJobs:
    def test_get_jobs_filtered_by_status(self, db):
        job_id1 = db.submit_job("job_a", priority=10, triggered_by="manual")
        db.submit_job("job_b", priority=10, triggered_by="manual")
        db.start_job(job_id1)

        running = db.get_jobs(status="running")
        assert len(running) == 1
        assert running[0]["type"] == "job_a"

        queued = db.get_jobs(status="queued")
        assert len(queued) == 1
        assert queued[0]["type"] == "job_b"

    def test_get_jobs_filtered_by_type(self, db):
        db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.submit_job("face_scan", priority=10, triggered_by="manual")

        jobs = db.get_jobs(type="upstream_sync")
        assert len(jobs) == 1
        assert jobs[0]["type"] == "upstream_sync"

    def test_get_jobs_limited(self, db):
        for i in range(10):
            db.submit_job(f"job_{i}", priority=10, triggered_by="manual")

        jobs = db.get_jobs(limit=3)
        assert len(jobs) == 3

    def test_get_jobs_default_order_is_created_desc(self, db):
        db.submit_job("first", priority=10, triggered_by="manual")
        db.submit_job("second", priority=10, triggered_by="manual")
        db.submit_job("third", priority=10, triggered_by="manual")

        jobs = db.get_jobs()
        assert jobs[0]["type"] == "third"
        assert jobs[2]["type"] == "first"


class TestSetJobStatus:
    def test_set_job_status(self, db):
        job_id = db.submit_job("upstream_sync", priority=10, triggered_by="manual")
        db.set_job_status(job_id, "stopping")
        job = db.get_job(job_id)
        assert job["status"] == "stopping"


class TestUpsertJobSchedule:
    def test_upsert_and_get_schedule(self, db):
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        schedule = db.get_job_schedule("upstream_sync")
        assert schedule is not None
        assert schedule["type"] == "upstream_sync"
        assert schedule["enabled"] == 1
        assert schedule["interval_hours"] == 6.0
        assert schedule["priority"] == 10

    def test_upsert_updates_existing(self, db):
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        db.upsert_job_schedule("upstream_sync", enabled=False, interval_hours=12.0, priority=5)
        schedule = db.get_job_schedule("upstream_sync")
        assert schedule["enabled"] == 0
        assert schedule["interval_hours"] == 12.0
        assert schedule["priority"] == 5

    def test_get_schedule_nonexistent_returns_none(self, db):
        assert db.get_job_schedule("nonexistent") is None


class TestUpsertJobScheduleNextRun:
    def test_enabling_sets_next_run_at(self, db):
        """Enabling a schedule should set next_run_at = now + interval."""
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        schedule = db.get_job_schedule("upstream_sync")
        assert schedule["next_run_at"] is not None
        last = datetime.fromisoformat(schedule["next_run_at"])
        now = datetime.utcnow()
        diff = (last - now).total_seconds()
        # Should be approximately 6 hours in the future
        assert abs(diff - 6 * 3600) < 10

    def test_disabling_clears_next_run_at(self, db):
        """Disabling a schedule should clear next_run_at."""
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        db.upsert_job_schedule("upstream_sync", enabled=False, interval_hours=6.0, priority=10)
        schedule = db.get_job_schedule("upstream_sync")
        assert schedule["next_run_at"] is None

    def test_re_enabling_preserves_existing_next_run(self, db):
        """Re-enabling an already-enabled schedule should not reset next_run_at."""
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        original = db.get_job_schedule("upstream_sync")["next_run_at"]
        # Update interval but keep enabled
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=12.0, priority=10)
        updated = db.get_job_schedule("upstream_sync")
        assert updated["next_run_at"] == original
        assert updated["interval_hours"] == 12.0

    def test_enabling_after_disabled_sets_new_next_run(self, db):
        """Enabling a previously disabled schedule should set a new next_run_at."""
        db.upsert_job_schedule("upstream_sync", enabled=False, interval_hours=6.0, priority=10)
        assert db.get_job_schedule("upstream_sync")["next_run_at"] is None
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        schedule = db.get_job_schedule("upstream_sync")
        assert schedule["next_run_at"] is not None


class TestGetAllJobSchedules:
    def test_get_all_schedules(self, db):
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        db.upsert_job_schedule("face_scan", enabled=False, interval_hours=24.0, priority=50)

        schedules = db.get_all_job_schedules()
        assert len(schedules) == 2
        types = {s["type"] for s in schedules}
        assert types == {"upstream_sync", "face_scan"}

    def test_get_all_schedules_empty(self, db):
        assert db.get_all_job_schedules() == []


class TestUpdateScheduleLastRun:
    def test_update_schedule_last_run_sets_times(self, db):
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        db.update_schedule_last_run("upstream_sync")
        schedule = db.get_job_schedule("upstream_sync")
        assert schedule["last_run_at"] is not None
        assert schedule["next_run_at"] is not None

        # Verify next_run_at is approximately 6 hours after last_run_at
        last_run = datetime.fromisoformat(schedule["last_run_at"])
        next_run = datetime.fromisoformat(schedule["next_run_at"])
        diff = next_run - last_run
        # Allow a small tolerance
        assert abs(diff.total_seconds() - 6 * 3600) < 5


class TestGetDueSchedules:
    def test_get_due_schedules_returns_enabled_past_due(self, db):
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        db.update_schedule_last_run("upstream_sync")
        # Manually set next_run_at to past
        with db._connection() as conn:
            conn.execute(
                "UPDATE job_schedules SET next_run_at = datetime('now', '-1 hour') WHERE type = ?",
                ("upstream_sync",)
            )

        due = db.get_due_schedules()
        assert len(due) == 1
        assert due[0]["type"] == "upstream_sync"

    def test_get_due_schedules_enabled_has_future_next_run(self, db):
        """Enabling a schedule sets next_run_at in the future, so it's not immediately due."""
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        schedule = db.get_job_schedule("upstream_sync")
        assert schedule["next_run_at"] is not None
        due = db.get_due_schedules()
        assert len(due) == 0

    def test_disabled_schedule_not_due(self, db):
        db.upsert_job_schedule("upstream_sync", enabled=False, interval_hours=6.0, priority=10)
        due = db.get_due_schedules()
        assert len(due) == 0

    def test_future_schedule_not_due(self, db):
        db.upsert_job_schedule("upstream_sync", enabled=True, interval_hours=6.0, priority=10)
        with db._connection() as conn:
            conn.execute(
                "UPDATE job_schedules SET next_run_at = datetime('now', '+1 hour') WHERE type = ?",
                ("upstream_sync",)
            )
        due = db.get_due_schedules()
        assert len(due) == 0

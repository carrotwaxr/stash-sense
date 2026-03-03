"""Tests for analysis runs and watermarks in RecommendationsDB."""

import pytest
from recommendations_db import RecommendationsDB, AnalysisRun


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "test.db"))


# ==================== start_analysis_run ====================


class TestStartAnalysisRun:
    def test_returns_int_id(self, db):
        run_id = db.start_analysis_run("upstream_performer_changes")
        assert isinstance(run_id, int)
        assert run_id > 0

    def test_initial_status_is_running(self, db):
        run_id = db.start_analysis_run("upstream_performer_changes")
        run = db.get_analysis_run(run_id)
        assert run.status == "running"

    def test_stores_type(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        run = db.get_analysis_run(run_id)
        assert run.type == "duplicate_scenes"

    def test_stores_items_total(self, db):
        run_id = db.start_analysis_run("upstream", items_total=500)
        run = db.get_analysis_run(run_id)
        assert run.items_total == 500

    def test_items_total_none_by_default(self, db):
        run_id = db.start_analysis_run("upstream")
        run = db.get_analysis_run(run_id)
        assert run.items_total is None

    def test_sets_started_at(self, db):
        run_id = db.start_analysis_run("upstream")
        run = db.get_analysis_run(run_id)
        assert run.started_at is not None

    def test_initial_recommendations_created_is_zero(self, db):
        run_id = db.start_analysis_run("upstream")
        run = db.get_analysis_run(run_id)
        assert run.recommendations_created == 0


# ==================== get_analysis_run ====================


class TestGetAnalysisRun:
    def test_returns_analysis_run_dataclass(self, db):
        run_id = db.start_analysis_run("upstream")
        run = db.get_analysis_run(run_id)
        assert isinstance(run, AnalysisRun)

    def test_returns_none_for_missing(self, db):
        assert db.get_analysis_run(9999) is None


# ==================== update_analysis_progress ====================


class TestUpdateAnalysisProgress:
    def test_updates_items_processed(self, db):
        run_id = db.start_analysis_run("upstream", items_total=100)
        db.update_analysis_progress(run_id, items_processed=42, recommendations_created=5)
        run = db.get_analysis_run(run_id)
        assert run.items_processed == 42
        assert run.recommendations_created == 5

    def test_updates_cursor(self, db):
        run_id = db.start_analysis_run("upstream")
        db.update_analysis_progress(run_id, items_processed=10, recommendations_created=2, cursor="performer:100")
        run = db.get_analysis_run(run_id)
        assert run.cursor == "performer:100"


# ==================== update_analysis_items_total ====================


class TestUpdateAnalysisItemsTotal:
    def test_updates_items_total(self, db):
        run_id = db.start_analysis_run("upstream")
        db.update_analysis_items_total(run_id, 250)
        run = db.get_analysis_run(run_id)
        assert run.items_total == 250


# ==================== complete_analysis_run ====================


class TestCompleteAnalysisRun:
    def test_sets_completed_status(self, db):
        run_id = db.start_analysis_run("upstream")
        db.complete_analysis_run(run_id, recommendations_created=10)
        run = db.get_analysis_run(run_id)
        assert run.status == "completed"

    def test_sets_completed_at(self, db):
        run_id = db.start_analysis_run("upstream")
        db.complete_analysis_run(run_id, recommendations_created=0)
        run = db.get_analysis_run(run_id)
        assert run.completed_at is not None

    def test_stores_recommendations_created(self, db):
        run_id = db.start_analysis_run("upstream")
        db.complete_analysis_run(run_id, recommendations_created=15)
        run = db.get_analysis_run(run_id)
        assert run.recommendations_created == 15


# ==================== fail_analysis_run ====================


class TestFailAnalysisRun:
    def test_sets_failed_status(self, db):
        run_id = db.start_analysis_run("upstream")
        db.fail_analysis_run(run_id, "connection timeout")
        run = db.get_analysis_run(run_id)
        assert run.status == "failed"

    def test_stores_error_message(self, db):
        run_id = db.start_analysis_run("upstream")
        db.fail_analysis_run(run_id, "rate limit exceeded")
        run = db.get_analysis_run(run_id)
        assert run.error_message == "rate limit exceeded"

    def test_sets_completed_at(self, db):
        run_id = db.start_analysis_run("upstream")
        db.fail_analysis_run(run_id, "error")
        run = db.get_analysis_run(run_id)
        assert run.completed_at is not None


# ==================== fail_stale_analysis_runs ====================


class TestFailStaleAnalysisRuns:
    def test_marks_running_as_failed(self, db):
        run_id1 = db.start_analysis_run("upstream")
        run_id2 = db.start_analysis_run("scenes")
        count = db.fail_stale_analysis_runs()
        assert count == 2
        assert db.get_analysis_run(run_id1).status == "failed"
        assert db.get_analysis_run(run_id2).status == "failed"

    def test_does_not_affect_completed(self, db):
        run_id = db.start_analysis_run("upstream")
        db.complete_analysis_run(run_id, recommendations_created=0)
        count = db.fail_stale_analysis_runs()
        assert count == 0
        assert db.get_analysis_run(run_id).status == "completed"

    def test_returns_zero_when_none_stale(self, db):
        assert db.fail_stale_analysis_runs() == 0

    def test_sets_error_message(self, db):
        run_id = db.start_analysis_run("upstream")
        db.fail_stale_analysis_runs()
        run = db.get_analysis_run(run_id)
        assert "restart" in run.error_message.lower()


# ==================== get_recent_analysis_runs ====================


class TestGetRecentAnalysisRuns:
    def test_returns_all_runs(self, db):
        run_id1 = db.start_analysis_run("upstream")
        db.complete_analysis_run(run_id1, recommendations_created=0)
        run_id2 = db.start_analysis_run("upstream")

        runs = db.get_recent_analysis_runs()
        assert len(runs) == 2
        returned_ids = {r.id for r in runs}
        assert returned_ids == {run_id1, run_id2}

    def test_filter_by_type(self, db):
        db.start_analysis_run("upstream")
        db.start_analysis_run("scenes")

        runs = db.get_recent_analysis_runs(type="upstream")
        assert len(runs) == 1
        assert runs[0].type == "upstream"

    def test_limit(self, db):
        for _ in range(5):
            db.start_analysis_run("upstream")
            db.fail_stale_analysis_runs()  # so we can start another
        runs = db.get_recent_analysis_runs(limit=3)
        assert len(runs) == 3

    def test_empty_when_none(self, db):
        assert db.get_recent_analysis_runs() == []

    def test_returns_analysis_run_dataclass(self, db):
        db.start_analysis_run("upstream")
        runs = db.get_recent_analysis_runs()
        assert isinstance(runs[0], AnalysisRun)


# ==================== Watermarks ====================


class TestWatermarks:
    def test_get_returns_none_when_no_watermark(self, db):
        assert db.get_watermark("upstream") is None

    def test_set_and_get_watermark(self, db):
        db.set_watermark("upstream", last_cursor="performer:100", last_stash_updated_at="2026-01-15T10:00:00Z")
        wm = db.get_watermark("upstream")
        assert wm is not None
        assert wm["type"] == "upstream"
        assert wm["last_cursor"] == "performer:100"
        assert wm["last_stash_updated_at"] == "2026-01-15T10:00:00Z"
        assert wm["last_completed_at"] is not None

    def test_set_watermark_updates_existing(self, db):
        db.set_watermark("upstream", last_cursor="performer:100")
        db.set_watermark("upstream", last_cursor="performer:200")
        wm = db.get_watermark("upstream")
        assert wm["last_cursor"] == "performer:200"

    def test_set_watermark_with_logic_version(self, db):
        db.set_watermark("upstream", logic_version=3)
        wm = db.get_watermark("upstream")
        assert wm["logic_version"] == 3

    def test_default_logic_version_is_1(self, db):
        db.set_watermark("upstream")
        wm = db.get_watermark("upstream")
        assert wm["logic_version"] == 1

    def test_delete_watermark(self, db):
        db.set_watermark("upstream", last_cursor="performer:100")
        db.delete_watermark("upstream")
        assert db.get_watermark("upstream") is None

    def test_delete_nonexistent_is_noop(self, db):
        db.delete_watermark("nonexistent")  # should not raise

    def test_coalesce_preserves_existing_values_on_update(self, db):
        db.set_watermark("upstream", last_cursor="performer:100", last_stash_updated_at="2026-01-15")
        # Update only cursor, stash_updated_at should be preserved
        db.set_watermark("upstream", last_cursor="performer:200")
        wm = db.get_watermark("upstream")
        assert wm["last_cursor"] == "performer:200"
        assert wm["last_stash_updated_at"] == "2026-01-15"

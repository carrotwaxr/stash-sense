"""Tests for database update job wrapper."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from base_job import JobContext


class TestDatabaseUpdateJob:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.update_job_progress = MagicMock()
        return db

    @pytest.fixture
    def ctx(self, mock_db):
        return JobContext(job_id=1, db=mock_db, queue_manager=None)

    @pytest.mark.asyncio
    async def test_runs_update_check(self, ctx):
        from jobs.database_update_job import DatabaseUpdateJob
        mock_updater = MagicMock()
        mock_updater.check_update = AsyncMock(return_value={
            "update_available": False,
        })
        with patch('jobs.database_update_job.get_db_updater', return_value=mock_updater):
            job = DatabaseUpdateJob()
            result = await job.run(ctx)
        mock_updater.check_update.assert_called_once()

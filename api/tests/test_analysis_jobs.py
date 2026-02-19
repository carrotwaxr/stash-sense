"""Tests for analysis job wrappers."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from base_job import JobContext
from jobs.analysis_jobs import AnalysisJob


class TestAnalysisJob:
    @pytest.fixture
    def mock_db(self):
        db = MagicMock()
        db.update_job_progress = MagicMock()
        return db

    @pytest.fixture
    def mock_stash(self):
        return MagicMock()

    @pytest.fixture
    def ctx(self, mock_db):
        return JobContext(job_id=1, db=mock_db, queue_manager=None)

    @pytest.mark.asyncio
    async def test_runs_analyzer(self, ctx, mock_stash, mock_db):
        mock_analyzer = MagicMock()
        mock_result = MagicMock()
        mock_result.recommendations_created = 5
        mock_result.items_processed = 10
        mock_analyzer.run = AsyncMock(return_value=mock_result)

        with patch('jobs.analysis_jobs.ANALYZERS') as mock_reg, \
             patch('jobs.analysis_jobs.get_rec_db', return_value=mock_db), \
             patch('jobs.analysis_jobs.get_stash_client', return_value=mock_stash):
            mock_cls = MagicMock(return_value=mock_analyzer)
            mock_reg.get.return_value = mock_cls

            job = AnalysisJob("duplicate_performer")
            result = await job.run(ctx)

        mock_analyzer.run.assert_called_once()
        assert result is None

    @pytest.mark.asyncio
    async def test_respects_stop_request(self, ctx, mock_stash, mock_db):
        ctx.request_stop()
        with patch('jobs.analysis_jobs.get_rec_db', return_value=mock_db), \
             patch('jobs.analysis_jobs.get_stash_client', return_value=mock_stash):
            job = AnalysisJob("duplicate_performer")
            result = await job.run(ctx)
        assert result is None

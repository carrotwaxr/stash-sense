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


class _StopAwareAnalyzer:
    """Records whether it saw a stop that arrived after it started."""

    def __init__(self, *_args, **_kwargs):
        from analyzers.base import BaseAnalyzer
        self._stop_requested = False
        self._stop_check = None
        self.is_stop_requested = BaseAnalyzer.is_stop_requested.__get__(self)
        self.request_stop = BaseAnalyzer.request_stop.__get__(self)
        self.saw_stop = None
        self.context = None

    async def run(self, incremental=True):
        assert not self.is_stop_requested()
        self.context.request_stop()          # the user presses Stop mid-run
        self.saw_stop = self.is_stop_requested()
        return MagicMock(recommendations_created=0, items_processed=1, extra=None)


@pytest.mark.asyncio
async def test_stop_requested_mid_run_reaches_the_analyzer():
    db = MagicMock()
    db.get_user_setting.return_value = []
    ctx = JobContext(job_id=1, db=db, queue_manager=None)
    analyzer = _StopAwareAnalyzer()
    analyzer.context = ctx
    excl = MagicMock(scene_ids=set())
    with patch('jobs.analysis_jobs.ANALYZERS') as reg, \
         patch('jobs.analysis_jobs.get_rec_db', return_value=db), \
         patch('jobs.analysis_jobs.get_stash_client', return_value=MagicMock()), \
         patch('jobs.analysis_jobs.curation_settings.current', AsyncMock(return_value=excl)):
        reg.get.return_value = MagicMock(return_value=analyzer)
        await AnalysisJob("scene_duplicates").run(ctx)
    assert analyzer.saw_stop is True

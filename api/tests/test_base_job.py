"""Tests for BaseJob ABC and JobContext."""

import pytest
from typing import Optional
from unittest.mock import MagicMock, AsyncMock

from base_job import BaseJob, JobContext


class ConcreteJob(BaseJob):
    """A simple concrete job for testing."""

    def __init__(self, items_to_process: int = 5):
        self.items_to_process = items_to_process
        self.processed = []

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        start = int(cursor) if cursor else 0
        for i in range(start, self.items_to_process):
            if context.is_stop_requested():
                return str(i)
            self.processed.append(i)
        return None


class TestJobContextInitialState:
    """Test JobContext initial state."""

    def test_not_stopped_initially(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)
        assert ctx.is_stop_requested() is False

    def test_job_id_property(self):
        db = MagicMock()
        ctx = JobContext(job_id=42, db=db, queue_manager=None)
        assert ctx.job_id == 42


class TestJobContextRequestStop:
    """Test request_stop() behavior."""

    def test_request_stop_sets_flag(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)
        ctx.request_stop()
        assert ctx.is_stop_requested() is True

    def test_request_stop_idempotent(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)
        ctx.request_stop()
        ctx.request_stop()
        assert ctx.is_stop_requested() is True


class TestJobContextCheckpoint:
    """Test checkpoint() calls db correctly."""

    @pytest.mark.asyncio
    async def test_checkpoint_calls_db(self):
        db = MagicMock()
        ctx = JobContext(job_id=7, db=db, queue_manager=None)
        await ctx.checkpoint(cursor="page_3", items_processed=30)
        db.update_job_progress.assert_called_once_with(
            7, items_processed=30, cursor="page_3"
        )


class TestJobContextReportProgress:
    """Test report_progress() calls db correctly."""

    @pytest.mark.asyncio
    async def test_report_progress_with_total(self):
        db = MagicMock()
        ctx = JobContext(job_id=5, db=db, queue_manager=None)
        await ctx.report_progress(items_processed=10, items_total=100)
        db.update_job_progress.assert_called_once_with(
            5, items_processed=10, items_total=100
        )

    @pytest.mark.asyncio
    async def test_report_progress_without_total(self):
        db = MagicMock()
        ctx = JobContext(job_id=5, db=db, queue_manager=None)
        await ctx.report_progress(items_processed=10)
        db.update_job_progress.assert_called_once_with(
            5, items_processed=10, items_total=None
        )


class TestJobContextShouldYield:
    """Test should_yield() delegation."""

    @pytest.mark.asyncio
    async def test_should_yield_returns_false_without_manager(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)
        result = await ctx.should_yield()
        assert result is False

    @pytest.mark.asyncio
    async def test_should_yield_delegates_to_manager(self):
        db = MagicMock()
        manager = MagicMock()
        manager.should_job_yield = AsyncMock(return_value=True)
        ctx = JobContext(job_id=9, db=db, queue_manager=manager)
        result = await ctx.should_yield()
        assert result is True
        manager.should_job_yield.assert_called_once_with(9)

    @pytest.mark.asyncio
    async def test_should_yield_delegates_false(self):
        db = MagicMock()
        manager = MagicMock()
        manager.should_job_yield = AsyncMock(return_value=False)
        ctx = JobContext(job_id=3, db=db, queue_manager=manager)
        result = await ctx.should_yield()
        assert result is False
        manager.should_job_yield.assert_called_once_with(3)


class TestConcreteJob:
    """Test that a BaseJob subclass can run."""

    @pytest.mark.asyncio
    async def test_run_completes_all_items(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)
        job = ConcreteJob(items_to_process=5)
        result = await job.run(ctx)
        assert result is None
        assert job.processed == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_run_resumes_from_cursor(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)
        job = ConcreteJob(items_to_process=5)
        result = await job.run(ctx, cursor="3")
        assert result is None
        assert job.processed == [3, 4]

    @pytest.mark.asyncio
    async def test_run_stops_early_when_stop_requested(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)
        ctx.request_stop()
        job = ConcreteJob(items_to_process=5)
        result = await job.run(ctx)
        assert result == "0"
        assert job.processed == []

    @pytest.mark.asyncio
    async def test_run_stops_midway(self):
        db = MagicMock()
        ctx = JobContext(job_id=1, db=db, queue_manager=None)

        class StopAfterTwo(BaseJob):
            def __init__(self):
                self.processed = []

            async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
                for i in range(5):
                    if context.is_stop_requested():
                        return str(i)
                    self.processed.append(i)
                    if i == 1:
                        context.request_stop()
                return None

        job2 = StopAfterTwo()
        result = await job2.run(ctx)
        # Items 0 and 1 processed, stop requested after 1, checked at 2
        assert result == "2"
        assert job2.processed == [0, 1]

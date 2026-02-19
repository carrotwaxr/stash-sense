"""Tests for QueueManager dispatch loop."""
import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from queue_manager import QueueManager
from base_job import BaseJob, JobContext
from job_models import ResourceType, JOB_REGISTRY, JobDefinition, JobPriority


class FakeJob(BaseJob):
    """A job that completes immediately."""
    def __init__(self):
        self.ran = False
    async def run(self, context, cursor=None):
        self.ran = True
        await context.report_progress(1, 1)
        return None


class SlowJob(BaseJob):
    """A job that takes a while."""
    def __init__(self):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
    async def run(self, context, cursor=None):
        self.started.set()
        await self.finish.wait()
        return None


class TestQueueDispatch:
    @pytest.fixture
    def db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(str(tmp_path / "test.db"))

    @pytest.fixture
    def mgr(self, db):
        return QueueManager(db)

    @pytest.mark.asyncio
    async def test_dispatch_runs_queued_job(self, mgr):
        fake = FakeJob()
        with patch.object(mgr, '_create_job_instance', return_value=fake):
            mgr.submit("duplicate_performer", triggered_by="user")
            await mgr._dispatch_once()
            # Give the task a moment to complete
            await asyncio.sleep(0.05)
            await mgr._check_completed()
        assert fake.ran

    @pytest.mark.asyncio
    async def test_dispatch_respects_resource_slots(self, mgr):
        """Only one GPU job at a time."""
        slow1 = SlowJob()
        slow2 = SlowJob()
        jobs = [slow1, slow2]
        call_count = 0
        def make_job(type_id):
            nonlocal call_count
            j = jobs[call_count]
            call_count += 1
            return j

        with patch.object(mgr, '_create_job_instance', side_effect=make_job):
            mgr.submit("fingerprint_generation", triggered_by="user")
            mgr.submit("fingerprint_generation", triggered_by="schedule")
            # Second submit returns None (duplicate prevention)
            # So only one GPU job should start
            await mgr._dispatch_once()
            await asyncio.sleep(0.05)
            assert slow1.started.is_set()
            # Only 1 job should be running
            assert len(mgr._running_tasks) == 1
            slow1.finish.set()
            await asyncio.sleep(0.05)
            await mgr._check_completed()

    @pytest.mark.asyncio
    async def test_dispatch_different_resources_parallel(self, mgr):
        """GPU and LIGHT jobs can run in parallel."""
        slow_gpu = SlowJob()
        slow_light = SlowJob()
        instances = {"fingerprint_generation": slow_gpu, "duplicate_performer": slow_light}
        with patch.object(mgr, '_create_job_instance', side_effect=lambda t: instances[t]):
            mgr.submit("fingerprint_generation", triggered_by="user")
            mgr.submit("duplicate_performer", triggered_by="user")
            await mgr._dispatch_once()
            await asyncio.sleep(0.05)
            assert slow_gpu.started.is_set()
            assert slow_light.started.is_set()
            assert len(mgr._running_tasks) == 2
            # Cleanup
            slow_gpu.finish.set()
            slow_light.finish.set()
            await asyncio.sleep(0.05)
            await mgr._check_completed()

    @pytest.mark.asyncio
    async def test_failed_job_records_error(self, mgr, db):
        class FailingJob(BaseJob):
            async def run(self, context, cursor=None):
                raise RuntimeError("boom")

        with patch.object(mgr, '_create_job_instance', return_value=FailingJob()):
            job_id = mgr.submit("duplicate_performer", triggered_by="user")
            await mgr._dispatch_once()
            await asyncio.sleep(0.05)
            await mgr._check_completed()
        job = db.get_job(job_id)
        assert job["status"] == "failed"
        assert "boom" in job["error_message"]

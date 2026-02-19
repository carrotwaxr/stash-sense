"""Queue manager — singleton orchestrator for background jobs."""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from job_models import (
    JOB_REGISTRY, JobDefinition, JobPriority, JobStatus, ResourceType,
)

if TYPE_CHECKING:
    from recommendations_db import RecommendationsDB

logger = logging.getLogger(__name__)

DEFAULT_RESOURCE_SLOTS = {
    ResourceType.GPU: 1,
    ResourceType.CPU_HEAVY: 1,
    ResourceType.NETWORK: 2,
    ResourceType.LIGHT: 3,
}


class QueueManager:
    """Central job queue orchestrator."""

    def __init__(self, db: RecommendationsDB):
        self._db = db
        self.resource_slots = dict(DEFAULT_RESOURCE_SLOTS)
        self._running_tasks: dict[int, asyncio.Task] = {}
        self._running_contexts: dict[int, object] = {}
        self._loop_task: Optional[asyncio.Task] = None
        self.is_shutting_down = False

    # ========================================================================
    # Public API
    # ========================================================================

    def submit(
        self,
        type_id: str,
        triggered_by: str,
        priority: Optional[JobPriority] = None,
        cursor: Optional[str] = None,
    ) -> Optional[int]:
        """Submit a job to the queue. Returns job_id or None if duplicate."""
        defn = JOB_REGISTRY.get(type_id)
        if not defn:
            raise ValueError(f"Unknown job type: {type_id}")
        actual_priority = priority if priority is not None else defn.default_priority
        return self._db.submit_job(
            type=type_id,
            priority=int(actual_priority),
            triggered_by=triggered_by,
            cursor=cursor,
        )

    def get_job(self, job_id: int) -> Optional[dict]:
        """Get a single job."""
        return self._db.get_job(job_id)

    def get_jobs(self, **kwargs) -> list[dict]:
        """Get jobs with filters."""
        return self._db.get_jobs(**kwargs)

    def cancel(self, job_id: int):
        """Cancel a queued job. Stop a running job."""
        job = self._db.get_job(job_id)
        if not job:
            return
        if job["status"] == "queued":
            self._db.cancel_job(job_id)
        elif job["status"] == "running":
            self._request_job_stop(job_id)

    def get_status(self) -> dict:
        """Get queue summary counts."""
        queued = self._db.get_jobs(status="queued")
        running = self._db.get_jobs(status="running")
        return {
            "queued": len(queued),
            "running": len(running),
            "running_jobs": [dict(j) for j in running],
        }

    def request_shutdown(self):
        """Signal all running jobs to stop for graceful shutdown."""
        self.is_shutting_down = True
        for job_id, ctx in self._running_contexts.items():
            ctx.request_stop()
            self._db.set_job_status(job_id, "stopping")
        if self._loop_task and not self._loop_task.done():
            self._loop_task.cancel()

    def recover_on_startup(self):
        """Re-queue any jobs interrupted by a crash."""
        count = self._db.requeue_interrupted_jobs()
        if count:
            logger.warning(f"Re-queued {count} interrupted job(s) for resume")

    # ========================================================================
    # Yield protocol
    # ========================================================================

    async def should_job_yield(self, job_id: int) -> bool:
        """Check if a higher-priority job is waiting for the same resource slot."""
        job = self._db.get_job(job_id)
        if not job:
            return False
        defn = JOB_REGISTRY.get(job["type"])
        if not defn:
            return False
        queued = self._db.get_queued_jobs()
        for queued_job in queued:
            queued_defn = JOB_REGISTRY.get(queued_job["type"])
            if not queued_defn:
                continue
            if queued_defn.resource == defn.resource and queued_job["priority"] < job["priority"]:
                return True
        return False

    # ========================================================================
    # Internal
    # ========================================================================

    def _request_job_stop(self, job_id: int):
        """Request a running job to stop."""
        ctx = self._running_contexts.get(job_id)
        if ctx:
            ctx.request_stop()
        self._db.set_job_status(job_id, "stopping")

    # ========================================================================
    # Dispatch loop
    # ========================================================================

    async def start(self):
        """Start the background dispatch loop."""
        self._loop_task = asyncio.create_task(self._dispatch_loop())

    async def stop(self, timeout: float = 30.0):
        """Stop the dispatch loop and wait for running jobs."""
        self.request_shutdown()
        if self._running_tasks:
            logger.warning(f"Waiting up to {timeout}s for {len(self._running_tasks)} running job(s)...")
            done, pending = await asyncio.wait(
                self._running_tasks.values(), timeout=timeout
            )
            for task in pending:
                task.cancel()
            for job_id in list(self._running_tasks.keys()):
                job = self._db.get_job(job_id)
                if job and job["status"] == "stopping":
                    self._db.set_job_status(job_id, "queued")
                    logger.warning(f"Job {job_id} ({job['type']}) re-queued for resume")
        self._running_tasks.clear()
        self._running_contexts.clear()

    async def _dispatch_loop(self):
        """Main loop — dispatch jobs every second."""
        while not self.is_shutting_down:
            try:
                await self._dispatch_once()
                await self._check_completed()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dispatch loop error: {e}")
            await asyncio.sleep(1.0)

    async def _dispatch_once(self):
        """Check queue and start eligible jobs."""
        if self.is_shutting_down:
            return
        running_per_resource: dict[ResourceType, int] = {r: 0 for r in ResourceType}
        for job_id in list(self._running_tasks.keys()):
            job = self._db.get_job(job_id)
            if job:
                defn = JOB_REGISTRY.get(job["type"])
                if defn:
                    running_per_resource[defn.resource] += 1

        queued = self._db.get_queued_jobs()
        for job_row in queued:
            defn = JOB_REGISTRY.get(job_row["type"])
            if not defn:
                continue
            resource = defn.resource
            max_slots = self.resource_slots.get(resource, 1)
            if running_per_resource[resource] >= max_slots:
                continue
            self._start_job(job_row, defn)
            running_per_resource[resource] += 1

    async def _check_completed(self):
        """Check for completed/failed tasks and clean up."""
        for job_id in list(self._running_tasks.keys()):
            task = self._running_tasks[job_id]
            if task.done():
                self._running_tasks.pop(job_id, None)
                self._running_contexts.pop(job_id, None)
                try:
                    task.result()
                except Exception:
                    pass  # Already handled in _run_job

    def _start_job(self, job_row: dict, defn: JobDefinition):
        """Start a job — create task and track it."""
        from base_job import JobContext
        job_id = job_row["id"]
        self._db.start_job(job_id)
        ctx = JobContext(job_id=job_id, db=self._db, queue_manager=self)
        self._running_contexts[job_id] = ctx
        task = asyncio.create_task(self._run_job(job_id, job_row, ctx))
        self._running_tasks[job_id] = task

    async def _run_job(self, job_id: int, job_row: dict, ctx):
        """Execute a single job and handle completion/failure."""
        type_id = job_row["type"]
        try:
            job_instance = self._create_job_instance(type_id)
            cursor = job_row.get("cursor")
            final_cursor = await job_instance.run(ctx, cursor=cursor)
            job = self._db.get_job(job_id)
            if job and job["status"] == "stopping":
                self._db.set_job_status(job_id, "queued")
                logger.warning(f"Job {job_id} ({type_id}) yielded, re-queued")
            elif job and job["status"] == "running":
                self._db.complete_job(job_id)
                logger.warning(f"Job {job_id} ({type_id}) completed")
        except Exception as e:
            logger.error(f"Job {job_id} ({type_id}) failed: {e}")
            self._db.fail_job(job_id, str(e))

    def _create_job_instance(self, type_id: str):
        """Create a job instance by type. Override in tests."""
        raise NotImplementedError(f"No job implementation registered for {type_id}")

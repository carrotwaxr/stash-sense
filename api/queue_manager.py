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

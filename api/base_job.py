"""BaseJob ABC and JobContext for the operation queue system."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from recommendations_db import RecommendationsDB


class JobContext:
    """Execution context passed to jobs, providing stop signaling, yielding, and progress reporting."""

    def __init__(
        self,
        job_id: int,
        db: RecommendationsDB,
        queue_manager: Any,
        job_type: Optional[str] = None,
    ) -> None:
        self._job_id = job_id
        self._db = db
        self._queue_manager = queue_manager
        self._job_type = job_type
        self._stop_requested = False

    @property
    def job_id(self) -> int:
        return self._job_id

    def request_stop(self) -> None:
        """Signal that this job should stop at its next checkpoint."""
        self._stop_requested = True

    def is_stop_requested(self) -> bool:
        """Check whether a stop has been requested."""
        return self._stop_requested

    def is_shutting_down(self) -> bool:
        """True when the stop comes from the sidecar shutting down rather than from the user.

        A job that returns a cursor then is re-queued and resumes on the next start; one that
        returns None is marked completed.
        """
        return bool(getattr(self._queue_manager, "is_shutting_down", False))

    async def should_yield(self) -> bool:
        """Check whether this job should yield to a higher-priority job.

        Returns False if no queue manager is set; otherwise delegates to the manager.
        """
        if self._queue_manager is None:
            return False
        return await self._queue_manager.should_job_yield(self._job_id)

    async def checkpoint(self, cursor: str, items_processed: int) -> None:
        """Persist progress so the job can be resumed from this point."""
        self._db.update_job_progress(
            self._job_id, items_processed=items_processed, cursor=cursor
        )

    async def run_media_command(self, cmd: list[str], timeout: Optional[float] = None):
        """Run an ffmpeg/ffprobe command for this job.

        CPU-heavy jobs get the media_job_nice / media_job_threads settings
        applied; other jobs run the command unchanged.
        """
        from job_models import JOB_REGISTRY, ResourceType
        from media_exec import media_job_command, run_command

        defn = JOB_REGISTRY.get(self._job_type) if self._job_type else None
        if defn is not None and defn.resource == ResourceType.CPU_HEAVY:
            cmd = media_job_command(cmd)
        return await run_command(cmd, timeout=timeout)

    def set_result(self, result: dict) -> None:
        """Record a short summary of what the job did (the driver API's job result)."""
        self._db.set_job_result(self._job_id, result)

    async def report_progress(self, items_processed: int, items_total: Optional[int] = None) -> None:
        """Report current progress without saving a resumption cursor."""
        self._db.update_job_progress(
            self._job_id, items_processed=items_processed, items_total=items_total
        )


class BaseJob(ABC):
    """Abstract base class for all queue-managed jobs."""

    @abstractmethod
    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        """Execute the job.

        Args:
            context: The JobContext providing stop signals, yielding, and progress reporting.
            cursor: Optional cursor string to resume from a previous checkpoint.

        Returns:
            A cursor string if the job was interrupted and can be resumed, or None if completed.
        """
        ...

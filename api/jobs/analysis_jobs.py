"""Wraps existing analyzers as BaseJob subclasses."""
from __future__ import annotations

import logging
from typing import Optional

from base_job import BaseJob, JobContext
from recommendations_router import ANALYZERS, get_rec_db, get_stash_client

logger = logging.getLogger(__name__)


class AnalysisJob(BaseJob):
    """Generic wrapper that runs any registered analyzer as a queue job."""

    def __init__(self, type_id: str):
        self._type_id = type_id

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return cursor

        db = get_rec_db()
        stash = get_stash_client()
        analyzer_class = ANALYZERS.get(self._type_id)
        if not analyzer_class:
            raise ValueError(f"Unknown analyzer type: {self._type_id}")

        # Create a real analysis_runs entry (FK target for duplicate_candidates.run_id)
        run_id = db.start_analysis_run(self._type_id)
        logger.warning("Starting analysis job %s (run_id=%d, job_id=%d)", self._type_id, run_id, context.job_id)

        # Bridge progress from analyzer → job queue for frontend polling
        def progress_callback(items_processed: int, items_total: Optional[int]) -> None:
            import asyncio
            loop = asyncio.get_event_loop()
            loop.create_task(context.report_progress(items_processed, items_total))

        analyzer = analyzer_class(stash, db, run_id=run_id)
        analyzer._job_progress_callback = progress_callback

        # Wire stop signal from job context to analyzer
        if context.is_stop_requested():
            analyzer.request_stop()

        try:
            result = await analyzer.run(incremental=True)

            db.complete_analysis_run(run_id, result.recommendations_created)
            await context.report_progress(result.items_processed, result.items_processed)
            logger.warning(
                "Analysis job %s completed (run_id=%d): %d processed, %d recommendations",
                self._type_id, run_id, result.items_processed, result.recommendations_created,
            )
        except Exception:
            db.fail_analysis_run(run_id, "Analysis job failed with exception")
            logger.warning("Analysis job %s failed (run_id=%d)", self._type_id, run_id, exc_info=True)
            raise

        return None

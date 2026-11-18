"""Wraps existing analyzers as BaseJob subclasses."""
from __future__ import annotations

import logging
from typing import Optional

import curation_settings
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

        # Bridge progress from analyzer → job queue for frontend polling. Written
        # synchronously: a scheduled write could land after the final count below.
        def progress_callback(items_processed: int, items_total: Optional[int]) -> None:
            db.update_job_progress(context.job_id, items_processed=items_processed, items_total=items_total)

        analyzer = analyzer_class(stash, db, run_id=run_id)
        analyzer._job_progress_callback = progress_callback

        # Wire stop signal from job context to analyzer
        if context.is_stop_requested():
            analyzer.request_stop()

        try:
            # Fresh read: a run never works from an exclusion list older than its start.
            analyzer.exclusion = await curation_settings.current(stash, max_age=0)
            # Incremental runs skip excluded scenes and still move their watermark past
            # them. When the last run excluded scenes that are now allowed, run in full
            # once so those scenes are analyzed.
            seen_key = f"_analysis_excluded_scenes:{self._type_id}"
            previously = set(db.get_user_setting(seen_key) or [])
            released = previously - analyzer.exclusion.scene_ids
            if released:
                logger.warning(
                    "Analysis job %s: %d scene(s) left the exclusion list; running in full",
                    self._type_id, len(released),
                )
            result = await analyzer.run(incremental=not released)
            db.set_user_setting(seen_key, sorted(analyzer.exclusion.scene_ids))
            if getattr(analyzer, "excluded_skipped", 0):
                logger.warning(
                    "Analysis job %s skipped %d recommendation(s) naming excluded scenes",
                    self._type_id, analyzer.excluded_skipped,
                )

            db.complete_analysis_run(run_id, result.recommendations_created)
            await context.report_progress(result.items_processed, result.items_processed)
            context.set_result({
                "analysis_run_id": run_id,
                "items_processed": result.items_processed,
                "recommendations_created": result.recommendations_created,
                "excluded_skipped": int(getattr(analyzer, "excluded_skipped", 0) or 0),
            })
            logger.warning(
                "Analysis job %s completed (run_id=%d): %d processed, %d recommendations",
                self._type_id, run_id, result.items_processed, result.recommendations_created,
            )
        except Exception:
            db.fail_analysis_run(run_id, "Analysis job failed with exception")
            logger.warning("Analysis job %s failed (run_id=%d)", self._type_id, run_id, exc_info=True)
            raise

        return None

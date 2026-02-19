"""Wraps existing analyzers as BaseJob subclasses."""
from __future__ import annotations

from typing import Optional

from base_job import BaseJob, JobContext
from recommendations_router import ANALYZERS, get_rec_db, get_stash_client


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

        analyzer = analyzer_class(stash, db, run_id=None)
        result = await analyzer.run(incremental=True)

        await context.report_progress(
            result.items_processed,
            result.items_processed,
        )
        return None

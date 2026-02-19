"""Fingerprint generation as a queue job."""
from __future__ import annotations

from typing import Optional

from base_job import BaseJob, JobContext
from fingerprint_generator import SceneFingerprintGenerator
from recommendations_router import get_rec_db, get_stash_client


class FingerprintGenerationJob(BaseJob):
    """Wraps SceneFingerprintGenerator as a queue-managed job."""

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return None

        stash = get_stash_client()
        db = get_rec_db()
        generator = SceneFingerprintGenerator(stash_client=stash, rec_db=db)

        async for progress in generator.generate_all():
            await context.report_progress(
                progress.processed_scenes,
                progress.total_scenes,
            )
            if context.is_stop_requested():
                generator.request_stop()
                break

        return None

"""Fingerprint generation as a queue job."""
from __future__ import annotations

from typing import Optional

from base_job import BaseJob, JobContext
from fingerprint_generator import SceneFingerprintGenerator
from recommendations_router import get_db_version, get_rec_db, get_stash_client


class FingerprintGenerationJob(BaseJob):
    """Wraps SceneFingerprintGenerator as a queue-managed job."""

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return None

        db_version = get_db_version()
        if db_version is None:
            raise RuntimeError("No face recognition database loaded; cannot generate fingerprints")

        stash = get_stash_client()
        db = get_rec_db()
        generator = SceneFingerprintGenerator(stash_client=stash, rec_db=db, db_version=db_version)

        async for progress in generator.generate_all():
            await context.report_progress(
                progress.processed_scenes,
                progress.total_scenes,
            )
            if context.is_stop_requested():
                generator.request_stop()
                break

        return None

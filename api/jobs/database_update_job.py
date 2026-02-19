"""Database update as a queue job."""
from __future__ import annotations

from typing import Optional

from base_job import BaseJob, JobContext


def get_db_updater():
    """Get the global DatabaseUpdater instance from main."""
    from main import db_updater
    return db_updater


class DatabaseUpdateJob(BaseJob):
    """Wraps DatabaseUpdater check-and-update as a queue job."""

    async def run(self, context: JobContext, cursor: Optional[str] = None) -> Optional[str]:
        if context.is_stop_requested():
            return None

        updater = get_db_updater()
        if not updater:
            raise RuntimeError("Database updater not initialized")

        result = await updater.check_for_update()
        if not result.get("update_available"):
            return None

        await updater.start_update(
            result["download_url"],
            result["latest_version"],
        )
        return None

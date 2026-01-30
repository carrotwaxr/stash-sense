"""Tests for database.py iter_performers_with_site_urls method."""

import pytest


class TestIterPerformersWithSiteUrls:
    """Tests for iter_performers_with_site_urls."""

    def test_returns_performers_with_matching_urls(self, tmp_path):
        from database import PerformerDatabase

        db = PerformerDatabase(tmp_path / "test.db")

        # Create performers with different URLs
        pid1 = db.add_performer("Mia Malkova")
        db.add_url(pid1, "https://www.babepedia.com/babe/Mia_Malkova", "stashdb")

        pid2 = db.add_performer("Angela White")
        db.add_url(pid2, "https://www.iafd.com/person.rme/perfid=angelawhite/Angela-White.htm", "stashdb")

        pid3 = db.add_performer("No URLs")
        # No URL added

        # Query for babepedia URLs
        results = list(db.iter_performers_with_site_urls("babepedia"))

        assert len(results) == 1
        assert results[0][0] == pid1  # performer_id
        assert results[0][1] == "Mia Malkova"  # name
        assert "babepedia.com" in results[0][2]  # url

    def test_respects_after_id_for_resume(self, tmp_path):
        from database import PerformerDatabase

        db = PerformerDatabase(tmp_path / "test.db")

        pid1 = db.add_performer("Performer A")
        db.add_url(pid1, "https://www.babepedia.com/babe/A", "stashdb")

        pid2 = db.add_performer("Performer B")
        db.add_url(pid2, "https://www.babepedia.com/babe/B", "stashdb")

        pid3 = db.add_performer("Performer C")
        db.add_url(pid3, "https://www.babepedia.com/babe/C", "stashdb")

        # Resume after pid1
        results = list(db.iter_performers_with_site_urls("babepedia", after_id=pid1))

        assert len(results) == 2
        assert results[0][0] == pid2
        assert results[1][0] == pid3

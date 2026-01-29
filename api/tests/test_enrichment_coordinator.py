"""Tests for enrichment coordinator."""
import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch


class TestEnrichmentCoordinator:
    """Test coordinator orchestrates scrapers."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create mock database."""
        from database import PerformerDatabase
        db = PerformerDatabase(tmp_path / "test.db")
        return db

    @pytest.fixture
    def mock_scraper(self):
        """Create mock scraper."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class MockScraper(BaseScraper):
            source_name = "mock"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)
                self.performers = [
                    ScrapedPerformer(
                        id=f"p{i}",
                        name=f"Performer {i}",
                        image_urls=[f"https://example.com/p{i}.jpg"],
                        gender="FEMALE",
                        stash_ids={"mock": f"p{i}"},
                    )
                    for i in range(5)
                ]

            def get_performer(self, performer_id):
                for p in self.performers:
                    if p.id == performer_id:
                        return p
                return None

            def query_performers(self, page=1, per_page=25):
                start = (page - 1) * per_page
                end = start + per_page
                return (len(self.performers), self.performers[start:end])

            def download_image(self, url, max_retries=3):
                # Return fake image bytes
                return b"fake_image_data"

        return MockScraper()

    @pytest.mark.asyncio
    async def test_coordinator_processes_performers(self, mock_db, mock_scraper):
        """Coordinator processes all performers from a scraper."""
        from enrichment_coordinator import EnrichmentCoordinator

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        await coordinator.run()

        # Check that performers were created
        assert mock_db.get_stats()['performer_count'] == 5

    @pytest.mark.asyncio
    async def test_coordinator_uses_write_queue(self, mock_db, mock_scraper):
        """Coordinator writes through the write queue."""
        from enrichment_coordinator import EnrichmentCoordinator

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        await coordinator.run()

        # Queue should have processed messages
        assert coordinator.write_queue.stats.processed >= 5

    @pytest.mark.asyncio
    async def test_coordinator_saves_progress(self, mock_db, mock_scraper):
        """Coordinator saves progress for resume."""
        from enrichment_coordinator import EnrichmentCoordinator

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        await coordinator.run()

        # Check progress was saved
        progress = mock_db.get_scrape_progress("mock")
        assert progress is not None
        assert progress['performers_processed'] == 5

    @pytest.mark.asyncio
    async def test_coordinator_resumes_from_checkpoint(self, mock_db, mock_scraper):
        """Coordinator resumes from last processed ID."""
        from enrichment_coordinator import EnrichmentCoordinator

        # Simulate previous progress - processed first 2 performers
        mock_db.save_scrape_progress(
            source="mock",
            last_processed_id="p1",
            performers_processed=2,
        )

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        await coordinator.run()

        # Should have processed p2, p3, p4 (3 more)
        progress = mock_db.get_scrape_progress("mock")
        # The total should be the new count from this run
        assert progress['performers_processed'] == 3

    @pytest.mark.asyncio
    async def test_coordinator_handles_scraper_errors(self, mock_db):
        """Coordinator continues despite individual scraper errors."""
        from enrichment_coordinator import EnrichmentCoordinator
        from base_scraper import BaseScraper, ScrapedPerformer

        class FailingScraper(BaseScraper):
            source_name = "failing"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                raise Exception("Simulated failure")

            def download_image(self, url, max_retries=3):
                return None

        class WorkingScraper(BaseScraper):
            source_name = "working"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                if page > 1:
                    return (2, [])
                return (2, [
                    ScrapedPerformer(id="w1", name="Working 1", stash_ids={"working": "w1"}),
                    ScrapedPerformer(id="w2", name="Working 2", stash_ids={"working": "w2"}),
                ])

            def download_image(self, url, max_retries=3):
                return None

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[FailingScraper(), WorkingScraper()],
        )

        # Should not raise despite failing scraper
        await coordinator.run()

        # Working scraper should have succeeded
        assert mock_db.get_stats()['performer_count'] == 2

    @pytest.mark.asyncio
    async def test_coordinator_tracks_stats(self, mock_db, mock_scraper):
        """Coordinator tracks processing statistics."""
        from enrichment_coordinator import EnrichmentCoordinator

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        await coordinator.run()

        assert coordinator.stats.performers_processed == 5
        assert coordinator.stats.by_source.get("mock", 0) == 5

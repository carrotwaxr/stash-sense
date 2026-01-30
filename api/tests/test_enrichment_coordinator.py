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

    @pytest.mark.asyncio
    async def test_coordinator_processes_images_for_faces(self, mock_db, tmp_path):
        """Coordinator downloads images and extracts faces."""
        from enrichment_coordinator import EnrichmentCoordinator
        from base_scraper import BaseScraper, ScrapedPerformer
        from unittest.mock import patch
        import numpy as np
        import voyager

        # Create mock scraper
        class ImageScraper(BaseScraper):
            source_name = "imagesource"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                if page > 1:
                    return (1, [])
                return (1, [
                    ScrapedPerformer(
                        id="p1",
                        name="Test Performer",
                        image_urls=["https://example.com/face1.jpg"],
                        stash_ids={"imagesource": "p1"},
                    ),
                ])

            def download_image(self, url, max_retries=3):
                from PIL import Image
                import io
                img = Image.new("RGB", (500, 500), color="white")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()

        # Create empty indices
        facenet = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        arcface = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        facenet.add_item(np.zeros(512, dtype=np.float32))  # Need at least 1 item
        arcface.add_item(np.zeros(512, dtype=np.float32))
        facenet.save(str(tmp_path / "face_facenet512.voy"))
        arcface.save(str(tmp_path / "face_arcface.voy"))

        with patch('face_processor.FaceProcessor.process_image') as mock_process:
            from face_processor import ProcessedFace
            from embeddings import FaceEmbedding

            mock_process.return_value = [ProcessedFace(
                embedding=FaceEmbedding(
                    facenet=np.random.rand(512).astype(np.float32),
                    arcface=np.random.rand(512).astype(np.float32),
                ),
                quality_score=0.9,
                bbox={"x": 100, "y": 100, "w": 150, "h": 150},
            )]

            coordinator = EnrichmentCoordinator(
                database=mock_db,
                scrapers=[ImageScraper()],
                data_dir=tmp_path,
                enable_face_processing=True,
                source_trust_levels={"imagesource": "high"},
            )

            await coordinator.run()

        assert coordinator.stats.images_processed >= 1
        assert coordinator.stats.faces_added >= 1

    @pytest.mark.asyncio
    async def test_coordinator_respects_face_limits(self, mock_db, tmp_path):
        """Coordinator stops processing when face limit reached."""
        from enrichment_coordinator import EnrichmentCoordinator
        from base_scraper import BaseScraper, ScrapedPerformer
        from unittest.mock import patch
        import numpy as np
        import voyager

        # Create mock scraper with many images
        class ImageScraper(BaseScraper):
            source_name = "imagesource"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                if page > 1:
                    return (1, [])
                return (1, [
                    ScrapedPerformer(
                        id="p1",
                        name="Test Performer",
                        # More images than the limit
                        image_urls=[f"https://example.com/face{i}.jpg" for i in range(10)],
                        stash_ids={"imagesource": "p1"},
                    ),
                ])

            def download_image(self, url, max_retries=3):
                from PIL import Image
                import io
                img = Image.new("RGB", (500, 500), color="white")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()

        # Create empty indices
        facenet = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        arcface = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        facenet.add_item(np.zeros(512, dtype=np.float32))  # Need at least 1 item
        arcface.add_item(np.zeros(512, dtype=np.float32))
        facenet.save(str(tmp_path / "face_facenet512.voy"))
        arcface.save(str(tmp_path / "face_arcface.voy"))

        with patch('face_processor.FaceProcessor.process_image') as mock_process:
            from face_processor import ProcessedFace
            from embeddings import FaceEmbedding

            # Return a new face for each call
            def make_face(*args, **kwargs):
                return [ProcessedFace(
                    embedding=FaceEmbedding(
                        facenet=np.random.rand(512).astype(np.float32),
                        arcface=np.random.rand(512).astype(np.float32),
                    ),
                    quality_score=0.9,
                    bbox={"x": 100, "y": 100, "w": 150, "h": 150},
                )]

            mock_process.side_effect = make_face

            coordinator = EnrichmentCoordinator(
                database=mock_db,
                scrapers=[ImageScraper()],
                data_dir=tmp_path,
                max_faces_per_source=3,  # Limit to 3 faces per source
                max_faces_total=20,
                enable_face_processing=True,
                source_trust_levels={"imagesource": "high"},
            )

            await coordinator.run()

        # Should have stopped at 3 faces due to per-source limit
        assert coordinator.stats.faces_added == 3

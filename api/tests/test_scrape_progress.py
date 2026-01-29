"""Tests for scrape progress tracking."""
import pytest
import tempfile
from pathlib import Path


class TestScrapeProgress:
    """Test scrape progress persistence."""

    @pytest.fixture
    def db(self):
        from database import PerformerDatabase

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = PerformerDatabase(db_path)
        yield db

        Path(db_path).unlink(missing_ok=True)

    def test_save_and_load_progress(self, db):
        """Can save and retrieve scrape progress."""
        db.save_scrape_progress(
            source="stashdb",
            last_processed_id="abc-123",
            performers_processed=1000,
            faces_added=2500,
            errors=5,
        )

        progress = db.get_scrape_progress("stashdb")

        assert progress is not None
        assert progress["last_processed_id"] == "abc-123"
        assert progress["performers_processed"] == 1000
        assert progress["faces_added"] == 2500
        assert progress["errors"] == 5

    def test_update_progress(self, db):
        """Progress can be updated."""
        db.save_scrape_progress(source="babepedia", last_processed_id="p1", performers_processed=100)
        db.save_scrape_progress(source="babepedia", last_processed_id="p2", performers_processed=200)

        progress = db.get_scrape_progress("babepedia")

        assert progress["last_processed_id"] == "p2"
        assert progress["performers_processed"] == 200

    def test_no_progress_returns_none(self, db):
        """Unknown source returns None."""
        progress = db.get_scrape_progress("unknown_source")
        assert progress is None

    def test_clear_progress(self, db):
        """Can clear progress for a source."""
        db.save_scrape_progress(source="test", last_processed_id="x", performers_processed=50)
        db.clear_scrape_progress("test")

        progress = db.get_scrape_progress("test")
        assert progress is None

    def test_multiple_sources_independent(self, db):
        """Progress for different sources is tracked independently."""
        db.save_scrape_progress(source="stashdb", last_processed_id="s1", performers_processed=100)
        db.save_scrape_progress(source="theporndb", last_processed_id="t1", performers_processed=200)
        db.save_scrape_progress(source="babepedia", last_processed_id="b1", performers_processed=300)

        assert db.get_scrape_progress("stashdb")["performers_processed"] == 100
        assert db.get_scrape_progress("theporndb")["performers_processed"] == 200
        assert db.get_scrape_progress("babepedia")["performers_processed"] == 300

    def test_get_all_progress(self, db):
        """Can get progress for all sources."""
        db.save_scrape_progress(source="stashdb", last_processed_id="s1", performers_processed=100)
        db.save_scrape_progress(source="theporndb", last_processed_id="t1", performers_processed=200)

        all_progress = db.get_all_scrape_progress()

        assert len(all_progress) == 2
        assert "stashdb" in all_progress
        assert "theporndb" in all_progress

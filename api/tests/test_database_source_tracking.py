"""Tests for per-source face tracking in database."""
import pytest
import tempfile
from pathlib import Path


class TestSourceTracking:
    """Test source tracking for faces."""

    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        from database import PerformerDatabase

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = PerformerDatabase(db_path)
        yield db

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_get_face_count_by_source(self, db):
        """Can get face count per source for a performer."""
        # Create performer
        performer_id = db.add_performer(
            canonical_name="Test Performer",
            gender="FEMALE",
        )
        db.add_stashbox_id(performer_id, "stashdb", "test-uuid")

        # Add faces from different sources
        db.add_face(performer_id, facenet_index=0, arcface_index=0, image_url="url1", source_endpoint="stashdb")
        db.add_face(performer_id, facenet_index=1, arcface_index=1, image_url="url2", source_endpoint="stashdb")
        db.add_face(performer_id, facenet_index=2, arcface_index=2, image_url="url3", source_endpoint="babepedia")
        db.add_face(performer_id, facenet_index=3, arcface_index=3, image_url="url4", source_endpoint="babepedia")
        db.add_face(performer_id, facenet_index=4, arcface_index=4, image_url="url5", source_endpoint="freeones")

        counts = db.get_face_counts_by_source(performer_id)

        assert counts["stashdb"] == 2
        assert counts["babepedia"] == 2
        assert counts["freeones"] == 1

    def test_check_source_limit(self, db):
        """Can check if source limit is reached."""
        performer_id = db.add_performer(canonical_name="Test", gender="FEMALE")

        # Add 5 faces from stashdb
        for i in range(5):
            db.add_face(performer_id, facenet_index=i, arcface_index=i, image_url=f"url{i}", source_endpoint="stashdb")

        # Check limits
        assert db.source_limit_reached(performer_id, "stashdb", max_faces=5) is True
        assert db.source_limit_reached(performer_id, "stashdb", max_faces=10) is False
        assert db.source_limit_reached(performer_id, "babepedia", max_faces=5) is False

    def test_get_performer_faces(self, db):
        """Faces are returned with their source."""
        performer_id = db.add_performer(canonical_name="Test", gender="FEMALE")

        db.add_face(
            performer_id,
            facenet_index=0,
            arcface_index=0,
            image_url="https://example.com/image.jpg",
            source_endpoint="stashdb",
            quality_score=0.95,
        )

        faces = db.get_faces(performer_id)
        assert len(faces) == 1
        assert faces[0].source_endpoint == "stashdb"
        assert faces[0].quality_score == 0.95
        assert faces[0].image_url == "https://example.com/image.jpg"

    def test_total_limit_check(self, db):
        """Can check if total face limit is reached."""
        performer_id = db.add_performer(canonical_name="Test", gender="FEMALE")

        # Add faces from multiple sources
        for i in range(3):
            db.add_face(performer_id, facenet_index=i, arcface_index=i, image_url=f"url{i}", source_endpoint="stashdb")
        for i in range(3, 6):
            db.add_face(performer_id, facenet_index=i, arcface_index=i, image_url=f"url{i}", source_endpoint="babepedia")

        assert db.total_limit_reached(performer_id, max_faces=5) is True
        assert db.total_limit_reached(performer_id, max_faces=6) is True
        assert db.total_limit_reached(performer_id, max_faces=10) is False

    def test_face_count_empty(self, db):
        """Empty face counts returns empty dict."""
        performer_id = db.add_performer(canonical_name="Test", gender="FEMALE")

        counts = db.get_face_counts_by_source(performer_id)
        assert counts == {}

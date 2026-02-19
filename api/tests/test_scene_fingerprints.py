"""Tests for scene fingerprint storage in recommendations DB."""



class TestSceneFingerprintSchema:
    """Tests for scene fingerprint table operations."""

    def test_create_fingerprint(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_scene_fingerprint(
            stash_scene_id=123,
            total_faces=5,
            frames_analyzed=40,
            fingerprint_status="complete",
        )

        assert fp_id is not None
        assert fp_id > 0

    def test_get_fingerprint_by_scene_id(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_scene_fingerprint(
            stash_scene_id=456,
            total_faces=3,
            frames_analyzed=40,
        )

        fp = db.get_scene_fingerprint(stash_scene_id=456)

        assert fp is not None
        assert fp["stash_scene_id"] == 456
        assert fp["total_faces"] == 3
        assert fp["frames_analyzed"] == 40

    def test_add_fingerprint_face(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_scene_fingerprint(stash_scene_id=789, total_faces=2, frames_analyzed=40)

        db.add_fingerprint_face(
            fingerprint_id=fp_id,
            performer_id="stashdb:abc-123",
            face_count=10,
            avg_confidence=0.85,
            proportion=0.5,
        )

        faces = db.get_fingerprint_faces(fp_id)

        assert len(faces) == 1
        assert faces[0]["performer_id"] == "stashdb:abc-123"
        assert faces[0]["face_count"] == 10
        assert faces[0]["proportion"] == 0.5

    def test_get_all_fingerprints(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_scene_fingerprint(stash_scene_id=1, total_faces=2, frames_analyzed=40)
        db.create_scene_fingerprint(stash_scene_id=2, total_faces=3, frames_analyzed=40)
        db.create_scene_fingerprint(stash_scene_id=3, total_faces=0, frames_analyzed=40)

        fps = db.get_all_scene_fingerprints()

        assert len(fps) == 3

    def test_fingerprint_upsert_updates_existing(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id1 = db.create_scene_fingerprint(stash_scene_id=100, total_faces=2, frames_analyzed=40)
        fp_id2 = db.create_scene_fingerprint(stash_scene_id=100, total_faces=5, frames_analyzed=40)

        # Should update, not create new
        assert fp_id1 == fp_id2

        fp = db.get_scene_fingerprint(stash_scene_id=100)
        assert fp["total_faces"] == 5

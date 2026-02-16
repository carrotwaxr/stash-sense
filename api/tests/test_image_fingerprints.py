"""Tests for image fingerprint storage in recommendations DB."""

import sqlite3
import pytest
from pathlib import Path


class TestImageFingerprintSchema:
    """Tests for image fingerprint table operations."""

    def test_create_image_fingerprint(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_image_fingerprint(
            stash_image_id="img-001",
            gallery_id="gallery-1",
            faces_detected=3,
            db_version="v1.0",
        )

        assert fp_id is not None
        assert fp_id > 0

    def test_create_image_fingerprint_without_gallery(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_image_fingerprint(
            stash_image_id="img-002",
            gallery_id=None,
            faces_detected=1,
        )

        assert fp_id is not None
        fp = db.get_image_fingerprint("img-002")
        assert fp is not None
        assert fp["gallery_id"] is None

    def test_create_image_fingerprint_upsert(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id1 = db.create_image_fingerprint(
            stash_image_id="img-100",
            gallery_id="gallery-1",
            faces_detected=2,
            db_version="v1.0",
        )
        fp_id2 = db.create_image_fingerprint(
            stash_image_id="img-100",
            gallery_id="gallery-1",
            faces_detected=5,
            db_version="v1.1",
        )

        # Should update, not create new
        assert fp_id1 == fp_id2

        fp = db.get_image_fingerprint("img-100")
        assert fp["faces_detected"] == 5
        assert fp["db_version"] == "v1.1"

    def test_get_image_fingerprint(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(
            stash_image_id="img-456",
            gallery_id="gallery-2",
            faces_detected=3,
            db_version="v1.0",
        )

        fp = db.get_image_fingerprint("img-456")

        assert fp is not None
        assert fp["stash_image_id"] == "img-456"
        assert fp["gallery_id"] == "gallery-2"
        assert fp["faces_detected"] == 3
        assert fp["db_version"] == "v1.0"
        assert fp["created_at"] is not None
        assert fp["updated_at"] is not None

    def test_get_image_fingerprint_not_found(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp = db.get_image_fingerprint("nonexistent")

        assert fp is None

    def test_get_gallery_image_fingerprints(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="img-1", gallery_id="gallery-A", faces_detected=1)
        db.create_image_fingerprint(stash_image_id="img-2", gallery_id="gallery-A", faces_detected=2)
        db.create_image_fingerprint(stash_image_id="img-3", gallery_id="gallery-B", faces_detected=0)

        fps = db.get_gallery_image_fingerprints("gallery-A")

        assert len(fps) == 2
        image_ids = {fp["stash_image_id"] for fp in fps}
        assert image_ids == {"img-1", "img-2"}

    def test_get_gallery_image_fingerprints_empty(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fps = db.get_gallery_image_fingerprints("nonexistent-gallery")

        assert fps == []

    def test_add_image_fingerprint_face(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="img-789", gallery_id="gallery-1", faces_detected=1)

        face_id = db.add_image_fingerprint_face(
            stash_image_id="img-789",
            performer_id="performer-abc",
            confidence=0.95,
            distance=0.32,
            bbox_x=100.0,
            bbox_y=50.0,
            bbox_w=200.0,
            bbox_h=250.0,
        )

        assert face_id is not None
        assert face_id > 0

    def test_add_image_fingerprint_face_upsert(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="img-upsert", gallery_id="gallery-1", faces_detected=1)

        face_id1 = db.add_image_fingerprint_face(
            stash_image_id="img-upsert",
            performer_id="performer-1",
            confidence=0.80,
            distance=0.45,
            bbox_x=10.0, bbox_y=20.0, bbox_w=30.0, bbox_h=40.0,
        )
        face_id2 = db.add_image_fingerprint_face(
            stash_image_id="img-upsert",
            performer_id="performer-1",
            confidence=0.95,
            distance=0.25,
            bbox_x=15.0, bbox_y=25.0, bbox_w=35.0, bbox_h=45.0,
        )

        # Should update, not create new
        assert face_id1 == face_id2

        faces = db.get_image_fingerprint_faces("img-upsert")
        assert len(faces) == 1
        assert faces[0]["confidence"] == 0.95
        assert faces[0]["distance"] == 0.25

    def test_get_image_fingerprint_faces(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="img-faces", gallery_id="gallery-1", faces_detected=2)

        db.add_image_fingerprint_face(
            stash_image_id="img-faces",
            performer_id="performer-1",
            confidence=0.95,
            distance=0.30,
            bbox_x=10.0, bbox_y=20.0, bbox_w=100.0, bbox_h=120.0,
        )
        db.add_image_fingerprint_face(
            stash_image_id="img-faces",
            performer_id="performer-2",
            confidence=0.85,
            distance=0.40,
            bbox_x=300.0, bbox_y=50.0, bbox_w=90.0, bbox_h=110.0,
        )

        faces = db.get_image_fingerprint_faces("img-faces")

        assert len(faces) == 2
        performer_ids = {f["performer_id"] for f in faces}
        assert performer_ids == {"performer-1", "performer-2"}

        # Check all fields present
        face = next(f for f in faces if f["performer_id"] == "performer-1")
        assert face["confidence"] == 0.95
        assert face["distance"] == 0.30
        assert face["bbox_x"] == 10.0
        assert face["bbox_y"] == 20.0
        assert face["bbox_w"] == 100.0
        assert face["bbox_h"] == 120.0
        assert face["created_at"] is not None

    def test_get_image_fingerprint_faces_empty(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="img-empty", gallery_id="gallery-1", faces_detected=0)

        faces = db.get_image_fingerprint_faces("img-empty")

        assert faces == []

    def test_delete_image_fingerprint_faces(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="img-del", gallery_id="gallery-1", faces_detected=2)
        db.add_image_fingerprint_face(
            stash_image_id="img-del", performer_id="p1",
            confidence=0.9, distance=0.3,
            bbox_x=0, bbox_y=0, bbox_w=50, bbox_h=50,
        )
        db.add_image_fingerprint_face(
            stash_image_id="img-del", performer_id="p2",
            confidence=0.8, distance=0.4,
            bbox_x=100, bbox_y=100, bbox_w=50, bbox_h=50,
        )

        deleted_count = db.delete_image_fingerprint_faces("img-del")

        assert deleted_count == 2
        assert db.get_image_fingerprint_faces("img-del") == []

    def test_delete_image_fingerprint_faces_none(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="img-nodel", gallery_id="gallery-1", faces_detected=0)

        deleted_count = db.delete_image_fingerprint_faces("img-nodel")

        assert deleted_count == 0


class TestImageFingerprintMigration:
    """Tests for schema migration from v5 to v6."""

    def _create_v5_database(self, db_path: Path):
        """Create a minimal v5 database with just the schema_version table and core tables."""
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript("""
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            );
            INSERT INTO schema_version (version) VALUES (5);

            CREATE TABLE recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details JSON NOT NULL,
                resolution_action TEXT,
                resolution_details JSON,
                resolved_at TEXT,
                confidence REAL,
                source_analysis_id INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(type, target_type, target_id)
            );

            CREATE TABLE analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                items_total INTEGER,
                items_processed INTEGER,
                recommendations_created INTEGER DEFAULT 0,
                cursor TEXT,
                error_message TEXT
            );

            CREATE TABLE recommendation_settings (
                type TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 1,
                auto_dismiss_threshold REAL,
                notify INTEGER DEFAULT 1,
                interval_hours INTEGER,
                last_run_at TEXT,
                next_run_at TEXT,
                config JSON
            );

            CREATE TABLE dismissed_targets (
                type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                dismissed_at TEXT DEFAULT (datetime('now')),
                reason TEXT,
                permanent INTEGER DEFAULT 0,
                PRIMARY KEY (type, target_type, target_id)
            );

            CREATE TABLE analysis_watermarks (
                type TEXT PRIMARY KEY,
                last_completed_at TEXT,
                last_cursor TEXT,
                last_stash_updated_at TEXT
            );

            CREATE TABLE scene_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_scene_id INTEGER NOT NULL UNIQUE,
                total_faces INTEGER NOT NULL DEFAULT 0,
                frames_analyzed INTEGER NOT NULL DEFAULT 0,
                fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                db_version TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE scene_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL,
                proportion REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(fingerprint_id, performer_id)
            );

            CREATE TABLE upstream_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                local_entity_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                stash_box_id TEXT NOT NULL,
                upstream_data JSON NOT NULL,
                upstream_updated_at TEXT NOT NULL,
                fetched_at TEXT DEFAULT (datetime('now')),
                UNIQUE(entity_type, endpoint, stash_box_id)
            );

            CREATE TABLE upstream_field_config (
                endpoint TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                PRIMARY KEY (endpoint, entity_type, field_name)
            );

            CREATE TABLE user_settings (
                key TEXT PRIMARY KEY,
                value JSON NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            INSERT INTO user_settings (key, value) VALUES ('normalize_enum_display', 'true');
        """)
        conn.commit()
        conn.close()

    def test_migration_v5_to_v6_creates_tables(self, tmp_path):
        """Migration from v5 should create image_fingerprints and image_fingerprint_faces tables."""
        from recommendations_db import RecommendationsDB

        db_path = tmp_path / "migrate.db"
        self._create_v5_database(db_path)

        # Opening the DB should trigger migration
        db = RecommendationsDB(db_path)

        # Verify schema version is now current
        from recommendations_db import SCHEMA_VERSION
        conn = sqlite3.connect(db_path)
        version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
        conn.close()
        assert version == SCHEMA_VERSION

    def test_migration_v5_to_v6_tables_exist(self, tmp_path):
        """After migration, image fingerprint tables should exist and be functional."""
        from recommendations_db import RecommendationsDB

        db_path = tmp_path / "migrate.db"
        self._create_v5_database(db_path)

        db = RecommendationsDB(db_path)

        # Should be able to use the new tables
        fp_id = db.create_image_fingerprint(
            stash_image_id="migrated-img-1",
            gallery_id="gallery-1",
            faces_detected=2,
            db_version="v1.0",
        )
        assert fp_id > 0

        face_id = db.add_image_fingerprint_face(
            stash_image_id="migrated-img-1",
            performer_id="perf-1",
            confidence=0.9,
            distance=0.3,
            bbox_x=10, bbox_y=20, bbox_w=30, bbox_h=40,
        )
        assert face_id > 0

    def test_migration_preserves_existing_data(self, tmp_path):
        """Migration should not affect existing v5 data."""
        from recommendations_db import RecommendationsDB

        db_path = tmp_path / "migrate.db"
        self._create_v5_database(db_path)

        # Insert some v5 data before migration
        conn = sqlite3.connect(db_path)
        conn.execute(
            "INSERT INTO scene_fingerprints (stash_scene_id, total_faces, frames_analyzed) VALUES (999, 5, 40)"
        )
        conn.commit()
        conn.close()

        # Trigger migration
        db = RecommendationsDB(db_path)

        # Existing data should still be there
        fp = db.get_scene_fingerprint(999)
        assert fp is not None
        assert fp["total_faces"] == 5

    def test_migration_indexes_created(self, tmp_path):
        """Migration should create the expected indexes."""
        from recommendations_db import RecommendationsDB

        db_path = tmp_path / "migrate.db"
        self._create_v5_database(db_path)

        db = RecommendationsDB(db_path)

        conn = sqlite3.connect(db_path)
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_image_fp%'"
        ).fetchall()
        conn.close()

        index_names = {row[0] for row in indexes}
        assert "idx_image_fp_gallery" in index_names
        assert "idx_image_fp_faces_image" in index_names
        assert "idx_image_fp_faces_performer" in index_names

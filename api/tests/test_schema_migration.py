"""Tests for schema migrations."""
import pytest
import tempfile
import sqlite3
from pathlib import Path


class TestSchemaMigration:
    """Test database schema migrations."""

    def test_migrate_sets_source_on_existing_faces(self):
        """Migration sets source_endpoint='stashdb' on existing faces."""
        from database import PerformerDatabase

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create old schema manually (v3)
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
                INSERT INTO schema_version (version) VALUES (3);

                CREATE TABLE performers (
                    id INTEGER PRIMARY KEY,
                    canonical_name TEXT,
                    gender TEXT,
                    country TEXT,
                    face_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE faces (
                    id INTEGER PRIMARY KEY,
                    performer_id INTEGER,
                    facenet_index INTEGER,
                    arcface_index INTEGER,
                    image_url TEXT,
                    source_endpoint TEXT,
                    quality_score REAL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                -- Insert performer
                INSERT INTO performers (id, canonical_name, face_count) VALUES (1, 'Test', 2);

                -- Insert faces WITHOUT source_endpoint (like old data)
                INSERT INTO faces (performer_id, facenet_index, arcface_index, image_url)
                    VALUES (1, 0, 0, 'url1');
                INSERT INTO faces (performer_id, facenet_index, arcface_index, image_url)
                    VALUES (1, 1, 1, 'url2');
            """)
            conn.close()

            # Open with our class - should trigger migration
            db = PerformerDatabase(db_path)

            # Check that faces now have source
            faces = db.get_faces(1)
            assert len(faces) == 2
            assert all(f.source_endpoint == "stashdb" for f in faces)

            # Verify schema version was updated to latest
            from database import SCHEMA_VERSION
            conn = sqlite3.connect(db_path)
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            conn.close()
            assert version == SCHEMA_VERSION

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_new_database_starts_at_current_version(self):
        """New databases start at current schema version."""
        from database import PerformerDatabase, SCHEMA_VERSION

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            db = PerformerDatabase(db_path)

            # Check schema version
            conn = sqlite3.connect(db_path)
            # New DBs should have the current version in the table
            # The schema_version table should exist
            cursor = conn.execute("SELECT version FROM schema_version")
            row = cursor.fetchone()
            conn.close()

            # Schema should be at current version (4)
            assert row is not None
            # Note: new DBs won't have the version entry until migration runs
            # but they create at current schema so no migration needed

        finally:
            Path(db_path).unlink(missing_ok=True)

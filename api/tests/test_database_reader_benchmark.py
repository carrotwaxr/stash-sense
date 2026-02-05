"""Tests for database reader benchmark methods."""
import os
import sqlite3
import tempfile

import pytest

from database_reader import PerformerDatabaseReader


@pytest.fixture
def db_reader():
    """Create a test database with performers and coverage data."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE performers (
            id INTEGER PRIMARY KEY,
            canonical_name TEXT,
            disambiguation TEXT,
            gender TEXT,
            country TEXT,
            ethnicity TEXT,
            birth_date TEXT,
            death_date TEXT,
            height_cm INTEGER,
            eye_color TEXT,
            hair_color TEXT,
            career_start_year INTEGER,
            career_end_year INTEGER,
            scene_count INTEGER,
            stashdb_updated_at TEXT,
            face_count INTEGER DEFAULT 0,
            image_url TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE stashbox_ids (
            performer_id INTEGER,
            endpoint TEXT,
            stashbox_performer_id TEXT
        );

        CREATE TABLE body_proportions (
            id INTEGER PRIMARY KEY,
            performer_id INTEGER,
            shoulder_hip_ratio REAL,
            leg_torso_ratio REAL,
            arm_span_height_ratio REAL,
            confidence REAL
        );

        CREATE TABLE tattoo_detections (
            id INTEGER PRIMARY KEY,
            performer_id INTEGER,
            has_tattoos BOOLEAN,
            confidence REAL,
            detections_json TEXT
        );

        -- Insert test performers
        INSERT INTO performers (id, canonical_name, face_count) VALUES (1, 'Performer With Everything', 15);
        INSERT INTO performers (id, canonical_name, face_count) VALUES (2, 'Performer No Body Data', 10);
        INSERT INTO performers (id, canonical_name, face_count) VALUES (3, 'Performer No Tattoo Data', 5);
        INSERT INTO performers (id, canonical_name, face_count) VALUES (4, 'Performer No Coverage', 0);

        -- Insert stashbox IDs
        INSERT INTO stashbox_ids VALUES (1, 'stashdb.org', 'abc-123');
        INSERT INTO stashbox_ids VALUES (2, 'stashdb.org', 'no-body-data');
        INSERT INTO stashbox_ids VALUES (3, 'stashdb.org', 'no-tattoo');
        INSERT INTO stashbox_ids VALUES (4, 'stashdb.org', 'no-coverage');

        -- Insert body proportions for performers 1 and 3
        INSERT INTO body_proportions (performer_id, shoulder_hip_ratio, leg_torso_ratio, arm_span_height_ratio, confidence)
        VALUES (1, 1.45, 1.32, 1.01, 0.89);
        INSERT INTO body_proportions (performer_id, shoulder_hip_ratio, leg_torso_ratio, arm_span_height_ratio, confidence)
        VALUES (3, 1.40, 1.30, 1.00, 0.85);

        -- Insert tattoo detections for performers 1 and 2
        INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
        VALUES (1, 1, 0.90, '[{"location_hint": "left arm"}]');
        INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
        VALUES (2, 0, 0.0, '[]');
    """)
    conn.close()

    reader = PerformerDatabaseReader(path)
    yield reader
    os.unlink(path)


class TestDatabaseReaderBenchmarkMethods:
    """Tests for benchmark-related database reader methods."""

    # ==================== _parse_universal_id ====================

    def test_parse_universal_id_valid(self, db_reader):
        """Parse a valid universal_id."""
        endpoint, stashbox_id = db_reader._parse_universal_id("stashdb.org:abc-123")
        assert endpoint == "stashdb.org"
        assert stashbox_id == "abc-123"

    def test_parse_universal_id_with_colons_in_id(self, db_reader):
        """Parse universal_id where the ID contains colons."""
        endpoint, stashbox_id = db_reader._parse_universal_id("stashdb.org:uuid:with:colons")
        assert endpoint == "stashdb.org"
        assert stashbox_id == "uuid:with:colons"

    def test_parse_universal_id_invalid_no_colon(self, db_reader):
        """Return empty strings for invalid universal_id (no colon)."""
        endpoint, stashbox_id = db_reader._parse_universal_id("invalid-no-colon")
        assert endpoint == ""
        assert stashbox_id == ""

    def test_parse_universal_id_empty_string(self, db_reader):
        """Return empty strings for empty universal_id."""
        endpoint, stashbox_id = db_reader._parse_universal_id("")
        assert endpoint == ""
        assert stashbox_id == ""

    # ==================== get_face_count_for_performer ====================

    def test_get_face_count_for_performer(self, db_reader):
        """Get face count for existing performer."""
        count = db_reader.get_face_count_for_performer("stashdb.org:abc-123")
        assert count == 15

    def test_get_face_count_for_performer_different_counts(self, db_reader):
        """Verify different performers have different face counts."""
        count1 = db_reader.get_face_count_for_performer("stashdb.org:no-body-data")
        count2 = db_reader.get_face_count_for_performer("stashdb.org:no-tattoo")
        assert count1 == 10
        assert count2 == 5

    def test_get_face_count_not_found(self, db_reader):
        """Return 0 for unknown performer."""
        count = db_reader.get_face_count_for_performer("stashdb.org:unknown")
        assert count == 0

    def test_get_face_count_zero_faces(self, db_reader):
        """Return 0 for performer with zero faces."""
        count = db_reader.get_face_count_for_performer("stashdb.org:no-coverage")
        assert count == 0

    def test_get_face_count_invalid_universal_id(self, db_reader):
        """Return 0 for invalid universal_id format."""
        count = db_reader.get_face_count_for_performer("invalid-format")
        assert count == 0

    # ==================== has_body_data ====================

    def test_has_body_data_true(self, db_reader):
        """Return True when body data exists."""
        has = db_reader.has_body_data("stashdb.org:abc-123")
        assert has is True

    def test_has_body_data_false(self, db_reader):
        """Return False when no body data exists."""
        has = db_reader.has_body_data("stashdb.org:no-body-data")
        assert has is False

    def test_has_body_data_unknown_performer(self, db_reader):
        """Return False for unknown performer."""
        has = db_reader.has_body_data("stashdb.org:unknown")
        assert has is False

    def test_has_body_data_invalid_universal_id(self, db_reader):
        """Return False for invalid universal_id format."""
        has = db_reader.has_body_data("invalid-format")
        assert has is False

    # ==================== has_tattoo_data ====================

    def test_has_tattoo_data_true(self, db_reader):
        """Return True when tattoo detection data exists."""
        has = db_reader.has_tattoo_data("stashdb.org:abc-123")
        assert has is True

    def test_has_tattoo_data_true_no_tattoos_detected(self, db_reader):
        """Return True even when has_tattoos=False (data still exists)."""
        # Performer 2 has tattoo_detections entry with has_tattoos=0
        has = db_reader.has_tattoo_data("stashdb.org:no-body-data")
        assert has is True

    def test_has_tattoo_data_false(self, db_reader):
        """Return False when no tattoo detection data exists."""
        has = db_reader.has_tattoo_data("stashdb.org:no-tattoo")
        assert has is False

    def test_has_tattoo_data_unknown_performer(self, db_reader):
        """Return False for unknown performer."""
        has = db_reader.has_tattoo_data("stashdb.org:unknown")
        assert has is False

    def test_has_tattoo_data_invalid_universal_id(self, db_reader):
        """Return False for invalid universal_id format."""
        has = db_reader.has_tattoo_data("invalid-format")
        assert has is False

    # ==================== Integration ====================

    def test_coverage_check_performer_with_full_coverage(self, db_reader):
        """Performer 1 has all coverage types."""
        uid = "stashdb.org:abc-123"
        assert db_reader.get_face_count_for_performer(uid) == 15
        assert db_reader.has_body_data(uid) is True
        assert db_reader.has_tattoo_data(uid) is True

    def test_coverage_check_performer_with_partial_coverage(self, db_reader):
        """Performer 2 has faces and tattoo data, but no body data."""
        uid = "stashdb.org:no-body-data"
        assert db_reader.get_face_count_for_performer(uid) == 10
        assert db_reader.has_body_data(uid) is False
        assert db_reader.has_tattoo_data(uid) is True

    def test_coverage_check_performer_with_no_coverage(self, db_reader):
        """Performer 4 has no faces or coverage data."""
        uid = "stashdb.org:no-coverage"
        assert db_reader.get_face_count_for_performer(uid) == 0
        assert db_reader.has_body_data(uid) is False
        assert db_reader.has_tattoo_data(uid) is False

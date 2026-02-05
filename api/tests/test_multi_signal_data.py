"""Tests for multi-signal data loading."""
import json
import os
import sqlite3
import tempfile

import pytest

from database_reader import PerformerDatabaseReader


@pytest.fixture
def test_db():
    """Create a temporary test database with body proportions and tattoo data."""
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

        CREATE TABLE tattoos (
            id INTEGER PRIMARY KEY,
            performer_id INTEGER,
            location TEXT,
            description TEXT
        );

        CREATE TABLE tattoo_detections (
            id INTEGER PRIMARY KEY,
            performer_id INTEGER,
            has_tattoos BOOLEAN,
            confidence REAL,
            detections_json TEXT
        );

        -- Insert test data
        INSERT INTO performers (id, canonical_name, face_count) VALUES (1, 'Test Performer 1', 5);
        INSERT INTO performers (id, canonical_name, face_count) VALUES (2, 'Test Performer 2', 3);

        INSERT INTO stashbox_ids VALUES (1, 'stashdb', 'uuid-1');
        INSERT INTO stashbox_ids VALUES (2, 'stashdb', 'uuid-2');

        INSERT INTO body_proportions (performer_id, shoulder_hip_ratio, leg_torso_ratio, arm_span_height_ratio, confidence)
        VALUES (1, 1.45, 1.32, 1.01, 0.89);

        INSERT INTO tattoos (performer_id, location, description) VALUES (1, 'left arm', 'sleeve');
        INSERT INTO tattoos (performer_id, location, description) VALUES (1, 'back', 'large piece');

        INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
        VALUES (1, 1, 0.85, '[{"location_hint": "left arm"}, {"location_hint": "torso"}]');
        INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
        VALUES (2, 0, 0.0, '[]');
    """)
    conn.close()

    yield path
    os.unlink(path)


class TestBulkDataLoading:
    """Test bulk data loading for multi-signal matching."""

    def test_get_all_body_proportions(self, test_db):
        """Load all body proportions keyed by universal_id."""
        db = PerformerDatabaseReader(test_db)
        data = db.get_all_body_proportions()

        assert 'stashdb.org:uuid-1' in data
        assert data['stashdb.org:uuid-1']['shoulder_hip_ratio'] == 1.45
        assert data['stashdb.org:uuid-1']['leg_torso_ratio'] == 1.32
        assert data['stashdb.org:uuid-1']['arm_span_height_ratio'] == 1.01
        assert data['stashdb.org:uuid-1']['confidence'] == 0.89

    def test_get_all_body_proportions_returns_highest_confidence(self, test_db):
        """When multiple records exist, returns the one with highest confidence."""
        # Add a second record with lower confidence
        conn = sqlite3.connect(test_db)
        conn.execute("""
            INSERT INTO body_proportions (performer_id, shoulder_hip_ratio, leg_torso_ratio, arm_span_height_ratio, confidence)
            VALUES (1, 1.50, 1.40, 1.05, 0.75)
        """)
        conn.commit()
        conn.close()

        db = PerformerDatabaseReader(test_db)
        data = db.get_all_body_proportions()

        # Should return the record with 0.89 confidence, not 0.75
        assert data['stashdb.org:uuid-1']['confidence'] == 0.89
        assert data['stashdb.org:uuid-1']['shoulder_hip_ratio'] == 1.45

    def test_get_all_body_proportions_empty_result(self, test_db):
        """Returns empty dict when no body proportion data exists."""
        # Clear body_proportions table
        conn = sqlite3.connect(test_db)
        conn.execute("DELETE FROM body_proportions")
        conn.commit()
        conn.close()

        db = PerformerDatabaseReader(test_db)
        data = db.get_all_body_proportions()

        assert data == {}

    def test_get_all_tattoo_info(self, test_db):
        """Load all tattoo info keyed by universal_id."""
        db = PerformerDatabaseReader(test_db)
        data = db.get_all_tattoo_info()

        # Performer 1 has tattoos
        assert 'stashdb.org:uuid-1' in data
        assert data['stashdb.org:uuid-1']['has_tattoos'] is True
        assert 'left arm' in data['stashdb.org:uuid-1']['locations']
        assert 'back' in data['stashdb.org:uuid-1']['locations']
        assert 'torso' in data['stashdb.org:uuid-1']['locations']  # from detections
        assert data['stashdb.org:uuid-1']['count'] >= 3  # At least 3 unique locations

        # Performer 2 has no tattoos
        assert 'stashdb.org:uuid-2' in data
        assert data['stashdb.org:uuid-2']['has_tattoos'] is False
        assert data['stashdb.org:uuid-2']['locations'] == []

    def test_get_all_tattoo_info_from_tattoos_table_only(self, test_db):
        """Load tattoo info when only tattoos table has data."""
        # Add performer 3 with only tattoos table data (no detections)
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO performers (id, canonical_name, face_count) VALUES (3, 'Test Performer 3', 2)")
        conn.execute("INSERT INTO stashbox_ids VALUES (3, 'stashdb', 'uuid-3')")
        conn.execute("INSERT INTO tattoos (performer_id, location, description) VALUES (3, 'chest', 'small tattoo')")
        conn.commit()
        conn.close()

        db = PerformerDatabaseReader(test_db)
        data = db.get_all_tattoo_info()

        assert 'stashdb.org:uuid-3' in data
        assert data['stashdb.org:uuid-3']['has_tattoos'] is True
        assert 'chest' in data['stashdb.org:uuid-3']['locations']

    def test_get_all_tattoo_info_from_detections_only(self, test_db):
        """Load tattoo info when only tattoo_detections table has data."""
        # Add performer 4 with only detection data
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO performers (id, canonical_name, face_count) VALUES (4, 'Test Performer 4', 1)")
        conn.execute("INSERT INTO stashbox_ids VALUES (4, 'stashdb', 'uuid-4')")
        conn.execute("""
            INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
            VALUES (4, 1, 0.90, '[{"location_hint": "right arm"}]')
        """)
        conn.commit()
        conn.close()

        db = PerformerDatabaseReader(test_db)
        data = db.get_all_tattoo_info()

        assert 'stashdb.org:uuid-4' in data
        assert data['stashdb.org:uuid-4']['has_tattoos'] is True
        assert 'right arm' in data['stashdb.org:uuid-4']['locations']

    def test_get_all_tattoo_info_handles_invalid_json(self, test_db):
        """Gracefully handles invalid JSON in detections_json."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO performers (id, canonical_name, face_count) VALUES (5, 'Test Performer 5', 1)")
        conn.execute("INSERT INTO stashbox_ids VALUES (5, 'stashdb', 'uuid-5')")
        conn.execute("""
            INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
            VALUES (5, 1, 0.80, 'invalid json{')
        """)
        conn.commit()
        conn.close()

        db = PerformerDatabaseReader(test_db)
        # Should not raise an exception
        data = db.get_all_tattoo_info()

        # Performer 5 should still be present with has_tattoos=True (from has_tattoos column)
        assert 'stashdb.org:uuid-5' in data
        assert data['stashdb.org:uuid-5']['has_tattoos'] is True
        # But locations may be empty due to invalid JSON
        assert isinstance(data['stashdb.org:uuid-5']['locations'], list)

    def test_get_all_tattoo_info_deduplicates_locations(self, test_db):
        """Locations from different sources are deduplicated."""
        # Performer 1 has "left arm" in both tattoos table and detections_json
        db = PerformerDatabaseReader(test_db)
        data = db.get_all_tattoo_info()

        # Should only appear once (lowercase normalized)
        locations = data['stashdb.org:uuid-1']['locations']
        assert locations.count('left arm') == 1

    def test_get_all_tattoo_info_normalizes_case(self, test_db):
        """Locations are normalized to lowercase."""
        conn = sqlite3.connect(test_db)
        conn.execute("INSERT INTO performers (id, canonical_name, face_count) VALUES (6, 'Test Performer 6', 1)")
        conn.execute("INSERT INTO stashbox_ids VALUES (6, 'stashdb', 'uuid-6')")
        conn.execute("INSERT INTO tattoos (performer_id, location, description) VALUES (6, 'LEFT ARM', 'uppercase location')")
        conn.commit()
        conn.close()

        db = PerformerDatabaseReader(test_db)
        data = db.get_all_tattoo_info()

        assert 'stashdb.org:uuid-6' in data
        assert 'left arm' in data['stashdb.org:uuid-6']['locations']
        # Uppercase should not be present
        assert 'LEFT ARM' not in data['stashdb.org:uuid-6']['locations']

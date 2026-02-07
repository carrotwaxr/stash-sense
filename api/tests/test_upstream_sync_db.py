"""Tests for upstream sync database tables and methods."""

import pytest
from recommendations_db import RecommendationsDB


class TestUpstreamSyncSchema:
    """Tests for V4 schema additions."""

    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_upstream_snapshots_table_exists(self, db):
        with db._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='upstream_snapshots'"
            )
            assert cursor.fetchone() is not None

    def test_upstream_field_config_table_exists(self, db):
        with db._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='upstream_field_config'"
            )
            assert cursor.fetchone() is not None

    def test_dismissed_targets_has_permanent_column(self, db):
        with db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id, permanent) VALUES (?, ?, ?, ?)",
                ("test", "performer", "1", 1),
            )
            row = conn.execute(
                "SELECT permanent FROM dismissed_targets WHERE target_id = '1'"
            ).fetchone()
            assert row["permanent"] == 1

    def test_dismissed_targets_permanent_defaults_to_zero(self, db):
        with db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id) VALUES (?, ?, ?)",
                ("test", "performer", "2"),
            )
            row = conn.execute(
                "SELECT permanent FROM dismissed_targets WHERE target_id = '2'"
            ).fetchone()
            assert row["permanent"] == 0

    def test_schema_version_is_4(self, db):
        with db._connection() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 4

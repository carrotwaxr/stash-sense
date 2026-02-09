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

    def test_schema_version_is_6(self, db):
        with db._connection() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 6


class TestUpstreamSnapshots:
    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_upsert_and_get_snapshot(self, db):
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="42",
            endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
            upstream_data={"name": "Jane Doe", "height": 165},
            upstream_updated_at="2026-01-15T10:00:00Z",
        )
        snapshot = db.get_upstream_snapshot(
            entity_type="performer", endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
        )
        assert snapshot is not None
        assert snapshot["local_entity_id"] == "42"
        assert snapshot["upstream_data"]["name"] == "Jane Doe"
        assert snapshot["upstream_updated_at"] == "2026-01-15T10:00:00Z"

    def test_upsert_snapshot_updates_existing(self, db):
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="42",
            endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
            upstream_data={"name": "Jane Doe"}, upstream_updated_at="2026-01-15T10:00:00Z",
        )
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="42",
            endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
            upstream_data={"name": "Jane Smith"}, upstream_updated_at="2026-01-16T10:00:00Z",
        )
        snapshot = db.get_upstream_snapshot(
            entity_type="performer", endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
        )
        assert snapshot["upstream_data"]["name"] == "Jane Smith"
        assert snapshot["upstream_updated_at"] == "2026-01-16T10:00:00Z"

    def test_get_snapshot_returns_none_when_missing(self, db):
        snapshot = db.get_upstream_snapshot(
            entity_type="performer", endpoint="https://stashdb.org/graphql", stash_box_id="nonexistent",
        )
        assert snapshot is None


class TestUpstreamFieldConfig:
    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_get_enabled_fields_returns_none_when_no_config(self, db):
        fields = db.get_enabled_fields(endpoint="https://stashdb.org/graphql", entity_type="performer")
        assert fields is None

    def test_set_and_get_field_config(self, db):
        db.set_field_config(
            endpoint="https://stashdb.org/graphql", entity_type="performer",
            field_configs={"name": True, "height": True, "tattoos": False},
        )
        fields = db.get_enabled_fields(endpoint="https://stashdb.org/graphql", entity_type="performer")
        assert fields is not None
        assert "name" in fields
        assert "height" in fields
        assert "tattoos" not in fields

    def test_set_field_config_replaces_existing(self, db):
        db.set_field_config(
            endpoint="https://stashdb.org/graphql", entity_type="performer",
            field_configs={"name": True, "height": True},
        )
        db.set_field_config(
            endpoint="https://stashdb.org/graphql", entity_type="performer",
            field_configs={"name": True, "height": False, "gender": True},
        )
        fields = db.get_enabled_fields(endpoint="https://stashdb.org/graphql", entity_type="performer")
        assert "name" in fields
        assert "gender" in fields
        assert "height" not in fields


class TestDismissedTargetsPermanent:
    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_dismiss_with_permanent_flag(self, db):
        rec_id = db.create_recommendation(
            type="upstream_performer_changes", target_type="performer", target_id="42",
            details={"changes": []}, confidence=1.0,
        )
        db.dismiss_recommendation(rec_id, reason="not interested", permanent=True)
        assert db.is_dismissed("upstream_performer_changes", "performer", "42")
        assert db.is_permanently_dismissed("upstream_performer_changes", "performer", "42")

    def test_dismiss_without_permanent_is_soft(self, db):
        rec_id = db.create_recommendation(
            type="upstream_performer_changes", target_type="performer", target_id="43",
            details={"changes": []}, confidence=1.0,
        )
        db.dismiss_recommendation(rec_id, reason="not now")
        assert db.is_dismissed("upstream_performer_changes", "performer", "43")
        assert not db.is_permanently_dismissed("upstream_performer_changes", "performer", "43")

    def test_undismiss_soft_dismissed(self, db):
        rec_id = db.create_recommendation(
            type="upstream_performer_changes", target_type="performer", target_id="44",
            details={"changes": []}, confidence=1.0,
        )
        db.dismiss_recommendation(rec_id)
        db.undismiss("upstream_performer_changes", "performer", "44")
        assert not db.is_dismissed("upstream_performer_changes", "performer", "44")


class TestUpdateRecommendationDetails:
    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_update_details_on_pending(self, db):
        rec_id = db.create_recommendation(
            type="upstream_performer_changes", target_type="performer", target_id="50",
            details={"changes": ["name"]}, confidence=0.8,
        )
        result = db.update_recommendation_details(rec_id, {"changes": ["name", "height"]})
        assert result is True
        rec = db.get_recommendation(rec_id)
        assert rec.details == {"changes": ["name", "height"]}

    def test_update_details_on_resolved_returns_false(self, db):
        rec_id = db.create_recommendation(
            type="upstream_performer_changes", target_type="performer", target_id="51",
            details={"changes": ["name"]}, confidence=0.8,
        )
        db.resolve_recommendation(rec_id, action="accepted")
        result = db.update_recommendation_details(rec_id, {"changes": ["name", "height"]})
        assert result is False

    def test_update_details_nonexistent_returns_false(self, db):
        result = db.update_recommendation_details(9999, {"changes": []})
        assert result is False

"""Tests for user settings, recommendation settings, field config, and upstream snapshots in RecommendationsDB."""

import pytest
from recommendations_db import RecommendationsDB, RecommendationSettings


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "test.db"))


# ==================== User Settings (key-value store) ====================


class TestGetUserSetting:
    def test_returns_none_for_missing_key(self, db):
        assert db.get_user_setting("nonexistent") is None

    def test_default_setting_exists(self, db):
        val = db.get_user_setting("normalize_enum_display")
        assert val is True

    def test_returns_parsed_json(self, db):
        db.set_user_setting("my_list", [1, 2, 3])
        assert db.get_user_setting("my_list") == [1, 2, 3]


class TestSetUserSetting:
    def test_creates_new_setting(self, db):
        db.set_user_setting("theme", "dark")
        assert db.get_user_setting("theme") == "dark"

    def test_updates_existing_setting(self, db):
        db.set_user_setting("theme", "dark")
        db.set_user_setting("theme", "light")
        assert db.get_user_setting("theme") == "light"

    def test_stores_dict(self, db):
        db.set_user_setting("config", {"key": "value", "nested": [1, 2]})
        assert db.get_user_setting("config") == {"key": "value", "nested": [1, 2]}

    def test_stores_bool(self, db):
        db.set_user_setting("enabled", True)
        assert db.get_user_setting("enabled") is True

    def test_stores_int(self, db):
        db.set_user_setting("count", 42)
        assert db.get_user_setting("count") == 42

    def test_stores_null(self, db):
        db.set_user_setting("empty", None)
        assert db.get_user_setting("empty") is None


class TestGetAllUserSettings:
    def test_returns_dict(self, db):
        result = db.get_all_user_settings()
        assert isinstance(result, dict)

    def test_includes_default_settings(self, db):
        result = db.get_all_user_settings()
        assert "normalize_enum_display" in result

    def test_includes_custom_settings(self, db):
        db.set_user_setting("theme", "dark")
        db.set_user_setting("count", 10)
        result = db.get_all_user_settings()
        assert result["theme"] == "dark"
        assert result["count"] == 10


class TestDeleteUserSetting:
    def test_deletes_setting(self, db):
        db.set_user_setting("temp", "value")
        db.delete_user_setting("temp")
        assert db.get_user_setting("temp") is None

    def test_delete_nonexistent_is_noop(self, db):
        db.delete_user_setting("nonexistent")  # should not raise


# ==================== Endpoint Priorities ====================


class TestEndpointPriorities:
    def test_default_is_empty_list(self, db):
        assert db.get_endpoint_priorities() == []

    def test_set_and_get(self, db):
        urls = ["https://stashdb.org/graphql", "https://fansdb.cc/graphql"]
        db.set_endpoint_priorities(urls)
        assert db.get_endpoint_priorities() == urls

    def test_overwrites_previous(self, db):
        db.set_endpoint_priorities(["a", "b"])
        db.set_endpoint_priorities(["c"])
        assert db.get_endpoint_priorities() == ["c"]


# ==================== Disabled Endpoints ====================


class TestDisabledEndpoints:
    def test_default_is_empty_list(self, db):
        assert db.get_disabled_endpoints() == []

    def test_set_and_get(self, db):
        db.set_disabled_endpoints(["https://example.com/graphql"])
        assert db.get_disabled_endpoints() == ["https://example.com/graphql"]

    def test_is_endpoint_enabled_default_true(self, db):
        assert db.is_endpoint_enabled("https://stashdb.org/graphql") is True

    def test_is_endpoint_enabled_when_disabled(self, db):
        db.set_disabled_endpoints(["https://stashdb.org/graphql"])
        assert db.is_endpoint_enabled("https://stashdb.org/graphql") is False

    def test_is_endpoint_enabled_other_endpoint_still_enabled(self, db):
        db.set_disabled_endpoints(["https://stashdb.org/graphql"])
        assert db.is_endpoint_enabled("https://fansdb.cc/graphql") is True


# ==================== Recommendation Settings ====================


class TestRecommendationSettings:
    def test_get_returns_none_when_not_set(self, db):
        assert db.get_settings("nonexistent") is None

    def test_upsert_creates_new(self, db):
        db.upsert_settings("upstream_performer_changes", enabled=True, interval_hours=6)
        settings = db.get_settings("upstream_performer_changes")
        assert settings is not None
        assert isinstance(settings, RecommendationSettings)
        assert settings.type == "upstream_performer_changes"
        assert settings.enabled is True
        assert settings.interval_hours == 6

    def test_upsert_updates_existing(self, db):
        db.upsert_settings("upstream", enabled=True, interval_hours=6)
        db.upsert_settings("upstream", enabled=False, interval_hours=12)
        settings = db.get_settings("upstream")
        assert settings.enabled is False
        assert settings.interval_hours == 12

    def test_upsert_partial_update(self, db):
        db.upsert_settings("upstream", enabled=True, interval_hours=6, notify=True)
        db.upsert_settings("upstream", interval_hours=12)
        settings = db.get_settings("upstream")
        assert settings.enabled is True  # unchanged
        assert settings.interval_hours == 12

    def test_upsert_with_config(self, db):
        db.upsert_settings("upstream", config={"threshold": 0.9})
        settings = db.get_settings("upstream")
        assert settings.config == {"threshold": 0.9}

    def test_get_all_settings(self, db):
        db.upsert_settings("type_a", enabled=True)
        db.upsert_settings("type_b", enabled=False)
        all_settings = db.get_all_settings()
        assert len(all_settings) == 2
        types = {s.type for s in all_settings}
        assert types == {"type_a", "type_b"}

    def test_get_all_settings_empty(self, db):
        assert db.get_all_settings() == []

    def test_defaults_when_inserting(self, db):
        db.upsert_settings("test_type")
        settings = db.get_settings("test_type")
        assert settings.enabled is True  # default
        assert settings.notify is True  # default


# ==================== Upstream Field Config ====================


class TestUpstreamFieldConfig:
    def test_get_enabled_fields_none_when_no_config(self, db):
        assert db.get_enabled_fields("https://stashdb.org/graphql", "performer") is None

    def test_set_and_get_field_config(self, db):
        db.set_field_config(
            "https://stashdb.org/graphql", "performer",
            {"name": True, "height": True, "tattoos": False},
        )
        fields = db.get_enabled_fields("https://stashdb.org/graphql", "performer")
        assert "name" in fields
        assert "height" in fields
        assert "tattoos" not in fields

    def test_set_replaces_existing(self, db):
        db.set_field_config("https://stashdb.org/graphql", "performer", {"name": True})
        db.set_field_config("https://stashdb.org/graphql", "performer", {"height": True})
        fields = db.get_enabled_fields("https://stashdb.org/graphql", "performer")
        assert "height" in fields
        assert "name" not in fields

    def test_different_endpoints_independent(self, db):
        db.set_field_config("https://stashdb.org/graphql", "performer", {"name": True})
        db.set_field_config("https://fansdb.cc/graphql", "performer", {"aliases": True})

        fields1 = db.get_enabled_fields("https://stashdb.org/graphql", "performer")
        fields2 = db.get_enabled_fields("https://fansdb.cc/graphql", "performer")
        assert "name" in fields1
        assert "aliases" not in fields1
        assert "aliases" in fields2
        assert "name" not in fields2


# ==================== Upstream Snapshots ====================


class TestUpstreamSnapshots:
    def test_upsert_and_get(self, db):
        snap_id = db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="42",
            endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
            upstream_data={"name": "Jane Doe"}, upstream_updated_at="2026-01-15T10:00:00Z",
        )
        assert isinstance(snap_id, int)
        snapshot = db.get_upstream_snapshot("performer", "https://stashdb.org/graphql", "abc-123")
        assert snapshot is not None
        assert snapshot["upstream_data"] == {"name": "Jane Doe"}
        assert snapshot["local_entity_id"] == "42"

    def test_upsert_updates_existing(self, db):
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="42",
            endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
            upstream_data={"name": "Jane Doe"}, upstream_updated_at="2026-01-15",
        )
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="42",
            endpoint="https://stashdb.org/graphql", stash_box_id="abc-123",
            upstream_data={"name": "Jane Smith"}, upstream_updated_at="2026-01-16",
        )
        snapshot = db.get_upstream_snapshot("performer", "https://stashdb.org/graphql", "abc-123")
        assert snapshot["upstream_data"]["name"] == "Jane Smith"

    def test_get_returns_none_for_missing(self, db):
        assert db.get_upstream_snapshot("performer", "https://stashdb.org/graphql", "no-exist") is None

    def test_delete_snapshots_for_endpoint(self, db):
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="1",
            endpoint="https://stashdb.org/graphql", stash_box_id="a",
            upstream_data={}, upstream_updated_at="2026-01-15",
        )
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="2",
            endpoint="https://stashdb.org/graphql", stash_box_id="b",
            upstream_data={}, upstream_updated_at="2026-01-15",
        )
        db.upsert_upstream_snapshot(
            entity_type="performer", local_entity_id="3",
            endpoint="https://fansdb.cc/graphql", stash_box_id="c",
            upstream_data={}, upstream_updated_at="2026-01-15",
        )
        deleted = db.delete_snapshots_for_endpoint("performer", "https://stashdb.org/graphql")
        assert deleted == 2
        assert db.get_upstream_snapshot("performer", "https://fansdb.cc/graphql", "c") is not None

"""Tests for studio field mapping and 3-way diff engine."""

from upstream_field_mapper import (
    DEFAULT_STUDIO_FIELDS,
    STUDIO_FIELD_MERGE_TYPES,
    STUDIO_FIELD_LABELS,
    ENTITY_FIELD_CONFIGS,
    normalize_upstream_studio,
    diff_studio_fields,
)


class TestStudioFieldConfig:
    def test_studio_registered_in_entity_configs(self):
        assert "studio" in ENTITY_FIELD_CONFIGS
        config = ENTITY_FIELD_CONFIGS["studio"]
        assert config["default_fields"] == DEFAULT_STUDIO_FIELDS
        assert config["labels"] == STUDIO_FIELD_LABELS
        assert config["merge_types"] == STUDIO_FIELD_MERGE_TYPES

    def test_default_studio_fields(self):
        assert DEFAULT_STUDIO_FIELDS == {"name", "url", "parent_studio"}

    def test_all_default_fields_have_merge_types(self):
        for field_name in DEFAULT_STUDIO_FIELDS:
            assert field_name in STUDIO_FIELD_MERGE_TYPES

    def test_all_default_fields_have_labels(self):
        for field_name in DEFAULT_STUDIO_FIELDS:
            assert field_name in STUDIO_FIELD_LABELS

    def test_merge_types_correct(self):
        assert STUDIO_FIELD_MERGE_TYPES["name"] == "name"
        assert STUDIO_FIELD_MERGE_TYPES["url"] == "simple"
        assert STUDIO_FIELD_MERGE_TYPES["parent_studio"] == "simple"


class TestNormalizeUpstreamStudio:
    def test_maps_name(self):
        upstream = {"name": "Brazzers", "urls": [], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["name"] == "Brazzers"

    def test_extracts_first_url(self):
        upstream = {"name": "Brazzers", "urls": [{"url": "https://brazzers.com"}, {"url": "https://brazzers.net"}], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["url"] == "https://brazzers.com"

    def test_empty_url_list_gives_none(self):
        upstream = {"name": "Brazzers", "urls": [], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["url"] is None

    def test_none_url_list_gives_none(self):
        upstream = {"name": "Brazzers", "urls": None, "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["url"] is None

    def test_extracts_parent_id(self):
        upstream = {"name": "Brazzers Exxtra", "urls": [], "parent": {"id": "parent-uuid-1", "name": "Brazzers"}}
        result = normalize_upstream_studio(upstream)
        assert result["parent_studio"] == "parent-uuid-1"

    def test_no_parent_gives_none(self):
        upstream = {"name": "Brazzers", "urls": [], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["parent_studio"] is None

    def test_handles_missing_fields(self):
        upstream = {}
        result = normalize_upstream_studio(upstream)
        assert result == {}


class TestDiffStudioFields:
    def test_detects_name_change(self):
        local = {"name": "Brazzrs", "url": "https://brazzers.com", "parent_studio": None}
        upstream = {"name": "Brazzers", "url": "https://brazzers.com", "parent_studio": None}
        snapshot = {"name": "Brazzrs", "url": "https://brazzers.com", "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "url", "parent_studio"})
        assert len(changes) == 1
        assert changes[0]["field"] == "name"
        assert changes[0]["merge_type"] == "name"

    def test_detects_url_change(self):
        local = {"name": "Brazzers", "url": "https://old.com", "parent_studio": None}
        upstream = {"name": "Brazzers", "url": "https://brazzers.com", "parent_studio": None}
        snapshot = {"name": "Brazzers", "url": "https://old.com", "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "url", "parent_studio"})
        assert len(changes) == 1
        assert changes[0]["field"] == "url"

    def test_detects_parent_change(self):
        local = {"name": "Sub Studio", "url": None, "parent_studio": None}
        upstream = {"name": "Sub Studio", "url": None, "parent_studio": "parent-uuid-1"}
        snapshot = {"name": "Sub Studio", "url": None, "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "url", "parent_studio"})
        assert len(changes) == 1
        assert changes[0]["field"] == "parent_studio"

    def test_no_changes_when_in_sync(self):
        data = {"name": "Brazzers", "url": "https://brazzers.com", "parent_studio": None}
        changes = diff_studio_fields(data, data, data, {"name", "url", "parent_studio"})
        assert len(changes) == 0

    def test_skips_unchanged_upstream_with_snapshot(self):
        local = {"name": "My Custom Name", "url": None, "parent_studio": None}
        upstream = {"name": "Brazzers", "url": None, "parent_studio": None}
        snapshot = {"name": "Brazzers", "url": None, "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "url", "parent_studio"})
        assert len(changes) == 0

    def test_first_run_no_snapshot_flags_all_differences(self):
        local = {"name": "Brazzrs", "url": None, "parent_studio": None}
        upstream = {"name": "Brazzers", "url": "https://brazzers.com", "parent_studio": "parent-uuid"}
        changes = diff_studio_fields(local, upstream, None, {"name", "url", "parent_studio"})
        assert len(changes) == 3

    def test_respects_enabled_fields_filter(self):
        local = {"name": "Old", "url": "https://old.com", "parent_studio": None}
        upstream = {"name": "New", "url": "https://new.com", "parent_studio": "uuid"}
        snapshot = {"name": "Older", "url": "https://older.com", "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"url"})
        assert len(changes) == 1
        assert changes[0]["field"] == "url"

    def test_empty_values_treated_as_equal(self):
        local = {"url": None}
        upstream = {"url": ""}
        changes = diff_studio_fields(local, upstream, None, {"url"})
        assert len(changes) == 0

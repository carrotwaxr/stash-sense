"""Tests for tag field mapping and 3-way diff engine."""

from upstream_field_mapper import (
    DEFAULT_TAG_FIELDS,
    TAG_FIELD_MERGE_TYPES,
    TAG_FIELD_LABELS,
    ENTITY_FIELD_CONFIGS,
    normalize_upstream_tag,
    diff_tag_fields,
)


class TestTagFieldConfig:
    def test_tag_registered_in_entity_configs(self):
        assert "tag" in ENTITY_FIELD_CONFIGS
        config = ENTITY_FIELD_CONFIGS["tag"]
        assert config["default_fields"] == DEFAULT_TAG_FIELDS
        assert config["labels"] == TAG_FIELD_LABELS
        assert config["merge_types"] == TAG_FIELD_MERGE_TYPES

    def test_default_tag_fields(self):
        assert DEFAULT_TAG_FIELDS == {"name", "description", "aliases"}

    def test_all_default_fields_have_merge_types(self):
        for field_name in DEFAULT_TAG_FIELDS:
            assert field_name in TAG_FIELD_MERGE_TYPES, (
                f"Field '{field_name}' missing from TAG_FIELD_MERGE_TYPES"
            )

    def test_all_default_fields_have_labels(self):
        for field_name in DEFAULT_TAG_FIELDS:
            assert field_name in TAG_FIELD_LABELS, (
                f"Field '{field_name}' missing from TAG_FIELD_LABELS"
            )

    def test_merge_types_correct(self):
        assert TAG_FIELD_MERGE_TYPES["name"] == "name"
        assert TAG_FIELD_MERGE_TYPES["description"] == "text"
        assert TAG_FIELD_MERGE_TYPES["aliases"] == "alias_list"


class TestNormalizeUpstreamTag:
    def test_maps_basic_fields(self):
        upstream = {
            "name": "Blowjob",
            "description": "Oral sex performed on a male",
        }
        result = normalize_upstream_tag(upstream)
        assert result["name"] == "Blowjob"
        assert result["description"] == "Oral sex performed on a male"

    def test_normalizes_aliases_from_list(self):
        upstream = {"aliases": ["BJ", "Fellatio"]}
        result = normalize_upstream_tag(upstream)
        assert result["aliases"] == ["BJ", "Fellatio"]

    def test_normalizes_none_aliases_to_empty_list(self):
        upstream = {"aliases": None}
        result = normalize_upstream_tag(upstream)
        assert result["aliases"] == []

    def test_omits_category(self):
        """Category is StashBox-only and should not be included in normalized output."""
        upstream = {
            "name": "Blowjob",
            "description": "Desc",
            "aliases": [],
            "category": {"id": "cat-1", "name": "Action", "group": "ACTION"},
        }
        result = normalize_upstream_tag(upstream)
        assert "category" not in result

    def test_handles_missing_fields(self):
        upstream = {}
        result = normalize_upstream_tag(upstream)
        assert result == {}

    def test_handles_none_description(self):
        upstream = {"name": "Tag", "description": None}
        result = normalize_upstream_tag(upstream)
        assert result["description"] is None


class TestDiffTagFields:
    def test_detects_description_change(self):
        local = {"name": "Blowjob", "description": "Old desc", "aliases": []}
        upstream = {"name": "Blowjob", "description": "New desc", "aliases": []}
        snapshot = {"name": "Blowjob", "description": "Old desc", "aliases": []}
        enabled = {"name", "description", "aliases"}

        changes = diff_tag_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "description"
        assert changes[0]["local_value"] == "Old desc"
        assert changes[0]["upstream_value"] == "New desc"
        assert changes[0]["merge_type"] == "text"

    def test_detects_name_change(self):
        local = {"name": "BJ", "description": "", "aliases": []}
        upstream = {"name": "Blowjob", "description": "", "aliases": []}
        snapshot = {"name": "BJ", "description": "", "aliases": []}
        enabled = {"name", "description", "aliases"}

        changes = diff_tag_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "name"
        assert changes[0]["merge_type"] == "name"

    def test_detects_alias_addition(self):
        local = {"name": "Blowjob", "description": "", "aliases": ["BJ"]}
        upstream = {"name": "Blowjob", "description": "", "aliases": ["BJ", "Fellatio"]}
        snapshot = {"name": "Blowjob", "description": "", "aliases": ["BJ"]}
        enabled = {"name", "description", "aliases"}

        changes = diff_tag_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "aliases"
        assert changes[0]["merge_type"] == "alias_list"

    def test_alias_diff_is_case_insensitive(self):
        local = {"aliases": ["BJ", "Fellatio"]}
        upstream = {"aliases": ["bj", "fellatio"]}
        snapshot = {"aliases": ["bj", "fellatio"]}
        enabled = {"aliases"}

        changes = diff_tag_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 0

    def test_skips_unchanged_upstream_with_snapshot(self):
        """If upstream == snapshot, user intentionally set local differently."""
        local = {"name": "My Custom Name", "description": "", "aliases": []}
        upstream = {"name": "Blowjob", "description": "", "aliases": []}
        snapshot = {"name": "Blowjob", "description": "", "aliases": []}
        enabled = {"name", "description", "aliases"}

        changes = diff_tag_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 0

    def test_first_run_no_snapshot_flags_all_differences(self):
        local = {"name": "BJ", "description": "Short", "aliases": []}
        upstream = {"name": "Blowjob", "description": "Long desc", "aliases": ["BJ"]}
        enabled = {"name", "description", "aliases"}

        changes = diff_tag_fields(local, upstream, None, enabled)
        assert len(changes) == 3
        field_names = {c["field"] for c in changes}
        assert field_names == {"name", "description", "aliases"}

    def test_respects_enabled_fields_filter(self):
        local = {"name": "BJ", "description": "Short", "aliases": []}
        upstream = {"name": "Blowjob", "description": "Long desc", "aliases": ["BJ"]}
        snapshot = {"name": "Old", "description": "Old", "aliases": []}
        enabled = {"description"}  # only monitoring description

        changes = diff_tag_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "description"

    def test_no_changes_when_all_in_sync(self):
        local = {"name": "Blowjob", "description": "Desc", "aliases": ["BJ"]}
        upstream = {"name": "Blowjob", "description": "Desc", "aliases": ["BJ"]}
        snapshot = {"name": "Blowjob", "description": "Desc", "aliases": ["BJ"]}
        enabled = {"name", "description", "aliases"}

        changes = diff_tag_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 0

    def test_empty_values_treated_as_equal(self):
        """None, empty string, and empty list should all be treated as equal."""
        local = {"description": None, "aliases": []}
        upstream = {"description": "", "aliases": []}
        enabled = {"description", "aliases"}

        changes = diff_tag_fields(local, upstream, None, enabled)
        assert len(changes) == 0

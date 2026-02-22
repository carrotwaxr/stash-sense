"""Tests for studio field mapping and 3-way diff engine."""

from upstream_field_mapper import (
    DEFAULT_STUDIO_FIELDS,
    STUDIO_FIELD_MERGE_TYPES,
    STUDIO_FIELD_LABELS,
    ENTITY_FIELD_CONFIGS,
    normalize_upstream_studio,
    diff_studio_fields,
)
from analyzers.upstream_studio import _build_local_studio_data


class TestStudioFieldConfig:
    def test_studio_registered_in_entity_configs(self):
        assert "studio" in ENTITY_FIELD_CONFIGS
        config = ENTITY_FIELD_CONFIGS["studio"]
        assert config["default_fields"] == DEFAULT_STUDIO_FIELDS
        assert config["labels"] == STUDIO_FIELD_LABELS
        assert config["merge_types"] == STUDIO_FIELD_MERGE_TYPES

    def test_default_studio_fields(self):
        assert DEFAULT_STUDIO_FIELDS == {"name", "urls", "parent_studio"}

    def test_all_default_fields_have_merge_types(self):
        for field_name in DEFAULT_STUDIO_FIELDS:
            assert field_name in STUDIO_FIELD_MERGE_TYPES

    def test_all_default_fields_have_labels(self):
        for field_name in DEFAULT_STUDIO_FIELDS:
            assert field_name in STUDIO_FIELD_LABELS

    def test_merge_types_correct(self):
        assert STUDIO_FIELD_MERGE_TYPES["name"] == "name"
        assert STUDIO_FIELD_MERGE_TYPES["urls"] == "alias_list"
        assert STUDIO_FIELD_MERGE_TYPES["parent_studio"] == "simple"


class TestNormalizeUpstreamStudio:
    def test_maps_name(self):
        upstream = {"name": "Brazzers", "urls": [], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["name"] == "Brazzers"

    def test_extracts_all_urls(self):
        """Studios can have multiple URLs, all should be preserved."""
        upstream = {"name": "Brazzers", "urls": [{"url": "https://brazzers.com"}, {"url": "https://brazzers.net"}], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["urls"] == ["https://brazzers.com", "https://brazzers.net"]

    def test_single_url(self):
        upstream = {"name": "Brazzers", "urls": [{"url": "https://brazzers.com"}], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["urls"] == ["https://brazzers.com"]

    def test_empty_url_list_gives_empty_list(self):
        upstream = {"name": "Brazzers", "urls": [], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["urls"] == []

    def test_none_url_list_gives_empty_list(self):
        upstream = {"name": "Brazzers", "urls": None, "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["urls"] == []

    def test_four_urls_vouyermedia_case(self):
        """Reproduces: Vouyer Media has 4 URLs on StashDB, all should be preserved."""
        upstream = {
            "name": "Vouyer Media",
            "urls": [
                {"url": "https://vouyermedia.com/"},
                {"url": "https://twitter.com/vouyermedia"},
                {"url": "https://www.instagram.com/vouyermedia/"},
                {"url": "https://www.imdb.com/company/co0139498/"},
            ],
            "parent": None,
        }
        result = normalize_upstream_studio(upstream)
        assert len(result["urls"]) == 4
        assert result["urls"][0] == "https://vouyermedia.com/"

    def test_extracts_parent_id(self):
        upstream = {"name": "Brazzers Exxtra", "urls": [], "parent": {"id": "parent-uuid-1", "name": "Brazzers"}}
        result = normalize_upstream_studio(upstream)
        assert result["parent_studio"] == "parent-uuid-1"

    def test_extracts_parent_display_name(self):
        upstream = {"name": "Brazzers Exxtra", "urls": [], "parent": {"id": "parent-uuid-1", "name": "Brazzers"}}
        result = normalize_upstream_studio(upstream)
        assert result["_parent_studio_name"] == "Brazzers"

    def test_no_parent_gives_none_display_name(self):
        upstream = {"name": "Brazzers", "urls": [], "parent": None}
        result = normalize_upstream_studio(upstream)
        assert result["_parent_studio_name"] is None

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
        local = {"name": "Brazzrs", "urls": ["https://brazzers.com"], "parent_studio": None}
        upstream = {"name": "Brazzers", "urls": ["https://brazzers.com"], "parent_studio": None}
        snapshot = {"name": "Brazzrs", "urls": ["https://brazzers.com"], "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "urls", "parent_studio"})
        assert len(changes) == 1
        assert changes[0]["field"] == "name"
        assert changes[0]["merge_type"] == "name"

    def test_detects_url_addition(self):
        """When upstream adds a new URL, it should be detected as a change."""
        local = {"name": "Brazzers", "urls": ["https://brazzers.com"], "parent_studio": None}
        upstream = {"name": "Brazzers", "urls": ["https://brazzers.com", "https://brazzers.net"], "parent_studio": None}
        snapshot = {"name": "Brazzers", "urls": ["https://brazzers.com"], "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "urls", "parent_studio"})
        assert len(changes) == 1
        assert changes[0]["field"] == "urls"
        assert changes[0]["merge_type"] == "alias_list"

    def test_no_diff_when_urls_match_regardless_of_order(self):
        """URL comparison should be set-based (order doesn't matter)."""
        local = {"name": "X", "urls": ["https://b.com", "https://a.com"], "parent_studio": None}
        upstream = {"name": "X", "urls": ["https://a.com", "https://b.com"], "parent_studio": None}
        snapshot = {"name": "X", "urls": ["https://a.com", "https://b.com"], "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"urls"})
        assert len(changes) == 0

    def test_detects_parent_change(self):
        local = {"name": "Sub Studio", "urls": [], "parent_studio": None}
        upstream = {"name": "Sub Studio", "urls": [], "parent_studio": "parent-uuid-1"}
        snapshot = {"name": "Sub Studio", "urls": [], "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "urls", "parent_studio"})
        assert len(changes) == 1
        assert changes[0]["field"] == "parent_studio"

    def test_no_changes_when_in_sync(self):
        data = {"name": "Brazzers", "urls": ["https://brazzers.com"], "parent_studio": None}
        changes = diff_studio_fields(data, data, data, {"name", "urls", "parent_studio"})
        assert len(changes) == 0

    def test_skips_unchanged_upstream_with_snapshot(self):
        local = {"name": "My Custom Name", "urls": [], "parent_studio": None}
        upstream = {"name": "Brazzers", "urls": [], "parent_studio": None}
        snapshot = {"name": "Brazzers", "urls": [], "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"name", "urls", "parent_studio"})
        assert len(changes) == 0

    def test_first_run_no_snapshot_flags_all_differences(self):
        local = {"name": "Brazzrs", "urls": [], "parent_studio": None}
        upstream = {"name": "Brazzers", "urls": ["https://brazzers.com"], "parent_studio": "parent-uuid"}
        changes = diff_studio_fields(local, upstream, None, {"name", "urls", "parent_studio"})
        assert len(changes) == 3

    def test_respects_enabled_fields_filter(self):
        local = {"name": "Old", "urls": ["https://old.com"], "parent_studio": None}
        upstream = {"name": "New", "urls": ["https://new.com"], "parent_studio": "uuid"}
        snapshot = {"name": "Older", "urls": ["https://older.com"], "parent_studio": None}
        changes = diff_studio_fields(local, upstream, snapshot, {"urls"})
        assert len(changes) == 1
        assert changes[0]["field"] == "urls"

    def test_empty_values_treated_as_equal(self):
        local = {"urls": []}
        upstream = {"urls": []}
        changes = diff_studio_fields(local, upstream, None, {"urls"})
        assert len(changes) == 0

    def test_parent_change_includes_display_names(self):
        local = {"name": "Sub", "urls": [], "parent_studio": None, "_parent_studio_name": None}
        upstream = {"name": "Sub", "urls": [], "parent_studio": "parent-uuid-1", "_parent_studio_name": "Brazzers"}
        changes = diff_studio_fields(local, upstream, None, {"parent_studio"})
        assert len(changes) == 1
        assert changes[0]["field"] == "parent_studio"
        assert changes[0]["upstream_display"] == "Brazzers"
        assert changes[0]["local_display"] is None

    def test_parent_change_display_fallback_to_raw_value(self):
        """When no display name is available, display falls back to the raw value."""
        local = {"name": "Sub", "urls": [], "parent_studio": "old-uuid"}
        upstream = {"name": "Sub", "urls": [], "parent_studio": "new-uuid"}
        changes = diff_studio_fields(local, upstream, None, {"parent_studio"})
        assert len(changes) == 1
        assert changes[0]["local_display"] == "old-uuid"
        assert changes[0]["upstream_display"] == "new-uuid"

    def test_non_parent_fields_have_no_display_keys(self):
        """Only parent_studio gets display enrichment."""
        local = {"name": "Old", "urls": [], "parent_studio": None}
        upstream = {"name": "New", "urls": [], "parent_studio": None}
        changes = diff_studio_fields(local, upstream, None, {"name"})
        assert len(changes) == 1
        assert "local_display" not in changes[0]


class TestBuildLocalStudioData:
    def test_resolves_parent_stashbox_id_for_endpoint(self):
        """Parent's stashbox ID should be used instead of local numeric ID."""
        studio = {
            "name": "Sub Studio",
            "urls": [],
            "parent_studio": {
                "id": "10",
                "name": "Parent Studio",
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "parent-uuid-1"},
                ],
            },
        }
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["parent_studio"] == "parent-uuid-1"
        assert result["_parent_studio_name"] == "Parent Studio"

    def test_none_when_parent_not_linked_to_endpoint(self):
        """If parent has no stash_id for this endpoint, parent_studio is None."""
        studio = {
            "name": "Sub Studio",
            "urls": [],
            "parent_studio": {
                "id": "10",
                "name": "Parent Studio",
                "stash_ids": [
                    {"endpoint": "https://other.org/graphql", "stash_id": "other-uuid"},
                ],
            },
        }
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["parent_studio"] is None
        assert result["_parent_studio_name"] == "Parent Studio"

    def test_none_when_parent_has_empty_stash_ids(self):
        """Parent with no stash_ids results in None parent_studio."""
        studio = {
            "name": "Sub Studio",
            "urls": [],
            "parent_studio": {
                "id": "10",
                "name": "Parent Studio",
                "stash_ids": [],
            },
        }
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["parent_studio"] is None
        assert result["_parent_studio_name"] == "Parent Studio"

    def test_no_parent(self):
        studio = {"name": "Studio", "urls": ["https://example.com"], "parent_studio": None}
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["parent_studio"] is None
        assert result["_parent_studio_name"] is None

    def test_urls_from_local_studio(self):
        """Local studio URLs should be passed through as array."""
        studio = {
            "name": "Studio",
            "urls": ["https://example.com", "https://twitter.com/studio"],
            "parent_studio": None,
        }
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["urls"] == ["https://example.com", "https://twitter.com/studio"]

    def test_empty_urls(self):
        studio = {"name": "Studio", "urls": [], "parent_studio": None}
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["urls"] == []

    def test_none_urls(self):
        studio = {"name": "Studio", "parent_studio": None}
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["urls"] == []

    def test_parent_multiple_endpoints_picks_correct(self):
        """When parent is linked to multiple endpoints, picks the right one."""
        studio = {
            "name": "Sub",
            "urls": [],
            "parent_studio": {
                "id": "10",
                "name": "Parent",
                "stash_ids": [
                    {"endpoint": "https://fansdb.cc/graphql", "stash_id": "fansdb-uuid"},
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "stashdb-uuid"},
                ],
            },
        }
        result = _build_local_studio_data(studio, "https://stashdb.org/graphql")
        assert result["parent_studio"] == "stashdb-uuid"

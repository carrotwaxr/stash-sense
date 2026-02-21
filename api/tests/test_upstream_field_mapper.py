"""Tests for upstream field mapper and 3-way diff engine."""

from upstream_field_mapper import (
    DEFAULT_PERFORMER_FIELDS,
    FIELD_MERGE_TYPES,
    _values_equal,
    diff_performer_fields,
    normalize_upstream_performer,
)


class TestNormalizeUpstreamPerformer:
    def test_maps_basic_fields(self):
        upstream = {
            "name": "Jane Doe",
            "disambiguation": "actress",
            "gender": "FEMALE",
            "country": "US",
            "eye_color": "BROWN",
            "hair_color": "BLACK",
            "height": 165,
            "details": "Some bio text",
        }
        result = normalize_upstream_performer(upstream)
        assert result["name"] == "Jane Doe"
        assert result["disambiguation"] == "actress"
        assert result["gender"] == "FEMALE"
        assert result["country"] == "US"
        assert result["eye_color"] == "BROWN"
        assert result["hair_color"] == "BLACK"
        assert result["height"] == 165
        assert result["details"] == "Some bio text"

    def test_maps_birth_date_field_name(self):
        upstream = {"birth_date": "1990-05-15"}
        result = normalize_upstream_performer(upstream)
        assert result["birthdate"] == "1990-05-15"
        assert "birth_date" not in result

    def test_formats_tattoos_from_body_modifications(self):
        upstream = {
            "tattoos": [
                {"location": "Left arm", "description": "Dragon"},
                {"location": "Back", "description": "Wings"},
            ]
        }
        result = normalize_upstream_performer(upstream)
        assert result["tattoos"] == "Left arm: Dragon; Back: Wings"

    def test_formats_piercings_from_body_modifications(self):
        upstream = {
            "piercings": [
                {"location": "Navel", "description": ""},
            ]
        }
        result = normalize_upstream_performer(upstream)
        assert result["piercings"] == "Navel"

    def test_extracts_urls_from_objects(self):
        upstream = {
            "urls": [
                {"url": "https://twitter.com/jane", "type": "TWITTER"},
                {"url": "https://instagram.com/jane", "type": "INSTAGRAM"},
            ]
        }
        result = normalize_upstream_performer(upstream)
        assert result["urls"] == [
            "https://twitter.com/jane",
            "https://instagram.com/jane",
        ]

    def test_ignores_is_favorite(self):
        """favorite is local-only metadata, not synced from upstream."""
        upstream = {"is_favorite": True}
        result = normalize_upstream_performer(upstream)
        assert "favorite" not in result
        assert "is_favorite" not in result

    def test_handles_none_values(self):
        upstream = {
            "name": None,
            "aliases": None,
            "tattoos": None,
            "piercings": None,
            "urls": None,
        }
        result = normalize_upstream_performer(upstream)
        assert result["name"] is None
        assert result["aliases"] == []
        assert result["tattoos"] is None
        assert result["piercings"] is None
        assert result["urls"] == []

    def test_maps_measurements(self):
        upstream = {
            "cup_size": "C",
            "band_size": 34,
            "waist_size": 24,
            "hip_size": 34,
            "breast_type": "NATURAL",
        }
        result = normalize_upstream_performer(upstream)
        assert result["cup_size"] == "C"
        assert result["band_size"] == 34
        assert result["waist_size"] == 24
        assert result["hip_size"] == 34
        assert result["breast_type"] == "NATURAL"


class TestDiffPerformerFields:
    def test_detects_simple_change(self):
        local = {"name": "Jane Doe", "height": 165}
        upstream = {"name": "Jane Doe", "height": 170}
        snapshot = {"name": "Jane Doe", "height": 165}
        enabled = {"name", "height"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "height"
        assert changes[0]["local_value"] == 165
        assert changes[0]["upstream_value"] == 170
        assert changes[0]["previous_upstream_value"] == 165

    def test_skips_unchanged_upstream(self):
        """If upstream == snapshot, user intentionally set local differently."""
        local = {"name": "Jane Smith"}
        upstream = {"name": "Jane Doe"}
        snapshot = {"name": "Jane Doe"}
        enabled = {"name"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 0

    def test_first_run_no_snapshot_flags_all_differences(self):
        local = {"name": "Jane Smith", "height": 165}
        upstream = {"name": "Jane Doe", "height": 170}
        enabled = {"name", "height"}

        changes = diff_performer_fields(local, upstream, None, enabled)
        assert len(changes) == 2
        field_names = {c["field"] for c in changes}
        assert field_names == {"name", "height"}
        for c in changes:
            assert c["previous_upstream_value"] is None

    def test_skips_fields_already_in_sync(self):
        local = {"name": "Jane Doe", "height": 165}
        upstream = {"name": "Jane Doe", "height": 170}
        snapshot = {"name": "Jane Doe", "height": 165}
        enabled = {"name", "height"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "height"

    def test_respects_enabled_fields_filter(self):
        local = {"name": "Jane Smith", "height": 165}
        upstream = {"name": "Jane Doe", "height": 170}
        snapshot = {"name": "Old Name", "height": 160}
        enabled = {"height"}  # name is NOT enabled

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "height"

    def test_assigns_correct_merge_types(self):
        local = {"name": "Jane Smith", "aliases": ["JD"], "tattoos": "Arm"}
        upstream = {"name": "Jane Doe", "aliases": ["Jane"], "tattoos": "Arm: Dragon"}
        snapshot = {"name": "Old", "aliases": ["Old"], "tattoos": ""}
        enabled = {"name", "aliases", "tattoos"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        change_map = {c["field"]: c for c in changes}
        assert change_map["name"]["merge_type"] == "name"
        assert change_map["aliases"]["merge_type"] == "alias_list"
        assert change_map["tattoos"]["merge_type"] == "text"

    def test_alias_diff_is_case_insensitive(self):
        local = {"aliases": ["Jane Doe", "JD"]}
        upstream = {"aliases": ["jane doe", "jd"]}
        snapshot = {"aliases": ["jane doe", "jd"]}
        enabled = {"aliases"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 0

    def test_url_diff_treats_as_alias_list(self):
        local = {"urls": ["https://twitter.com/jane"]}
        upstream = {"urls": ["https://twitter.com/jane", "https://instagram.com/jane"]}
        snapshot = {"urls": ["https://twitter.com/jane"]}
        enabled = {"urls"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["merge_type"] == "alias_list"


class TestValuesEqual:
    def test_string_case_insensitive(self):
        assert _values_equal("BROWN", "Brown", "simple") is True
        assert _values_equal("NATURAL", "Natural", "simple") is True
        assert _values_equal("female", "FEMALE", "simple") is True

    def test_string_different_values(self):
        assert _values_equal("BROWN", "BLACK", "simple") is False

    def test_int_strict_equality(self):
        assert _values_equal(165, 165, "simple") is True
        assert _values_equal(165, 170, "simple") is False

    def test_none_handling(self):
        assert _values_equal(None, None, "simple") is True
        assert _values_equal(None, "BROWN", "simple") is False
        assert _values_equal("BROWN", None, "simple") is False

    def test_none_vs_empty_string(self):
        # None and "" are both semantically empty, so treated as equal
        assert _values_equal(None, "", "simple") is True

    def test_alias_list_case_insensitive(self):
        assert _values_equal(["Jane", "JD"], ["jane", "jd"], "alias_list") is True

    def test_alias_list_different_sets(self):
        assert _values_equal(["Jane"], ["Jane", "JD"], "alias_list") is False


class TestDiffCaseInsensitiveStrings:
    def test_case_only_difference_is_not_flagged(self):
        """BROWN vs Brown should NOT be flagged as a change."""
        local = {"eye_color": "Brown", "hair_color": "Black"}
        upstream = {"eye_color": "BROWN", "hair_color": "BLACK"}
        snapshot = {"eye_color": "BROWN", "hair_color": "BLACK"}
        enabled = {"eye_color", "hair_color"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 0

    def test_real_value_change_still_flagged(self):
        """BROWN vs GREEN should still be flagged."""
        local = {"eye_color": "Brown"}
        upstream = {"eye_color": "GREEN"}
        snapshot = {"eye_color": "BROWN"}
        enabled = {"eye_color"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled)
        assert len(changes) == 1
        assert changes[0]["field"] == "eye_color"


class TestFieldMergeTypes:
    def test_all_default_fields_have_merge_types(self):
        for field_name in DEFAULT_PERFORMER_FIELDS:
            assert field_name in FIELD_MERGE_TYPES, (
                f"Field '{field_name}' missing from FIELD_MERGE_TYPES"
            )

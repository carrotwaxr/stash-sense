"""Tests for upstream field mapper and 3-way diff engine."""

from upstream_field_mapper import (
    DEFAULT_PERFORMER_FIELDS,
    FIELD_MERGE_TYPES,
    _normalize_date,
    _values_equal,
    diff_performer_fields,
    diff_tag_fields,
    diff_scene_fields,
    normalize_upstream_performer,
    normalize_upstream_scene,
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
        }
        result = normalize_upstream_performer(upstream)
        assert result["name"] == "Jane Doe"
        assert result["disambiguation"] == "actress"
        assert result["gender"] == "FEMALE"
        assert result["country"] == "US"
        assert result["eye_color"] == "BROWN"
        assert result["hair_color"] == "BLACK"
        assert result["height"] == 165
        # details is not in StashBox schema — local-only field, not normalized
        assert "details" not in result

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

    def test_alias_list_trailing_slash_normalized(self):
        assert _values_equal(
            ["https://www.kink.com/"],
            ["https://www.kink.com"],
            "alias_list",
        ) is True

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


class TestDateNormalization:
    def test_strips_time_component_space_separated(self):
        assert _normalize_date("1978-01-01 00:00:00") == "1978-01-01"

    def test_strips_time_component_t_separated(self):
        assert _normalize_date("2007-06-15T00:00:00Z") == "2007-06-15"

    def test_expands_year_only(self):
        assert _normalize_date("2007") == "2007-01-01"

    def test_expands_year_month(self):
        assert _normalize_date("2007-06") == "2007-06-01"

    def test_full_date_unchanged(self):
        assert _normalize_date("1990-05-15") == "1990-05-15"

    def test_none_passthrough(self):
        assert _normalize_date(None) is None

    def test_empty_string_passthrough(self):
        assert _normalize_date("") == ""


class TestDateComparisonInDiff:
    def test_partial_year_matches_padded_local(self):
        """StashDB '2007' should match Stash's padded '2007-01-01'."""
        local = {"name": "Test", "birthdate": "2007-01-01"}
        upstream = {"name": "Test", "birthdate": "2007"}
        changes = diff_performer_fields(local, upstream, None, {"birthdate"})
        assert len(changes) == 0

    def test_partial_year_differs_from_real_date(self):
        """StashDB '2007' should NOT match '2007-06-15'."""
        local = {"name": "Test", "birthdate": "2007-06-15"}
        upstream = {"name": "Test", "birthdate": "2007"}
        changes = diff_performer_fields(local, upstream, None, {"birthdate"})
        assert len(changes) == 1

    def test_time_component_stripped_for_comparison(self):
        """ThePornDB '1978-01-01 00:00:00' should match '1978-01-01'."""
        local = {"name": "Test", "birthdate": "1978-01-01"}
        upstream = {"name": "Test", "birthdate": "1978-01-01 00:00:00"}
        changes = diff_performer_fields(local, upstream, None, {"birthdate"})
        assert len(changes) == 0

    def test_real_date_difference_still_flagged(self):
        """Genuine date differences should still be flagged."""
        local = {"name": "Test", "birthdate": "1978-08-10"}
        upstream = {"name": "Test", "birthdate": "1978-01-01 00:00:00"}
        changes = diff_performer_fields(local, upstream, None, {"birthdate"})
        assert len(changes) == 1
        assert changes[0]["field"] == "birthdate"

    def test_death_date_also_normalized(self):
        local = {"name": "Test", "death_date": "2017-12-05"}
        upstream = {"name": "Test", "death_date": "2017-12-05 00:00:00"}
        changes = diff_performer_fields(local, upstream, None, {"death_date"})
        assert len(changes) == 0

    def test_both_none_dates_equal(self):
        local = {"name": "Test", "birthdate": None}
        upstream = {"name": "Test", "birthdate": None}
        changes = diff_performer_fields(local, upstream, None, {"birthdate"})
        assert len(changes) == 0


class TestAliasSelfReferenceFiltering:
    def test_upstream_alias_matching_name_ignored(self):
        """Upstream alias 'JAMIE BARRY' should be ignored when name is 'Jamie Barry'."""
        local = {"name": "Jamie Barry", "aliases": ["Ed Wood", "Jamie"]}
        upstream = {"name": "Jamie Barry", "aliases": ["JAMIE BARRY"]}
        changes = diff_performer_fields(local, upstream, None, {"aliases"})
        # After filtering "JAMIE BARRY", upstream aliases is empty set
        # Local has {"ed wood", "jamie"} - these differ, so change IS flagged
        assert len(changes) == 1
        # The output should not contain the self-referencing alias
        assert "JAMIE BARRY" not in changes[0]["upstream_value"]

    def test_no_false_positive_when_only_self_ref_alias(self):
        """If upstream only has self-referencing alias, it equals empty local aliases."""
        local = {"name": "Jamie Barry", "aliases": []}
        upstream = {"name": "Jamie Barry", "aliases": ["JAMIE BARRY"]}
        changes = diff_performer_fields(local, upstream, None, {"aliases"})
        assert len(changes) == 0

    def test_self_ref_in_both_sides_ignored(self):
        """Self-referencing aliases filtered from both sides."""
        local = {"name": "Jane Doe", "aliases": ["Jane Doe", "JD"]}
        upstream = {"name": "Jane Doe", "aliases": ["jane doe", "JD"]}
        changes = diff_performer_fields(local, upstream, None, {"aliases"})
        assert len(changes) == 0

    def test_real_alias_differences_still_flagged(self):
        """Non-self-referencing alias differences should still be detected."""
        local = {"name": "Jane Doe", "aliases": ["JD"]}
        upstream = {"name": "Jane Doe", "aliases": ["Jane Doe", "JD", "Janet"]}
        changes = diff_performer_fields(local, upstream, None, {"aliases"})
        assert len(changes) == 1
        # Self-ref alias filtered out, but "Janet" is a real addition
        assert "Janet" in changes[0]["upstream_value"]
        assert "Jane Doe" not in changes[0]["upstream_value"]


class TestTagAliasSelfReferenceFiltering:
    def test_tag_alias_matching_name_filtered(self):
        """Tag alias matching its own name should be filtered out."""
        local = {"name": "Brunette", "aliases": []}
        upstream = {"name": "Brunette", "aliases": ["Brunette", "Brown Hair"]}
        changes = diff_tag_fields(local, upstream, None, {"aliases"})
        # "Brunette" filtered, only "Brown Hair" remains as real difference
        assert len(changes) == 1
        assert "Brunette" not in changes[0]["upstream_value"]
        assert "Brown Hair" in changes[0]["upstream_value"]

    def test_tag_no_false_positive_self_ref_only(self):
        """Tag with only self-referencing alias should not flag a change."""
        local = {"name": "Anal", "aliases": []}
        upstream = {"name": "Anal", "aliases": ["Anal"]}
        changes = diff_tag_fields(local, upstream, None, {"aliases"})
        assert len(changes) == 0


class TestSceneDateNormalization:
    def test_scene_date_time_stripped(self):
        """Scene date with time component should match date-only."""
        local = {"title": "Test", "date": "2024-01-15"}
        upstream = {"title": "Test", "date": "2024-01-15 00:00:00"}
        result = diff_scene_fields(local, upstream, None, {"date"})
        assert len(result["changes"]) == 0

    def test_scene_date_partial_year_expanded(self):
        """Scene date year-only should match padded date."""
        local = {"title": "Test", "date": "2024-01-01"}
        upstream = {"title": "Test", "date": "2024"}
        result = diff_scene_fields(local, upstream, None, {"date"})
        assert len(result["changes"]) == 0

    def test_scene_date_real_difference_flagged(self):
        """Genuine scene date difference should be flagged."""
        local = {"title": "Test", "date": "2024-01-15"}
        upstream = {"title": "Test", "date": "2024-06-01"}
        result = diff_scene_fields(local, upstream, None, {"date"})
        assert len(result["changes"]) == 1


class TestDateComparisonWithSnapshot:
    def test_normalized_dates_match_snapshot_skips_change(self):
        """If upstream date matches snapshot after normalization, skip (user set local differently)."""
        local = {"name": "Test", "birthdate": "1990-06-15"}
        upstream = {"name": "Test", "birthdate": "1978-01-01 00:00:00"}
        snapshot = {"name": "Test", "birthdate": "1978-01-01 00:00:00"}
        changes = diff_performer_fields(local, upstream, snapshot, {"birthdate"})
        # upstream == snapshot (both normalize to 1978-01-01), so no change flagged
        assert len(changes) == 0

    def test_upstream_changed_from_snapshot_flags_change(self):
        """If upstream date differs from snapshot, flag it."""
        local = {"name": "Test", "birthdate": "1978-01-01"}
        upstream = {"name": "Test", "birthdate": "1978-06-15"}
        snapshot = {"name": "Test", "birthdate": "1978-01-01 00:00:00"}
        changes = diff_performer_fields(local, upstream, snapshot, {"birthdate"})
        assert len(changes) == 1
        assert changes[0]["field"] == "birthdate"


class TestNormalizeUpstreamScene:
    def test_uses_release_date_field(self):
        """StashBox canonical field is release_date, should map to 'date'."""
        upstream = {"title": "Test Scene", "release_date": "2024-06-15"}
        result = normalize_upstream_scene(upstream)
        assert result["date"] == "2024-06-15"

    def test_falls_back_to_date_field(self):
        """Legacy 'date' field should still work when release_date is absent."""
        upstream = {"title": "Test Scene", "date": "2024-06-15"}
        result = normalize_upstream_scene(upstream)
        assert result["date"] == "2024-06-15"

    def test_release_date_takes_precedence_over_date(self):
        """When both exist, release_date should win."""
        upstream = {"title": "Test Scene", "release_date": "2024-06-15", "date": "2023-01-01"}
        result = normalize_upstream_scene(upstream)
        assert result["date"] == "2024-06-15"

    def test_missing_date_fields_default_to_empty(self):
        """When neither date field is present, should default to empty string."""
        upstream = {"title": "Test Scene"}
        result = normalize_upstream_scene(upstream)
        assert result["date"] == ""


class TestFieldMergeTypes:
    def test_all_default_fields_have_merge_types(self):
        for field_name in DEFAULT_PERFORMER_FIELDS:
            assert field_name in FIELD_MERGE_TYPES, (
                f"Field '{field_name}' missing from FIELD_MERGE_TYPES"
            )

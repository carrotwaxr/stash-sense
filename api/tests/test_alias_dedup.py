"""Tests for alias deduplication logic."""

import pytest
from recommendations_router import deduplicate_aliases


class TestDeduplicateAliases:
    def test_removes_alias_matching_own_name(self):
        result = deduplicate_aliases(
            aliases=["Jane Doe", "JD", "jane doe"],
            entity_name="Jane Doe",
            other_entity_names=set(),
        )
        assert result == ["JD"]

    def test_removes_duplicate_aliases_case_insensitive(self):
        result = deduplicate_aliases(
            aliases=["Alpha", "Beta", "alpha", "BETA", "Gamma"],
            entity_name="Test",
            other_entity_names=set(),
        )
        assert result == ["Alpha", "Beta", "Gamma"]

    def test_removes_alias_matching_other_entity_name(self):
        result = deduplicate_aliases(
            aliases=["Brunette", "Dark Hair"],
            entity_name="Brown Hair",
            other_entity_names={"brunette", "blonde"},
        )
        assert result == ["Dark Hair"]

    def test_combined_dedup(self):
        result = deduplicate_aliases(
            aliases=["Brown Hair", "Brunette", "brown hair", "Dark", "brunette"],
            entity_name="Brown Hair",
            other_entity_names={"brunette"},
        )
        assert result == ["Dark"]

    def test_empty_aliases(self):
        result = deduplicate_aliases(
            aliases=[],
            entity_name="Test",
            other_entity_names=set(),
        )
        assert result == []

    def test_none_and_empty_string_aliases_filtered(self):
        result = deduplicate_aliases(
            aliases=["Valid", "", None, "  "],
            entity_name="Test",
            other_entity_names=set(),
        )
        assert result == ["Valid"]

    def test_preserves_order_keeps_first_occurrence(self):
        result = deduplicate_aliases(
            aliases=["Zeta", "Alpha", "zeta", "Beta"],
            entity_name="Test",
            other_entity_names=set(),
        )
        assert result == ["Zeta", "Alpha", "Beta"]

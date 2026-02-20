"""Tests for endpoint priority storage and retrieval."""

import pytest
from recommendations_db import RecommendationsDB


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(tmp_path / "test.db")


class TestEndpointPriority:
    def test_get_priorities_returns_empty_list_when_unset(self, db):
        """No priorities configured returns empty list."""
        assert db.get_endpoint_priorities() == []

    def test_set_and_get_priorities(self, db):
        """Setting priorities and retrieving them preserves order."""
        endpoints = [
            "https://stashdb.org/graphql",
            "https://fansdb.cc/graphql",
            "https://theporndb.net/graphql",
        ]
        db.set_endpoint_priorities(endpoints)
        assert db.get_endpoint_priorities() == endpoints

    def test_set_priorities_overwrites_previous(self, db):
        """Setting priorities again replaces the old list."""
        db.set_endpoint_priorities(["https://a.com/graphql", "https://b.com/graphql"])
        db.set_endpoint_priorities(["https://b.com/graphql", "https://a.com/graphql"])
        assert db.get_endpoint_priorities() == ["https://b.com/graphql", "https://a.com/graphql"]

    def test_get_endpoint_priority_rank(self, db):
        """Priority rank returns 0-indexed position, or None if not in list."""
        db.set_endpoint_priorities([
            "https://stashdb.org/graphql",
            "https://fansdb.cc/graphql",
        ])
        assert db.get_endpoint_priority_rank("https://stashdb.org/graphql") == 0
        assert db.get_endpoint_priority_rank("https://fansdb.cc/graphql") == 1
        assert db.get_endpoint_priority_rank("https://unknown.com/graphql") is None

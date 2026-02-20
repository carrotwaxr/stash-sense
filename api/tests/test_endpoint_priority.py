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

    def test_empty_list_clears_priorities(self, db):
        """Setting an empty list clears any previously stored priorities."""
        db.set_endpoint_priorities(["https://a.com/graphql"])
        db.set_endpoint_priorities([])
        assert db.get_endpoint_priorities() == []


from httpx import AsyncClient, ASGITransport
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.fixture
def mock_rec_db(db):
    """Patch get_rec_db to return our test DB."""
    with patch("settings_router.get_rec_db", return_value=db):
        yield db


@pytest.fixture
def mock_connection_manager():
    """Patch connection manager to return test connections."""
    mgr = MagicMock()
    mgr.get_connections.return_value = [
        {"endpoint": "https://stashdb.org/graphql", "name": "StashDB", "domain": "stashdb.org"},
        {"endpoint": "https://fansdb.cc/graphql", "name": "FansDB", "domain": "fansdb.cc"},
    ]
    with patch("settings_router.get_connection_manager", return_value=mgr):
        yield mgr


@pytest.fixture
async def client(mock_rec_db, mock_connection_manager):
    from main import app
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestEndpointPriorityAPI:
    @pytest.mark.asyncio
    async def test_get_priorities_empty(self, client):
        """GET returns connections in default order when no priorities set."""
        resp = await client.get("/settings/endpoint-priorities")
        assert resp.status_code == 200
        data = resp.json()
        assert "endpoints" in data
        assert len(data["endpoints"]) == 2
        assert data["endpoints"][0]["endpoint"] == "https://stashdb.org/graphql"

    @pytest.mark.asyncio
    async def test_set_and_get_priorities(self, client):
        """POST sets priority order, GET reflects it."""
        resp = await client.post("/settings/endpoint-priorities", json={
            "endpoints": ["https://fansdb.cc/graphql", "https://stashdb.org/graphql"]
        })
        assert resp.status_code == 200

        resp = await client.get("/settings/endpoint-priorities")
        data = resp.json()
        assert data["endpoints"][0]["endpoint"] == "https://fansdb.cc/graphql"
        assert data["endpoints"][1]["endpoint"] == "https://stashdb.org/graphql"


from tests.test_base_upstream_analyzer import ConcreteUpstreamAnalyzer


class TestPriorityAwareProcessing:
    """Test that BaseUpstreamAnalyzer respects endpoint priorities."""

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.fixture
    def mock_stash_multi_endpoint(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://fansdb.cc/graphql", "api_key": "key1"},
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key2"},
        ])
        # Entity "42" is linked to BOTH endpoints
        stash.get_test_entities_for_endpoint = AsyncMock(side_effect=lambda ep: [
            {
                "id": "42",
                "name": "Local Name",
                "description": "Local desc",
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-uuid-1"},
                    {"endpoint": "https://fansdb.cc/graphql", "stash_id": "fb-uuid-1"},
                ],
            }
        ])
        return stash

    @pytest.mark.asyncio
    async def test_entity_on_multiple_endpoints_only_processed_by_highest_priority(
        self, mock_stash_multi_endpoint, rec_db
    ):
        """Entity linked to 2 endpoints should only generate a recommendation
        from the highest-priority endpoint."""
        # Set StashDB as highest priority
        rec_db.set_endpoint_priorities([
            "https://stashdb.org/graphql",
            "https://fansdb.cc/graphql",
        ])

        upstream_data = {
            "name": "Upstream Name",
            "description": "Upstream desc",
            "updated": "2026-01-01T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc_instance = MagicMock()
            mock_sbc_instance.get_test_entity = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc_instance

            analyzer = ConcreteUpstreamAnalyzer(mock_stash_multi_endpoint, rec_db)
            result = await analyzer.run()

        # Should have processed entity once (from StashDB), not twice
        recs = rec_db.get_recommendations(type="upstream_test_entity_changes", status="pending")
        assert len(recs) == 1
        assert recs[0].details["endpoint"] == "https://stashdb.org/graphql"

    @pytest.mark.asyncio
    async def test_no_priority_set_processes_all_endpoints(
        self, mock_stash_multi_endpoint, rec_db
    ):
        """Without priorities, all endpoints are processed (backward compatible)."""
        upstream_data = {
            "name": "Upstream Name",
            "description": "Upstream desc",
            "updated": "2026-01-01T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc_instance = MagicMock()
            mock_sbc_instance.get_test_entity = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc_instance

            analyzer = ConcreteUpstreamAnalyzer(mock_stash_multi_endpoint, rec_db)
            result = await analyzer.run()

        # Without priority, both endpoints process the entity.
        # The UNIQUE(type, target_type, target_id) constraint means only one rec exists,
        # but it should have been processed by both endpoints (second overwrites first)
        recs = rec_db.get_recommendations(type="upstream_test_entity_changes", status="pending")
        assert len(recs) >= 1


class TestPriorityIntegration:
    """Integration test: entity linked to 2 endpoints, priority set,
    verify only 1 recommendation generated from the correct endpoint."""

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_full_priority_flow(self, rec_db):
        """Set priority, run analyzer, verify correct endpoint wins."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://fansdb.cc/graphql", "api_key": "key1"},
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key2"},
        ])

        # Same entity on both endpoints
        entity = {
            "id": "42",
            "name": "Local Name",
            "description": "Local desc",
            "stash_ids": [
                {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-1"},
                {"endpoint": "https://fansdb.cc/graphql", "stash_id": "fb-1"},
            ],
        }
        stash.get_test_entities_for_endpoint = AsyncMock(return_value=[entity])

        # StashDB has a change, FansDB also has a (different) change
        stashdb_data = {"name": "StashDB Name", "description": "desc", "updated": "2026-01-01T00:00:00Z"}
        fansdb_data = {"name": "FansDB Name", "description": "desc", "updated": "2026-01-01T00:00:00Z"}

        def get_upstream(stash_id):
            if stash_id == "sb-1":
                return stashdb_data
            return fansdb_data

        # Set StashDB as higher priority
        rec_db.set_endpoint_priorities([
            "https://stashdb.org/graphql",
            "https://fansdb.cc/graphql",
        ])

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(side_effect=get_upstream)
            MockSBC.return_value = mock_sbc

            analyzer = ConcreteUpstreamAnalyzer(stash, rec_db)
            result = await analyzer.run()

        recs = rec_db.get_recommendations(
            type="upstream_test_entity_changes", status="pending"
        )
        assert len(recs) == 1
        assert recs[0].details["endpoint"] == "https://stashdb.org/graphql"
        # Verify it used StashDB's data (name changed to "StashDB Name")
        changes = recs[0].details.get("changes", [])
        name_change = next((c for c in changes if c["field"] == "name"), None)
        assert name_change is not None
        assert name_change["upstream_value"] == "StashDB Name"

    @pytest.mark.asyncio
    async def test_entity_only_on_low_priority_still_processed(self, rec_db):
        """Entity only linked to a lower-priority endpoint should still get processed."""
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1"},
            {"endpoint": "https://fansdb.cc/graphql", "api_key": "key2"},
        ])

        # Entity only on FansDB (lower priority)
        def get_entities(ep):
            if ep == "https://fansdb.cc/graphql":
                return [{
                    "id": "99",
                    "name": "Local Name",
                    "description": "Local desc",
                    "stash_ids": [
                        {"endpoint": "https://fansdb.cc/graphql", "stash_id": "fb-99"},
                    ],
                }]
            return []  # StashDB has no entities

        stash.get_test_entities_for_endpoint = AsyncMock(side_effect=get_entities)

        # Set StashDB as higher priority
        rec_db.set_endpoint_priorities([
            "https://stashdb.org/graphql",
            "https://fansdb.cc/graphql",
        ])

        upstream_data = {"name": "Upstream Name", "description": "desc", "updated": "2026-01-01T00:00:00Z"}

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            analyzer = ConcreteUpstreamAnalyzer(stash, rec_db)
            result = await analyzer.run()

        # Should still generate a recommendation from FansDB
        recs = rec_db.get_recommendations(type="upstream_test_entity_changes", status="pending")
        assert len(recs) == 1
        assert recs[0].details["endpoint"] == "https://fansdb.cc/graphql"

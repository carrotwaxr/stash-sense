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

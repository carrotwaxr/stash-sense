"""Tests for StashBox client studio methods."""

import pytest
from unittest.mock import AsyncMock


class TestGetStudio:
    @pytest.mark.asyncio
    async def test_get_studio_returns_data(self):
        from stashbox_client import StashBoxClient
        client = StashBoxClient("https://stashdb.org/graphql", "test-key")
        client._execute = AsyncMock(return_value={
            "findStudio": {
                "id": "studio-uuid-1", "name": "Brazzers",
                "urls": [{"url": "https://brazzers.com"}],
                "parent": None, "deleted": False,
                "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
            }
        })
        result = await client.get_studio("studio-uuid-1")
        assert result is not None
        assert result["name"] == "Brazzers"
        client._execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_studio_returns_none_when_not_found(self):
        from stashbox_client import StashBoxClient
        client = StashBoxClient("https://stashdb.org/graphql", "test-key")
        client._execute = AsyncMock(return_value={"findStudio": None})
        result = await client.get_studio("nonexistent-uuid")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_studio_includes_parent(self):
        from stashbox_client import StashBoxClient
        client = StashBoxClient("https://stashdb.org/graphql", "test-key")
        client._execute = AsyncMock(return_value={
            "findStudio": {
                "id": "sub-uuid", "name": "Brazzers Exxtra", "urls": [],
                "parent": {"id": "parent-uuid", "name": "Brazzers"},
                "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
            }
        })
        result = await client.get_studio("sub-uuid")
        assert result["parent"]["id"] == "parent-uuid"

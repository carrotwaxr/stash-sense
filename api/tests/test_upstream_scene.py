"""Tests for upstream scene sync."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStashBoxSceneQuery:
    @pytest.mark.asyncio
    async def test_get_scene_returns_scene_data(self):
        """get_scene fetches a scene by ID with all required fields."""
        from stashbox_client import StashBoxClient

        mock_response = {
            "findScene": {
                "id": "scene-uuid-1",
                "title": "Test Scene",
                "details": "A test scene",
                "date": "2025-01-15",
                "urls": [{"url": "https://example.com/scene1", "site": {"name": "Example"}}],
                "studio": {"id": "studio-uuid-1", "name": "Test Studio"},
                "tags": [{"id": "tag-uuid-1", "name": "HD"}],
                "performers": [
                    {"performer": {"id": "perf-uuid-1", "name": "Jane Doe"}, "as": "Jane Smith"}
                ],
                "director": "John Director",
                "code": "TS-001",
                "deleted": False,
                "created": "2025-01-01T00:00:00Z",
                "updated": "2025-01-15T00:00:00Z",
            }
        }

        client = StashBoxClient("https://test.box/graphql", "key")
        client._execute = AsyncMock(return_value=mock_response)

        result = await client.get_scene("scene-uuid-1")
        assert result is not None
        assert result["title"] == "Test Scene"
        assert result["studio"]["id"] == "studio-uuid-1"
        assert len(result["performers"]) == 1
        assert result["performers"][0]["as"] == "Jane Smith"

    @pytest.mark.asyncio
    async def test_get_scene_returns_none_for_missing(self):
        """get_scene returns None when scene not found."""
        from stashbox_client import StashBoxClient

        client = StashBoxClient("https://test.box/graphql", "key")
        client._execute = AsyncMock(return_value={"findScene": None})

        result = await client.get_scene("nonexistent")
        assert result is None

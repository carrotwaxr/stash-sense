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


class TestStashSceneQueries:
    @pytest.mark.asyncio
    async def test_get_scenes_for_endpoint(self):
        """get_scenes_for_endpoint returns scenes linked to a specific endpoint."""
        from stash_client_unified import StashClientUnified

        mock_response = {
            "findScenes": {
                "scenes": [
                    {
                        "id": "1",
                        "title": "Scene One",
                        "date": "2025-01-01",
                        "details": "Details here",
                        "director": "Director",
                        "code": "SC-001",
                        "urls": ["https://example.com/1"],
                        "studio": {"id": "10", "name": "Studio A", "stash_ids": []},
                        "performers": [{"id": "20", "name": "Perf A", "stash_ids": []}],
                        "tags": [{"id": "30", "name": "Tag A", "stash_ids": []}],
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                        ],
                    }
                ]
            }
        }

        client = StashClientUnified("http://localhost:9999", "key")
        client._execute = AsyncMock(return_value=mock_response)

        scenes = await client.get_scenes_for_endpoint("https://stashdb.org/graphql")
        assert len(scenes) == 1
        assert scenes[0]["title"] == "Scene One"
        assert scenes[0]["stash_ids"][0]["stash_id"] == "sb-scene-1"

    @pytest.mark.asyncio
    async def test_update_scene(self):
        """update_scene sends mutation with correct fields."""
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999", "key")
        client._execute = AsyncMock(return_value={"sceneUpdate": {"id": "1"}})

        result = await client.update_scene("1", title="New Title", date="2025-02-01")
        assert result["id"] == "1"
        # Verify the input dict was passed correctly
        call_args = client._execute.call_args
        # _execute is called as (query, {"input": input_dict}, priority=Priority.CRITICAL)
        # positional args: call_args[0][1] is the variables dict
        variables = call_args[0][1]
        input_dict = variables["input"]
        assert input_dict["id"] == "1"
        assert input_dict["title"] == "New Title"
        assert input_dict["date"] == "2025-02-01"

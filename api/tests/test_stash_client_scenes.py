"""Tests for Stash client scene query methods."""

import pytest
from unittest.mock import AsyncMock, patch


class TestGetScenesForFingerprinting:
    """Tests for get_scenes_for_fingerprinting method."""

    @pytest.mark.asyncio
    async def test_returns_scenes_with_metadata(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {
            "findScenes": {
                "count": 2,
                "scenes": [
                    {
                        "id": "1",
                        "title": "Test Scene",
                        "date": "2024-01-15",
                        "updated_at": "2024-01-20T00:00:00Z",
                        "studio": {"id": "s1", "name": "Test Studio"},
                        "performers": [{"id": "p1", "name": "Performer One"}],
                        "files": [{"duration": 1800}],
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}
                        ],
                    },
                    {
                        "id": "2",
                        "title": "Another Scene",
                        "date": None,
                        "updated_at": "2024-01-21T00:00:00Z",
                        "studio": None,
                        "performers": [],
                        "files": [],
                        "stash_ids": [],
                    },
                ],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            scenes, count = await client.get_scenes_for_fingerprinting(limit=10)

            assert count == 2
            assert len(scenes) == 2
            assert scenes[0]["id"] == "1"
            assert scenes[0]["studio"]["name"] == "Test Studio"

    @pytest.mark.asyncio
    async def test_filters_by_updated_after(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"findScenes": {"count": 0, "scenes": []}}

            await client.get_scenes_for_fingerprinting(
                updated_after="2024-01-15T00:00:00Z"
            )

            # Check that the filter was passed
            call_args = mock_execute.call_args
            variables = call_args[1]["variables"] if call_args[1] else call_args[0][1]
            assert variables["scene_filter"]["updated_at"]["value"] == "2024-01-15T00:00:00Z"


class TestGetSceneStreamUrl:
    """Tests for get_scene_stream_url method."""

    @pytest.mark.asyncio
    async def test_returns_stream_url(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {
            "findScene": {
                "id": "123",
                "sceneStreams": [
                    {"url": "http://localhost:9999/scene/123/stream.mp4", "label": "Direct stream"},
                ],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            url = await client.get_scene_stream_url("123")

            assert url == "http://localhost:9999/scene/123/stream.mp4"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_streams(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {"findScene": {"id": "123", "sceneStreams": []}}

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            url = await client.get_scene_stream_url("123")

            assert url is None

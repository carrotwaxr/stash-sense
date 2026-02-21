"""Tests for scene fingerprint query in Stash client."""

import pytest
from unittest.mock import AsyncMock, patch


@pytest.mark.asyncio
async def test_get_scenes_with_fingerprints_returns_expected_shape():
    from stash_client_unified import StashClientUnified

    mock_response = {
        "findScenes": {
            "count": 1,
            "scenes": [
                {
                    "id": "42",
                    "title": "Test Scene",
                    "updated_at": "2026-01-15T00:00:00Z",
                    "files": [
                        {
                            "id": "f1",
                            "duration": 1835.5,
                            "fingerprints": [
                                {"type": "md5", "value": "abc123"},
                                {"type": "oshash", "value": "def456"},
                                {"type": "phash", "value": "0011223344556677"},
                            ],
                        }
                    ],
                    "stash_ids": [
                        {"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-1"}
                    ],
                }
            ],
        }
    }

    client = StashClientUnified.__new__(StashClientUnified)
    client._execute = AsyncMock(return_value=mock_response)

    scenes, total = await client.get_scenes_with_fingerprints()

    assert total == 1
    assert len(scenes) == 1
    scene = scenes[0]
    assert scene["id"] == "42"
    assert len(scene["files"]) == 1
    assert len(scene["files"][0]["fingerprints"]) == 3
    assert scene["files"][0]["fingerprints"][0]["type"] == "md5"


@pytest.mark.asyncio
async def test_get_scenes_with_fingerprints_incremental():
    from stash_client_unified import StashClientUnified

    mock_response = {"findScenes": {"count": 0, "scenes": []}}

    client = StashClientUnified.__new__(StashClientUnified)
    client._execute = AsyncMock(return_value=mock_response)

    await client.get_scenes_with_fingerprints(updated_after="2026-01-01T00:00:00Z")

    call_args = client._execute.call_args
    variables = call_args[0][1]
    assert "scene_filter" in variables
    assert "updated_at" in variables["scene_filter"]
    assert variables["scene_filter"]["updated_at"]["value"] == "2026-01-01T00:00:00Z"

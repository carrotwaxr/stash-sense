"""Tests for stash-box batch fingerprint lookup."""

import pytest
from unittest.mock import AsyncMock


@pytest.mark.asyncio
async def test_find_scenes_by_fingerprints_returns_matched_scenes():
    from stashbox_client import StashBoxClient

    mock_response = {
        "findScenesBySceneFingerprints": [
            [
                {
                    "id": "sb-uuid-1",
                    "title": "Matched Scene",
                    "date": "2024-01-15",
                    "studio": {"id": "st-1", "name": "Studio A"},
                    "performers": [{"performer": {"id": "p-1", "name": "Actor A"}, "as": None}],
                    "urls": [{"url": "https://example.com", "site": {"name": "Example"}}],
                    "fingerprints": [
                        {"hash": "abc123", "algorithm": "MD5", "duration": 1834, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"},
                        {"hash": "def456", "algorithm": "OSHASH", "duration": 1834, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
                    ],
                    "duration": 1834,
                }
            ],
            [],
        ]
    }

    client = StashBoxClient.__new__(StashBoxClient)
    client._execute = AsyncMock(return_value=mock_response)

    fingerprint_sets = [
        [{"hash": "abc123", "algorithm": "MD5"}, {"hash": "def456", "algorithm": "OSHASH"}],
        [{"hash": "xyz789", "algorithm": "MD5"}],
    ]

    results = await client.find_scenes_by_fingerprints(fingerprint_sets)

    assert len(results) == 2
    assert len(results[0]) == 1
    assert results[0][0]["id"] == "sb-uuid-1"
    assert len(results[0][0]["fingerprints"]) == 2
    assert len(results[1]) == 0


@pytest.mark.asyncio
async def test_find_scenes_by_fingerprints_empty_input():
    from stashbox_client import StashBoxClient

    client = StashBoxClient.__new__(StashBoxClient)
    client._execute = AsyncMock()

    results = await client.find_scenes_by_fingerprints([])

    assert results == []
    client._execute.assert_not_called()

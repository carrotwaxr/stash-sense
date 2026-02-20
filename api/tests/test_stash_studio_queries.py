"""Tests for Stash client studio query/mutation methods."""

import pytest
from unittest.mock import AsyncMock


class TestGetStudiosForEndpoint:
    @pytest.mark.asyncio
    async def test_returns_studios_matching_endpoint(self):
        from stash_client_unified import StashClientUnified
        client = StashClientUnified("http://localhost:9999", "test-key")
        client._execute = AsyncMock(return_value={
            "findStudios": {"studios": [
                {"id": "1", "name": "Brazzers", "url": "https://brazzers.com",
                 "parent_studio": None,
                 "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}]}
            ]}
        })
        result = await client.get_studios_for_endpoint("https://stashdb.org/graphql")
        assert len(result) == 1
        assert result[0]["name"] == "Brazzers"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_none(self):
        from stash_client_unified import StashClientUnified
        client = StashClientUnified("http://localhost:9999", "test-key")
        client._execute = AsyncMock(return_value={"findStudios": {"studios": []}})
        result = await client.get_studios_for_endpoint("https://stashdb.org/graphql")
        assert result == []


class TestUpdateStudio:
    @pytest.mark.asyncio
    async def test_sends_update_mutation(self):
        from stash_client_unified import StashClientUnified
        client = StashClientUnified("http://localhost:9999", "test-key")
        client._execute = AsyncMock(return_value={"studioUpdate": {"id": "1"}})
        result = await client.update_studio("1", name="Updated Name")
        assert result["id"] == "1"
        call_args = client._execute.call_args
        assert "studioUpdate" in call_args[0][0]
        input_dict = call_args[1].get("variables") or call_args[0][1]
        assert input_dict["input"]["id"] == "1"
        assert input_dict["input"]["name"] == "Updated Name"


class TestCreateStudio:
    @pytest.mark.asyncio
    async def test_sends_create_mutation(self):
        from stash_client_unified import StashClientUnified
        client = StashClientUnified("http://localhost:9999", "test-key")
        client._execute = AsyncMock(return_value={"studioCreate": {"id": "99", "name": "New Studio"}})
        result = await client.create_studio(
            name="New Studio", url="https://new.com",
            stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-1"}],
        )
        assert result["id"] == "99"
        assert "studioCreate" in client._execute.call_args[0][0]

    @pytest.mark.asyncio
    async def test_create_with_parent(self):
        from stash_client_unified import StashClientUnified
        client = StashClientUnified("http://localhost:9999", "test-key")
        client._execute = AsyncMock(return_value={"studioCreate": {"id": "100", "name": "Sub Studio"}})
        result = await client.create_studio(
            name="Sub Studio",
            stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-2"}],
            parent_id="99",
        )
        assert result["id"] == "100"
        input_dict = (client._execute.call_args[1].get("variables") or client._execute.call_args[0][1])
        assert input_dict["input"]["parent_id"] == "99"

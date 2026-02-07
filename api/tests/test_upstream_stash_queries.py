"""Tests for upstream Stash client performer query and update methods."""

import pytest
from unittest.mock import AsyncMock, patch


class TestGetPerformersForEndpoint:
    """Tests for get_performers_for_endpoint method."""

    @pytest.mark.asyncio
    async def test_returns_performers_with_stash_ids(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {
            "findPerformers": {
                "performers": [
                    {
                        "id": "1",
                        "name": "Jane Doe",
                        "disambiguation": "",
                        "alias_list": ["JD"],
                        "gender": "FEMALE",
                        "birthdate": "1990-01-15",
                        "death_date": None,
                        "ethnicity": "caucasian",
                        "country": "US",
                        "eye_color": "blue",
                        "hair_color": "blonde",
                        "height_cm": 170,
                        "measurements": "34-26-36",
                        "fake_tits": "No",
                        "career_length": "2010-2020",
                        "tattoos": "None",
                        "piercings": "Ears",
                        "details": "Some bio text",
                        "urls": ["https://example.com"],
                        "favorite": False,
                        "image_path": "/images/1.jpg",
                        "stash_ids": [
                            {
                                "endpoint": "https://stashdb.org/graphql",
                                "stash_id": "abc-123",
                            }
                        ],
                    },
                    {
                        "id": "2",
                        "name": "John Smith",
                        "disambiguation": "actor",
                        "alias_list": [],
                        "gender": "MALE",
                        "birthdate": None,
                        "death_date": None,
                        "ethnicity": None,
                        "country": "UK",
                        "eye_color": None,
                        "hair_color": None,
                        "height_cm": None,
                        "measurements": None,
                        "fake_tits": None,
                        "career_length": None,
                        "tattoos": None,
                        "piercings": None,
                        "details": None,
                        "urls": [],
                        "favorite": True,
                        "image_path": None,
                        "stash_ids": [
                            {
                                "endpoint": "https://stashdb.org/graphql",
                                "stash_id": "def-456",
                            }
                        ],
                    },
                ]
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            performers = await client.get_performers_for_endpoint(
                "https://stashdb.org/graphql"
            )

            assert len(performers) == 2
            assert performers[0]["id"] == "1"
            assert performers[0]["name"] == "Jane Doe"
            assert performers[0]["stash_ids"][0]["stash_id"] == "abc-123"
            assert performers[1]["id"] == "2"
            assert performers[1]["name"] == "John Smith"
            assert performers[1]["favorite"] is True

    @pytest.mark.asyncio
    async def test_uses_stash_ids_endpoint_filter(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"findPerformers": {"performers": []}}

            await client.get_performers_for_endpoint("https://stashdb.org/graphql")

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args
            # Variables are the second positional argument
            variables = call_args[0][1]
            assert "performer_filter" in variables
            pf = variables["performer_filter"]
            assert "stash_id_endpoint" in pf
            assert pf["stash_id_endpoint"]["endpoint"] == "https://stashdb.org/graphql"
            assert pf["stash_id_endpoint"]["modifier"] == "NOT_NULL"

    @pytest.mark.asyncio
    async def test_returns_empty_list_when_no_performers(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"findPerformers": {"performers": []}}

            performers = await client.get_performers_for_endpoint(
                "https://stashdb.org/graphql"
            )

            assert performers == []


class TestUpdatePerformer:
    """Tests for update_performer method."""

    @pytest.mark.asyncio
    async def test_sends_performer_update_mutation(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {
            "performerUpdate": {
                "id": "42",
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            result = await client.update_performer(
                "42", name="Jane Smith", height_cm=168
            )

            mock_execute.assert_called_once()
            call_args = mock_execute.call_args

            # Verify the query contains performerUpdate
            query = call_args[0][0]
            assert "performerUpdate" in query

            # Verify variables include id and the provided fields
            variables = call_args[0][1]
            assert variables["input"]["id"] == "42"
            assert variables["input"]["name"] == "Jane Smith"
            assert variables["input"]["height_cm"] == 168

            # Verify result is returned
            assert result == {"id": "42"}

    @pytest.mark.asyncio
    async def test_uses_critical_priority(self):
        from stash_client_unified import StashClientUnified
        from rate_limiter import Priority

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"performerUpdate": {"id": "42"}}

            await client.update_performer("42", name="Updated Name")

            call_kwargs = mock_execute.call_args[1]
            assert call_kwargs["priority"] == Priority.CRITICAL

    @pytest.mark.asyncio
    async def test_handles_single_field_update(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"performerUpdate": {"id": "10"}}

            result = await client.update_performer("10", country="US")

            variables = mock_execute.call_args[0][1]
            assert variables["input"] == {"id": "10", "country": "US"}
            assert result == {"id": "10"}

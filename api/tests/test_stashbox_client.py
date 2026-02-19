"""Tests for StashBoxClient - queries stash-box endpoints (StashDB, FansDB, etc.)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from rate_limiter import RateLimiter


class TestStashBoxClientInit:
    """Tests for StashBoxClient initialization."""

    def test_init_sets_endpoint(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")
        assert client.endpoint == "https://stashdb.org/graphql"

    def test_init_sets_headers_with_api_key(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")
        assert client.headers["ApiKey"] == "test-key"
        assert client.headers["Content-Type"] == "application/json"
        assert client.headers["Accept"] == "application/json"

    def test_init_sets_headers_without_api_key(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql")
        assert "ApiKey" not in client.headers
        assert client.headers["Content-Type"] == "application/json"
        assert client.headers["Accept"] == "application/json"

    def test_init_with_empty_api_key_omits_header(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="")
        assert "ApiKey" not in client.headers

    def test_init_stores_rate_limiter(self):
        from stashbox_client import StashBoxClient

        limiter = RateLimiter(requests_per_second=2.0)
        client = StashBoxClient(
            endpoint="https://stashdb.org/graphql", api_key="key",
            rate_limiter=limiter,
        )
        assert client._rate_limiter is limiter

    def test_init_without_rate_limiter_defaults_to_none(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="key")
        assert client._rate_limiter is None


class TestStashBoxClientExecute:
    """Tests for StashBoxClient._execute method."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        RateLimiter.reset_instance()
        yield
        RateLimiter.reset_instance()

    def _make_mock_limiter(self):
        """Create a mock RateLimiter that works with async context manager."""
        mock_limiter = MagicMock()
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=None)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        # acquire() is a regular method returning a context manager, not a coroutine
        mock_limiter.acquire.return_value = mock_ctx
        return mock_limiter

    def _make_mock_response(self, json_data):
        """Create a mock httpx Response with sync json() and raise_for_status()."""
        mock_response = MagicMock()
        mock_response.json.return_value = json_data
        mock_response.raise_for_status.return_value = None
        return mock_response

    def _make_mock_http_client(self, mock_response):
        """Create a mock httpx.AsyncClient that returns mock_response on post."""
        mock_async_client = AsyncMock()
        mock_async_client.post.return_value = mock_response
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        return mock_async_client

    @pytest.mark.asyncio
    async def test_execute_posts_to_endpoint(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_response = self._make_mock_response({"data": {"test": "value"}})

        with patch("stashbox_client.httpx.AsyncClient") as mock_client_cls:
            mock_async_client = self._make_mock_http_client(mock_response)
            mock_client_cls.return_value = mock_async_client

            mock_limiter = self._make_mock_limiter()
            with patch("stashbox_client.RateLimiter.get_instance", return_value=mock_limiter):
                result = await client._execute("query { test }")

                mock_async_client.post.assert_called_once()
                call_args = mock_async_client.post.call_args
                assert call_args[0][0] == "https://stashdb.org/graphql"
                assert call_args[1]["json"]["query"] == "query { test }"
                assert result == {"test": "value"}

    @pytest.mark.asyncio
    async def test_execute_raises_on_graphql_errors(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_response = self._make_mock_response(
            {"errors": [{"message": "Something went wrong"}]}
        )

        with patch("stashbox_client.httpx.AsyncClient") as mock_client_cls:
            mock_async_client = self._make_mock_http_client(mock_response)
            mock_client_cls.return_value = mock_async_client

            mock_limiter = self._make_mock_limiter()
            with patch("stashbox_client.RateLimiter.get_instance", return_value=mock_limiter):
                with pytest.raises(RuntimeError, match="GraphQL error"):
                    await client._execute("query { bad }")

    @pytest.mark.asyncio
    async def test_execute_uses_injected_rate_limiter(self):
        """When a rate_limiter is injected, it should be used instead of the singleton."""
        from stashbox_client import StashBoxClient

        injected_limiter = self._make_mock_limiter()
        client = StashBoxClient(
            endpoint="https://stashdb.org/graphql", api_key="test-key",
            rate_limiter=injected_limiter,
        )

        mock_response = self._make_mock_response({"data": {"test": "value"}})

        with patch("stashbox_client.httpx.AsyncClient") as mock_client_cls:
            mock_async_client = self._make_mock_http_client(mock_response)
            mock_client_cls.return_value = mock_async_client

            # Should NOT call RateLimiter.get_instance since we have an injected limiter
            with patch("stashbox_client.RateLimiter.get_instance") as mock_get_instance:
                await client._execute("query { test }")
                mock_get_instance.assert_not_called()

            # The injected limiter should have been used
            injected_limiter.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_falls_back_to_global_limiter(self):
        """When no rate_limiter is injected, it should fall back to the global singleton."""
        from stashbox_client import StashBoxClient

        client = StashBoxClient(
            endpoint="https://stashdb.org/graphql", api_key="test-key",
        )

        mock_response = self._make_mock_response({"data": {"test": "value"}})
        global_limiter = self._make_mock_limiter()

        with patch("stashbox_client.httpx.AsyncClient") as mock_client_cls:
            mock_async_client = self._make_mock_http_client(mock_response)
            mock_client_cls.return_value = mock_async_client

            with patch("stashbox_client.RateLimiter.get_instance", return_value=global_limiter):
                await client._execute("query { test }")

            global_limiter.acquire.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_sends_variables(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_response = self._make_mock_response({"data": {"result": True}})

        with patch("stashbox_client.httpx.AsyncClient") as mock_client_cls:
            mock_async_client = self._make_mock_http_client(mock_response)
            mock_client_cls.return_value = mock_async_client

            mock_limiter = self._make_mock_limiter()
            with patch("stashbox_client.RateLimiter.get_instance", return_value=mock_limiter):
                await client._execute("query { test }", variables={"id": "abc"})

                call_args = mock_async_client.post.call_args
                assert call_args[1]["json"]["variables"] == {"id": "abc"}


class TestQueryPerformers:
    """Tests for StashBoxClient.query_performers method."""

    @pytest.mark.asyncio
    async def test_query_performers_returns_performers_and_count(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_data = {
            "queryPerformers": {
                "count": 2,
                "performers": [
                    {
                        "id": "perf-1",
                        "name": "Performer One",
                        "disambiguation": "",
                        "aliases": ["Alias A"],
                        "gender": "FEMALE",
                        "birth_date": "1990-01-15",
                        "death_date": None,
                        "ethnicity": "CAUCASIAN",
                        "country": "US",
                        "eye_color": "BROWN",
                        "hair_color": "BLACK",
                        "height": 165,
                        "cup_size": "C",
                        "band_size": 32,
                        "waist_size": 24,
                        "hip_size": 34,
                        "breast_type": "NATURAL",
                        "career_start_year": 2015,
                        "career_end_year": None,
                        "tattoos": [{"location": "arm", "description": "rose"}],
                        "piercings": [{"location": "navel", "description": "ring"}],
                        "urls": [{"url": "https://example.com", "type": "HOMEPAGE"}],
                        "is_favorite": False,
                        "deleted": False,
                        "merged_into_id": None,
                        "created": "2024-01-01T00:00:00Z",
                        "updated": "2024-06-15T00:00:00Z",
                    },
                    {
                        "id": "perf-2",
                        "name": "Performer Two",
                        "disambiguation": "the second",
                        "aliases": [],
                        "gender": "MALE",
                        "birth_date": None,
                        "death_date": None,
                        "ethnicity": None,
                        "country": None,
                        "eye_color": None,
                        "hair_color": None,
                        "height": None,
                        "cup_size": None,
                        "band_size": None,
                        "waist_size": None,
                        "hip_size": None,
                        "breast_type": None,
                        "career_start_year": None,
                        "career_end_year": None,
                        "tattoos": [],
                        "piercings": [],
                        "urls": [],
                        "is_favorite": False,
                        "deleted": False,
                        "merged_into_id": None,
                        "created": "2024-02-01T00:00:00Z",
                        "updated": "2024-05-10T00:00:00Z",
                    },
                ],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_data

            performers, total_count = await client.query_performers(page=1, per_page=25)

            assert total_count == 2
            assert len(performers) == 2
            assert performers[0]["id"] == "perf-1"
            assert performers[0]["name"] == "Performer One"
            assert performers[1]["id"] == "perf-2"

    @pytest.mark.asyncio
    async def test_query_performers_sorts_by_updated(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_data = {
            "queryPerformers": {
                "count": 0,
                "performers": [],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_data

            await client.query_performers()

            # Verify the call includes sort by UPDATED_AT DESC
            call_args = mock_execute.call_args
            query = call_args[0][0]
            variables = call_args[1].get("variables") or call_args[0][1]

            assert variables["input"]["sort"] == "UPDATED_AT"
            assert variables["input"]["direction"] == "DESC"

    @pytest.mark.asyncio
    async def test_query_performers_pagination(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_data = {
            "queryPerformers": {
                "count": 100,
                "performers": [],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_data

            await client.query_performers(page=3, per_page=10)

            call_args = mock_execute.call_args
            variables = call_args[1].get("variables") or call_args[0][1]

            assert variables["input"]["page"] == 3
            assert variables["input"]["per_page"] == 10


class TestGetPerformer:
    """Tests for StashBoxClient.get_performer method."""

    @pytest.mark.asyncio
    async def test_get_performer_returns_performer(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_data = {
            "findPerformer": {
                "id": "perf-123",
                "name": "Test Performer",
                "disambiguation": "",
                "aliases": ["Alt Name"],
                "gender": "FEMALE",
                "birth_date": "1992-03-20",
                "death_date": None,
                "ethnicity": "ASIAN",
                "country": "JP",
                "eye_color": "BROWN",
                "hair_color": "BLACK",
                "height": 160,
                "cup_size": "B",
                "band_size": 30,
                "waist_size": 23,
                "hip_size": 33,
                "breast_type": "NATURAL",
                "career_start_year": 2018,
                "career_end_year": None,
                "tattoos": [],
                "piercings": [],
                "urls": [{"url": "https://twitter.com/test", "type": "TWITTER"}],
                "is_favorite": True,
                "deleted": False,
                "merged_into_id": None,
                "created": "2024-01-01T00:00:00Z",
                "updated": "2024-06-15T00:00:00Z",
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_data

            performer = await client.get_performer("perf-123")

            assert performer is not None
            assert performer["id"] == "perf-123"
            assert performer["name"] == "Test Performer"
            assert performer["country"] == "JP"
            assert performer["urls"][0]["type"] == "TWITTER"

            # Verify variables passed
            call_args = mock_execute.call_args
            variables = call_args[1].get("variables") or call_args[0][1]
            assert variables["id"] == "perf-123"

    @pytest.mark.asyncio
    async def test_get_performer_returns_none_when_not_found(self):
        from stashbox_client import StashBoxClient

        client = StashBoxClient(endpoint="https://stashdb.org/graphql", api_key="test-key")

        mock_data = {"findPerformer": None}

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_data

            performer = await client.get_performer("nonexistent-id")

            assert performer is None

"""Tests for stash_client_unified.py - URL/header setup, sync execution, connection testing."""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import httpx


class TestClientInit:
    def test_url_trailing_slash_stripped(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999/")
        assert client.base_url == "http://localhost:9999"
        assert client.graphql_url == "http://localhost:9999/graphql"

    def test_url_without_trailing_slash(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://10.0.0.4:6969")
        assert client.base_url == "http://10.0.0.4:6969"
        assert client.graphql_url == "http://10.0.0.4:6969/graphql"

    def test_headers_without_api_key(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")
        assert client.headers["Content-Type"] == "application/json"
        assert "ApiKey" not in client.headers

    def test_headers_with_api_key(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999", api_key="my-secret-key")
        assert client.headers["ApiKey"] == "my-secret-key"
        assert client.headers["Content-Type"] == "application/json"

    def test_empty_api_key_not_added(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999", api_key="")
        assert "ApiKey" not in client.headers

    def test_api_key_stored(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999", api_key="test-key")
        assert client.api_key == "test-key"


class TestExecuteSync:
    def test_success_returns_data(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "data": {"systemStatus": {"databaseSchema": 60}}
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.post.return_value = mock_response
            MockClient.return_value = mock_instance

            result = client._execute_sync("query { systemStatus { databaseSchema } }")

            assert result == {"systemStatus": {"databaseSchema": 60}}
            mock_instance.post.assert_called_once()

    def test_sends_correct_payload_without_variables(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"test": True}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.post.return_value = mock_response
            MockClient.return_value = mock_instance

            client._execute_sync("query { test }")

            call_kwargs = mock_instance.post.call_args
            assert call_kwargs[1]["json"] == {"query": "query { test }"}

    def test_sends_variables_when_provided(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"findPerformer": {"id": "1"}}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.post.return_value = mock_response
            MockClient.return_value = mock_instance

            client._execute_sync("query GetPerformer($id: ID!) { findPerformer(id: $id) { id } }", {"id": "1"})

            call_kwargs = mock_instance.post.call_args
            payload = call_kwargs[1]["json"]
            assert payload["variables"] == {"id": "1"}

    def test_graphql_error_raises_runtime_error(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = MagicMock()
        mock_response.json.return_value = {
            "errors": [{"message": "Field not found"}]
        }
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.post.return_value = mock_response
            MockClient.return_value = mock_instance

            with pytest.raises(RuntimeError, match="GraphQL error"):
                client._execute_sync("query { badField }")

    def test_http_error_raises(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Server Error", request=MagicMock(), response=MagicMock(status_code=500)
        )

        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.post.return_value = mock_response
            MockClient.return_value = mock_instance

            with pytest.raises(httpx.HTTPStatusError):
                client._execute_sync("query { test }")

    def test_sends_headers_with_api_key(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999", api_key="secret-123")

        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"test": True}}
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.Client") as MockClient:
            mock_instance = MagicMock()
            mock_instance.__enter__ = MagicMock(return_value=mock_instance)
            mock_instance.__exit__ = MagicMock(return_value=False)
            mock_instance.post.return_value = mock_response
            MockClient.return_value = mock_instance

            client._execute_sync("query { test }")

            call_kwargs = mock_instance.post.call_args
            assert call_kwargs[1]["headers"]["ApiKey"] == "secret-123"


class TestConnectionSync:
    def test_success_returns_true(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute_sync") as mock_exec:
            mock_exec.return_value = {"systemStatus": {"databaseSchema": 60, "databasePath": "/data"}}

            result = client.test_connection_sync()

            assert result is True
            mock_exec.assert_called_once()

    def test_failure_propagates_exception(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute_sync") as mock_exec:
            mock_exec.side_effect = RuntimeError("Connection refused")

            with pytest.raises(RuntimeError, match="Connection refused"):
                client.test_connection_sync()


class TestAsyncConnection:
    @pytest.mark.asyncio
    async def test_success_returns_true(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"systemStatus": {"databaseSchema": 60}}

            result = await client.test_connection()

            assert result is True
            mock_exec.assert_called_once()
            # Verify skip_rate_limit=True is passed
            _, kwargs = mock_exec.call_args
            assert kwargs.get("skip_rate_limit") is True

    @pytest.mark.asyncio
    async def test_failure_propagates(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.side_effect = RuntimeError("Connection failed")

            with pytest.raises(RuntimeError):
                await client.test_connection()

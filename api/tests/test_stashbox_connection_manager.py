"""Tests for StashBoxConnectionManager — reads stash-box config from Stash."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from stashbox_connection_manager import (
    StashBoxConnection,
    StashBoxConnectionManager,
    get_connection_manager,
    set_connection_manager,
)


# ==================== StashBoxConnection ====================


class TestStashBoxConnection:
    """Tests for the StashBoxConnection data class."""

    def test_domain_extraction(self):
        conn = StashBoxConnection("https://stashdb.org/graphql", "key", "StashDB")
        assert conn.domain == "stashdb.org"

    def test_domain_extraction_http(self):
        conn = StashBoxConnection("http://localhost:9999/graphql", "key", "Local")
        assert conn.domain == "localhost:9999"

    def test_domain_extraction_no_graphql(self):
        conn = StashBoxConnection("https://custom.example.com", "key", "Custom")
        assert conn.domain == "custom.example.com"

    def test_to_dict(self):
        conn = StashBoxConnection("https://fansdb.cc/graphql", "key", "FansDB")
        d = conn.to_dict()
        assert d == {
            "endpoint": "https://fansdb.cc/graphql",
            "name": "FansDB",
            "domain": "fansdb.cc",
        }
        # api_key should NOT be in the dict (security)
        assert "api_key" not in d


# ==================== StashBoxConnectionManager ====================


class TestStashBoxConnectionManager:
    """Tests for the connection manager."""

    def _make_manager(self):
        return StashBoxConnectionManager("http://stash:9999", "test-key")

    @pytest.mark.asyncio
    async def test_load_parses_connections(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
            {"endpoint": "https://fansdb.cc/graphql", "api_key": "key2", "name": "FansDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            count = await mgr.load()

        assert count == 2
        assert mgr.loaded is True
        connections = mgr.get_connections()
        assert len(connections) == 2
        assert connections[0]["domain"] == "stashdb.org"
        assert connections[1]["domain"] == "fansdb.cc"

    @pytest.mark.asyncio
    async def test_load_skips_empty_endpoints(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
            {"endpoint": "", "api_key": "key2", "name": "Empty"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            count = await mgr.load()

        assert count == 1

    @pytest.mark.asyncio
    async def test_get_client_returns_client_for_configured_endpoint(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

        client = mgr.get_client("stashdb.org")
        assert client is not None
        assert client.endpoint == "https://stashdb.org/graphql"
        assert client.headers["ApiKey"] == "key1"

    @pytest.mark.asyncio
    async def test_get_client_by_full_url(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

        client = mgr.get_client("https://stashdb.org/graphql")
        assert client is not None
        assert client.endpoint == "https://stashdb.org/graphql"

    @pytest.mark.asyncio
    async def test_get_client_returns_none_for_unknown_endpoint(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

        client = mgr.get_client("unknown.org")
        assert client is None

    @pytest.mark.asyncio
    async def test_get_client_returns_none_when_no_api_key(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

        client = mgr.get_client("stashdb.org")
        assert client is None

    @pytest.mark.asyncio
    async def test_get_client_caches_clients(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

        client1 = mgr.get_client("stashdb.org")
        client2 = mgr.get_client("stashdb.org")
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_refresh_clears_cache(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()
            client_before = mgr.get_client("stashdb.org")

            # Refresh with updated config
            mock_stash.get_stashbox_connections.return_value = [
                {"endpoint": "https://stashdb.org/graphql", "api_key": "key2", "name": "StashDB"},
            ]
            await mgr.refresh()
            client_after = mgr.get_client("stashdb.org")

        # Client should be a new instance after refresh
        assert client_before is not client_after
        assert client_after.headers["ApiKey"] == "key2"

    @pytest.mark.asyncio
    async def test_get_endpoint_url(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

        assert mgr.get_endpoint_url("stashdb.org") == "https://stashdb.org/graphql"
        assert mgr.get_endpoint_url("unknown.org") is None

    @pytest.mark.asyncio
    async def test_get_connections_excludes_api_keys(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "secret-key", "name": "StashDB"},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

        connections = mgr.get_connections()
        for conn in connections:
            assert "api_key" not in conn


# ==================== Module singleton ====================


class TestModuleSingleton:
    """Tests for the module-level singleton functions."""

    def test_get_connection_manager_raises_before_init(self):
        import stashbox_connection_manager as mod
        original = mod._manager
        mod._manager = None
        try:
            with pytest.raises(RuntimeError, match="not initialized"):
                get_connection_manager()
        finally:
            mod._manager = original

    def test_set_connection_manager(self):
        import stashbox_connection_manager as mod
        original = mod._manager

        mock_mgr = MagicMock()
        set_connection_manager(mock_mgr)
        assert get_connection_manager() is mock_mgr

        mod._manager = original


# ==================== Refresh endpoint integration ====================


class TestRefreshEndpoint:
    """Integration tests for the /system/refresh-stashbox-config endpoint."""

    @pytest.fixture
    def client(self):
        """Create a test client with mocked connection manager."""
        import stashbox_connection_manager as mod
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from settings_router import router, init_settings_router

        # Mock the settings and hardware dependencies
        import settings as settings_mod
        import hardware
        from recommendations_db import RecommendationsDB
        from hardware import HardwareProfile
        from settings import init_settings

        original_mgr = settings_mod._settings_manager
        original_hw = hardware._hardware_profile
        original_conn = mod._manager

        hardware._hardware_profile = HardwareProfile(
            gpu_available=True, gpu_name="Test GPU", gpu_vram_mb=8192,
            cpu_cores=8, memory_total_mb=32768, memory_available_mb=16384,
            storage_free_mb=500000, tier="gpu-high",
        )

        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            db = RecommendationsDB(os.path.join(tmp, "test.db"))
            init_settings(db, "gpu-high")
            init_settings_router()

            # Set up a mock connection manager
            mock_mgr = MagicMock(spec=StashBoxConnectionManager)
            mock_mgr.refresh = AsyncMock(return_value=2)
            mock_mgr.get_connections.return_value = [
                {"endpoint": "https://stashdb.org/graphql", "name": "StashDB", "domain": "stashdb.org"},
                {"endpoint": "https://fansdb.cc/graphql", "name": "FansDB", "domain": "fansdb.cc"},
            ]
            mod._manager = mock_mgr

            app = FastAPI()
            app.include_router(router)
            yield TestClient(app)

        settings_mod._settings_manager = original_mgr
        hardware._hardware_profile = original_hw
        mod._manager = original_conn

    def test_refresh_returns_count(self, client):
        resp = client.post("/system/refresh-stashbox-config")
        assert resp.status_code == 200
        data = resp.json()
        assert data["endpoints_loaded"] == 2
        assert len(data["connections"]) == 2

    def test_list_connections(self, client):
        resp = client.get("/system/stashbox-connections")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["connections"]) == 2
        assert data["connections"][0]["domain"] == "stashdb.org"

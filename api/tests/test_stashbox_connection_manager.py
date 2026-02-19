"""Tests for StashBoxConnectionManager — reads stash-box config from Stash."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from rate_limiter import RateLimiter
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
            "max_requests_per_minute": 0,
        }
        # api_key should NOT be in the dict (security)
        assert "api_key" not in d

    def test_to_dict_with_rate_limit(self):
        conn = StashBoxConnection(
            "https://stashdb.org/graphql", "key", "StashDB", max_requests_per_minute=120
        )
        d = conn.to_dict()
        assert d["max_requests_per_minute"] == 120

    def test_default_max_requests_per_minute_is_zero(self):
        conn = StashBoxConnection("https://stashdb.org/graphql", "key", "StashDB")
        assert conn.max_requests_per_minute == 0


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
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB", "max_requests_per_minute": 120},
            {"endpoint": "https://fansdb.cc/graphql", "api_key": "key2", "name": "FansDB", "max_requests_per_minute": 0},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            count = await mgr.load()

        assert count == 2
        assert mgr.loaded is True
        connections = mgr.get_connections()
        assert len(connections) == 2
        assert connections[0]["domain"] == "stashdb.org"
        assert connections[0]["max_requests_per_minute"] == 120
        assert connections[1]["domain"] == "fansdb.cc"
        assert connections[1]["max_requests_per_minute"] == 0

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


# ==================== Per-endpoint rate limiting ====================


class TestPerEndpointRateLimiting:
    """Tests for per-endpoint rate limiter creation."""

    def _make_manager(self):
        return StashBoxConnectionManager("http://stash:9999", "test-key")

    async def _load_with_connections(self, mgr, connections):
        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = connections
        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()

    @pytest.mark.asyncio
    async def test_client_gets_per_endpoint_rate_limiter(self):
        mgr = self._make_manager()
        await self._load_with_connections(mgr, [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB", "max_requests_per_minute": 120},
        ])

        client = mgr.get_client("stashdb.org")
        assert client is not None
        assert client._rate_limiter is not None
        # 120/min = 2 req/s
        assert client._rate_limiter.requests_per_second == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_zero_rpm_uses_default(self):
        mgr = self._make_manager()
        await self._load_with_connections(mgr, [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB", "max_requests_per_minute": 0},
        ])

        client = mgr.get_client("stashdb.org")
        assert client is not None
        # 0 = use default (240/min = 4 req/s)
        expected_rps = StashBoxConnectionManager.DEFAULT_REQUESTS_PER_MINUTE / 60.0
        assert client._rate_limiter.requests_per_second == pytest.approx(expected_rps)

    @pytest.mark.asyncio
    async def test_missing_rpm_field_uses_default(self):
        mgr = self._make_manager()
        await self._load_with_connections(mgr, [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"},
        ])

        client = mgr.get_client("stashdb.org")
        assert client is not None
        expected_rps = StashBoxConnectionManager.DEFAULT_REQUESTS_PER_MINUTE / 60.0
        assert client._rate_limiter.requests_per_second == pytest.approx(expected_rps)

    @pytest.mark.asyncio
    async def test_different_endpoints_get_separate_limiters(self):
        mgr = self._make_manager()
        await self._load_with_connections(mgr, [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB", "max_requests_per_minute": 120},
            {"endpoint": "https://fansdb.cc/graphql", "api_key": "key2", "name": "FansDB", "max_requests_per_minute": 60},
        ])

        client1 = mgr.get_client("stashdb.org")
        client2 = mgr.get_client("fansdb.cc")

        assert client1._rate_limiter is not client2._rate_limiter
        assert client1._rate_limiter.requests_per_second == pytest.approx(2.0)  # 120/60
        assert client2._rate_limiter.requests_per_second == pytest.approx(1.0)  # 60/60

    @pytest.mark.asyncio
    async def test_same_endpoint_reuses_limiter(self):
        mgr = self._make_manager()
        await self._load_with_connections(mgr, [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB", "max_requests_per_minute": 120},
        ])

        client1 = mgr.get_client("stashdb.org")
        client2 = mgr.get_client("stashdb.org")
        assert client1._rate_limiter is client2._rate_limiter

    @pytest.mark.asyncio
    async def test_refresh_creates_fresh_limiters(self):
        mgr = self._make_manager()

        mock_stash = AsyncMock()
        mock_stash.get_stashbox_connections.return_value = [
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB", "max_requests_per_minute": 120},
        ]

        with patch("stash_client_unified.StashClientUnified", return_value=mock_stash):
            await mgr.load()
            limiter_before = mgr.get_client("stashdb.org")._rate_limiter

            # Refresh with different rate limit
            mock_stash.get_stashbox_connections.return_value = [
                {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB", "max_requests_per_minute": 60},
            ]
            await mgr.refresh()
            limiter_after = mgr.get_client("stashdb.org")._rate_limiter

        assert limiter_before is not limiter_after
        assert limiter_before.requests_per_second == pytest.approx(2.0)  # 120/60
        assert limiter_after.requests_per_second == pytest.approx(1.0)   # 60/60


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

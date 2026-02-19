"""StashBox Connection Manager.

Reads stash-box endpoint configuration from Stash's GraphQL API and caches
it in memory. Replaces the previous approach of hardcoded endpoint URLs and
env var API keys with auto-discovery from whatever the user has configured
in Stash's Settings > Metadata Providers.

Usage:
    # At startup
    mgr = StashBoxConnectionManager(stash_url, stash_api_key)
    await mgr.load()

    # Get a client for an endpoint
    client = mgr.get_client("stashdb.org")
    client = mgr.get_client("https://stashdb.org/graphql")

    # List configured endpoints
    connections = mgr.get_connections()

    # Refresh from Stash
    await mgr.refresh()
"""

import logging
from typing import Optional

from rate_limiter import RateLimiter
from stashbox_client import StashBoxClient

logger = logging.getLogger(__name__)


class StashBoxConnection:
    """A single stash-box endpoint connection with its config."""

    __slots__ = ("endpoint", "api_key", "name", "max_requests_per_minute")

    def __init__(self, endpoint: str, api_key: str, name: str, max_requests_per_minute: int = 0):
        self.endpoint = endpoint
        self.api_key = api_key
        self.name = name
        self.max_requests_per_minute = max_requests_per_minute

    @property
    def domain(self) -> str:
        """Extract domain from endpoint URL (e.g. 'stashdb.org' from 'https://stashdb.org/graphql')."""
        return self.endpoint.replace("https://", "").replace("http://", "").replace("/graphql", "").rstrip("/")

    def to_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "name": self.name,
            "domain": self.domain,
            "max_requests_per_minute": self.max_requests_per_minute,
        }


class StashBoxConnectionManager:
    """Manages stash-box endpoint connections read from Stash's configuration.

    Provides client creation, lookup by domain or URL, and config refresh.
    """

    # Default rate limit when Stash reports 0 (unlimited).
    # 240/min = 4 req/s — matches Stash's suggested default for stash-box servers.
    DEFAULT_REQUESTS_PER_MINUTE = 240

    def __init__(self, stash_url: str, stash_api_key: str):
        self._stash_url = stash_url
        self._stash_api_key = stash_api_key
        self._connections: list[StashBoxConnection] = []
        self._clients: dict[str, StashBoxClient] = {}
        self._rate_limiters: dict[str, RateLimiter] = {}
        self._loaded = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> int:
        """Load stash-box connections from Stash's configuration.

        Returns the number of endpoints discovered.
        """
        from stash_client_unified import StashClientUnified

        stash = StashClientUnified(self._stash_url, self._stash_api_key)
        raw_connections = await stash.get_stashbox_connections()

        self._connections = []
        self._clients = {}
        self._rate_limiters = {}

        for conn in raw_connections:
            endpoint = conn.get("endpoint", "")
            api_key = conn.get("api_key", "")
            name = conn.get("name", "")
            max_rpm = conn.get("max_requests_per_minute", 0)
            if not endpoint:
                continue
            self._connections.append(
                StashBoxConnection(endpoint, api_key, name, max_rpm)
            )

        self._loaded = True

        endpoints_info = []
        for c in self._connections:
            rpm = c.max_requests_per_minute
            rate_str = f"{rpm}/min" if rpm > 0 else "unlimited"
            endpoints_info.append(f"{c.domain} ({rate_str})")
        logger.warning(
            f"Loaded {len(self._connections)} stash-box endpoint(s) from Stash: "
            f"{', '.join(endpoints_info)}"
        )
        return len(self._connections)

    async def refresh(self) -> int:
        """Re-read stash-box config from Stash. Returns endpoint count."""
        return await self.load()

    def get_connections(self) -> list[dict]:
        """Return all configured connections as dicts (safe for API responses)."""
        return [c.to_dict() for c in self._connections]

    def _find_connection(self, endpoint_key: str) -> Optional[StashBoxConnection]:
        """Find a connection by domain name or full URL.

        Args:
            endpoint_key: Either a domain like "stashdb.org" or a full URL
                          like "https://stashdb.org/graphql"
        """
        for conn in self._connections:
            if conn.domain == endpoint_key or conn.endpoint == endpoint_key:
                return conn
        return None

    def _get_rate_limiter(self, conn: StashBoxConnection) -> RateLimiter:
        """Get or create a per-endpoint rate limiter for a stash-box connection."""
        if conn.endpoint not in self._rate_limiters:
            rpm = conn.max_requests_per_minute
            if rpm <= 0:
                rpm = self.DEFAULT_REQUESTS_PER_MINUTE
            rps = rpm / 60.0
            self._rate_limiters[conn.endpoint] = RateLimiter(requests_per_second=rps)
            logger.warning(
                f"Rate limiter for {conn.domain}: {rpm}/min ({rps:.2f} req/s)"
            )
        return self._rate_limiters[conn.endpoint]

    def get_client(self, endpoint_key: str) -> Optional[StashBoxClient]:
        """Get or create a StashBoxClient for the given endpoint.

        Args:
            endpoint_key: Domain (e.g. "stashdb.org") or full URL.

        Returns:
            StashBoxClient if the endpoint is configured with an API key, None otherwise.
        """
        conn = self._find_connection(endpoint_key)
        if not conn:
            return None
        if not conn.api_key:
            return None

        if conn.endpoint not in self._clients:
            limiter = self._get_rate_limiter(conn)
            self._clients[conn.endpoint] = StashBoxClient(
                conn.endpoint, conn.api_key, rate_limiter=limiter
            )

        return self._clients[conn.endpoint]

    def get_endpoint_url(self, endpoint_key: str) -> Optional[str]:
        """Get the full endpoint URL for a domain or URL key.

        Args:
            endpoint_key: Domain (e.g. "stashdb.org") or full URL.

        Returns:
            The full GraphQL endpoint URL, or None if not configured.
        """
        conn = self._find_connection(endpoint_key)
        return conn.endpoint if conn else None


# Module-level singleton
_manager: Optional[StashBoxConnectionManager] = None


async def init_connection_manager(stash_url: str, stash_api_key: str) -> StashBoxConnectionManager:
    """Initialize the global connection manager. Called once at startup."""
    global _manager
    _manager = StashBoxConnectionManager(stash_url, stash_api_key)
    await _manager.load()
    return _manager


def get_connection_manager() -> StashBoxConnectionManager:
    """Get the global connection manager. Must be called after init."""
    if _manager is None:
        raise RuntimeError(
            "StashBoxConnectionManager not initialized. "
            "Call init_connection_manager() during startup."
        )
    return _manager


def set_connection_manager(mgr: StashBoxConnectionManager):
    """Set the global connection manager (for testing)."""
    global _manager
    _manager = mgr

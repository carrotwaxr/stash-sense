"""Shared StashBox client utilities.

Provides lazy client creation and endpoint extraction used by both
the identification and stashbox routers.
"""

import os
from typing import Optional

from stashbox_client import StashBoxClient

# StashBox endpoint configuration
ENDPOINT_URLS = {
    "stashdb.org": "https://stashdb.org/graphql",
    "fansdb.cc": "https://fansdb.cc/graphql",
    "theporndb.net": "https://theporndb.net/graphql",
    "pmvstash.org": "https://pmvstash.org/graphql",
    "javstash.org": "https://javstash.org/graphql",
}

ENDPOINT_API_KEY_ENVS = {
    "stashdb.org": "STASHDB_API_KEY",
    "fansdb.cc": "FANSDB_API_KEY",
    "theporndb.net": "THEPORNDB_API_KEY",
    "pmvstash.org": "PMVSTASH_API_KEY",
    "javstash.org": "JAVSTASH_API_KEY",
}

# Lazily initialized StashBox clients
_stashbox_clients: dict[str, StashBoxClient] = {}


def _get_stashbox_client(endpoint_domain: str) -> Optional[StashBoxClient]:
    """Get or create a StashBox client for the given endpoint domain."""
    if endpoint_domain in _stashbox_clients:
        return _stashbox_clients[endpoint_domain]

    url = ENDPOINT_URLS.get(endpoint_domain)
    env_key = ENDPOINT_API_KEY_ENVS.get(endpoint_domain)
    if not url or not env_key:
        return None

    api_key = os.environ.get(env_key, "")
    if not api_key:
        return None

    client = StashBoxClient(url, api_key)
    _stashbox_clients[endpoint_domain] = client
    return client


def _extract_endpoint(universal_id: str | None) -> str | None:
    """Extract endpoint domain from universal_id (e.g. 'stashdb.org:uuid' -> 'stashdb.org')."""
    if universal_id and ":" in universal_id:
        return universal_id.split(":")[0]
    return None

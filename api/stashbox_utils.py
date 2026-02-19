"""Shared StashBox client utilities.

Provides client creation and endpoint extraction used by both
the identification and stashbox routers.

Clients are sourced from the StashBoxConnectionManager which reads
endpoint config from Stash's settings API (auto-discovery).
"""

from typing import Optional

from stashbox_client import StashBoxClient
from stashbox_connection_manager import get_connection_manager


def _get_stashbox_client(endpoint_domain: str) -> Optional[StashBoxClient]:
    """Get a StashBox client for the given endpoint domain or URL.

    Reads from the StashBoxConnectionManager (backed by Stash config).
    """
    mgr = get_connection_manager()
    return mgr.get_client(endpoint_domain)


def _get_endpoint_url(endpoint_domain: str) -> Optional[str]:
    """Get the full endpoint URL for a domain.

    Reads from the StashBoxConnectionManager (backed by Stash config).
    """
    mgr = get_connection_manager()
    return mgr.get_endpoint_url(endpoint_domain)


def _extract_endpoint(universal_id: str | None) -> str | None:
    """Extract endpoint domain from universal_id (e.g. 'stashdb.org:uuid' -> 'stashdb.org')."""
    if universal_id and ":" in universal_id:
        return universal_id.split(":")[0]
    return None

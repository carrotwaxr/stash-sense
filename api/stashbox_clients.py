"""
Stash-box client wrappers for PMVStash, JAVStash, and FansDB.

These all use the same GraphQL schema as StashDB, so they inherit from StashDBClient
and just override source_name and default URL.

See: docs/plans/2026-01-27-data-sources-catalog.md
"""
from typing import Optional

from stashdb_client import StashDBClient


class PMVStashClient(StashDBClient):
    """Client for PMVStash GraphQL API.

    PMVStash uses the standard stash-box GraphQL schema.
    ~6,500 performers with 3.7 images/performer average.
    """

    source_name = "pmvstash"
    source_type = "stash_box"

    DEFAULT_URL = "https://pmvstash.org/graphql"

    def __init__(
        self,
        api_key: str,
        url: Optional[str] = None,
        rate_limit_delay: float = 0.2,  # 300 req/min
    ):
        super().__init__(
            url=url or self.DEFAULT_URL,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
        )


class JAVStashClient(StashDBClient):
    """Client for JAVStash GraphQL API.

    JAVStash uses the standard stash-box GraphQL schema.
    ~21,700 performers but only ~1.0 image/performer (major limitation).

    Special considerations:
    - Japanese names may have family name first vs romanized Western order
    - Many performers won't exist in Western databases
    - Single image per performer limits face recognition accuracy
    """

    source_name = "javstash"
    source_type = "stash_box"

    DEFAULT_URL = "https://javstash.org/graphql"

    def __init__(
        self,
        api_key: str,
        url: Optional[str] = None,
        rate_limit_delay: float = 0.2,  # 300 req/min
    ):
        super().__init__(
            url=url or self.DEFAULT_URL,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
        )


class FansDBClient(StashDBClient):
    """Client for FansDB GraphQL API.

    FansDB uses the standard stash-box GraphQL schema.
    Estimated ~50k performers (untested).
    """

    source_name = "fansdb"
    source_type = "stash_box"

    DEFAULT_URL = "https://fansdb.cc/graphql"

    def __init__(
        self,
        api_key: str,
        url: Optional[str] = None,
        rate_limit_delay: float = 0.25,  # 240 req/min (conservative)
    ):
        super().__init__(
            url=url or self.DEFAULT_URL,
            api_key=api_key,
            rate_limit_delay=rate_limit_delay,
        )

"""
Stash-Box GraphQL Client

Client for querying stash-box endpoints (StashDB, FansDB, etc.).
This is separate from StashClientUnified which only talks to local Stash.

All requests go through the RateLimiter with Priority.LOW to avoid
overwhelming external stash-box servers during bulk operations.
"""

import httpx
import logging
from typing import Optional

from rate_limiter import RateLimiter, Priority

logger = logging.getLogger(__name__)

# Fragment with all performer fields used by both query_performers and get_performer
PERFORMER_FIELDS = """
    id
    name
    disambiguation
    aliases
    gender
    birth_date
    death_date
    ethnicity
    country
    eye_color
    hair_color
    height
    cup_size
    band_size
    waist_size
    hip_size
    breast_type
    career_start_year
    career_end_year
    tattoos { location description }
    piercings { location description }
    urls { url type }
    is_favorite
    deleted
    merged_into_id
    created
    updated
"""


class StashBoxClient:
    """
    Client for querying stash-box GraphQL endpoints.

    Supports StashDB, FansDB, and other stash-box compatible servers.
    """

    def __init__(self, endpoint: str, api_key: str = ""):
        """
        Initialize the stash-box client.

        Args:
            endpoint: The GraphQL URL (e.g. "https://stashdb.org/graphql")
            api_key: API key for authentication (optional)
        """
        self.endpoint = endpoint
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            self.headers["ApiKey"] = api_key

    async def _execute(
        self,
        query: str,
        variables: dict | None = None,
        priority: Priority = Priority.LOW,
    ) -> dict:
        """
        Execute a GraphQL query against the stash-box endpoint.

        Args:
            query: GraphQL query string
            variables: Query variables
            priority: Request priority for rate limiting (default: LOW)

        Returns:
            The 'data' portion of the GraphQL response.

        Raises:
            RuntimeError: If the GraphQL response contains errors.
            httpx.HTTPStatusError: If the HTTP request fails.
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        limiter = await RateLimiter.get_instance()
        async with limiter.acquire(priority):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint,
                    json=payload,
                    headers=self.headers,
                )
                response.raise_for_status()

                result = response.json()
                if "errors" in result:
                    raise RuntimeError(f"GraphQL error: {result['errors']}")

                return result["data"]

    async def query_performers(
        self, page: int = 1, per_page: int = 25
    ) -> tuple[list[dict], int]:
        """
        Query performers from the stash-box endpoint.

        Results are sorted by UPDATED_AT descending.

        Args:
            page: Page number (1-indexed)
            per_page: Number of results per page

        Returns:
            Tuple of (performers list, total count)
        """
        query = f"""
        query QueryPerformers($input: PerformerQueryInput!) {{
            queryPerformers(input: $input) {{
                count
                performers {{
                    {PERFORMER_FIELDS}
                }}
            }}
        }}
        """

        variables = {
            "input": {
                "page": page,
                "per_page": per_page,
                "sort": "UPDATED_AT",
                "direction": "DESC",
            }
        }

        data = await self._execute(query, variables=variables)
        result = data["queryPerformers"]
        return result["performers"], result["count"]

    async def get_performer(self, performer_id: str) -> Optional[dict]:
        """
        Get a single performer by ID from the stash-box endpoint.

        Args:
            performer_id: The stash-box performer UUID

        Returns:
            Performer dict if found, None otherwise
        """
        query = f"""
        query FindPerformer($id: ID!) {{
            findPerformer(id: $id) {{
                {PERFORMER_FIELDS}
            }}
        }}
        """

        data = await self._execute(query, variables={"id": performer_id})
        return data.get("findPerformer")

"""
Stash-Box GraphQL Client

Client for querying stash-box endpoints (StashDB, FansDB, etc.).
This is separate from StashClientUnified which only talks to local Stash.

Each client accepts a per-endpoint RateLimiter (created by the
StashBoxConnectionManager from Stash's max_requests_per_minute config).
Falls back to the global shared RateLimiter singleton if none is provided.
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
    images { id url }
    is_favorite
    deleted
    merged_into_id
    created
    updated
"""

# Fragment with all tag fields used by get_tag
TAG_FIELDS = """
    id
    name
    description
    aliases
    category {
        id
        name
        group
    }
    deleted
    created
    updated
"""

# Fragment with all studio fields used by get_studio
STUDIO_FIELDS = """
    id
    name
    urls { url }
    parent { id name }
    deleted
    created
    updated
"""

# Fragment with all scene fields used by get_scene
SCENE_FIELDS = """
    id
    title
    details
    date
    urls { url site { name } }
    studio { id name }
    tags { id name }
    performers { performer { id name } as }
    director
    code
    deleted
    created
    updated
"""


class StashBoxClient:
    """
    Client for querying stash-box GraphQL endpoints.

    Supports StashDB, FansDB, and other stash-box compatible servers.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str = "",
        rate_limiter: Optional[RateLimiter] = None,
    ):
        """
        Initialize the stash-box client.

        Args:
            endpoint: The GraphQL URL (e.g. "https://stashdb.org/graphql")
            api_key: API key for authentication (optional)
            rate_limiter: Per-endpoint rate limiter. If None, falls back to
                the global shared singleton.
        """
        self.endpoint = endpoint
        self._rate_limiter = rate_limiter
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

        limiter = self._rate_limiter or await RateLimiter.get_instance()
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

    async def get_tag(self, tag_id: str) -> Optional[dict]:
        """
        Get a single tag by ID from the stash-box endpoint.

        Args:
            tag_id: The stash-box tag UUID

        Returns:
            Tag dict if found, None otherwise
        """
        query = f"""
        query FindTag($id: ID!) {{
            findTag(id: $id) {{
                {TAG_FIELDS}
            }}
        }}
        """

        data = await self._execute(query, variables={"id": tag_id})
        return data.get("findTag")

    async def get_studio(self, studio_id: str) -> Optional[dict]:
        """
        Get a single studio by ID from the stash-box endpoint.

        Args:
            studio_id: The stash-box studio UUID

        Returns:
            Studio dict if found, None otherwise
        """
        query = f"""
        query FindStudio($id: ID!) {{
            findStudio(id: $id) {{
                {STUDIO_FIELDS}
            }}
        }}
        """

        data = await self._execute(query, variables={"id": studio_id})
        return data.get("findStudio")

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

    async def get_scene(self, scene_id: str) -> Optional[dict]:
        """
        Get a single scene by ID from the stash-box endpoint.

        Args:
            scene_id: The stash-box scene UUID

        Returns:
            Scene dict if found, None otherwise
        """
        query = f"""
        query FindScene($id: ID!) {{
            findScene(id: $id) {{
                {SCENE_FIELDS}
            }}
        }}
        """

        data = await self._execute(query, variables={"id": scene_id})
        return data.get("findScene")

    async def find_scenes_by_fingerprints(
        self, fingerprint_sets: list[list[dict]]
    ) -> list[list[dict]]:
        """Batch lookup scenes by fingerprint sets.

        Uses stash-box's findScenesBySceneFingerprints query which accepts
        multiple fingerprint sets (one per local scene) and returns matched
        stash-box scenes for each.

        Args:
            fingerprint_sets: List of fingerprint lists. Each inner list has
                dicts with keys: hash (str), algorithm (str: MD5/OSHASH/PHASH)

        Returns:
            List of match-lists, one per input fingerprint set. Each match
            includes scene metadata and matched fingerprints.
        """
        if not fingerprint_sets:
            return []

        query = """
        query FindScenesByFingerprints($fingerprints: [[FingerprintQueryInput!]!]!) {
          findScenesBySceneFingerprints(fingerprints: $fingerprints) {
            id
            title
            date
            duration
            studio { id name }
            performers { performer { id name } as }
            urls { url site { name } }
            images { id url }
            fingerprints {
              hash
              algorithm
              duration
              submissions
              created
              updated
            }
          }
        }
        """

        data = await self._execute(
            query, variables={"fingerprints": fingerprint_sets}
        )
        return data.get("findScenesBySceneFingerprints", [])

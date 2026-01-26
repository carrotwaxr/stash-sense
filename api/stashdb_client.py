"""Client for interacting with StashDB."""
import requests
import time
from typing import Iterator, Optional
from dataclasses import dataclass

@dataclass
class StashDBPerformer:
    """Performer data from StashDB."""
    id: str
    name: str
    image_urls: list[str]
    country: Optional[str]

class StashDBClient:
    """Client for the StashDB GraphQL API."""

    def __init__(self, url: str, api_key: str, rate_limit_delay: float = 0.5):
        """
        Initialize the StashDB client.

        Args:
            url: StashDB GraphQL endpoint
            api_key: API key for authentication
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        self.url = url
        self.headers = {
            "ApiKey": api_key,
            "Content-Type": "application/json",
        }
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _query(self, query: str, variables: dict = None, max_retries: int = 5) -> dict:
        """Execute a GraphQL query with retry on rate limit."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        for attempt in range(max_retries):
            self._rate_limit()
            response = requests.post(self.url, json=payload, headers=self.headers)

            if response.status_code == 429:
                wait_time = 2 ** attempt * 10  # 10s, 20s, 40s, 80s, 160s
                print(f"  Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            result = response.json()
            if "errors" in result:
                raise Exception(f"GraphQL errors: {result['errors']}")
            return result["data"]

        raise Exception(f"Max retries ({max_retries}) exceeded due to rate limiting")

    def get_performer(self, stashdb_id: str) -> Optional[StashDBPerformer]:
        """Get a single performer by ID with their images."""
        query = """
        query FindPerformer($id: ID!) {
            findPerformer(id: $id) {
                id
                name
                country
                images {
                    url
                }
            }
        }
        """
        result = self._query(query, {"id": stashdb_id})
        performer = result.get("findPerformer")
        if not performer:
            return None

        return StashDBPerformer(
            id=performer["id"],
            name=performer["name"],
            image_urls=[img["url"] for img in performer.get("images", [])],
            country=performer.get("country"),
        )

    def get_performers_batch(
        self,
        stashdb_ids: list[str],
    ) -> dict[str, StashDBPerformer]:
        """
        Get multiple performers by their IDs.

        Note: StashDB doesn't have a native batch query, so this fetches one at a time
        with rate limiting. For large batches, consider using iter_performers instead.
        """
        result = {}
        for stashdb_id in stashdb_ids:
            performer = self.get_performer(stashdb_id)
            if performer:
                result[stashdb_id] = performer
        return result

    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
        sort: str = "CREATED_AT",
        direction: str = "ASC",  # Ascending = oldest first, stable for pagination
    ) -> tuple[int, list[StashDBPerformer]]:
        """
        Query performers with pagination.

        Returns: (total_count, list of StashDBPerformer objects)
        """
        query = """
        query QueryPerformers($input: PerformerQueryInput!) {
            queryPerformers(input: $input) {
                count
                performers {
                    id
                    name
                    country
                    images {
                        url
                    }
                }
            }
        }
        """
        variables = {
            "input": {
                "page": page,
                "per_page": per_page,
                "sort": sort,
                "direction": direction,
            }
        }

        result = self._query(query, variables)["queryPerformers"]

        performers = [
            StashDBPerformer(
                id=p["id"],
                name=p["name"],
                image_urls=[img["url"] for img in p.get("images", [])],
                country=p.get("country"),
            )
            for p in result["performers"]
        ]

        return result["count"], performers

    def iter_all_performers(
        self,
        per_page: int = 25,
        max_performers: int = None,
    ) -> Iterator[StashDBPerformer]:
        """
        Iterate through all performers in StashDB.

        Warning: StashDB has 100k+ performers. Use max_performers to limit.
        """
        page = 1
        count_fetched = 0

        while True:
            count, performers = self.query_performers(page=page, per_page=per_page)
            if not performers:
                break

            for performer in performers:
                yield performer
                count_fetched += 1
                if max_performers and count_fetched >= max_performers:
                    return

            if page * per_page >= count:
                break
            page += 1

    def iter_performers_updated_since(
        self,
        since: str,  # ISO format date string
        per_page: int = 25,
        max_performers: int = None,
    ) -> Iterator[StashDBPerformer]:
        """
        Iterate through performers updated since a given date.

        Uses UPDATED_AT sort to efficiently find recently modified performers.
        """
        page = 1
        count_fetched = 0

        while True:
            count, performers = self.query_performers(
                page=page,
                per_page=per_page,
                sort="UPDATED_AT",
                direction="DESC",  # Most recently updated first
            )
            if not performers:
                break

            for performer in performers:
                # Note: We'd need to add updated_at to the query to filter properly
                # For now, this just iterates in update order
                yield performer
                count_fetched += 1
                if max_performers and count_fetched >= max_performers:
                    return

            if page * per_page >= count:
                break
            page += 1

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from StashDB with retry on rate limit."""
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, timeout=30)

                if response.status_code == 429:
                    wait_time = 2 ** attempt * 5  # 5s, 10s, 20s
                    print(f"  Image download rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                response.raise_for_status()
                return response.content
            except requests.exceptions.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"  Image download failed (attempt {attempt + 1}), retrying: {e}")
                    time.sleep(2 ** attempt)
                    continue
                print(f"Failed to download image {url}: {e}")
                return None
        return None


if __name__ == "__main__":
    # Quick test
    import os
    from dotenv import load_dotenv
    load_dotenv()

    client = StashDBClient(
        url=os.environ["STASHDB_URL"],
        api_key=os.environ["STASHDB_API_KEY"],
    )

    # Test single performer lookup
    performer = client.get_performer("50459d16-787c-47c9-8ce9-a4cac9404324")
    if performer:
        print(f"Performer: {performer.name}")
        print(f"  Images: {len(performer.image_urls)}")
        for url in performer.image_urls[:3]:
            print(f"    - {url}")

    # Test pagination
    count, performers = client.query_performers(per_page=5)
    print(f"\nTotal performers in StashDB: {count}")
    print("First 5:")
    for p in performers:
        print(f"  - {p.name} ({len(p.image_urls)} images)")

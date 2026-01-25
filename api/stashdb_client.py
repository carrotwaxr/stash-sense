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

    def _query(self, query: str, variables: dict = None) -> dict:
        """Execute a GraphQL query."""
        self._rate_limit()
        payload = {"query": query}
        if variables:
            payload["variables"] = variables
        response = requests.post(self.url, json=payload, headers=self.headers)
        response.raise_for_status()
        result = response.json()
        if "errors" in result:
            raise Exception(f"GraphQL errors: {result['errors']}")
        return result["data"]

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

    def download_image(self, url: str) -> Optional[bytes]:
        """Download an image from StashDB."""
        try:
            self._rate_limit()
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except Exception as e:
            print(f"Failed to download image {url}: {e}")
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

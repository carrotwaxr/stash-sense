"""Client for interacting with ThePornDB REST API.

ThePornDB has a different API than StashDB:
- REST API at api.theporndb.net (not GraphQL)
- Has pre-cropped face thumbnails
- ~10,000 performers with pagination support

This client mirrors the StashDBClient interface for compatibility with database_builder.py.
"""
import requests
import time
from typing import Iterator, Optional
from dataclasses import dataclass


@dataclass
class ThePornDBPerformer:
    """Performer data from ThePornDB.

    Compatible with StashDBPerformer for use with database_builder.py.
    """
    id: str
    name: str
    image_urls: list[str]
    country: Optional[str]
    # ThePornDB extras
    face_url: Optional[str] = None  # Pre-cropped face image
    stashdb_id: Optional[str] = None  # Cross-reference if available


# Alias for compatibility with database_builder.py
StashDBPerformer = ThePornDBPerformer


class ThePornDBClient:
    """Client for ThePornDB REST API.

    Mirrors the StashDBClient interface for compatibility with database_builder.py.
    """

    BASE_URL = "https://api.theporndb.net"

    def __init__(self, api_key: str, rate_limit_delay: float = 0.25):
        """
        Initialize the ThePornDB client.

        Args:
            api_key: API key for authentication
            rate_limit_delay: Delay between requests (default 0.25s = 240/min)
        """
        self.url = f"{self.BASE_URL}/performers"  # For compatibility
        self.api_key = api_key
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time = 0
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
        })

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _request(self, endpoint: str, params: dict = None, max_retries: int = 5) -> dict:
        """Make a REST API request with retry on rate limit."""
        url = f"{self.BASE_URL}/{endpoint}"

        for attempt in range(max_retries):
            self._rate_limit()
            response = self.session.get(url, params=params, timeout=30)

            if response.status_code == 429:
                wait_time = 2 ** attempt * 10  # 10s, 20s, 40s, 80s, 160s
                print(f"  Rate limited (429), waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait_time)
                continue

            response.raise_for_status()
            return response.json()

        raise Exception(f"Max retries ({max_retries}) exceeded due to rate limiting")

    def _parse_performer(self, data: dict) -> ThePornDBPerformer:
        """Parse performer data from API response."""
        # Collect image URLs - prioritize posters, fall back to main image
        image_urls = []

        # Add poster images (usually multiple)
        for poster in data.get("posters", []):
            if poster.get("url"):
                image_urls.append(poster["url"])

        # Add main image if not already included
        if data.get("image") and data["image"] not in image_urls:
            image_urls.insert(0, data["image"])

        # Extract country from extras
        extras = data.get("extras", {})
        country = None
        if extras.get("nationality"):
            country = extras["nationality"]
        elif extras.get("birthplace_code"):
            country = extras["birthplace_code"]

        # Extract StashDB ID from links if available
        stashdb_id = None
        links = extras.get("links", {})
        if isinstance(links, dict) and links.get("StashDB"):
            # Extract ID from URL like "https://stashdb.org/performers/uuid"
            stashdb_url = links["StashDB"]
            if "/performers/" in stashdb_url:
                stashdb_id = stashdb_url.split("/performers/")[-1]

        return ThePornDBPerformer(
            id=data["id"],
            name=data.get("name") or data.get("full_name", "Unknown"),
            image_urls=image_urls,
            country=country,
            face_url=data.get("face"),  # Pre-cropped face thumbnail
            stashdb_id=stashdb_id,
        )

    def get_performer(self, performer_id: str) -> Optional[ThePornDBPerformer]:
        """Get a single performer by ID."""
        try:
            result = self._request(f"performers/{performer_id}")
            data = result.get("data")
            if not data:
                return None
            return self._parse_performer(data)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
        sort: str = "CREATED_AT",  # Ignored - TPDB uses different sorting
        direction: str = "ASC",  # Ignored
    ) -> tuple[int, list[ThePornDBPerformer]]:
        """
        Query performers with pagination.

        Note: sort/direction params are for StashDB compatibility but ignored.
        ThePornDB returns performers in their default order.

        Returns: (total_count, list of ThePornDBPerformer objects)
        """
        result = self._request("performers", params={
            "page": page,
            "per_page": per_page,
        })

        meta = result.get("meta", {})
        total_count = meta.get("total", 0)

        performers = [
            self._parse_performer(p)
            for p in result.get("data", [])
        ]

        return total_count, performers

    def iter_all_performers(
        self,
        per_page: int = 25,
        max_performers: int = None,
    ) -> Iterator[ThePornDBPerformer]:
        """
        Iterate through all performers in ThePornDB.

        Args:
            per_page: Number of performers per page (default 25)
            max_performers: Maximum number to return (default: all ~10,000)
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

    def iter_performers_with_faces(
        self,
        per_page: int = 25,
        max_performers: int = None,
    ) -> Iterator[ThePornDBPerformer]:
        """
        Iterate through performers that have face thumbnails.

        This is more efficient for face recognition since TPDB provides
        pre-cropped face images.
        """
        for performer in self.iter_all_performers(per_page, max_performers):
            if performer.face_url or performer.image_urls:
                yield performer

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image with retry on rate limit."""
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self.session.get(url, timeout=30)

                if response.status_code == 429:
                    wait_time = 2 ** attempt * 5
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

    def download_face(self, performer: ThePornDBPerformer) -> Optional[bytes]:
        """
        Download the pre-cropped face image for a performer.

        Falls back to main image if no face URL available.
        """
        if performer.face_url:
            return self.download_image(performer.face_url)
        elif performer.image_urls:
            return self.download_image(performer.image_urls[0])
        return None


# Alias for compatibility - allows database_builder.py to use either client
StashDBClient = ThePornDBClient


if __name__ == "__main__":
    # Quick test
    import os
    from dotenv import load_dotenv
    load_dotenv()

    api_key = os.environ.get("THEPORNDB_API_KEY")
    if not api_key:
        print("Set THEPORNDB_API_KEY environment variable")
        exit(1)

    client = ThePornDBClient(api_key=api_key)

    # Test pagination
    print("Testing pagination...")
    count, performers = client.query_performers(per_page=5)
    print(f"Total performers in ThePornDB: {count}")
    print("First 5:")
    for p in performers:
        print(f"  - {p.name}")
        print(f"    Images: {len(p.image_urls)}")
        print(f"    Face URL: {p.face_url is not None}")
        print(f"    StashDB ID: {p.stashdb_id}")
        if p.image_urls:
            print(f"    First image: {p.image_urls[0][:60]}...")

    # Test iteration
    print("\nTesting iteration (first 10)...")
    for i, p in enumerate(client.iter_all_performers(max_performers=10)):
        print(f"  {i+1}. {p.name} ({len(p.image_urls)} images)")

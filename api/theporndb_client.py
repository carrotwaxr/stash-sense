"""Client for interacting with ThePornDB REST API.

ThePornDB has a different API than StashDB:
- REST API at api.theporndb.net (not GraphQL)
- Has pre-cropped face thumbnails
- ~10,000 performers with pagination support

This client inherits from BaseScraper for use with enrichment_coordinator.py.
"""
import requests
import time
from typing import Iterator, Optional
from dataclasses import dataclass

from base_scraper import BaseScraper, ScrapedPerformer


@dataclass
class ThePornDBPerformer:
    """Performer data from ThePornDB (legacy - use ScrapedPerformer instead)."""
    id: str
    name: str
    image_urls: list[str]
    country: Optional[str]
    face_url: Optional[str] = None
    stashdb_id: Optional[str] = None


class ThePornDBClient(BaseScraper):
    """Client for ThePornDB REST API.

    Inherits from BaseScraper for use with enrichment_coordinator.py.
    """

    source_name = "theporndb"
    source_type = "stash_box"

    BASE_URL = "https://api.theporndb.net"

    def __init__(self, api_key: str, rate_limit_delay: float = 0.25):
        """
        Initialize the ThePornDB client.

        Args:
            api_key: API key for authentication
            rate_limit_delay: Delay between requests (default 0.25s = 240/min)
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self.url = f"{self.BASE_URL}/performers"
        self.api_key = api_key
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

    def _parse_performer(self, data: dict) -> ScrapedPerformer:
        """Parse performer data from API response."""
        # Collect image URLs - prioritize posters, fall back to main image
        image_urls = []

        # Add face URL first if available (pre-cropped, best for face recognition)
        if data.get("face"):
            image_urls.append(data["face"])

        # Add poster images (usually multiple)
        for poster in data.get("posters", []):
            if poster.get("url") and poster["url"] not in image_urls:
                image_urls.append(poster["url"])

        # Add main image if not already included
        if data.get("image") and data["image"] not in image_urls:
            image_urls.append(data["image"])

        # Extract data from extras
        extras = data.get("extras", {}) or {}
        country = extras.get("nationality") or extras.get("birthplace_code")
        gender = extras.get("gender", "").upper() or None

        # Extract external URLs
        external_urls = {}
        links = extras.get("links", {})
        if isinstance(links, dict):
            for site, url in links.items():
                if url:
                    external_urls[site] = [url] if isinstance(url, str) else url

        # Extract StashDB ID for cross-reference
        stash_ids = {"theporndb": str(data["id"])}
        if isinstance(links, dict) and links.get("StashDB"):
            stashdb_url = links["StashDB"]
            if "/performers/" in stashdb_url:
                stash_ids["stashdb"] = stashdb_url.split("/performers/")[-1]

        # Parse aliases
        aliases = []
        if data.get("aliases"):
            if isinstance(data["aliases"], list):
                aliases = data["aliases"]
            elif isinstance(data["aliases"], str):
                aliases = [a.strip() for a in data["aliases"].split(",")]

        return ScrapedPerformer(
            id=str(data["id"]),
            name=data.get("name") or data.get("full_name", "Unknown"),
            image_urls=image_urls,
            country=country,
            gender=gender,
            aliases=aliases,
            external_urls=external_urls,
            stash_ids=stash_ids,
            birth_date=extras.get("birthday"),
            ethnicity=extras.get("ethnicity"),
            height_cm=extras.get("height"),
            eye_color=extras.get("eye_colour"),
            hair_color=extras.get("hair_colour"),
        )

    def get_performer(self, performer_id: str) -> Optional[ScrapedPerformer]:
        """Get a single performer by ID."""
        try:
            self._rate_limit()
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
    ) -> tuple[int, list[ScrapedPerformer]]:
        """
        Query performers with pagination.

        Returns: (total_count, list of ScrapedPerformer objects)
        """
        self._rate_limit()
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

    # iter_all_performers is inherited from BaseScraper

    def iter_performers_with_faces(
        self,
        per_page: int = 25,
        max_performers: int = None,
    ) -> Iterator[ScrapedPerformer]:
        """
        Iterate through performers that have images.

        ThePornDB provides pre-cropped face images for many performers.
        """
        for performer in self.iter_all_performers(per_page, max_performers):
            if performer.image_urls:
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

    def download_face(self, performer: ScrapedPerformer) -> Optional[bytes]:
        """
        Download the first image for a performer.

        For ThePornDB, the first image_url is the pre-cropped face if available.
        """
        if performer.image_urls:
            return self.download_image(performer.image_urls[0])
        return None


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
        print(f"    StashDB ID: {p.stash_ids.get('stashdb', 'N/A')}")
        print(f"    Gender: {p.gender}")
        if p.image_urls:
            print(f"    First image: {p.image_urls[0][:60]}...")

    # Test iteration
    print("\nTesting iteration (first 10)...")
    for i, p in enumerate(client.iter_all_performers(max_performers=10)):
        print(f"  {i+1}. {p.name} ({len(p.image_urls)} images, stashdb={p.stash_ids.get('stashdb', 'N/A')})")

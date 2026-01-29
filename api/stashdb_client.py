"""Client for interacting with StashDB."""
import requests
import time
from typing import Iterator, Optional
from dataclasses import dataclass, field

from base_scraper import BaseScraper, ScrapedPerformer


@dataclass
class StashDBPerformer:
    """Performer data from StashDB.

    Extended to capture fields useful for building the performer identity graph.
    See: docs/plans/2026-01-27-performer-identity-graph.md
    """
    id: str
    name: str
    image_urls: list[str]
    country: Optional[str]

    # Identity graph fields
    aliases: list[str] = field(default_factory=list)
    urls: dict[str, list[str]] = field(default_factory=dict)  # site_name -> [urls]
    birth_date: Optional[str] = None  # Can be "YYYY", "YYYY-MM", or "YYYY-MM-DD"
    death_date: Optional[str] = None
    gender: Optional[str] = None  # MALE, FEMALE, TRANSGENDER_MALE, TRANSGENDER_FEMALE, INTERSEX, NON_BINARY
    career_start_year: Optional[int] = None
    career_end_year: Optional[int] = None
    merged_ids: list[str] = field(default_factory=list)  # Previously merged StashDB entries

    # Identity confidence fields (stable physical attributes)
    disambiguation: Optional[str] = None  # Differentiates same-name performers
    ethnicity: Optional[str] = None  # CAUCASIAN, BLACK, ASIAN, INDIAN, LATIN, MIDDLE_EASTERN, MIXED, OTHER
    height_cm: Optional[int] = None  # Height in centimeters
    eye_color: Optional[str] = None  # BLUE, BROWN, GREY, GREEN, HAZEL, RED
    hair_color: Optional[str] = None  # BLONDE, BRUNETTE, BLACK, RED, AUBURN, GREY, BALD, VARIOUS, OTHER
    tattoos: list[dict] = field(default_factory=list)  # [{"location": "...", "description": "..."}]
    piercings: list[dict] = field(default_factory=list)  # [{"location": "...", "description": "..."}]

    # Sync/prioritization fields
    scene_count: Optional[int] = None  # Number of scenes (for prioritization)
    updated: Optional[str] = None  # StashDB update timestamp (ISO format)

class StashDBClient(BaseScraper):
    """Client for the StashDB GraphQL API."""

    source_name = "stashdb"
    source_type = "stash_box"

    def __init__(self, url: str, api_key: str, rate_limit_delay: float = 0.5):
        """
        Initialize the StashDB client.

        Args:
            url: StashDB GraphQL endpoint
            api_key: API key for authentication
            rate_limit_delay: Delay between requests to avoid rate limiting
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self.url = url
        self.headers = {
            "ApiKey": api_key,
            "Content-Type": "application/json",
        }

    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    def _query(self, query: str, variables: dict = None, max_retries: int = 5) -> dict:
        """Execute a GraphQL query with retry on rate limit and connection errors."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        last_error = None
        for attempt in range(max_retries):
            self._rate_limit()
            try:
                response = requests.post(self.url, json=payload, headers=self.headers, timeout=30)
            except requests.exceptions.RequestException as e:
                wait_time = 2 ** attempt * 5  # 5s, 10s, 20s, 40s, 80s
                print(f"  Request error: {e}, waiting {wait_time}s before retry {attempt + 1}/{max_retries}...")
                last_error = e
                time.sleep(wait_time)
                continue

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

        raise Exception(f"Max retries ({max_retries}) exceeded: {last_error}")

    def get_performer(self, stashdb_id: str) -> Optional[ScrapedPerformer]:
        """Get a single performer by ID with all identity graph fields."""
        query = """
        query FindPerformer($id: ID!) {
            findPerformer(id: $id) {
                id
                name
                disambiguation
                country
                ethnicity
                aliases
                birth_date
                death_date
                gender
                height
                eye_color
                hair_color
                career_start_year
                career_end_year
                scene_count
                updated
                merged_ids
                tattoos {
                    location
                    description
                }
                piercings {
                    location
                    description
                }
                images {
                    url
                }
                urls {
                    url
                    site {
                        name
                    }
                }
            }
        }
        """
        result = self._query(query, {"id": stashdb_id})
        performer = result.get("findPerformer")
        if not performer:
            return None

        return self._parse_performer(performer)

    def _parse_performer(self, p: dict) -> ScrapedPerformer:
        """Parse a performer dict from GraphQL response into ScrapedPerformer."""
        # Group URLs by site name
        urls_by_site: dict[str, list[str]] = {}
        for url_entry in p.get("urls", []):
            site_name = url_entry.get("site", {}).get("name", "Unknown")
            url = url_entry.get("url")
            if url:
                if site_name not in urls_by_site:
                    urls_by_site[site_name] = []
                urls_by_site[site_name].append(url)

        # Parse tattoos and piercings
        tattoos = [
            {"location": t.get("location"), "description": t.get("description")}
            for t in p.get("tattoos") or []
        ]
        piercings = [
            {"location": t.get("location"), "description": t.get("description")}
            for t in p.get("piercings") or []
        ]

        return ScrapedPerformer(
            id=p["id"],
            name=p["name"],
            disambiguation=p.get("disambiguation"),
            image_urls=[img["url"] for img in p.get("images", [])],
            country=p.get("country"),
            ethnicity=p.get("ethnicity"),
            aliases=p.get("aliases") or [],
            external_urls=urls_by_site,
            birth_date=p.get("birth_date"),
            gender=p.get("gender"),
            height_cm=p.get("height"),  # StashDB returns as "height" in cm
            eye_color=p.get("eye_color"),
            hair_color=p.get("hair_color"),
            career_start_year=p.get("career_start_year"),
            career_end_year=p.get("career_end_year"),
            merged_ids=p.get("merged_ids") or [],
            tattoos=tattoos,
            piercings=piercings,
            scene_count=p.get("scene_count"),
            updated_at=p.get("updated"),
            stash_ids={"stashdb": p["id"]},  # Include self reference
        )

    def get_performers_batch(
        self,
        stashdb_ids: list[str],
    ) -> dict[str, ScrapedPerformer]:
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
        sort: str = "NAME",
        direction: str = "ASC",  # NAME sort is stable (unique values), unlike CREATED_AT which has 50k+ duplicates
    ) -> tuple[int, list[ScrapedPerformer]]:
        """
        Query performers with pagination.

        Returns: (total_count, list of StashDBPerformer objects)

        Fetches all identity graph fields for cross-source matching.
        """
        query = """
        query QueryPerformers($input: PerformerQueryInput!) {
            queryPerformers(input: $input) {
                count
                performers {
                    id
                    name
                    disambiguation
                    country
                    ethnicity
                    aliases
                    birth_date
                    death_date
                    gender
                    height
                    eye_color
                    hair_color
                    career_start_year
                    career_end_year
                    scene_count
                    updated
                    merged_ids
                    tattoos {
                        location
                        description
                    }
                    piercings {
                        location
                        description
                    }
                    images {
                        url
                    }
                    urls {
                        url
                        site {
                            name
                        }
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
        performers = [self._parse_performer(p) for p in result["performers"]]

        return result["count"], performers

    def iter_all_performers(
        self,
        per_page: int = 25,
        max_performers: int = None,
    ) -> Iterator[ScrapedPerformer]:
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
    ) -> Iterator[ScrapedPerformer]:
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
        print(f"  Aliases: {performer.aliases}")
        print(f"  Birth date: {performer.birth_date}")
        print(f"  Gender: {performer.gender}")
        print(f"  Country: {performer.country}")
        print(f"  Career: {performer.career_start_year} - {performer.career_end_year}")
        print(f"  Merged IDs: {performer.merged_ids}")
        print(f"  URLs by site:")
        for site, urls in performer.urls.items():
            for url in urls:
                print(f"    [{site}] {url}")

    # Test pagination with identity graph fields
    count, performers = client.query_performers(per_page=5)
    print(f"\nTotal performers in StashDB: {count}")
    print("First 5 (with identity graph data):")
    for p in performers:
        url_count = sum(len(urls) for urls in p.urls.values())
        print(f"  - {p.name}: {len(p.image_urls)} images, {len(p.aliases)} aliases, {url_count} URLs")

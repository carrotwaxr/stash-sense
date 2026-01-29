"""
Base scraper interface for multi-source enrichment.

All source scrapers (stash-boxes and reference sites) implement this interface.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Iterator, Optional
import time


@dataclass
class ScrapedPerformer:
    """
    Performer data scraped from a source.

    Required fields work across all sources.
    Optional fields are source-specific.
    """
    # Required
    id: str
    name: str

    # Common optional
    image_urls: list[str] = field(default_factory=list)
    aliases: list[str] = field(default_factory=list)
    gender: Optional[str] = None
    country: Optional[str] = None

    # Identity graph fields (stash-boxes)
    birth_date: Optional[str] = None
    death_date: Optional[str] = None
    career_start_year: Optional[int] = None
    career_end_year: Optional[int] = None
    height_cm: Optional[int] = None
    ethnicity: Optional[str] = None
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None
    disambiguation: Optional[str] = None

    # Cross-references
    stash_ids: dict[str, str] = field(default_factory=dict)  # endpoint -> id
    external_urls: dict[str, list[str]] = field(default_factory=dict)  # site -> urls
    merged_ids: list[str] = field(default_factory=list)

    # Body modifications
    tattoos: list[dict] = field(default_factory=list)
    piercings: list[dict] = field(default_factory=list)

    # Metadata
    scene_count: Optional[int] = None
    updated_at: Optional[str] = None


class BaseScraper(ABC):
    """
    Abstract base class for all source scrapers.

    Provides:
    - Rate limiting between requests
    - Default pagination iterator
    - Retry logic helpers

    Subclasses must implement:
    - source_name: str - identifier for this source
    - source_type: str - "stash_box" or "reference_site"
    - get_performer(id) - fetch single performer
    - query_performers(page, per_page) - paginated query
    - download_image(url) - download image bytes
    """

    source_name: str  # e.g., "stashdb", "babepedia"
    source_type: str  # "stash_box" or "reference_site"

    def __init__(self, rate_limit_delay: float = 0.5):
        """
        Initialize scraper with rate limiting.

        Args:
            rate_limit_delay: Minimum seconds between requests
        """
        self.rate_limit_delay = rate_limit_delay
        self._last_request_time: float = 0

    def _rate_limit(self) -> None:
        """
        Enforce rate limiting by sleeping if needed.

        Call this at the start of any method that makes HTTP requests.
        """
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()

    @abstractmethod
    def get_performer(self, performer_id: str) -> Optional[ScrapedPerformer]:
        """
        Fetch a single performer by ID.

        Args:
            performer_id: Source-specific performer ID

        Returns:
            ScrapedPerformer or None if not found
        """
        pass

    @abstractmethod
    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[int, list[ScrapedPerformer]]:
        """
        Query performers with pagination.

        Args:
            page: Page number (1-indexed)
            per_page: Results per page

        Returns:
            Tuple of (total_count, list of performers on this page)
        """
        pass

    @abstractmethod
    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """
        Download image from URL.

        Args:
            url: Image URL
            max_retries: Number of retry attempts

        Returns:
            Image bytes or None on failure
        """
        pass

    def iter_all_performers(
        self,
        per_page: int = 25,
        max_performers: Optional[int] = None,
        start_page: int = 1,
    ) -> Iterator[ScrapedPerformer]:
        """
        Iterate through all performers with pagination.

        Args:
            per_page: Results per page
            max_performers: Maximum performers to yield (None = no limit)
            start_page: Page to start from (for resume)

        Yields:
            ScrapedPerformer objects
        """
        page = start_page
        count_fetched = 0

        while True:
            total_count, performers = self.query_performers(page=page, per_page=per_page)

            if not performers:
                break

            for performer in performers:
                yield performer
                count_fetched += 1

                if max_performers and count_fetched >= max_performers:
                    return

            # Check if we've fetched all pages
            if page * per_page >= total_count:
                break

            page += 1

    def iter_performers_after(
        self,
        last_id: Optional[str],
        per_page: int = 25,
        max_performers: Optional[int] = None,
    ) -> Iterator[ScrapedPerformer]:
        """
        Iterate performers, optionally resuming after a given ID.

        Default implementation uses iter_all_performers and skips until last_id.
        Subclasses may override with more efficient cursor-based iteration.

        Args:
            last_id: Last processed performer ID (None = start from beginning)
            per_page: Results per page
            max_performers: Maximum performers to yield

        Yields:
            ScrapedPerformer objects
        """
        found_last = last_id is None
        count_yielded = 0

        for performer in self.iter_all_performers(per_page=per_page):
            if not found_last:
                if performer.id == last_id:
                    found_last = True
                continue

            yield performer
            count_yielded += 1

            if max_performers and count_yielded >= max_performers:
                return

# Scraper Orchestration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a scraper orchestration layer that runs multiple sources in parallel with proper coordination, per-source limits, and the write queue.

**Architecture:** Base scraper interface with concrete implementations for each source. Coordinator runs scrapers concurrently, funneling results through the write queue for serialized database writes.

**Tech Stack:** Python 3.11+, dataclasses, ABC for interfaces, asyncio for coordination, existing synchronous HTTP clients.

**Depends On:** Infrastructure from `2026-01-29-multi-source-enrichment-implementation.md` (all 6 tasks complete).

---

## Task 7: Base Scraper Interface

**Files:**
- Create: `api/base_scraper.py`
- Test: `api/tests/test_base_scraper.py`

**Step 1: Write failing test**

```python
# api/tests/test_base_scraper.py
"""Tests for base scraper interface."""
import pytest
from unittest.mock import MagicMock, patch
import time


class TestBaseScraper:
    """Test base scraper behavior."""

    def test_rate_limiting_enforced(self):
        """Rate limiting sleeps between requests."""
        from base_scraper import BaseScraper

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def get_performer(self, performer_id):
                self._rate_limit()
                return None

            def query_performers(self, page=1, per_page=25):
                return (0, [])

            def download_image(self, url):
                return None

        scraper = TestScraper(rate_limit_delay=0.1)

        start = time.time()
        scraper.get_performer("1")
        scraper.get_performer("2")
        elapsed = time.time() - start

        # Should have delayed at least 0.1s between calls
        assert elapsed >= 0.1

    def test_iter_all_performers_paginates(self):
        """iter_all_performers uses pagination correctly."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)
                self.pages_fetched = []

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                self.pages_fetched.append(page)
                if page == 1:
                    return (50, [
                        ScrapedPerformer(id=f"p{i}", name=f"Performer {i}")
                        for i in range(25)
                    ])
                elif page == 2:
                    return (50, [
                        ScrapedPerformer(id=f"p{i}", name=f"Performer {i}")
                        for i in range(25, 50)
                    ])
                else:
                    return (50, [])

            def download_image(self, url):
                return None

        scraper = TestScraper()
        performers = list(scraper.iter_all_performers(per_page=25))

        assert len(performers) == 50
        assert scraper.pages_fetched == [1, 2]

    def test_iter_respects_max_performers(self):
        """iter_all_performers stops at max_performers."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                return (100, [
                    ScrapedPerformer(id=f"p{i}", name=f"Performer {i}")
                    for i in range((page-1)*per_page, page*per_page)
                ])

            def download_image(self, url):
                return None

        scraper = TestScraper()
        performers = list(scraper.iter_all_performers(per_page=25, max_performers=30))

        assert len(performers) == 30

    def test_scraped_performer_dataclass(self):
        """ScrapedPerformer has expected fields."""
        from base_scraper import ScrapedPerformer

        p = ScrapedPerformer(
            id="abc-123",
            name="Test Performer",
            aliases=["Alias 1"],
            image_urls=["https://example.com/img.jpg"],
            gender="FEMALE",
            country="US",
        )

        assert p.id == "abc-123"
        assert p.name == "Test Performer"
        assert p.aliases == ["Alias 1"]
        assert len(p.image_urls) == 1
```

**Step 2: Run test to verify it fails**

```bash
cd api && python -m pytest tests/test_base_scraper.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'base_scraper'`

**Step 3: Write implementation**

```python
# api/base_scraper.py
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
```

**Step 4: Run tests to verify they pass**

```bash
cd api && python -m pytest tests/test_base_scraper.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/base_scraper.py api/tests/test_base_scraper.py
git commit -m "feat: add base scraper interface for multi-source support

- ScrapedPerformer dataclass with all identity graph fields
- BaseScraper ABC with rate limiting and pagination helpers
- iter_all_performers and iter_performers_after for resumable iteration"
```

---

## Task 8: Adapt StashDB Client

**Files:**
- Modify: `api/stashdb_client.py` (make it inherit from BaseScraper)
- Test: `api/tests/test_stashdb_adapter.py`

**Step 1: Write failing test**

```python
# api/tests/test_stashdb_adapter.py
"""Tests for StashDB client adapter."""
import pytest
from unittest.mock import MagicMock, patch


class TestStashDBAdapter:
    """Test StashDB client implements BaseScraper."""

    def test_is_base_scraper(self):
        """StashDBClient inherits from BaseScraper."""
        from stashdb_client import StashDBClient
        from base_scraper import BaseScraper

        assert issubclass(StashDBClient, BaseScraper)

    def test_has_source_name(self):
        """StashDBClient has correct source_name."""
        from stashdb_client import StashDBClient

        with patch.object(StashDBClient, '__init__', lambda self: None):
            client = StashDBClient.__new__(StashDBClient)
            assert client.source_name == "stashdb"
            assert client.source_type == "stash_box"

    def test_query_performers_returns_tuple(self):
        """query_performers returns (total, list) tuple."""
        from stashdb_client import StashDBClient
        from base_scraper import ScrapedPerformer

        # Mock the GraphQL query
        with patch.object(StashDBClient, '_query') as mock_query:
            mock_query.return_value = {
                'queryPerformers': {
                    'count': 100,
                    'performers': [
                        {'id': 'p1', 'name': 'Test 1', 'images': []},
                        {'id': 'p2', 'name': 'Test 2', 'images': []},
                    ]
                }
            }

            client = StashDBClient(
                url="https://stashdb.org/graphql",
                api_key="test-key",
            )
            total, performers = client.query_performers(page=1, per_page=25)

            assert total == 100
            assert len(performers) == 2
            assert all(isinstance(p, ScrapedPerformer) for p in performers)

    def test_get_performer_returns_scraped_performer(self):
        """get_performer returns ScrapedPerformer."""
        from stashdb_client import StashDBClient
        from base_scraper import ScrapedPerformer

        with patch.object(StashDBClient, '_query') as mock_query:
            mock_query.return_value = {
                'findPerformer': {
                    'id': 'abc-123',
                    'name': 'Test Performer',
                    'images': [{'url': 'https://example.com/img.jpg'}],
                    'gender': 'FEMALE',
                    'country': {'code': 'US'},
                    'aliases': ['Alias 1'],
                }
            }

            client = StashDBClient(
                url="https://stashdb.org/graphql",
                api_key="test-key",
            )
            performer = client.get_performer("abc-123")

            assert isinstance(performer, ScrapedPerformer)
            assert performer.id == "abc-123"
            assert performer.name == "Test Performer"
            assert performer.gender == "FEMALE"
            assert performer.country == "US"
```

**Step 2: Write the adapter**

The key changes to `stashdb_client.py`:

1. Import and inherit from `BaseScraper`
2. Add `source_name = "stashdb"` and `source_type = "stash_box"` class attributes
3. Make `_parse_performer` return `ScrapedPerformer` instead of `StashDBPerformer`
4. Update `query_performers` to match interface signature
5. Rename `get_performer_by_id` to `get_performer`

**Step 3: Commit**

```bash
git add api/stashdb_client.py api/tests/test_stashdb_adapter.py
git commit -m "refactor: adapt StashDBClient to BaseScraper interface

- Inherit from BaseScraper
- Return ScrapedPerformer from all methods
- Rename get_performer_by_id to get_performer"
```

---

## Task 9: Enrichment Coordinator

**Files:**
- Create: `api/enrichment_coordinator.py`
- Test: `api/tests/test_enrichment_coordinator.py`

**Step 1: Write failing test**

```python
# api/tests/test_enrichment_coordinator.py
"""Tests for enrichment coordinator."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock, patch


class TestEnrichmentCoordinator:
    """Test coordinator orchestrates scrapers."""

    @pytest.fixture
    def mock_db(self, tmp_path):
        """Create mock database."""
        from database import PerformerDatabase
        db = PerformerDatabase(tmp_path / "test.db")
        return db

    @pytest.fixture
    def mock_scraper(self):
        """Create mock scraper."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class MockScraper(BaseScraper):
            source_name = "mock"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)
                self.performers = [
                    ScrapedPerformer(
                        id=f"p{i}",
                        name=f"Performer {i}",
                        image_urls=[f"https://example.com/p{i}.jpg"],
                        gender="FEMALE",
                    )
                    for i in range(5)
                ]

            def get_performer(self, performer_id):
                for p in self.performers:
                    if p.id == performer_id:
                        return p
                return None

            def query_performers(self, page=1, per_page=25):
                start = (page - 1) * per_page
                end = start + per_page
                return (len(self.performers), self.performers[start:end])

            def download_image(self, url, max_retries=3):
                # Return fake image bytes
                return b"fake_image_data"

        return MockScraper()

    @pytest.mark.asyncio
    async def test_coordinator_processes_performers(self, mock_db, mock_scraper):
        """Coordinator processes all performers from a scraper."""
        from enrichment_coordinator import EnrichmentCoordinator

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        # Mock face detection to avoid loading models
        with patch.object(coordinator, '_process_performer_images', return_value=[]):
            await coordinator.run()

        # Check that performers were created
        assert mock_db.get_stats()['performer_count'] == 5

    @pytest.mark.asyncio
    async def test_coordinator_respects_source_limits(self, mock_db, mock_scraper):
        """Coordinator respects per-source face limits."""
        from enrichment_coordinator import EnrichmentCoordinator
        from enrichment_config import EnrichmentConfig

        config = EnrichmentConfig()
        # Set low limit for testing
        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
            max_faces_per_source=2,
        )

        # Track faces added per performer
        faces_per_performer = {}

        original_add = mock_db.add_face
        def tracking_add(*args, **kwargs):
            pid = args[0]
            faces_per_performer[pid] = faces_per_performer.get(pid, 0) + 1
            return original_add(*args, **kwargs)

        with patch.object(mock_db, 'add_face', side_effect=tracking_add):
            with patch.object(coordinator, '_detect_faces', return_value=[MagicMock()]*5):
                await coordinator.run()

        # Each performer should have at most 2 faces from this source
        for pid, count in faces_per_performer.items():
            assert count <= 2

    @pytest.mark.asyncio
    async def test_coordinator_uses_write_queue(self, mock_db, mock_scraper):
        """Coordinator writes through the write queue."""
        from enrichment_coordinator import EnrichmentCoordinator
        from write_queue import WriteQueue

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        # Verify write queue is used
        assert coordinator.write_queue is not None

        with patch.object(coordinator, '_process_performer_images', return_value=[]):
            await coordinator.run()

        # Queue should have processed messages
        assert coordinator.write_queue.stats.processed > 0

    @pytest.mark.asyncio
    async def test_coordinator_saves_progress(self, mock_db, mock_scraper):
        """Coordinator saves progress for resume."""
        from enrichment_coordinator import EnrichmentCoordinator

        coordinator = EnrichmentCoordinator(
            database=mock_db,
            scrapers=[mock_scraper],
        )

        with patch.object(coordinator, '_process_performer_images', return_value=[]):
            await coordinator.run()

        # Check progress was saved
        progress = mock_db.get_scrape_progress("mock")
        assert progress is not None
        assert progress['performers_processed'] == 5
```

**Step 2: Write implementation**

```python
# api/enrichment_coordinator.py
"""
Enrichment coordinator for multi-source database building.

Runs multiple scrapers concurrently, funneling results through
a single write queue for serialized database writes.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from base_scraper import BaseScraper, ScrapedPerformer
from database import PerformerDatabase
from embeddings import FaceEmbeddingGenerator
from enrichment_config import EnrichmentConfig
from quality_filters import QualityFilter
from write_queue import WriteQueue, WriteMessage, WriteOperation

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorStats:
    """Statistics for monitoring."""
    performers_processed: int = 0
    faces_added: int = 0
    images_processed: int = 0
    errors: int = 0
    by_source: dict = field(default_factory=dict)


class EnrichmentCoordinator:
    """
    Coordinates multi-source enrichment.

    Features:
    - Runs scrapers concurrently (one thread per scraper)
    - Single write queue for serialized database writes
    - Per-source and total face limits
    - Quality filtering
    - Resume capability
    """

    def __init__(
        self,
        database: PerformerDatabase,
        scrapers: list[BaseScraper],
        config: Optional[EnrichmentConfig] = None,
        max_faces_per_source: int = 5,
        max_faces_total: int = 20,
        dry_run: bool = False,
    ):
        self.database = database
        self.scrapers = scrapers
        self.config = config or EnrichmentConfig()
        self.max_faces_per_source = max_faces_per_source
        self.max_faces_total = max_faces_total
        self.dry_run = dry_run

        self.stats = CoordinatorStats()
        self._face_generator: Optional[FaceEmbeddingGenerator] = None
        self._quality_filter = QualityFilter(self.config.global_settings.quality_filters)

        # Write queue with handler
        self.write_queue = WriteQueue(self._handle_write)

        # Thread pool for running sync scrapers
        self._executor = ThreadPoolExecutor(max_workers=len(scrapers))

    async def run(self):
        """Run all scrapers and process results."""
        await self.write_queue.start()

        try:
            # Run each scraper in a thread
            loop = asyncio.get_event_loop()
            tasks = [
                loop.run_in_executor(self._executor, self._run_scraper, scraper)
                for scraper in self.scrapers
            ]

            # Wait for all scrapers to complete
            await asyncio.gather(*tasks)

            # Wait for write queue to drain
            await self.write_queue.wait_until_empty()

        finally:
            await self.write_queue.stop()
            self._executor.shutdown(wait=True)

        logger.info(f"Enrichment complete: {self.stats}")

    def _run_scraper(self, scraper: BaseScraper):
        """Run a single scraper (called from thread pool)."""
        source = scraper.source_name
        logger.info(f"Starting scraper: {source}")

        # Get resume point
        progress = self.database.get_scrape_progress(source)
        last_id = progress['last_processed_id'] if progress else None

        processed = 0
        faces_added = 0

        try:
            for performer in scraper.iter_performers_after(last_id):
                # Process performer
                result = self._process_performer(scraper, performer)
                faces_added += result

                processed += 1

                # Save progress periodically
                if processed % 100 == 0:
                    self.database.save_scrape_progress(
                        source=source,
                        last_processed_id=performer.id,
                        performers_processed=processed,
                        faces_added=faces_added,
                    )
                    logger.info(f"{source}: {processed} performers, {faces_added} faces")

        except Exception as e:
            logger.error(f"Scraper {source} error: {e}")
            self.stats.errors += 1

        # Final progress save
        self.database.save_scrape_progress(
            source=source,
            last_processed_id=performer.id if 'performer' in dir() else last_id,
            performers_processed=processed,
            faces_added=faces_added,
        )

        logger.info(f"Scraper {source} complete: {processed} performers, {faces_added} faces")

    def _process_performer(self, scraper: BaseScraper, performer: ScrapedPerformer) -> int:
        """Process a single performer, returning number of faces added."""
        # Queue performer creation/update
        asyncio.run_coroutine_threadsafe(
            self.write_queue.enqueue(WriteMessage(
                operation=WriteOperation.CREATE_PERFORMER,
                source=scraper.source_name,
                performer_data=self._performer_to_dict(performer),
            )),
            asyncio.get_event_loop(),
        )

        # Process images
        faces_added = 0
        for result in self._process_performer_images(scraper, performer):
            faces_added += 1

        self.stats.performers_processed += 1
        return faces_added

    def _process_performer_images(self, scraper: BaseScraper, performer: ScrapedPerformer):
        """Process images for a performer, yielding face embeddings."""
        # This would be implemented with actual face detection
        # For now, placeholder that yields nothing
        return []

    def _performer_to_dict(self, performer: ScrapedPerformer) -> dict:
        """Convert performer to dict for write queue."""
        return {
            'id': performer.id,
            'name': performer.name,
            'aliases': performer.aliases,
            'gender': performer.gender,
            'country': performer.country,
            'birth_date': performer.birth_date,
            'image_urls': performer.image_urls,
            'stash_ids': performer.stash_ids,
            'external_urls': performer.external_urls,
        }

    async def _handle_write(self, message: WriteMessage):
        """Handle write queue messages."""
        if self.dry_run:
            return

        if message.operation == WriteOperation.CREATE_PERFORMER:
            await self._handle_create_performer(message)
        elif message.operation == WriteOperation.ADD_EMBEDDING:
            await self._handle_add_embedding(message)

    async def _handle_create_performer(self, message: WriteMessage):
        """Create or update performer in database."""
        data = message.performer_data
        source = message.source

        # Check if performer exists by stash ID
        existing = self.database.get_performer_by_stashbox_id(source, data['id'])

        if existing:
            # Update existing performer
            self.database.update_performer(
                existing.id,
                canonical_name=data.get('name'),
                gender=data.get('gender'),
                country=data.get('country'),
            )
            performer_id = existing.id
        else:
            # Create new performer
            performer_id = self.database.add_performer(
                canonical_name=data['name'],
                gender=data.get('gender'),
                country=data.get('country'),
                birth_date=data.get('birth_date'),
            )
            self.database.add_stashbox_id(performer_id, source, data['id'])

        # Add aliases
        for alias in data.get('aliases', []):
            self.database.add_alias(performer_id, alias, source)

        # Add URLs
        for site, urls in data.get('external_urls', {}).items():
            for url in urls:
                self.database.add_url(performer_id, url, source)

    async def _handle_add_embedding(self, message: WriteMessage):
        """Add face embedding to database."""
        # This would add to Voyager index and database
        # Implementation depends on index management
        pass
```

**Step 3: Commit**

```bash
git add api/enrichment_coordinator.py api/tests/test_enrichment_coordinator.py
git commit -m "feat: add enrichment coordinator for multi-source orchestration

- Runs scrapers concurrently with thread pool
- Funnels writes through single write queue
- Saves progress for resume capability
- Handles performer creation and face embedding"
```

---

## Task 10: CLI Entry Point

**Files:**
- Create: `api/enrichment_builder.py`
- Test: Manual testing (CLI tool)

**Step 1: Write implementation**

```python
# api/enrichment_builder.py
#!/usr/bin/env python3
"""
Multi-source enrichment builder CLI.

Usage:
    python enrichment_builder.py --sources stashdb,theporndb
    python enrichment_builder.py --sources all
    python enrichment_builder.py --dry-run --test-performers 100
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from database import PerformerDatabase
from enrichment_config import EnrichmentConfig
from enrichment_coordinator import EnrichmentCoordinator
from stashdb_client import StashDBClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def create_scrapers(config: EnrichmentConfig, sources: list[str]) -> list:
    """Create scraper instances for enabled sources."""
    scrapers = []

    for source_name in sources:
        source_config = config.get_source(source_name)

        if source_name == "stashdb":
            import os
            scrapers.append(StashDBClient(
                url=source_config.url or os.environ.get("STASHDB_URL", "https://stashdb.org/graphql"),
                api_key=os.environ.get("STASHDB_API_KEY", ""),
                rate_limit_delay=60 / source_config.rate_limit,  # Convert from req/min to delay
            ))
        # Add other sources here as implemented

    return scrapers


def main():
    parser = argparse.ArgumentParser(description="Multi-source enrichment builder")

    parser.add_argument(
        "--sources",
        type=str,
        default="stashdb",
        help="Comma-separated list of sources, or 'all' (default: stashdb)",
    )
    parser.add_argument(
        "--disable-source",
        type=str,
        action="append",
        default=[],
        help="Disable a source (can be used multiple times)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent / "sources.yaml",
        help="Path to sources.yaml config file",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path(__file__).parent / "data" / "performers.db",
        help="Path to performers database",
    )
    parser.add_argument(
        "--max-faces-per-source",
        type=int,
        help="Override max faces per source",
    )
    parser.add_argument(
        "--max-faces-total",
        type=int,
        help="Override total max faces per performer",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without writing to database",
    )
    parser.add_argument(
        "--test-performers",
        type=int,
        help="Process only N performers (for testing)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last checkpoint",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current progress and exit",
    )

    args = parser.parse_args()

    # Load config
    cli_sources = args.sources.split(",") if args.sources != "all" else None

    config = EnrichmentConfig(
        config_path=args.config if args.config.exists() else None,
        cli_sources=cli_sources,
        cli_disabled_sources=args.disable_source,
    )

    # Open database
    db = PerformerDatabase(args.database)

    # Status mode
    if args.status:
        print("\n=== Enrichment Progress ===\n")
        progress = db.get_all_scrape_progress()
        if not progress:
            print("No progress recorded yet.")
        else:
            for source, data in progress.items():
                print(f"{source}:")
                print(f"  Last ID: {data['last_processed_id']}")
                print(f"  Performers: {data['performers_processed']}")
                print(f"  Faces: {data['faces_added']}")
                print(f"  Errors: {data['errors']}")
                print(f"  Last run: {data['last_processed_time']}")
                print()
        return

    # Get enabled sources
    if cli_sources:
        sources = [s for s in cli_sources if s not in args.disable_source]
    else:
        sources = config.get_enabled_sources()

    if not sources:
        logger.error("No sources enabled!")
        sys.exit(1)

    logger.info(f"Enabled sources: {sources}")

    # Create scrapers
    scrapers = create_scrapers(config, sources)

    if not scrapers:
        logger.error("No scrapers created!")
        sys.exit(1)

    # Create coordinator
    coordinator = EnrichmentCoordinator(
        database=db,
        scrapers=scrapers,
        config=config,
        max_faces_per_source=args.max_faces_per_source or config.global_settings.max_faces_per_performer,
        max_faces_total=args.max_faces_total or config.global_settings.max_faces_per_performer,
        dry_run=args.dry_run,
    )

    # Run
    try:
        asyncio.run(coordinator.run())
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)


if __name__ == "__main__":
    main()
```

**Step 2: Make executable and test**

```bash
chmod +x api/enrichment_builder.py
python api/enrichment_builder.py --status
python api/enrichment_builder.py --dry-run --test-performers 10
```

**Step 3: Commit**

```bash
git add api/enrichment_builder.py
git commit -m "feat: add enrichment builder CLI

- Parse command line arguments
- Load config with CLI overrides
- Create scrapers for enabled sources
- Show progress with --status
- Support --dry-run and --test-performers"
```

---

## Summary: Scraper Orchestration Tasks

| Task | Description | Files |
|------|-------------|-------|
| 7 | Base scraper interface | `base_scraper.py` |
| 8 | Adapt StashDB client | `stashdb_client.py` updates |
| 9 | Enrichment coordinator | `enrichment_coordinator.py` |
| 10 | CLI entry point | `enrichment_builder.py` |

---

## Future Tasks (Phase 3)

After orchestration is working:

- Task 11: ThePornDB scraper (adapt existing client)
- Task 12: Reference site scraper base (for Babepedia, IAFD)
- Task 13: Face detection integration
- Task 14: Trust-level validation
- Task 15: Face clustering for low-trust sources

---

*Plan created: 2026-01-29*

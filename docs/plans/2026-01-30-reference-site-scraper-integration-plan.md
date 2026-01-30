# Reference Site Scraper Integration - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable Babepedia and IAFD scrapers to work with the enrichment coordinator by looking up performers from our database.

**Architecture:** Reference site scrapers iterate our database (not the external site), extracting slugs from existing URLs or converting names. The coordinator routes to different iteration logic based on `scraper.source_type`.

**Tech Stack:** Python, SQLite, existing scraper infrastructure

---

### Task 1: Add Database Method for URL-Based Lookup

**Files:**
- Modify: `api/database.py`
- Test: `api/tests/test_database.py`

**Step 1: Write the failing test**

Add to `api/tests/test_database.py`:

```python
class TestIterPerformersWithSiteUrls:
    """Tests for iter_performers_with_site_urls."""

    def test_returns_performers_with_matching_urls(self, tmp_path):
        from database import PerformerDatabase

        db = PerformerDatabase(tmp_path / "test.db")

        # Create performers with different URLs
        pid1 = db.add_performer("Mia Malkova")
        db.add_url(pid1, "https://www.babepedia.com/babe/Mia_Malkova", "stashdb")

        pid2 = db.add_performer("Angela White")
        db.add_url(pid2, "https://www.iafd.com/person.rme/perfid=angelawhite/Angela-White.htm", "stashdb")

        pid3 = db.add_performer("No URLs")
        # No URL added

        # Query for babepedia URLs
        results = list(db.iter_performers_with_site_urls("babepedia"))

        assert len(results) == 1
        assert results[0][0] == pid1  # performer_id
        assert results[0][1] == "Mia Malkova"  # name
        assert "babepedia.com" in results[0][2]  # url

    def test_respects_after_id_for_resume(self, tmp_path):
        from database import PerformerDatabase

        db = PerformerDatabase(tmp_path / "test.db")

        pid1 = db.add_performer("Performer A")
        db.add_url(pid1, "https://www.babepedia.com/babe/A", "stashdb")

        pid2 = db.add_performer("Performer B")
        db.add_url(pid2, "https://www.babepedia.com/babe/B", "stashdb")

        pid3 = db.add_performer("Performer C")
        db.add_url(pid3, "https://www.babepedia.com/babe/C", "stashdb")

        # Resume after pid1
        results = list(db.iter_performers_with_site_urls("babepedia", after_id=pid1))

        assert len(results) == 2
        assert results[0][0] == pid2
        assert results[1][0] == pid3
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_database.py::TestIterPerformersWithSiteUrls -v`
Expected: FAIL with "has no attribute 'iter_performers_with_site_urls'"

**Step 3: Write minimal implementation**

Add to `api/database.py` after the `iter_performers` method (around line 1005):

```python
def iter_performers_with_site_urls(
    self,
    site: str,
    after_id: int = 0,
    batch_size: int = 1000,
) -> Iterator[tuple[int, str, str]]:
    """
    Iterate performers that have URLs matching the given site.

    Args:
        site: Site name to match in URL (e.g., 'babepedia', 'iafd')
        after_id: Resume after this performer ID
        batch_size: Batch size for queries

    Yields:
        (performer_id, performer_name, url) tuples ordered by performer_id
    """
    with self._connection() as conn:
        offset_id = after_id
        while True:
            rows = conn.execute(
                """
                SELECT p.id, p.canonical_name, u.url
                FROM performers p
                JOIN external_urls u ON p.id = u.performer_id
                WHERE u.url LIKE ? AND p.id > ?
                ORDER BY p.id ASC
                LIMIT ?
                """,
                (f"%{site}%", offset_id, batch_size),
            ).fetchall()

            if not rows:
                break

            for row in rows:
                yield (row[0], row[1], row[2])
                offset_id = row[0]
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_database.py::TestIterPerformersWithSiteUrls -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/database.py api/tests/test_database.py
git commit -m "feat: add iter_performers_with_site_urls for reference site lookup"
```

---

### Task 2: Add Database Method for Name-Based Iteration

**Files:**
- Modify: `api/database.py`
- Test: `api/tests/test_database.py`

**Step 1: Write the failing test**

Add to `api/tests/test_database.py`:

```python
class TestIterPerformersAfterId:
    """Tests for iter_performers_after_id."""

    def test_iterates_performers_after_id(self, tmp_path):
        from database import PerformerDatabase

        db = PerformerDatabase(tmp_path / "test.db")

        pid1 = db.add_performer("A", gender="FEMALE")
        pid2 = db.add_performer("B", gender="MALE")
        pid3 = db.add_performer("C", gender="FEMALE")

        # Get all after pid1
        results = list(db.iter_performers_after_id(after_id=pid1))

        assert len(results) == 2
        assert results[0].id == pid2
        assert results[1].id == pid3

    def test_returns_all_when_after_id_zero(self, tmp_path):
        from database import PerformerDatabase

        db = PerformerDatabase(tmp_path / "test.db")

        db.add_performer("A")
        db.add_performer("B")

        results = list(db.iter_performers_after_id(after_id=0))

        assert len(results) == 2
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_database.py::TestIterPerformersAfterId -v`
Expected: FAIL with "has no attribute 'iter_performers_after_id'"

**Step 3: Write minimal implementation**

Add to `api/database.py` after `iter_performers_with_site_urls`:

```python
def iter_performers_after_id(
    self,
    after_id: int = 0,
    batch_size: int = 1000,
) -> Iterator[Performer]:
    """
    Iterate all performers with ID greater than after_id.

    Args:
        after_id: Resume after this performer ID (0 = start from beginning)
        batch_size: Batch size for queries

    Yields:
        Performer objects ordered by id
    """
    with self._connection() as conn:
        offset_id = after_id
        while True:
            rows = conn.execute(
                """
                SELECT * FROM performers
                WHERE id > ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (offset_id, batch_size),
            ).fetchall()

            if not rows:
                break

            for row in rows:
                performer = Performer(**dict(row))
                yield performer
                offset_id = performer.id
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_database.py::TestIterPerformersAfterId -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/database.py api/tests/test_database.py
git commit -m "feat: add iter_performers_after_id for name-based lookup"
```

---

### Task 3: Add Slug Extraction to BaseScraper

**Files:**
- Modify: `api/base_scraper.py`
- Test: `api/tests/test_base_scraper.py`

**Step 1: Write the failing test**

Add to `api/tests/test_base_scraper.py`:

```python
class TestReferenceScraperMethods:
    """Tests for reference site scraper methods."""

    def test_base_scraper_has_source_type(self):
        from base_scraper import BaseScraper

        assert hasattr(BaseScraper, 'source_type')
        assert BaseScraper.source_type == "stash_box"

    def test_base_scraper_has_gender_filter(self):
        from base_scraper import BaseScraper

        assert hasattr(BaseScraper, 'gender_filter')
        assert BaseScraper.gender_filter is None

    def test_extract_slug_from_url_raises_not_implemented(self):
        from base_scraper import BaseScraper, ScrapedPerformer

        class MinimalScraper(BaseScraper):
            source_name = "test"

            def query_performers(self, page, per_page):
                return (0, [])

            def get_performer(self, performer_id):
                return None

            def download_image(self, url, max_retries=3):
                return None

        scraper = MinimalScraper()

        import pytest
        with pytest.raises(NotImplementedError):
            scraper.extract_slug_from_url("http://example.com")

    def test_name_to_slug_raises_not_implemented(self):
        from base_scraper import BaseScraper

        class MinimalScraper(BaseScraper):
            source_name = "test"

            def query_performers(self, page, per_page):
                return (0, [])

            def get_performer(self, performer_id):
                return None

            def download_image(self, url, max_retries=3):
                return None

        scraper = MinimalScraper()

        import pytest
        with pytest.raises(NotImplementedError):
            scraper.name_to_slug("Test Name")
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_base_scraper.py::TestReferenceScraperMethods -v`
Expected: FAIL with "has no attribute 'source_type'" or similar

**Step 3: Write minimal implementation**

Modify `api/base_scraper.py`. Add class attributes after line 50 (after `source_name`):

```python
class BaseScraper(ABC):
    """Base class for all scrapers."""

    source_name: str = ""  # Must be set by subclass
    source_type: str = "stash_box"  # "stash_box" or "reference_site"
    gender_filter: Optional[str] = None  # "FEMALE", "MALE", or None
```

Add methods after `download_image` (around line 110):

```python
def extract_slug_from_url(self, url: str) -> Optional[str]:
    """
    Extract performer slug/ID from a URL for this site.

    Only needed for reference_site scrapers in URL lookup mode.

    Args:
        url: Full URL to the performer page

    Returns:
        Performer slug/ID or None if URL doesn't match this site
    """
    raise NotImplementedError("Reference site scrapers must implement extract_slug_from_url")

def name_to_slug(self, name: str) -> str:
    """
    Convert a performer name to the site's slug format.

    Only needed for reference_site scrapers in name lookup mode.

    Args:
        name: Performer's canonical name

    Returns:
        Slug suitable for looking up on this site
    """
    raise NotImplementedError("Reference site scrapers must implement name_to_slug")
```

Also add `Optional` to the imports at the top if not already there.

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_base_scraper.py::TestReferenceScraperMethods -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/base_scraper.py api/tests/test_base_scraper.py
git commit -m "feat: add source_type and slug methods to BaseScraper"
```

---

### Task 4: Implement Babepedia Slug Methods

**Files:**
- Modify: `api/babepedia_client.py`
- Test: `api/tests/test_babepedia_scraper.py`

**Step 1: Write the failing test**

Add to `api/tests/test_babepedia_scraper.py`:

```python
class TestBabepediaSlugMethods:
    """Tests for Babepedia slug extraction and conversion."""

    def test_source_type_is_reference_site(self):
        from unittest.mock import patch, MagicMock

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        assert scraper.source_type == "reference_site"

    def test_gender_filter_is_female(self):
        from unittest.mock import patch

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        assert scraper.gender_filter == "FEMALE"

    def test_extract_slug_from_url(self):
        from unittest.mock import patch

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        # Standard URL
        assert scraper.extract_slug_from_url("https://www.babepedia.com/babe/Mia_Malkova") == "Mia_Malkova"

        # Without www
        assert scraper.extract_slug_from_url("https://babepedia.com/babe/Angela_White") == "Angela_White"

        # With query string
        assert scraper.extract_slug_from_url("https://www.babepedia.com/babe/Test_Name?ref=123") == "Test_Name"

        # Non-matching URL
        assert scraper.extract_slug_from_url("https://iafd.com/person/test") is None

    def test_name_to_slug(self):
        from unittest.mock import patch

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        assert scraper.name_to_slug("Mia Malkova") == "Mia_Malkova"
        assert scraper.name_to_slug("Angela White") == "Angela_White"
        assert scraper.name_to_slug("Single") == "Single"
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_babepedia_scraper.py::TestBabepediaSlugMethods -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Modify `api/babepedia_client.py`. Update class definition:

```python
class BabepediaScraper(BaseScraper):
    """Scraper for Babepedia performer images."""

    source_name = "babepedia"
    source_type = "reference_site"
    gender_filter = "FEMALE"

    BASE_URL = "https://www.babepedia.com"
```

Add methods after `download_image` (end of class):

```python
def extract_slug_from_url(self, url: str) -> Optional[str]:
    """Extract Babepedia slug from URL.

    Example: https://www.babepedia.com/babe/Mia_Malkova -> Mia_Malkova
    """
    match = re.search(r'babepedia\.com/babe/([^/?#]+)', url)
    return match.group(1) if match else None

def name_to_slug(self, name: str) -> str:
    """Convert name to Babepedia slug format.

    Example: "Mia Malkova" -> "Mia_Malkova"
    """
    return name.replace(" ", "_")
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_babepedia_scraper.py::TestBabepediaSlugMethods -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/babepedia_client.py api/tests/test_babepedia_scraper.py
git commit -m "feat: add slug methods to BabepediaScraper"
```

---

### Task 5: Implement IAFD Slug Methods

**Files:**
- Modify: `api/iafd_client.py`
- Test: `api/tests/test_iafd_scraper.py`

**Step 1: Write the failing test**

Add to `api/tests/test_iafd_scraper.py`:

```python
class TestIAFDSlugMethods:
    """Tests for IAFD slug extraction and conversion."""

    def test_source_type_is_reference_site(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        assert scraper.source_type == "reference_site"

    def test_gender_filter_is_none(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        assert scraper.gender_filter is None

    def test_extract_slug_from_url(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        # Standard URL with perfid
        url = "https://www.iafd.com/person.rme/perfid=miamalkova/Mia-Malkova.htm"
        assert scraper.extract_slug_from_url(url) == "miamalkova"

        # Different format
        url2 = "https://iafd.com/person.rme/perfid=angelawhite/Angela-White.htm"
        assert scraper.extract_slug_from_url(url2) == "angelawhite"

        # Non-matching URL
        assert scraper.extract_slug_from_url("https://babepedia.com/babe/test") is None

    def test_name_to_slug(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        assert scraper.name_to_slug("Mia Malkova") == "miamalkova"
        assert scraper.name_to_slug("Angela White") == "angelawhite"
        assert scraper.name_to_slug("Mary-Jane") == "maryjane"
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_iafd_scraper.py::TestIAFDSlugMethods -v`
Expected: FAIL

**Step 3: Write minimal implementation**

Modify `api/iafd_client.py`. Update class definition:

```python
class IAFDScraper(BaseScraper):
    """Scraper for IAFD performer data and images."""

    source_name = "iafd"
    source_type = "reference_site"
    gender_filter = None  # IAFD has both genders

    BASE_URL = "https://www.iafd.com"
```

Add methods after `download_image` (end of class):

```python
def extract_slug_from_url(self, url: str) -> Optional[str]:
    """Extract IAFD performer ID from URL.

    Example: https://www.iafd.com/person.rme/perfid=miamalkova/... -> miamalkova
    """
    match = re.search(r'perfid=([^/]+)', url)
    return match.group(1) if match else None

def name_to_slug(self, name: str) -> str:
    """Convert name to IAFD slug format.

    Example: "Mia Malkova" -> "miamalkova"
    """
    return name.lower().replace(" ", "").replace("-", "")
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_iafd_scraper.py::TestIAFDSlugMethods -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/iafd_client.py api/tests/test_iafd_scraper.py
git commit -m "feat: add slug methods to IAFDScraper"
```

---

### Task 6: Add ReferenceSiteMode Enum and Coordinator Routing

**Files:**
- Modify: `api/enrichment_coordinator.py`
- Test: `api/tests/test_enrichment_coordinator.py`

**Step 1: Write the failing test**

Add to `api/tests/test_enrichment_coordinator.py`:

```python
class TestReferenceSiteMode:
    """Tests for reference site mode enum and routing."""

    def test_reference_site_mode_enum_exists(self):
        from enrichment_coordinator import ReferenceSiteMode

        assert ReferenceSiteMode.URL_LOOKUP.value == "url"
        assert ReferenceSiteMode.NAME_LOOKUP.value == "name"

    def test_coordinator_accepts_reference_site_mode(self, tmp_path):
        from database import PerformerDatabase
        from enrichment_coordinator import EnrichmentCoordinator, ReferenceSiteMode

        db = PerformerDatabase(tmp_path / "test.db")

        coordinator = EnrichmentCoordinator(
            database=db,
            scrapers=[],
            reference_site_mode=ReferenceSiteMode.URL_LOOKUP,
        )

        assert coordinator.reference_site_mode == ReferenceSiteMode.URL_LOOKUP

    def test_coordinator_defaults_to_url_mode(self, tmp_path):
        from database import PerformerDatabase
        from enrichment_coordinator import EnrichmentCoordinator, ReferenceSiteMode

        db = PerformerDatabase(tmp_path / "test.db")

        coordinator = EnrichmentCoordinator(
            database=db,
            scrapers=[],
        )

        assert coordinator.reference_site_mode == ReferenceSiteMode.URL_LOOKUP
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_enrichment_coordinator.py::TestReferenceSiteMode -v`
Expected: FAIL with "cannot import name 'ReferenceSiteMode'"

**Step 3: Write minimal implementation**

Modify `api/enrichment_coordinator.py`. Add import and enum at top (after existing imports):

```python
from enum import Enum


class ReferenceSiteMode(Enum):
    """Mode for reference site scraper iteration."""
    URL_LOOKUP = "url"   # Look up performers by existing URLs
    NAME_LOOKUP = "name" # Try all performers by name
```

Update `__init__` to accept the parameter:

```python
def __init__(
    self,
    database: PerformerDatabase,
    scrapers: list[BaseScraper],
    data_dir: Optional[Path] = None,
    max_faces_per_source: int = 5,
    max_faces_total: int = 20,
    dry_run: bool = False,
    enable_face_processing: bool = False,
    source_trust_levels: Optional[dict[str, str]] = None,
    reference_site_mode: ReferenceSiteMode = ReferenceSiteMode.URL_LOOKUP,
):
    self.database = database
    self.scrapers = scrapers
    self.data_dir = Path(data_dir) if data_dir else None
    self.max_faces_per_source = max_faces_per_source
    self.max_faces_total = max_faces_total
    self.dry_run = dry_run
    self.enable_face_processing = enable_face_processing
    self.source_trust_levels = source_trust_levels or {}
    self.reference_site_mode = reference_site_mode
    # ... rest of init unchanged
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_enrichment_coordinator.py::TestReferenceSiteMode -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/enrichment_coordinator.py api/tests/test_enrichment_coordinator.py
git commit -m "feat: add ReferenceSiteMode enum to coordinator"
```

---

### Task 7: Implement Reference Site Scraper Runner

**Files:**
- Modify: `api/enrichment_coordinator.py`
- Test: `api/tests/test_enrichment_coordinator.py`

**Step 1: Write the failing test**

Add to `api/tests/test_enrichment_coordinator.py`:

```python
class TestReferenceSiteScraperRunner:
    """Tests for reference site scraper execution."""

    @pytest.mark.asyncio
    async def test_url_mode_processes_performers_with_urls(self, tmp_path):
        from unittest.mock import MagicMock, patch
        from database import PerformerDatabase
        from enrichment_coordinator import EnrichmentCoordinator, ReferenceSiteMode
        from base_scraper import ScrapedPerformer

        db = PerformerDatabase(tmp_path / "test.db")

        # Create performer with babepedia URL
        pid = db.add_performer("Mia Malkova", gender="FEMALE")
        db.add_url(pid, "https://www.babepedia.com/babe/Mia_Malkova", "stashdb")

        # Mock scraper
        mock_scraper = MagicMock()
        mock_scraper.source_name = "babepedia"
        mock_scraper.source_type = "reference_site"
        mock_scraper.gender_filter = "FEMALE"
        mock_scraper.extract_slug_from_url.return_value = "Mia_Malkova"
        mock_scraper.get_performer.return_value = ScrapedPerformer(
            id="Mia_Malkova",
            name="Mia Malkova",
            image_urls=["https://example.com/img.jpg"],
        )
        mock_scraper.download_image.return_value = None  # Skip image processing

        coordinator = EnrichmentCoordinator(
            database=db,
            scrapers=[mock_scraper],
            reference_site_mode=ReferenceSiteMode.URL_LOOKUP,
        )

        await coordinator.run()

        # Verify slug extraction was called
        mock_scraper.extract_slug_from_url.assert_called_once()
        # Verify get_performer was called with extracted slug
        mock_scraper.get_performer.assert_called_once_with("Mia_Malkova")

    @pytest.mark.asyncio
    async def test_name_mode_iterates_all_performers(self, tmp_path):
        from unittest.mock import MagicMock
        from database import PerformerDatabase
        from enrichment_coordinator import EnrichmentCoordinator, ReferenceSiteMode
        from base_scraper import ScrapedPerformer

        db = PerformerDatabase(tmp_path / "test.db")

        # Create performers (no URLs needed for name mode)
        db.add_performer("Mia Malkova", gender="FEMALE")
        db.add_performer("Male Performer", gender="MALE")
        db.add_performer("Angela White", gender="FEMALE")

        # Mock scraper with gender filter
        mock_scraper = MagicMock()
        mock_scraper.source_name = "babepedia"
        mock_scraper.source_type = "reference_site"
        mock_scraper.gender_filter = "FEMALE"
        mock_scraper.name_to_slug.side_effect = lambda n: n.replace(" ", "_")
        mock_scraper.get_performer.return_value = None  # All 404s for simplicity
        mock_scraper.download_image.return_value = None

        coordinator = EnrichmentCoordinator(
            database=db,
            scrapers=[mock_scraper],
            reference_site_mode=ReferenceSiteMode.NAME_LOOKUP,
        )

        await coordinator.run()

        # Should have tried 2 female performers (skipped male)
        assert mock_scraper.name_to_slug.call_count == 2
        assert mock_scraper.get_performer.call_count == 2
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_enrichment_coordinator.py::TestReferenceSiteScraperRunner -v`
Expected: FAIL (scrapers not being routed correctly)

**Step 3: Write minimal implementation**

Modify `api/enrichment_coordinator.py`. Update `_run_scraper` method:

```python
def _run_scraper(self, scraper: BaseScraper):
    """Run a single scraper (called from thread pool)."""
    if scraper.source_type == "reference_site":
        self._run_reference_scraper(scraper)
    else:
        self._run_stashbox_scraper(scraper)
```

Rename existing `_run_scraper` logic to `_run_stashbox_scraper`:

```python
def _run_stashbox_scraper(self, scraper: BaseScraper):
    """Run a stash-box scraper that iterates its own performers."""
    source = scraper.source_name
    logger.info(f"Starting stash-box scraper: {source}")

    # Get resume point
    progress = self.database.get_scrape_progress(source)
    last_id = progress['last_processed_id'] if progress else None

    processed = 0
    faces_added = 0

    try:
        for performer in scraper.iter_performers_after(last_id):
            # Process performer
            faces_from_performer = self._process_performer(scraper, performer)
            faces_added += faces_from_performer
            processed += 1

            # Save progress periodically
            if processed % 100 == 0:
                self.database.save_scrape_progress(
                    source=source,
                    last_processed_id=performer.id,
                    performers_processed=processed,
                    faces_added=faces_added,
                )
                logger.info(f"{source}: {processed} performers processed")

            # Record stats
            self.stats.record_performer(source)

    except Exception as e:
        logger.error(f"Scraper {source} error: {e}")
        self.stats.errors += 1

    # Final progress save
    if processed > 0:
        self.database.save_scrape_progress(
            source=source,
            last_processed_id=performer.id,
            performers_processed=processed,
            faces_added=faces_added,
        )

    logger.info(f"Scraper {source} complete: {processed} performers")
```

Add the reference site runner:

```python
def _run_reference_scraper(self, scraper: BaseScraper):
    """Run a reference site scraper that looks up performers from our DB."""
    source = scraper.source_name
    mode = self.reference_site_mode.value
    progress_key = f"{source}:{mode}"

    logger.info(f"Starting reference scraper: {source} (mode={mode})")

    # Get resume point
    progress = self.database.get_scrape_progress(progress_key)
    last_id = int(progress['last_processed_id']) if progress and progress['last_processed_id'] else 0

    processed = 0
    faces_added = 0
    last_performer_id = last_id

    try:
        if self.reference_site_mode == ReferenceSiteMode.URL_LOOKUP:
            iterator = self._iter_url_lookup(scraper, last_id)
        else:
            iterator = self._iter_name_lookup(scraper, last_id)

        for performer_id, slug in iterator:
            last_performer_id = performer_id

            # Fetch from reference site
            scraped = scraper.get_performer(slug)
            if not scraped:
                processed += 1
                continue  # Not found on site

            # Process images
            faces_from_performer = self._process_reference_performer(scraper, performer_id, scraped)
            faces_added += faces_from_performer
            processed += 1

            # Save progress periodically
            if processed % 100 == 0:
                self.database.save_scrape_progress(
                    source=progress_key,
                    last_processed_id=str(performer_id),
                    performers_processed=processed,
                    faces_added=faces_added,
                )
                logger.info(f"{source}: {processed} performers processed, {faces_added} faces added")

            # Record stats
            self.stats.record_performer(source)

    except Exception as e:
        logger.error(f"Reference scraper {source} error: {e}")
        import traceback
        traceback.print_exc()
        self.stats.errors += 1

    # Final progress save
    self.database.save_scrape_progress(
        source=progress_key,
        last_processed_id=str(last_performer_id),
        performers_processed=processed,
        faces_added=faces_added,
    )

    logger.info(f"Reference scraper {source} complete: {processed} performers, {faces_added} faces")

def _iter_url_lookup(self, scraper: BaseScraper, last_id: int):
    """Iterate performers with URLs for this reference site."""
    for performer_id, name, url in self.database.iter_performers_with_site_urls(
        site=scraper.source_name,
        after_id=last_id,
    ):
        slug = scraper.extract_slug_from_url(url)
        if slug:
            yield performer_id, slug

def _iter_name_lookup(self, scraper: BaseScraper, last_id: int):
    """Iterate all performers for name-based lookup."""
    for performer in self.database.iter_performers_after_id(after_id=last_id):
        # Apply gender filter
        if scraper.gender_filter and performer.gender != scraper.gender_filter:
            continue

        slug = scraper.name_to_slug(performer.canonical_name)
        yield performer.id, slug

def _process_reference_performer(self, scraper: BaseScraper, performer_id: int, scraped: ScrapedPerformer) -> int:
    """Process a performer fetched from a reference site. Returns faces added."""
    source = scraper.source_name
    trust_level = self.source_trust_levels.get(source, "high")

    # Check current face counts
    existing_source_faces = len([
        f for f in self.database.get_faces(performer_id)
        if f.source_endpoint == source
    ])
    existing_total_faces = len(self.database.get_faces(performer_id))

    if existing_total_faces >= self.max_faces_total:
        return 0

    # Get existing embeddings for validation
    existing_embeddings = self._get_existing_embeddings(performer_id)

    faces_added = 0

    for image_url in scraped.image_urls:
        # Check limits
        if existing_source_faces + faces_added >= self.max_faces_per_source:
            break
        if existing_total_faces + faces_added >= self.max_faces_total:
            break

        # Download image
        try:
            image_data = scraper.download_image(image_url)
            if not image_data:
                continue
        except Exception as e:
            logger.debug(f"Failed to download {image_url}: {e}")
            continue

        self.stats.images_processed += 1

        # Skip face processing if not enabled
        if not self.enable_face_processing or not self._face_processor:
            continue

        # Process image for faces
        processed_faces = self._face_processor.process_image(image_data, trust_level)

        if not processed_faces:
            continue

        # Validate and add each face
        for face in processed_faces:
            validation = self._face_validator.validate(
                new_embedding=face.embedding.facenet,
                existing_embeddings=existing_embeddings,
                trust_level=trust_level,
            )

            if not validation.accepted:
                self.stats.faces_rejected += 1
                continue

            # Add to index
            with self._index_lock:
                face_index = self._index_manager.add_embedding(face.embedding)
                self._faces_since_save += 1

                if self._faces_since_save >= self.SAVE_INTERVAL:
                    self._index_manager.save()
                    self._faces_since_save = 0

            # Queue database write
            future = asyncio.run_coroutine_threadsafe(
                self.write_queue.enqueue(WriteMessage(
                    operation=WriteOperation.ADD_EMBEDDING,
                    source=source,
                    performer_id=performer_id,
                    image_url=image_url,
                    quality_score=face.quality_score,
                    embedding_type=str(face_index),
                )),
                self._loop,
            )
            future.result(timeout=10.0)

            faces_added += 1
            self.stats.faces_added += 1
            existing_embeddings.append(face.embedding.facenet)

    return faces_added
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_enrichment_coordinator.py::TestReferenceSiteScraperRunner -v`
Expected: PASS

**Step 5: Commit**

```bash
git add api/enrichment_coordinator.py api/tests/test_enrichment_coordinator.py
git commit -m "feat: implement reference site scraper runner"
```

---

### Task 8: Add CLI Flag for Reference Mode

**Files:**
- Modify: `api/enrichment_builder.py`

**Step 1: Add argument parser flag**

Modify `api/enrichment_builder.py`. Add after other arguments:

```python
parser.add_argument(
    "--reference-mode",
    choices=["url", "name"],
    default="url",
    help="How to find performers for reference sites: 'url' (lookup by existing URLs) or 'name' (try all by name)",
)
```

**Step 2: Wire up to coordinator**

In the main function where coordinator is created, add:

```python
from enrichment_coordinator import ReferenceSiteMode

# Parse reference mode
reference_mode = ReferenceSiteMode.URL_LOOKUP
if args.reference_mode == "name":
    reference_mode = ReferenceSiteMode.NAME_LOOKUP

# Pass to coordinator
coordinator = EnrichmentCoordinator(
    database=database,
    scrapers=scrapers,
    data_dir=data_dir,
    max_faces_per_source=config.global_config.max_faces_per_performer,
    max_faces_total=args.max_faces_total or config.global_config.max_faces_per_performer,
    dry_run=args.dry_run,
    enable_face_processing=args.enable_faces,
    source_trust_levels=trust_levels,
    reference_site_mode=reference_mode,
)
```

**Step 3: Test manually**

Run: `cd api && python enrichment_builder.py --help`
Expected: Should show `--reference-mode` option

**Step 4: Commit**

```bash
git add api/enrichment_builder.py
git commit -m "feat: add --reference-mode CLI flag"
```

---

### Task 9: Run Full Test Suite

**Step 1: Run all tests**

Run: `cd api && python -m pytest tests/ -v`
Expected: All tests pass

**Step 2: Fix any failures**

If any tests fail, investigate and fix.

**Step 3: Commit any fixes**

```bash
git add -A
git commit -m "fix: resolve test failures"
```

---

### Task 10: Integration Test

**Step 1: Test with dry run**

```bash
cd api
python enrichment_builder.py --sources babepedia --dry-run
```

Expected: Should iterate performers with babepedia URLs and log what it would do.

**Step 2: Verify babepedia URL count**

```bash
cd api
python -c "
from database import PerformerDatabase
db = PerformerDatabase('./data/performers.db')
count = len(list(db.iter_performers_with_site_urls('babepedia')))
print(f'Performers with Babepedia URLs: {count}')
"
```

**Step 3: Commit**

No commit needed for integration test.

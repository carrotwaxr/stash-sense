# Reference Site Scraper Integration

## Overview

Integrate reference site scrapers (Babepedia, IAFD) into the enrichment coordinator. Unlike stash-box scrapers that iterate their own performer lists, reference sites require looking up performers from our database.

## Design

### Lookup Modes

```python
class ReferenceSiteMode(Enum):
    URL_LOOKUP = "url"   # Only performers with existing URLs for that site
    NAME_LOOKUP = "name" # Try all performers by name matching
```

**URL mode** (default): Fast, targeted. Only processes performers that already have URLs for the reference site (from StashDB metadata).

**Name mode**: Slow, comprehensive. Iterates all performers and attempts name-based lookup on the reference site.

### URL-Based Lookup Flow

1. Query DB for performers with URLs matching the site (e.g., `babepedia.com`)
2. Extract performer slug from URL using scraper's `extract_slug_from_url()`
3. Call `scraper.get_performer(slug)` to fetch images
4. Process images for face embeddings

### Name-Based Lookup Flow

1. Iterate all performers in DB (ordered by ID for deterministic resume)
2. Apply gender filter if scraper specifies one (babepedia = female only)
3. Convert name to slug using scraper's `name_to_slug()`
4. Call `scraper.get_performer(slug)` - returns None for 404s
5. Process images for face embeddings

### Progress Tracking

Reuse existing `scrape_progress` table with mode-specific keys:
- URL mode: `"babepedia:url"` with `last_processed_id` = our performer ID
- Name mode: `"babepedia:name"` with `last_processed_id` = our performer ID

### CLI Usage

```bash
# URL-based lookup (default, fast)
python enrichment_builder.py --sources babepedia,iafd --enable-faces

# Name-based lookup (slow, comprehensive)
python enrichment_builder.py --sources babepedia,iafd --enable-faces --reference-mode name
```

## Implementation

### database.py

Add method to iterate performers with URLs for a specific site:

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
        site: Site name to match (e.g., 'babepedia', 'iafd')
        after_id: Resume after this performer ID
        batch_size: Batch size for queries

    Yields:
        (performer_id, performer_name, url) tuples
    """
```

### base_scraper.py

Add attributes and methods for reference site scrapers:

```python
class BaseScraper:
    source_type: str = "stash_box"  # or "reference_site"
    gender_filter: Optional[str] = None  # "FEMALE", "MALE", or None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract performer slug from a URL for this site."""
        raise NotImplementedError

    def name_to_slug(self, name: str) -> str:
        """Convert a performer name to the site's slug format."""
        raise NotImplementedError
```

### babepedia_client.py

```python
class BabepediaScraper(BaseScraper):
    source_type = "reference_site"
    gender_filter = "FEMALE"

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        # https://www.babepedia.com/babe/Mia_Malkova -> Mia_Malkova
        match = re.search(r'babepedia\.com/babe/([^/?#]+)', url)
        return match.group(1) if match else None

    def name_to_slug(self, name: str) -> str:
        # "Mia Malkova" -> "Mia_Malkova"
        return name.replace(" ", "_")
```

### iafd_client.py

```python
class IAFDScraper(BaseScraper):
    source_type = "reference_site"
    gender_filter = None  # IAFD has both genders

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        # https://www.iafd.com/person.rme/perfid=miamalkova/... -> miamalkova
        match = re.search(r'perfid=([^/]+)', url)
        return match.group(1) if match else None

    def name_to_slug(self, name: str) -> str:
        # "Mia Malkova" -> "miamalkova" (lowercase, no spaces)
        return name.lower().replace(" ", "").replace("-", "")
```

### enrichment_coordinator.py

Add reference site handling:

```python
from enum import Enum

class ReferenceSiteMode(Enum):
    URL_LOOKUP = "url"
    NAME_LOOKUP = "name"

class EnrichmentCoordinator:
    def __init__(self, ..., reference_site_mode: ReferenceSiteMode = ReferenceSiteMode.URL_LOOKUP):
        self.reference_site_mode = reference_site_mode

    def _run_scraper(self, scraper: BaseScraper):
        if scraper.source_type == "reference_site":
            self._run_reference_scraper(scraper)
        else:
            # existing stash-box logic

    def _run_reference_scraper(self, scraper: BaseScraper):
        source = scraper.source_name
        mode = self.reference_site_mode.value
        progress_key = f"{source}:{mode}"

        progress = self.database.get_scrape_progress(progress_key)
        last_id = int(progress['last_processed_id']) if progress else 0

        if self.reference_site_mode == ReferenceSiteMode.URL_LOOKUP:
            self._run_url_lookup(scraper, progress_key, last_id)
        else:
            self._run_name_lookup(scraper, progress_key, last_id)

    def _run_url_lookup(self, scraper, progress_key, last_id):
        """Iterate performers with URLs for this site."""
        for performer_id, name, url in self.database.iter_performers_with_site_urls(
            site=scraper.source_name,
            after_id=last_id,
        ):
            slug = scraper.extract_slug_from_url(url)
            if not slug:
                continue

            self._process_reference_performer(scraper, performer_id, slug)
            # Save progress...

    def _run_name_lookup(self, scraper, progress_key, last_id):
        """Iterate all performers and try name lookup."""
        for performer in self.database.iter_performers_after_id(last_id):
            # Gender filter
            if scraper.gender_filter and performer.gender != scraper.gender_filter:
                continue

            slug = scraper.name_to_slug(performer.canonical_name)
            self._process_reference_performer(scraper, performer.id, slug)
            # Save progress...

    def _process_reference_performer(self, scraper, performer_id, slug):
        """Fetch performer from reference site and process images."""
        scraped = scraper.get_performer(slug)
        if not scraped:
            return  # Not found on site

        # Process images for faces (reuse existing logic)
        ...
```

### enrichment_builder.py

Add CLI flag:

```python
parser.add_argument(
    "--reference-mode",
    choices=["url", "name"],
    default="url",
    help="How to find performers for reference sites: 'url' (lookup by existing URLs) or 'name' (try all by name)",
)
```

"""Client for scraping FreeOnes performer images.

FreeOnes is a reference site with extensive performer galleries.
May need FlareSolverr for bulk scraping (Cloudflare protection).

Trust level: MEDIUM - many gallery images, need quality filtering
See: docs/plans/2026-01-27-data-sources-catalog.md
"""
import re
import time
import requests
from typing import Iterator, Optional
from urllib.parse import quote, urljoin

from base_scraper import BaseScraper, ScrapedPerformer
from flaresolverr_client import FlareSolverr


class FreeOnesScraper(BaseScraper):
    """Scraper for FreeOnes performer images."""

    source_name = "freeones"
    source_type = "reference_site"

    BASE_URL = "https://www.freeones.com"

    # CDN domains for FreeOnes images
    CDN_DOMAINS = [
        "thumbs.freeones.com",
        "ch-thumbs.freeones.com",
        "img.freeones.com",
    ]

    # Standard headers that work for single requests
    HEADERS = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(
        self,
        flaresolverr_url: Optional[str] = None,
        rate_limit_delay: float = 2.0,  # 30 req/min - conservative for bulk
        max_galleries: int = 5,
    ):
        """Initialize FreeOnes scraper.

        Args:
            flaresolverr_url: Optional FlareSolverr URL for Cloudflare bypass
            rate_limit_delay: Delay between requests (seconds)
            max_galleries: Maximum galleries to drill into per performer
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self.flaresolverr = FlareSolverr(flaresolverr_url) if flaresolverr_url else None
        self.max_galleries = max_galleries
        self._use_flaresolverr = False  # Start with direct requests
        self._session = requests.Session()
        self._session.headers.update(self.HEADERS)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML, falling back to FlareSolverr if needed."""
        self._rate_limit()

        # Try direct request first
        if not self._use_flaresolverr:
            try:
                response = self._session.get(url, timeout=30)
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 403:
                    # Cloudflare detected, switch to FlareSolverr
                    if self.flaresolverr and self.flaresolverr.is_available():
                        print("  Cloudflare detected, switching to FlareSolverr")
                        self._use_flaresolverr = True
                    else:
                        print("  Cloudflare blocked and FlareSolverr not available")
                        return None
            except requests.exceptions.RequestException as e:
                print(f"  Request failed: {e}")
                if self.flaresolverr and self.flaresolverr.is_available():
                    self._use_flaresolverr = True
                else:
                    return None

        # Use FlareSolverr
        if self._use_flaresolverr and self.flaresolverr:
            response = self.flaresolverr.get(url)
            if response and response.solution_status == 200:
                return response.solution_html

        return None

    def get_performer(self, performer_id: str) -> Optional[ScrapedPerformer]:
        """Get performer by FreeOnes slug.

        Args:
            performer_id: FreeOnes URL slug (e.g., 'angela-white')

        Returns:
            ScrapedPerformer or None if not found
        """
        # Fetch the photos page directly
        photos_url = f"{self.BASE_URL}/{performer_id}/photos"
        html = self._fetch_html(photos_url)

        if not html:
            return None

        # Check for 404
        if "Page Not Found" in html or "not found" in html.lower():
            return None

        return self._parse_performer(performer_id, html)

    def _parse_performer(self, slug: str, html: str) -> Optional[ScrapedPerformer]:
        """Parse performer data from photos page."""
        # Extract name from page
        name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
        if not name_match:
            name_match = re.search(r'<title>([^<|]+)', html)
        name = name_match.group(1).strip() if name_match else slug.replace("-", " ").title()

        # Clean up name (remove " Photos" suffix etc)
        name = re.sub(r'\s*(Photos|Pictures|Images|Gallery).*$', '', name, flags=re.IGNORECASE).strip()

        # Collect image URLs from galleries
        image_urls = []

        # Find gallery links on the photos page
        # Pattern: href="/{slug}/photos/{gallery-slug}"
        gallery_pattern = rf'href="(/{re.escape(slug)}/photos/[^"]+)"'
        galleries = list(set(re.findall(gallery_pattern, html, re.IGNORECASE)))

        # Also extract images directly from the photos page
        self._extract_images(html, image_urls)

        # Drill into galleries (limit to max_galleries)
        for gallery_path in galleries[:self.max_galleries]:
            gallery_url = urljoin(self.BASE_URL, gallery_path)
            gallery_html = self._fetch_html(gallery_url)
            if gallery_html:
                self._extract_images(gallery_html, image_urls)

            # Stop if we have enough images
            if len(image_urls) >= 20:
                break

        return ScrapedPerformer(
            id=slug,
            name=name,
            image_urls=image_urls[:20],  # Limit to 20 images
            gender="FEMALE",  # FreeOnes is primarily female
            external_urls={"FreeOnes": [f"{self.BASE_URL}/{slug}"]},
            stash_ids={"freeones": slug},
        )

    def _extract_images(self, html: str, image_urls: list[str]):
        """Extract CDN image URLs from HTML."""
        # Pattern for FreeOnes CDN images
        # Images are on: thumbs.freeones.com, ch-thumbs.freeones.com, img.freeones.com
        pattern = r'(https?://(?:thumbs|ch-thumbs|img)\.freeones\.com/[^"\'<>\s]+\.(?:jpg|jpeg|png|webp))'

        for match in re.findall(pattern, html, re.IGNORECASE):
            # Skip tiny thumbnails (usually have dimensions in URL)
            if "/tiny/" in match or "/small/" in match:
                continue
            # Prefer larger versions
            url = match
            # Try to get full size by modifying thumbnail URL
            if "/thumb/" in url:
                url = url.replace("/thumb/", "/full/")
            if url not in image_urls:
                image_urls.append(url)

    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[int, list[ScrapedPerformer]]:
        """Query performers from FreeOnes.

        FreeOnes has a search API but for enrichment we look up
        performers by name from our database.

        Returns:
            Tuple of (total_count, list of performers)
        """
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from FreeOnes CDN."""
        headers = {
            "User-Agent": self.HEADERS["User-Agent"],
            "Referer": self.BASE_URL,
            "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
        }

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = requests.get(url, headers=headers, timeout=30)

                if response.status_code == 200:
                    return response.content
                elif response.status_code == 403:
                    print(f"  FreeOnes CDN blocked (403)")
                    return None
                else:
                    print(f"  FreeOnes image error: {response.status_code}")

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"Failed to download {url}: {e}")
                return None

        return None

    def lookup_by_name(self, name: str) -> Optional[ScrapedPerformer]:
        """Look up a performer by name.

        Args:
            name: Performer name

        Returns:
            ScrapedPerformer or None if not found
        """
        # Convert name to FreeOnes slug format
        # e.g., "Angela White" -> "angela-white"
        slug = name.lower().replace(" ", "-")
        # Remove special characters
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        # Collapse multiple hyphens
        slug = re.sub(r'-+', '-', slug).strip('-')

        result = self.get_performer(slug)
        if result:
            return result

        # If direct lookup failed, try search
        return self._search_performer(name)

    def _search_performer(self, name: str) -> Optional[ScrapedPerformer]:
        """Search for a performer by name."""
        search_url = f"{self.BASE_URL}/babes?q={quote(name)}"
        html = self._fetch_html(search_url)

        if not html:
            return None

        # Find performer links in search results
        # Pattern: href="/{slug}" where slug doesn't contain /photos or other subpaths
        # Look for the babe profile links
        profile_pattern = r'href="/([a-z0-9-]+)"[^>]*class="[^"]*babe[^"]*"'

        # Also try data-follow links
        if not re.search(profile_pattern, html, re.IGNORECASE):
            profile_pattern = r'href="/([a-z0-9-]+)"[^>]*>'

        best_match = None
        best_score = 0
        name_lower = name.lower()

        for match in re.finditer(profile_pattern, html, re.IGNORECASE):
            slug = match.group(1)

            # Skip common non-profile paths
            if slug in ('babes', 'photos', 'videos', 'feed', 'search', 'login', 'signup'):
                continue

            # Score the match by comparing slug to name
            slug_name = slug.replace("-", " ")
            score = 0

            if slug_name == name_lower:
                score = 100
            elif name_lower in slug_name:
                score = 80
            elif slug_name in name_lower:
                score = 70
            else:
                # Word overlap
                name_words = set(name_lower.split())
                slug_words = set(slug_name.split())
                overlap = len(name_words & slug_words)
                if overlap > 0:
                    score = 50 + (overlap * 10)

            if score > best_score:
                best_score = score
                best_match = slug

        if not best_match or best_score < 50:
            return None

        return self.get_performer(best_match)


if __name__ == "__main__":
    # Test the scraper
    import os

    flaresolverr_url = os.environ.get("FLARESOLVERR_URL", "http://10.0.0.4:8191")
    scraper = FreeOnesScraper(flaresolverr_url=flaresolverr_url)

    print("Testing FreeOnes scraper...")

    # Test lookup by name
    performer = scraper.lookup_by_name("Angela White")
    if performer:
        print(f"\nFound: {performer.name}")
        print(f"  ID: {performer.id}")
        print(f"  Images: {len(performer.image_urls)}")
        for i, url in enumerate(performer.image_urls[:5]):
            print(f"    {i+1}. {url[:80]}...")
        if len(performer.image_urls) > 5:
            print(f"    ... and {len(performer.image_urls) - 5} more")
    else:
        print("Performer not found!")

"""Client for scraping TheNude performer images.

TheNude is a reference site with performer galleries.
Uses simple requests - no Cloudflare protection observed.

Trust level: MEDIUM - varied image quality
"""
import re
import time
import requests
from typing import Optional
from urllib.parse import unquote

from base_scraper import BaseScraper, ScrapedPerformer


class TheNudeScraper(BaseScraper):
    """Scraper for TheNude performer images."""

    source_name = "thenude"
    source_type = "reference_site"
    gender_filter = "FEMALE"

    BASE_URL = "https://www.thenude.com"

    def __init__(self, rate_limit_delay: float = 1.0):
        """Initialize TheNude scraper.

        Args:
            rate_limit_delay: Delay between requests (seconds)
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        })

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML via requests."""
        import logging
        logger = logging.getLogger(__name__)

        self._rate_limit()
        try:
            response = self._session.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            logger.debug(f"[thenude] HTTP {response.status_code} for {url}")
        except Exception as e:
            logger.warning(f"[thenude] Fetch error: {e}")
        return None

    def get_performer(self, slug: str) -> Optional[ScrapedPerformer]:
        """Get performer by TheNude slug.

        Args:
            slug: TheNude URL path (e.g., 'Angela%20White_12345.htm')

        Returns:
            ScrapedPerformer or None if not found
        """
        # The slug is the full path after the domain
        url = f"{self.BASE_URL}/{slug}"
        html = self._fetch_html(url)

        if not html:
            return None

        if "Page Not Found" in html or "404" in html:
            return None

        return self._parse_performer_page(slug, html)

    def _parse_performer_page(self, slug: str, html: str) -> Optional[ScrapedPerformer]:
        """Parse a TheNude performer page."""
        # Extract name from slug: Name%20Encoded_12345.htm
        decoded_slug = unquote(slug)
        name_match = re.match(r'([^_]+)_\d+\.htm', decoded_slug)
        name = name_match.group(1) if name_match else None

        if not name:
            # Try title: "Name nude from Site - TheNude"
            title_match = re.search(r'<title>([^<]+?)(?:\s+nude|\s*[-|])', html)
            name = title_match.group(1).strip() if title_match else decoded_slug

        # Normalize name for matching in URLs
        name_for_match = name.replace(" ", "-")

        # Find images with performer name in the URL
        # TheNude has two sizes:
        # - medheads: /admin/covers/{studio}/{date}/medheads/{filename}.jpg (266x400)
        # - full: /admin/covers/{studio}/{date}/{filename}.jpg (520x782)
        # We want the full-size images, so convert medheads to full
        image_urls = []

        # Look for cover images with performer name
        img_pattern = r'src="(https://static\.thenude\.com/admin/covers/[^"]+\.jpg)"'
        for match in re.findall(img_pattern, html, re.IGNORECASE):
            # Only include images that have the performer's name
            if name_for_match.lower() in match.lower():
                # Convert medheads to full-size by removing /medheads/ from path
                if '/medheads/' in match:
                    full_url = match.replace('/medheads/', '/')
                else:
                    full_url = match
                if full_url not in image_urls:
                    image_urls.append(full_url)

        # Also check for medheads pattern in case we missed any
        medhead_pattern = r'src="([^"]+/medheads/[^"]+\.jpg)"'
        for match in re.findall(medhead_pattern, html, re.IGNORECASE):
            if name_for_match.lower() in match.lower():
                if not match.startswith("http"):
                    match = f"https://static.thenude.com{match}"
                # Convert to full-size
                full_url = match.replace('/medheads/', '/')
                if full_url not in image_urls:
                    image_urls.append(full_url)

        if not image_urls:
            return None

        # Extract ID from slug
        id_match = re.search(r'_(\d+)\.htm', slug)
        performer_id = id_match.group(1) if id_match else slug

        return ScrapedPerformer(
            id=performer_id,
            name=name,
            image_urls=image_urls[:10],
            gender="FEMALE",
            stash_ids={"thenude": performer_id},
        )

    def query_performers(self, page: int = 1, per_page: int = 25) -> tuple[int, list[ScrapedPerformer]]:
        """TheNude doesn't support pagination queries."""
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from TheNude."""
        import logging
        logger = logging.getLogger(__name__)

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self._session.get(url, timeout=30)

                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type or len(response.content) > 1000:
                        return response.content
                elif response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"[thenude] Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.debug(f"[thenude] Image error: {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"[thenude] Download error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

        return None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract TheNude slug from URL.

        Example: https://www.thenude.com/Angela%20White_12345.htm -> Angela%20White_12345.htm
        """
        match = re.search(r'thenude\.com/([^/?#]+\.htm)', url)
        return match.group(1) if match else None

    def name_to_slug(self, name: str) -> str:
        """Convert name to TheNude slug format."""
        # TheNude URLs are stored with the full path, not just the name
        # This won't work for name lookup - use URL lookup instead
        return name.replace(" ", "%20")


if __name__ == "__main__":
    scraper = TheNudeScraper()

    print("Testing TheNude scraper...")

    # Test with a known URL slug
    test_slug = "Jayden%20Jaymes_13038.htm"
    performer = scraper.get_performer(test_slug)

    if performer:
        print(f"Found: {performer.name}")
        print(f"Images: {len(performer.image_urls)}")
        for url in performer.image_urls[:3]:
            print(f"  - {url}")

        if performer.image_urls:
            print("\nDownloading first image...")
            data = scraper.download_image(performer.image_urls[0])
            if data:
                print(f"SUCCESS: {len(data)} bytes")
            else:
                print("FAILED")
    else:
        print("Performer not found")

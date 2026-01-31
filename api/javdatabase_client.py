"""Client for scraping JavDatabase performer images.

JavDatabase is a JAV filmography database with idol images.
Uses simple requests - no Cloudflare protection observed.

Trust level: HIGH - single-face idol photos
"""
import re
import time
import requests
from typing import Optional

from base_scraper import BaseScraper, ScrapedPerformer


class JavDatabaseScraper(BaseScraper):
    """Scraper for JavDatabase performer images."""

    source_name = "javdatabase"
    source_type = "reference_site"
    gender_filter = "FEMALE"  # JAV idols are typically female

    BASE_URL = "https://www.javdatabase.com"

    def __init__(self, rate_limit_delay: float = 1.0):
        """Initialize JavDatabase scraper.

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
            logger.debug(f"[javdatabase] HTTP {response.status_code} for {url}")
        except Exception as e:
            logger.warning(f"[javdatabase] Fetch error: {e}")
        return None

    def get_performer(self, slug: str) -> Optional[ScrapedPerformer]:
        """Get performer by JavDatabase slug.

        Args:
            slug: JavDatabase URL slug (e.g., 'hikari')

        Returns:
            ScrapedPerformer or None if not found
        """
        url = f"{self.BASE_URL}/idols/{slug}/"
        html = self._fetch_html(url)

        if not html:
            return None

        if "Page Not Found" in html or "<title>404" in html or "idol not found" in html.lower():
            return None

        return self._parse_performer_page(slug, html)

    def _parse_performer_page(self, slug: str, html: str) -> Optional[ScrapedPerformer]:
        """Parse a JavDatabase performer page."""
        # Extract name from title or heading
        name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
        if not name_match:
            name_match = re.search(r'<title>([^|<-]+)', html)
        name = name_match.group(1).strip() if name_match else slug.replace("-", " ").title()

        # Clean up name
        name = re.sub(r'\s*(Biography|JAV|Movies|Videos).*$', '', name, flags=re.IGNORECASE).strip()

        image_urls = []

        # Find idol image matching this performer's slug
        # Pattern: /idolimages/full/{slug}.webp or .jpg
        # Only include images that match the slug to avoid related idol images
        idol_pattern = r'(https://www\.javdatabase\.com/idolimages/full/[^"\']+\.(?:webp|jpg|png))'
        for match in re.findall(idol_pattern, html, re.IGNORECASE):
            # Only include if URL contains the slug (the performer's name)
            if slug.lower() in match.lower():
                if match not in image_urls:
                    image_urls.append(match)

        # Also check for idol thumbs (convert to full) - same slug matching
        idol_thumb_pattern = r'(https://www\.javdatabase\.com/idolimages/thumb/[^"\']+\.(?:webp|jpg|png))'
        for match in re.findall(idol_thumb_pattern, html, re.IGNORECASE):
            if slug.lower() in match.lower():
                full_url = match.replace('/thumb/', '/full/')
                if full_url not in image_urls:
                    image_urls.append(full_url)

        if not image_urls:
            return None

        return ScrapedPerformer(
            id=slug,
            name=name,
            image_urls=image_urls[:10],
            gender="FEMALE",
            stash_ids={"javdatabase": slug},
        )

    def query_performers(self, page: int = 1, per_page: int = 25) -> tuple[int, list[ScrapedPerformer]]:
        """JavDatabase doesn't support pagination queries."""
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from JavDatabase."""
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
                elif response.status_code == 404:
                    # Full version might not exist
                    if '/full/' in url:
                        thumb_url = url.replace('/full/', '/thumb/')
                        logger.debug(f"[javdatabase] Full not found, trying thumb")
                        return self.download_image(thumb_url, max_retries=1)
                    return None
                elif response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"[javdatabase] Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.debug(f"[javdatabase] Image error: {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"[javdatabase] Download error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

        return None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract JavDatabase slug from URL.

        Example: https://www.javdatabase.com/idols/hikari/ -> hikari
        """
        match = re.search(r'javdatabase\.com/idols/([^/?#]+)', url)
        if match:
            return match.group(1).rstrip("/")
        return None

    def name_to_slug(self, name: str) -> str:
        """Convert name to JavDatabase slug format.

        Example: "Hikari" -> "hikari"
        """
        slug = name.lower().replace(" ", "-")
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        return slug


if __name__ == "__main__":
    scraper = JavDatabaseScraper()

    print("Testing JavDatabase scraper...")

    # Test with the one URL we have
    test_slug = "hikari"
    performer = scraper.get_performer(test_slug)

    if performer:
        print(f"Found: {performer.name}")
        print(f"Images: {len(performer.image_urls)}")
        for url in performer.image_urls[:3]:
            print(f"  - {url[:70]}...")

        if performer.image_urls:
            print("\nDownloading first image...")
            data = scraper.download_image(performer.image_urls[0])
            if data:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(data))
                print(f"SUCCESS: {img.size[0]}x{img.size[1]} ({len(data):,} bytes)")
            else:
                print("FAILED")
    else:
        print("Performer not found")

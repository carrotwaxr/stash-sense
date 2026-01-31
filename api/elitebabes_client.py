"""Client for scraping EliteBabes performer images.

EliteBabes has high-quality glamour photography - excellent for face recognition.
Uses simple requests - no Cloudflare protection observed.

Trust level: MEDIUM - high quality but some multi-person shots
"""
import re
import time
import requests
from typing import Optional

from base_scraper import BaseScraper, ScrapedPerformer


class EliteBabesScraper(BaseScraper):
    """Scraper for EliteBabes performer images."""

    source_name = "elitebabes"
    source_type = "reference_site"
    gender_filter = "FEMALE"

    BASE_URL = "https://www.elitebabes.com"

    def __init__(self, rate_limit_delay: float = 1.0):
        """Initialize EliteBabes scraper.

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
            logger.debug(f"[elitebabes] HTTP {response.status_code} for {url}")
        except Exception as e:
            logger.warning(f"[elitebabes] Fetch error: {e}")
        return None

    def get_performer(self, slug: str) -> Optional[ScrapedPerformer]:
        """Get performer by EliteBabes slug.

        Args:
            slug: EliteBabes URL slug (e.g., 'angela-white')

        Returns:
            ScrapedPerformer or None if not found
        """
        url = f"{self.BASE_URL}/model/{slug}/"
        html = self._fetch_html(url)

        if not html:
            return None

        # Check for actual 404 page (not just "404" appearing in URLs/scripts)
        if "Page Not Found" in html or "<title>404" in html or "page not found" in html.lower():
            return None

        return self._parse_performer_page(slug, html)

    def _parse_performer_page(self, slug: str, html: str) -> Optional[ScrapedPerformer]:
        """Parse an EliteBabes performer page."""
        # Extract name from title or heading
        name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
        if not name_match:
            name_match = re.search(r'<title>([^|<-]+)', html)
        name = name_match.group(1).strip() if name_match else slug.replace("-", " ").title()

        # Clean up name
        name = re.sub(r'\s*(Photos|Gallery|Nude|Model).*$', '', name, flags=re.IGNORECASE).strip()

        image_urls = []

        # Find CDN images - prefer 1200 size
        # Pattern: https://cdn.elitebabes.com/content/{id}/{id}_masonry_1200.jpg
        cdn_pattern = r'(https://cdn\.elitebabes\.com/content/[^"\']+_(?:masonry_)?1200\.(?:jpg|webp))'
        for match in re.findall(cdn_pattern, html, re.IGNORECASE):
            if match not in image_urls:
                image_urls.append(match)

        # Also get 800 size as fallback
        cdn_800_pattern = r'(https://cdn\.elitebabes\.com/content/[^"\']+_(?:masonry_)?800\.(?:jpg|webp))'
        for match in re.findall(cdn_800_pattern, html, re.IGNORECASE):
            # Upgrade to 1200 if possible
            large_url = match.replace('_800.', '_1200.')
            if large_url not in image_urls:
                image_urls.append(large_url)

        # Generic CDN pattern for any remaining images
        generic_cdn = r'(https://cdn\.elitebabes\.com/content/[^"\']+\.(?:jpg|webp))'
        for match in re.findall(generic_cdn, html, re.IGNORECASE):
            # Skip small thumbnails
            if '_thumb' in match or '_small' in match:
                continue
            if match not in image_urls:
                image_urls.append(match)

        if not image_urls:
            return None

        return ScrapedPerformer(
            id=slug,
            name=name,
            image_urls=image_urls[:15],
            gender="FEMALE",
            stash_ids={"elitebabes": slug},
        )

    def query_performers(self, page: int = 1, per_page: int = 25) -> tuple[int, list[ScrapedPerformer]]:
        """EliteBabes doesn't support pagination queries."""
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from EliteBabes."""
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
                    # 1200 might not exist, try 800
                    if '_1200.' in url:
                        fallback_url = url.replace('_1200.', '_800.')
                        logger.debug(f"[elitebabes] 1200 not found, trying 800")
                        return self.download_image(fallback_url, max_retries=1)
                    return None
                elif response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"[elitebabes] Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.debug(f"[elitebabes] Image error: {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"[elitebabes] Download error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

        return None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract EliteBabes slug from URL.

        Example: https://www.elitebabes.com/model/angela-white/ -> angela-white
        """
        match = re.search(r'elitebabes\.com/model/([^/?#]+)', url)
        if match:
            return match.group(1).rstrip("/")
        return None

    def name_to_slug(self, name: str) -> str:
        """Convert name to EliteBabes slug format.

        Example: "Angela White" -> "angela-white"
        """
        slug = name.lower().replace(" ", "-")
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        return slug


if __name__ == "__main__":
    scraper = EliteBabesScraper()

    print("Testing EliteBabes scraper...")

    test_slug = "angela-white"
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

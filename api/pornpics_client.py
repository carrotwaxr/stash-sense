"""Client for scraping PornPics performer images.

PornPics has large, high-quality gallery images perfect for face recognition.
Uses simple requests - no Cloudflare protection observed.

Trust level: MEDIUM - gallery images with some multi-person shots
"""
import re
import time
import requests
from typing import Optional

from base_scraper import BaseScraper, ScrapedPerformer


class PornPicsScraper(BaseScraper):
    """Scraper for PornPics performer images."""

    source_name = "pornpics"
    source_type = "reference_site"
    gender_filter = None  # PornPics has all genders

    BASE_URL = "https://www.pornpics.com"

    def __init__(self, rate_limit_delay: float = 1.0):
        """Initialize PornPics scraper.

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
            logger.debug(f"[pornpics] HTTP {response.status_code} for {url}")
        except Exception as e:
            logger.warning(f"[pornpics] Fetch error: {e}")
        return None

    def get_performer(self, slug: str) -> Optional[ScrapedPerformer]:
        """Get performer by PornPics slug.

        Args:
            slug: PornPics URL slug (e.g., 'angela-white')

        Returns:
            ScrapedPerformer or None if not found
        """
        url = f"{self.BASE_URL}/pornstars/{slug}/"
        html = self._fetch_html(url)

        if not html:
            return None

        if "Page Not Found" in html or "404" in html or "no results" in html.lower():
            return None

        return self._parse_performer_page(slug, html)

    def _parse_performer_page(self, slug: str, html: str) -> Optional[ScrapedPerformer]:
        """Parse a PornPics performer page."""
        # Extract name from title or heading
        name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html)
        if not name_match:
            name_match = re.search(r'<title>([^|<-]+)', html)
        name = name_match.group(1).strip() if name_match else slug.replace("-", " ").title()

        # Clean up name (remove "Pics" suffix etc)
        name = re.sub(r'\s*(Pics|Photos|Gallery|Nude).*$', '', name, flags=re.IGNORECASE).strip()

        image_urls = []

        # Find model headshot: https://cdni.pornpics.com/models/{letter}/{slug}.jpg
        headshot_pattern = r'(https://cdni\.pornpics\.com/models/[^"\']+\.jpg)'
        for match in re.findall(headshot_pattern, html, re.IGNORECASE):
            if match not in image_urls:
                image_urls.append(match)

        # Find gallery images and convert to large size
        # Pattern: https://cdni.pornpics.com/460/{path}.jpg -> /1280/{path}.jpg
        gallery_pattern = r'(https://cdni\.pornpics\.com/460/[^"\']+\.jpg)'
        for match in re.findall(gallery_pattern, html, re.IGNORECASE):
            # Convert to large size
            large_url = match.replace('/460/', '/1280/')
            if large_url not in image_urls:
                image_urls.append(large_url)

        # Also check for other CDN patterns
        cdn_pattern = r'(https://cdn[^"\']*\.pornpics\.com/[^"\']+\.(?:jpg|jpeg|png))'
        for match in re.findall(cdn_pattern, html, re.IGNORECASE):
            # Convert 460 to 1280 if present
            if '/460/' in match:
                match = match.replace('/460/', '/1280/')
            if match not in image_urls:
                image_urls.append(match)

        if not image_urls:
            return None

        return ScrapedPerformer(
            id=slug,
            name=name,
            image_urls=image_urls[:15],  # PornPics has many images, take more
            stash_ids={"pornpics": slug},
        )

    def query_performers(self, page: int = 1, per_page: int = 25) -> tuple[int, list[ScrapedPerformer]]:
        """PornPics doesn't support pagination queries."""
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from PornPics."""
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
                    # Large version might not exist, try medium
                    if '/1280/' in url:
                        medium_url = url.replace('/1280/', '/460/')
                        logger.debug(f"[pornpics] 1280 not found, trying 460")
                        return self.download_image(medium_url, max_retries=1)
                    return None
                elif response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"[pornpics] Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.debug(f"[pornpics] Image error: {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"[pornpics] Download error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

        return None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract PornPics slug from URL.

        Example: https://www.pornpics.com/pornstars/angela-white/ -> angela-white
        """
        match = re.search(r'pornpics\.com/pornstars/([^/?#]+)', url)
        if match:
            return match.group(1).rstrip("/")
        return None

    def name_to_slug(self, name: str) -> str:
        """Convert name to PornPics slug format.

        Example: "Angela White" -> "angela-white"
        """
        slug = name.lower().replace(" ", "-")
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        return slug


if __name__ == "__main__":
    scraper = PornPicsScraper()

    print("Testing PornPics scraper...")

    # Test with a known URL slug
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

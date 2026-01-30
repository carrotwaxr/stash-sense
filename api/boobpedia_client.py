"""Client for scraping Boobpedia performer images.

Boobpedia is a MediaWiki-based encyclopedia with performer profiles and images.
Uses cloudscraper - no FlareSolverr needed.

Trust level: MEDIUM - wiki-style, community edited
See: docs/url-domains-analysis.md
"""
import re
import time
from typing import Iterator, Optional
from urllib.parse import urljoin

import cloudscraper

from base_scraper import BaseScraper, ScrapedPerformer


class BoobpediaScraper(BaseScraper):
    """Scraper for Boobpedia performer data and images."""

    source_name = "boobpedia"
    source_type = "reference_site"
    gender_filter = "FEMALE"  # Boobpedia is female-focused

    BASE_URL = "https://www.boobpedia.com"

    def __init__(
        self,
        rate_limit_delay: float = 0.5,  # 120 req/min
    ):
        """Initialize Boobpedia scraper.

        Args:
            rate_limit_delay: Delay between requests (seconds)
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self._scraper = cloudscraper.create_scraper()

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML via cloudscraper."""
        import logging
        logger = logging.getLogger(__name__)

        self._rate_limit()
        try:
            response = self._scraper.get(url, timeout=30)
            if response.status_code == 200:
                return response.text
            elif response.status_code == 429:
                logger.warning(f"[boobpedia] Rate limited (429) on page fetch")
            else:
                logger.debug(f"[boobpedia] Page fetch failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"[boobpedia] Fetch error: {e}")
        return None

    def get_performer(self, slug: str) -> Optional[ScrapedPerformer]:
        """Get performer by Boobpedia page name.

        Args:
            slug: Boobpedia page name (e.g., 'Angela_White')

        Returns:
            ScrapedPerformer or None if not found
        """
        url = f"{self.BASE_URL}/boobs/{slug}"
        html = self._fetch_html(url)

        if not html:
            return None

        # Check for redirect or missing page
        if "There is currently no text in this page" in html:
            return None

        return self._parse_performer_page(url, html, slug)

    def _parse_performer_page(self, url: str, html: str, slug: str) -> Optional[ScrapedPerformer]:
        """Parse a Boobpedia performer page."""
        # Extract name from page title
        # Pattern: <title>Angela White - Boobpedia</title>
        name_match = re.search(r'<title>([^-<]+)', html, re.IGNORECASE)
        if not name_match:
            # Fallback to slug
            name = slug.replace("_", " ")
        else:
            name = name_match.group(1).strip()

        # Find the infobox image (main profile pic)
        image_urls = []

        # Look for infobox image first
        infobox_match = re.search(
            r'<table[^>]+class="[^"]*infobox[^"]*"[^>]*>(.*?)</table>',
            html,
            re.IGNORECASE | re.DOTALL
        )

        if infobox_match:
            infobox_html = infobox_match.group(1)
            # Find images in infobox (excluding icons)
            infobox_imgs = re.findall(
                r'<img[^>]+src="(/wiki/images/[^"]+)"',
                infobox_html,
                re.IGNORECASE
            )
            for img in infobox_imgs:
                if 'icon' not in img.lower() and '_icon' not in img.lower():
                    # Convert thumbnail to original for better quality
                    original = self._thumb_to_original(img)
                    if original not in image_urls:
                        image_urls.append(original)

        # Also find gallery images if present
        gallery_imgs = re.findall(
            r'<a[^>]+class="[^"]*image[^"]*"[^>]*href="[^"]*"[^>]*>.*?'
            r'<img[^>]+src="(/wiki/images/[^"]+)"',
            html,
            re.IGNORECASE | re.DOTALL
        )

        for img in gallery_imgs:
            if 'icon' not in img.lower() and '_icon' not in img.lower():
                original = self._thumb_to_original(img)
                if original not in image_urls:
                    image_urls.append(original)

        # Prepend base URL
        image_urls = [urljoin(self.BASE_URL, url) for url in image_urls]

        if not image_urls:
            return None

        return ScrapedPerformer(
            id=slug,
            name=name,
            image_urls=image_urls[:10],  # Limit to 10 images
            gender="FEMALE",
            aliases=[],
            external_urls=[url],
            stash_ids=[],
        )

    def _thumb_to_original(self, thumb_url: str) -> str:
        """Convert a MediaWiki thumbnail URL to the original image URL.

        Example:
            /wiki/images/thumb/2/20/Name.jpg/240px-Name.jpg
            -> /wiki/images/2/20/Name.jpg
        """
        # Pattern: /thumb/X/XX/filename.ext/NNNpx-filename.ext
        match = re.match(
            r'(/wiki/images)/thumb(/[a-f0-9]/[a-f0-9]{2}/[^/]+)/\d+px-[^/]+$',
            thumb_url,
            re.IGNORECASE
        )
        if match:
            return match.group(1) + match.group(2)
        return thumb_url

    def iter_performers_after(self, after_id: Optional[str] = None) -> Iterator[ScrapedPerformer]:
        """Boobpedia doesn't support iteration - use URL lookup mode."""
        return iter([])

    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[int, list[ScrapedPerformer]]:
        """Boobpedia doesn't support pagination queries."""
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from Boobpedia."""
        import logging
        logger = logging.getLogger(__name__)

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                response = self._scraper.get(url, timeout=30)

                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type or len(response.content) > 1000:
                        return response.content
                    logger.debug(f"[boobpedia] Image too small or wrong content-type")
                    return None
                elif response.status_code == 429:
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"[boobpedia] Rate limited (429), waiting {wait_time}s")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    return None
                else:
                    logger.debug(f"[boobpedia] Image error: {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"[boobpedia] Download error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return None

        return None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract Boobpedia page name from URL.

        Example: https://www.boobpedia.com/boobs/Angela_White -> Angela_White
        """
        match = re.search(r'boobpedia\.com/boobs/([^/?#]+)', url)
        return match.group(1) if match else None

    def name_to_slug(self, name: str) -> str:
        """Convert name to Boobpedia page name format.

        Example: "Angela White" -> "Angela_White"
        """
        return name.replace(" ", "_")


if __name__ == "__main__":
    # Test the scraper
    scraper = BoobpediaScraper()

    print("Testing Boobpedia scraper...")
    performer = scraper.get_performer("Angela_White")

    if performer:
        print(f"Found: {performer.name}")
        print(f"Images: {len(performer.image_urls)}")
        for url in performer.image_urls[:3]:
            print(f"  {url}")

        if performer.image_urls:
            print("\nDownloading first image...")
            data = scraper.download_image(performer.image_urls[0])
            if data:
                print(f"SUCCESS: {len(data)} bytes")
            else:
                print("FAILED")
    else:
        print("Could not find performer")

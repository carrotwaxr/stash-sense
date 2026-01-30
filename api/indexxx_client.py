"""Client for scraping Indexxx performer images.

Indexxx is a comprehensive adult performer database with profile photos.
Uses Playwright with remote Chrome for Cloudflare bypass (both main site
and img.indexxx.com CDN require browser-level access).

Trust level: HIGH - professional database with curated images
See: docs/url-domains-analysis.md
"""
import os
import re
import time
from typing import Iterator, Optional
from urllib.parse import urljoin

from base_scraper import BaseScraper, ScrapedPerformer


class IndexxxScraper(BaseScraper):
    """Scraper for Indexxx performer data and images."""

    source_name = "indexxx"
    source_type = "reference_site"
    gender_filter = None  # Indexxx has all genders

    BASE_URL = "https://www.indexxx.com"
    IMAGE_BASE = "https://img.indexxx.com"

    def __init__(
        self,
        chrome_cdp_url: str = None,
        rate_limit_delay: float = 3.0,  # Aggressive rate limit to avoid 429s
        flaresolverr_url: str = None,  # For compatibility, but not used
    ):
        """Initialize Indexxx scraper.

        Args:
            chrome_cdp_url: Chrome DevTools Protocol URL (e.g., http://10.0.0.4:9222)
            rate_limit_delay: Delay between requests (seconds)
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self.chrome_cdp_url = chrome_cdp_url or os.environ.get("CHROME_CDP_URL", "http://10.0.0.4:9222")
        self._browser = None
        self._context = None
        self._playwright = None

    def _ensure_browser(self):
        """Ensure browser connection is established."""
        if self._browser is None:
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[indexxx] Connecting to Chrome at {self.chrome_cdp_url}")

            from playwright.sync_api import sync_playwright
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.connect_over_cdp(self.chrome_cdp_url)
            self._context = self._browser.new_context()

            # Warm up by visiting main site to establish Cloudflare cookies
            logger.info("[indexxx] Warming up - visiting main site for cookies")
            self._visit_main_site()
            logger.info("[indexxx] Browser ready")

    def _fetch_html(self, url: str, max_retries: int = 3) -> Optional[str]:
        """Fetch HTML via Playwright/Chrome."""
        import logging
        logger = logging.getLogger(__name__)
        self._ensure_browser()

        for attempt in range(max_retries):
            self._rate_limit()
            try:
                page = self._context.new_page()
                # Use domcontentloaded for faster page loads
                response = page.goto(url, wait_until='domcontentloaded', timeout=60000)

                if response and response.status == 200:
                    html = page.content()
                    page.close()
                    return html

                status = response.status if response else "no response"
                page.close()

                # Handle rate limiting
                if response and response.status == 429:
                    wait_time = 30 * (attempt + 1)  # 30s, 60s, 90s
                    logger.warning(f"[indexxx] Rate limited (429), waiting {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                logger.debug(f"[indexxx] Page fetch failed: {status} for {url}")
                return None

            except Exception as e:
                logger.warning(f"[indexxx] Fetch error for {url}: {e}")
                try:
                    page.close()
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

        return None

    def get_performer(self, slug: str) -> Optional[ScrapedPerformer]:
        """Get performer by Indexxx slug.

        Args:
            slug: Indexxx model slug (e.g., 'angela-white')

        Returns:
            ScrapedPerformer or None if not found
        """
        url = f"{self.BASE_URL}/m/{slug}"
        html = self._fetch_html(url)

        if not html:
            return None

        return self._parse_performer_page(url, html, slug)

    def _parse_performer_page(self, url: str, html: str, slug: str) -> Optional[ScrapedPerformer]:
        """Parse an Indexxx performer page."""
        # Extract name from page title or heading
        # Pattern: <title>Angela White - Indexxx</title> or <h1>Angela White</h1>
        name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if not name_match:
            name_match = re.search(r'<title>([^-<]+)', html, re.IGNORECASE)

        if not name_match:
            return None

        name = name_match.group(1).strip()

        # Find profile/model images
        # Pattern: https://img.indexxx.com/images/models/angela-white-1488849.jpg
        image_urls = []

        # Main model image
        model_imgs = re.findall(
            r'(https://img\.indexxx\.com/images/models/[^"\'>\s]+\.(?:jpg|jpeg|png|webp))',
            html,
            re.IGNORECASE
        )
        image_urls.extend(model_imgs)

        # Also check for thumbnail images that might be higher quality versions
        thumb_imgs = re.findall(
            r'(https://img\.indexxx\.com/images/thumbs/[^"\'>\s]+\.(?:jpg|jpeg|png|webp))',
            html,
            re.IGNORECASE
        )
        image_urls.extend(thumb_imgs)

        # Deduplicate while preserving order
        seen = set()
        unique_urls = []
        for img_url in image_urls:
            if img_url not in seen:
                seen.add(img_url)
                unique_urls.append(img_url)

        if not unique_urls:
            return None

        return ScrapedPerformer(
            id=slug,
            name=name,
            image_urls=unique_urls[:10],  # Limit to 10 images
            gender=None,
            aliases=[],
            external_urls=[url],
            stash_ids=[],
        )

    def iter_performers_after(self, after_id: Optional[str] = None) -> Iterator[ScrapedPerformer]:
        """Indexxx doesn't support iteration - use URL lookup mode."""
        return iter([])

    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[int, list[ScrapedPerformer]]:
        """Indexxx doesn't support pagination queries."""
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from Indexxx via Playwright.

        After visiting the main site, cookies allow access to img.indexxx.com.
        """
        import logging
        logger = logging.getLogger(__name__)
        self._ensure_browser()

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                page = self._context.new_page()
                # Use 'load' instead of 'networkidle' - faster for images
                # Increase timeout to 60s for slow connections
                response = page.goto(url, wait_until='load', timeout=60000)

                if response and response.status == 200:
                    body = response.body()
                    page.close()
                    if len(body) > 1000:
                        return body
                    logger.debug(f"[indexxx] Image too small: {len(body)} bytes")
                    return None

                status = response.status if response else "no response"
                page.close()

                if response and response.status == 403:
                    # Might need to refresh cookies by visiting main site
                    if attempt == 0:
                        logger.warning(f"[indexxx] Got 403, refreshing cookies...")
                        self._visit_main_site()
                        continue
                    logger.warning(f"[indexxx] Still 403 after cookie refresh")
                    return None

                logger.debug(f"[indexxx] Image download failed: {status}")

            except Exception as e:
                logger.warning(f"[indexxx] Download error (attempt {attempt+1}/{max_retries}): {e}")
                try:
                    page.close()
                except Exception:
                    pass
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None

        return None

    def _visit_main_site(self):
        """Visit main site to refresh Cloudflare cookies."""
        import logging
        logger = logging.getLogger(__name__)
        try:
            page = self._context.new_page()
            page.goto(f"{self.BASE_URL}/", wait_until='networkidle', timeout=90000)
            time.sleep(3)  # Let Cloudflare fully process
            page.close()
        except Exception as e:
            logger.warning(f"[indexxx] Failed to visit main site: {e}")

    def close(self):
        """Clean up browser resources."""
        if self._context:
            self._context.close()
            self._context = None
        if self._browser:
            self._browser = None
        if self._playwright:
            self._playwright.stop()
            self._playwright = None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract Indexxx model slug from URL.

        Example: https://www.indexxx.com/m/angela-white -> angela-white
        """
        match = re.search(r'indexxx\.com/m/([^/?#]+)', url)
        return match.group(1) if match else None

    def name_to_slug(self, name: str) -> str:
        """Convert name to Indexxx slug format.

        Example: "Angela White" -> "angela-white"
        """
        # Lowercase and replace spaces with hyphens
        slug = name.lower().replace(" ", "-")
        # Remove special characters except hyphens
        slug = re.sub(r'[^a-z0-9-]', '', slug)
        # Collapse multiple hyphens
        slug = re.sub(r'-+', '-', slug)
        return slug.strip('-')


if __name__ == "__main__":
    # Test the scraper
    scraper = IndexxxScraper()

    print("Testing Indexxx scraper...")
    performer = scraper.get_performer("angela-white")

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

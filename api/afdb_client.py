"""Client for scraping Adult Film Database (AFDB) performer images.

AFDB is a comprehensive filmography database with performer headshots and gallery images.
Uses simple requests - no Cloudflare protection observed.

Trust level: HIGH - curated headshots
"""
import re
import time
import requests
from typing import Optional

from base_scraper import BaseScraper, ScrapedPerformer


class AFDBScraper(BaseScraper):
    """Scraper for Adult Film Database performer images."""

    source_name = "afdb"
    source_type = "reference_site"
    gender_filter = None  # AFDB has all genders
    url_site_pattern = "adultfilmdatabase"  # URL pattern differs from source_name

    BASE_URL = "https://www.adultfilmdatabase.com"

    def __init__(self, rate_limit_delay: float = 1.0):
        """Initialize AFDB scraper.

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
            logger.debug(f"[afdb] HTTP {response.status_code} for {url}")
        except Exception as e:
            logger.warning(f"[afdb] Fetch error: {e}")
        return None

    def get_performer(self, slug: str) -> Optional[ScrapedPerformer]:
        """Get performer by AFDB slug.

        Args:
            slug: AFDB URL path (e.g., 'jayden-james-45421')

        Returns:
            ScrapedPerformer or None if not found
        """
        url = f"{self.BASE_URL}/actor/{slug}/"
        html = self._fetch_html(url)

        if not html:
            return None

        if "Page Not Found" in html or "404" in html:
            return None

        return self._parse_performer_page(slug, html)

    def _parse_performer_page(self, slug: str, html: str) -> Optional[ScrapedPerformer]:
        """Parse an AFDB performer page."""
        # Extract name from title: "Jayden James | Adult Film Database"
        name_match = re.search(r'<title>([^|<]+)', html)
        name = name_match.group(1).strip() if name_match else slug.replace("-", " ").title()

        # Remove ID suffix from name if present (e.g., "Jayden James 45421" -> "Jayden James")
        name = re.sub(r'\s*\d+$', '', name).strip()

        image_urls = []

        # Find main performer headshot: /Graphics/PornStars/{slug}_{id}.jpg
        headshot_pattern = r'src="(/Graphics/PornStars/[^"]+\.(?:jpg|png))"'
        for match in re.findall(headshot_pattern, html, re.IGNORECASE):
            full_url = f"{self.BASE_URL}{match}"
            if full_url not in image_urls:
                image_urls.append(full_url)

        # Find gallery images: /graphics/galleries/{id}/{id}.jpg
        gallery_pattern = r'src="(/graphics/galleries/[^"]+\.(?:jpg|png))"'
        for match in re.findall(gallery_pattern, html, re.IGNORECASE):
            full_url = f"{self.BASE_URL}{match}"
            if full_url not in image_urls:
                image_urls.append(full_url)

        if not image_urls:
            return None

        # Extract ID from slug (e.g., 'jayden-james-45421' -> '45421')
        id_match = re.search(r'-(\d+)$', slug)
        performer_id = id_match.group(1) if id_match else slug

        # Parse additional metadata
        aliases = []
        country = None
        birth_date = None

        # Aliases pattern: "AKA: alias1, alias2"
        aka_match = re.search(r'AKA:\s*([^<]+)', html, re.IGNORECASE)
        if aka_match:
            aliases = [a.strip() for a in aka_match.group(1).split(",") if a.strip()]

        # Country/nationality
        country_match = re.search(r'Nationality:\s*([^<]+)', html, re.IGNORECASE)
        if country_match:
            country = country_match.group(1).strip()

        # Birth date
        birth_match = re.search(r'Born:\s*([^<]+)', html, re.IGNORECASE)
        if birth_match:
            birth_date = birth_match.group(1).strip()

        return ScrapedPerformer(
            id=performer_id,
            name=name,
            image_urls=image_urls[:10],
            aliases=aliases,
            country=country,
            birth_date=birth_date,
            stash_ids={"afdb": performer_id},
        )

    def query_performers(self, page: int = 1, per_page: int = 25) -> tuple[int, list[ScrapedPerformer]]:
        """AFDB doesn't support pagination queries."""
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from AFDB."""
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
                    logger.warning(f"[afdb] Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.debug(f"[afdb] Image error: {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"[afdb] Download error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(5)
                    continue
                return None

        return None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract AFDB slug from URL.

        Example: https://www.adultfilmdatabase.com/actor/jayden-james-45421/ -> jayden-james-45421
        """
        match = re.search(r'adultfilmdatabase\.com/actor/([^/?#]+)', url)
        if match:
            # Remove trailing slash if present
            return match.group(1).rstrip("/")
        return None

    def name_to_slug(self, name: str) -> str:
        """Convert name to AFDB slug format.

        Note: AFDB slugs include an ID suffix that we don't have from name alone.
        Use URL lookup instead of name lookup for AFDB.
        """
        return name.lower().replace(" ", "-")


if __name__ == "__main__":
    scraper = AFDBScraper()

    print("Testing AFDB scraper...")

    # Test with a known URL slug
    test_slug = "jayden-james-45421"
    performer = scraper.get_performer(test_slug)

    if performer:
        print(f"Found: {performer.name}")
        print(f"Images: {len(performer.image_urls)}")
        for url in performer.image_urls[:3]:
            print(f"  - {url}")
        print(f"Aliases: {performer.aliases}")
        print(f"Country: {performer.country}")

        if performer.image_urls:
            print("\nDownloading first image...")
            data = scraper.download_image(performer.image_urls[0])
            if data:
                print(f"SUCCESS: {len(data)} bytes")
            else:
                print("FAILED")
    else:
        print("Performer not found")

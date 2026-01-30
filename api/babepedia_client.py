"""Client for scraping Babepedia performer images.

Babepedia is a reference site with high-quality headshots of female performers.
Requires FlareSolverr for Cloudflare bypass.

Trust level: HIGH - curated headshots, single-face images
"""
import re
import time
from typing import Iterator, Optional
from urllib.parse import quote, unquote

from base_scraper import BaseScraper, ScrapedPerformer
from flaresolverr_client import FlareSolverr


class BabepediaScraper(BaseScraper):
    """Scraper for Babepedia performer images."""

    source_name = "babepedia"
    source_type = "reference_site"
    gender_filter = "FEMALE"

    BASE_URL = "https://www.babepedia.com"

    def __init__(
        self,
        flaresolverr_url: str = "http://10.0.0.4:8191",
        rate_limit_delay: float = 2.0,  # Be respectful - 30/min
    ):
        """Initialize Babepedia scraper.

        Args:
            flaresolverr_url: FlareSolverr proxy URL
            rate_limit_delay: Delay between requests (seconds)
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self.flaresolverr = FlareSolverr(flaresolverr_url)

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML via FlareSolverr."""
        self._rate_limit()
        response = self.flaresolverr.get(url)
        if response and response.solution_status == 200:
            return response.solution_html
        return None

    def get_performer(self, performer_id: str) -> Optional[ScrapedPerformer]:
        """Get performer by Babepedia slug (e.g., 'Mia_Malkova').

        Args:
            performer_id: Babepedia URL slug (name with underscores)

        Returns:
            ScrapedPerformer or None if not found
        """
        url = f"{self.BASE_URL}/babe/{quote(performer_id)}"
        html = self._fetch_html(url)

        if not html:
            return None

        # Check if performer exists (404 pages still return 200)
        if "Page Not Found" in html or "does not exist" in html.lower():
            return None

        return self._parse_performer_page(performer_id, html)

    def _parse_performer_page(self, slug: str, html: str) -> ScrapedPerformer:
        """Parse a Babepedia performer page."""
        # Extract name from title or heading
        name_match = re.search(r'<title>([^<]+)\s*[-|]', html)
        name = name_match.group(1).strip() if name_match else slug.replace("_", " ")

        # Find all thumbnail images in /pics/
        thumb_pattern = r'/pics/([^"\']+?)_thumb\d*\.jpg'
        thumbs = re.findall(thumb_pattern, html, re.IGNORECASE)

        # Convert to full-size image URLs (remove _thumb suffix)
        image_urls = []
        seen = set()
        for thumb in thumbs:
            # Decode URL encoding
            decoded = unquote(thumb)
            if decoded not in seen:
                seen.add(decoded)
                # Full size URL
                full_url = f"{self.BASE_URL}/pics/{quote(decoded)}.jpg"
                image_urls.append(full_url)

        # Also check for direct image references
        direct_pattern = r'src=["\']([^"\']+/pics/[^"\']+\.jpg)["\']'
        for match in re.findall(direct_pattern, html, re.IGNORECASE):
            if "_thumb" not in match.lower():
                if match.startswith("/"):
                    match = f"{self.BASE_URL}{match}"
                if match not in image_urls:
                    image_urls.append(match)

        # Parse info-items (label/value pairs)
        # Pattern: <span class="label">Key:</span> <span class="value">Value</span>
        info_pattern = r'<span class="label">([^<]+)</span>\s*<span class="value">([^<]+)'

        aliases = []
        country = None
        birth_date = None

        for match in re.finditer(info_pattern, html, re.IGNORECASE):
            label = match.group(1).strip().rstrip(":")
            value = match.group(2).strip()

            if "alias" in label.lower() or "known as" in label.lower():
                # Parse aliases
                aliases = [a.strip() for a in re.split(r'[,/]', value) if a.strip()]
            elif "birthplace" in label.lower():
                # Extract country from birthplace (last part usually)
                parts = value.split(",")
                if parts:
                    country = parts[-1].strip()
            elif "country" in label.lower():
                country = value

        # Extract birth year from the Born line (more complex HTML)
        birth_match = re.search(r'born-in-the-year/(\d{4})', html, re.IGNORECASE)
        if birth_match:
            birth_date = birth_match.group(1)

        return ScrapedPerformer(
            id=slug,
            name=name,
            image_urls=image_urls[:10],  # Limit to 10 images
            aliases=aliases,
            gender="FEMALE",  # Babepedia is female-only
            country=country,
            birth_date=birth_date,
            stash_ids={"babepedia": slug},
        )

    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[int, list[ScrapedPerformer]]:
        """Query performers from Babepedia.

        Note: Babepedia doesn't have great pagination. This uses the A-Z index.
        For enrichment, we'll look up performers by name from our database instead.

        Returns:
            Tuple of (total_count, list of performers)
        """
        # Babepedia has an A-Z index at /babes/letter/A etc
        # For now, return empty - we'll use lookup by name instead
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from Babepedia.

        Note: Uses FlareSolverr for the initial page load to get cookies,
        then uses those cookies for image downloads.
        """
        import requests

        for attempt in range(max_retries):
            try:
                self._rate_limit()
                # Use a regular request with common headers
                response = requests.get(
                    url,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Referer": self.BASE_URL,
                    },
                    timeout=30,
                )

                if response.status_code == 200:
                    return response.content
                elif response.status_code == 403:
                    # Cloudflare blocked - would need to use FlareSolverr
                    print(f"  Babepedia image blocked (403), may need FlareSolverr")
                    return None
                else:
                    print(f"  Babepedia image error: {response.status_code}")

            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                print(f"Failed to download {url}: {e}")
                return None

        return None

    def lookup_by_name(self, name: str) -> Optional[ScrapedPerformer]:
        """Look up a performer by name.

        This is the primary method for enrichment - we look up performers
        from our existing database by their name on Babepedia.

        Args:
            name: Performer name (will be converted to Babepedia slug)

        Returns:
            ScrapedPerformer or None if not found
        """
        # Convert name to Babepedia slug format
        # e.g., "Mia Malkova" -> "Mia_Malkova"
        slug = name.replace(" ", "_")
        return self.get_performer(slug)

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


if __name__ == "__main__":
    # Test the scraper
    scraper = BabepediaScraper()

    if not scraper.flaresolverr.is_available():
        print("FlareSolverr is not available!")
        exit(1)

    print("Testing Babepedia scraper...")

    # Test lookup
    performer = scraper.lookup_by_name("Mia Malkova")
    if performer:
        print(f"\nFound: {performer.name}")
        print(f"  ID: {performer.id}")
        print(f"  Images: {len(performer.image_urls)}")
        print(f"  Aliases: {performer.aliases}")
        print(f"  Country: {performer.country}")
        if performer.image_urls:
            print(f"  First image: {performer.image_urls[0]}")
    else:
        print("Performer not found!")

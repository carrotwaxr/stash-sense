"""Client for scraping IAFD (Internet Adult Film Database) performer images.

IAFD is a comprehensive reference site with performer data and headshots.
Requires FlareSolverr for Cloudflare bypass.

Trust level: HIGH - authoritative filmography database
See: docs/plans/2026-01-27-data-sources-catalog.md
"""
import re
import time
from typing import Iterator, Optional
from urllib.parse import quote, unquote, urljoin

from base_scraper import BaseScraper, ScrapedPerformer
from flaresolverr_client import FlareSolverr


class IAFDScraper(BaseScraper):
    """Scraper for IAFD performer data and images."""

    source_name = "iafd"
    source_type = "reference_site"
    gender_filter = None  # IAFD has both genders

    BASE_URL = "https://www.iafd.com"

    def __init__(
        self,
        flaresolverr_url: str = "http://10.0.0.4:8191",
        rate_limit_delay: float = 0.5,  # 120 req/min
    ):
        """Initialize IAFD scraper.

        Args:
            flaresolverr_url: FlareSolverr proxy URL
            rate_limit_delay: Delay between requests (seconds)
        """
        super().__init__(rate_limit_delay=rate_limit_delay)
        self.flaresolverr = FlareSolverr(flaresolverr_url)
        self._last_cookies = []
        self._last_user_agent = ""
        self._last_page_url = ""  # Track the page URL for Referer

    def _fetch_html(self, url: str) -> Optional[str]:
        """Fetch HTML via FlareSolverr."""
        import logging
        logger = logging.getLogger(__name__)

        self._rate_limit()
        response = self.flaresolverr.get(url)
        if response and response.solution_status == 200:
            # Store cookies, user-agent, and page URL for image downloads
            self._last_cookies = response.cookies
            self._last_user_agent = response.user_agent
            self._last_page_url = response.solution_url or url
            return response.solution_html
        elif response:
            logger.debug(f"[iafd] FlareSolverr returned status {response.solution_status} for {url}")
        else:
            logger.debug(f"[iafd] FlareSolverr returned None for {url}")
        return None

    def get_performer(self, performer_id: str) -> Optional[ScrapedPerformer]:
        """Get performer by IAFD perfid slug.

        The performer_id should be the IAFD perfid (e.g., 'aaliyahhadid').
        This is typically found in IAFD URLs from StashDB's urls field.

        Args:
            performer_id: IAFD perfid slug

        Returns:
            ScrapedPerformer or None if not found
        """
        # IAFD URLs can vary, try the common format first
        # Pattern: /person.rme/perfid={slug}/{Display-Name}.htm
        # We'll search for the performer instead to get the correct URL
        return self.lookup_by_name(performer_id.replace("-", " ").title())

    def _parse_performer_page(self, url: str, html: str) -> Optional[ScrapedPerformer]:
        """Parse an IAFD performer page."""
        # Extract ID from URL - can be perfid= or id= (UUID) format
        perfid_match = re.search(r'(?:perfid|id)=([^/&]+)', url, re.IGNORECASE)
        if not perfid_match:
            return None
        perfid = perfid_match.group(1)

        # Extract name from heading
        name_match = re.search(r'<h1[^>]*>([^<]+)</h1>', html, re.IGNORECASE)
        if not name_match:
            # Try title
            name_match = re.search(r'<title>([^<|]+)', html, re.IGNORECASE)

        name = name_match.group(1).strip() if name_match else perfid.replace("-", " ").title()

        # Extract headshot image
        # IAFD stores headshots at: https://www.iafd.com/graphics/headshots/...
        image_urls = []

        # Primary headshot patterns
        headshot_patterns = [
            # Direct src with full URL
            r'<img[^>]+src="(https?://[^"]+/graphics/headshots/[^"]+\.jpg)"',
            # Relative path
            r'<img[^>]+src="(/graphics/headshots/[^"]+\.jpg)"',
            # Any image with headshots in path
            r'(https?://www\.iafd\.com/graphics/headshots/[^"\'<>\s]+\.jpg)',
        ]

        for pattern in headshot_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            for match in matches:
                if match and "/nophoto" not in match.lower():
                    url = match if match.startswith("http") else urljoin(self.BASE_URL, match)
                    if url not in image_urls:
                        image_urls.append(url)

        # Parse biodata
        aliases = []
        country = None
        birth_date = None
        ethnicity = None
        gender = None

        # IAFD uses a biodata section with label/value pairs
        # Patterns vary:
        # - <p class="bioheading">Label</p><p class="biodata">Value</p>
        # - <p class="bioheading">AKA</p><div class="biodata">...</div>
        # - Birthday may contain links: <p class="biodata"><a href=...>March 4, 1985</a></p>

        # Pattern for standard p/p format - handles content with or without links
        bio_pattern = r'<p class="bioheading">([^<]+)</p>\s*<(?:p|div) class="biodata"[^>]*>(.*?)</(?:p|div)>'

        for match in re.finditer(bio_pattern, html, re.IGNORECASE | re.DOTALL):
            label = match.group(1).strip().lower()
            raw_value = match.group(2)

            # Replace <br> tags with newlines before stripping other HTML
            value_with_breaks = re.sub(r'<br\s*/?\s*>', '\n', raw_value, flags=re.IGNORECASE)
            # Strip remaining HTML tags
            value = re.sub(r'<[^>]+>', '', value_with_breaks).strip()

            if not value or value.lower() in ("no data", "n/a", "-", "none"):
                continue

            if label == "aka" or "alias" in label:
                # Parse comma/semicolon/linebreak separated aliases
                raw_aliases = re.split(r'[,;\n]', value)
                for alias in raw_aliases:
                    # Remove parenthetical notes like "(Abby Winters)"
                    clean = re.sub(r'\([^)]*\)', '', alias).strip()
                    if clean and clean.lower() != name.lower() and len(clean) > 1:
                        aliases.append(clean)
            elif "nationality" in label:
                country = value
            elif "birthplace" in label:
                # Extract country (last part of birthplace)
                if "," in value:
                    country = value.split(",")[-1].strip()
                else:
                    country = value
            elif "birthday" in label:
                # Parse date - IAFD displays as "March 4, 1985" etc
                # Try to extract year
                year_match = re.search(r'(\d{4})', value)
                if year_match:
                    birth_date = year_match.group(1)
            elif "ethnicity" in label:
                ethnicity = value
            elif "gender" in label:
                gender = value.upper()
                if gender not in ("MALE", "FEMALE"):
                    gender = None

        # Also try to extract AKA from the div format specifically
        aka_div_match = re.search(r'<p class="bioheading">\s*AKA\s*</p>\s*<div class="biodata"[^>]*>(.*?)</div>', html, re.IGNORECASE | re.DOTALL)
        if aka_div_match and not aliases:
            aka_content = aka_div_match.group(1)
            # Replace <br>, <br/>, <br /> with newlines
            aka_text = re.sub(r'<br\s*/?\s*>', '\n', aka_content, flags=re.IGNORECASE)
            # Remove any remaining HTML tags
            aka_text = re.sub(r'<[^>]+>', '', aka_text)
            for line in aka_text.split('\n'):
                alias = line.strip()
                # Remove parenthetical notes
                alias = re.sub(r'\([^)]*\)', '', alias).strip()
                if alias and alias.lower() != name.lower() and len(alias) > 1:
                    aliases.append(alias)

        # Build external URLs dict with the IAFD profile
        external_urls = {"IAFD": [url]}

        return ScrapedPerformer(
            id=perfid,
            name=name,
            image_urls=image_urls[:3],  # IAFD usually has 1-3 images
            aliases=aliases,
            gender=gender,
            country=country,
            birth_date=birth_date,
            ethnicity=ethnicity,
            external_urls=external_urls,
            stash_ids={"iafd": perfid},
        )

    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[int, list[ScrapedPerformer]]:
        """Query performers from IAFD.

        IAFD doesn't have a good pagination API.
        For enrichment, we look up performers by name from our database.

        Returns:
            Tuple of (total_count, list of performers)
        """
        return (0, [])

    def download_image(self, url: str, max_retries: int = 3) -> Optional[bytes]:
        """Download an image from IAFD.

        IAFD has Cloudflare protection. We use cloudscraper which handles
        the JavaScript challenges automatically.
        """
        import logging
        import cloudscraper

        logger = logging.getLogger(__name__)

        for attempt in range(max_retries):
            try:
                self._rate_limit()

                # Use cloudscraper which handles Cloudflare challenges
                scraper = cloudscraper.create_scraper()
                response = scraper.get(url, timeout=30)

                if response.status_code == 200:
                    content_type = response.headers.get("content-type", "")
                    if "image" in content_type or len(response.content) > 1000:
                        return response.content
                    logger.debug(f"[iafd] Image too small or wrong content-type: {len(response.content)} bytes")
                    return None
                elif response.status_code == 403:
                    # Cloudflare blocked - retry with backoff
                    wait_time = 10 * (attempt + 1)
                    logger.warning(f"[iafd] 403 on image (attempt {attempt+1}/{max_retries}), waiting {wait_time}s")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    return None
                elif response.status_code == 429:
                    # Rate limited - longer backoff
                    wait_time = 30 * (attempt + 1)
                    logger.warning(f"[iafd] Rate limited (429), waiting {wait_time}s")
                    if attempt < max_retries - 1:
                        time.sleep(wait_time)
                        continue
                    return None
                else:
                    logger.debug(f"[iafd] Image download failed: {response.status_code}")
                    return None

            except Exception as e:
                logger.warning(f"[iafd] Download error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                return None

        return None

    def extract_slug_from_url(self, url: str) -> Optional[str]:
        """Extract IAFD performer ID from URL.

        Example: https://www.iafd.com/person.rme/perfid=miamalkova/... -> miamalkova
        """
        match = re.search(r'perfid=([^/]+)', url)
        return match.group(1) if match else None

    def name_to_slug(self, name: str) -> str:
        """Convert name to IAFD slug format.

        Example: "Mia Malkova" -> "miamalkova"
        """
        return name.lower().replace(" ", "").replace("-", "")

    def lookup_by_name(self, name: str) -> Optional[ScrapedPerformer]:
        """Look up a performer by name using IAFD search.

        Args:
            name: Performer name

        Returns:
            ScrapedPerformer or None if not found
        """
        # Use IAFD's search API
        search_url = f"{self.BASE_URL}/results.asp?searchtype=comprehensive&searchstring={quote(name)}"
        html = self._fetch_html(search_url)

        if not html:
            return None

        # Parse search results to find performer link
        # IAFD uses two patterns:
        # Old: /person.rme/perfid={slug}/{Name}.htm
        # New: /person.rme/id={uuid}
        performer_patterns = [
            r'<a href="(/person\.rme/perfid=[^"]+)"[^>]*>([^<]+)</a>',
            r'<a href="(/person\.rme/id=[^"]+)"[^>]*>([^<]+)</a>',
        ]

        best_match = None
        best_score = 0
        name_lower = name.lower()

        for pattern in performer_patterns:
            for match in re.finditer(pattern, html, re.IGNORECASE):
                link = match.group(1)
                found_name = match.group(2).strip()

                # Skip non-performer links
                if not found_name or found_name.lower() in ("performer", "details", "view"):
                    continue

                # Score the match
                score = 0
                if found_name.lower() == name_lower:
                    score = 100  # Exact match
                elif name_lower in found_name.lower():
                    score = 80  # Name contained
                elif found_name.lower() in name_lower:
                    score = 70  # Partial match
                else:
                    # Check word overlap
                    name_words = set(name_lower.split())
                    found_words = set(found_name.lower().split())
                    overlap = len(name_words & found_words)
                    if overlap > 0:
                        score = 50 + (overlap * 10)

                if score > best_score:
                    best_score = score
                    best_match = link

        if not best_match or best_score < 50:
            return None

        # Fetch the performer page
        performer_url = urljoin(self.BASE_URL, best_match)
        performer_html = self._fetch_html(performer_url)

        if not performer_html:
            return None

        return self._parse_performer_page(performer_url, performer_html)

    def lookup_by_url(self, iafd_url: str) -> Optional[ScrapedPerformer]:
        """Look up a performer by their IAFD URL.

        This is the preferred method when we have the URL from StashDB.

        Args:
            iafd_url: Full IAFD performer URL

        Returns:
            ScrapedPerformer or None if not found
        """
        # Normalize URL
        if not iafd_url.startswith("http"):
            iafd_url = urljoin(self.BASE_URL, iafd_url)

        html = self._fetch_html(iafd_url)
        if not html:
            return None

        return self._parse_performer_page(iafd_url, html)


if __name__ == "__main__":
    # Test the scraper
    scraper = IAFDScraper()

    if not scraper.flaresolverr.is_available():
        print("FlareSolverr is not available!")
        exit(1)

    print("Testing IAFD scraper...")

    # Test lookup by name
    performer = scraper.lookup_by_name("Angela White")
    if performer:
        print(f"\nFound: {performer.name}")
        print(f"  ID: {performer.id}")
        print(f"  Images: {len(performer.image_urls)}")
        for url in performer.image_urls:
            print(f"    - {url}")
        print(f"  Aliases: {performer.aliases}")
        print(f"  Country: {performer.country}")
        print(f"  Birth date: {performer.birth_date}")
        print(f"  Ethnicity: {performer.ethnicity}")
    else:
        print("Performer not found!")

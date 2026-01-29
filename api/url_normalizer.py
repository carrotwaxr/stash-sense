"""
URL Normalization Library for Performer Identity Graph

Parses URLs from various sites into canonical (site, id) tuples for matching.
Used by the URL-first resolution strategy to link performers across databases.

See: docs/plans/2026-01-27-performer-identity-graph.md
"""

import re
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, unquote


@dataclass
class NormalizedURL:
    """A normalized URL with site identifier and canonical ID."""
    site: str           # e.g., "iafd", "twitter", "stashdb"
    normalized_id: str  # e.g., "aaliyahhadid", "stashdb-uuid"
    original_url: str   # The original URL for reference
    confidence: float   # How confident we are in the normalization (0-1)


# High-confidence sites (unique IDs, reliable for identity matching)
HIGH_CONFIDENCE_SITES = {
    "stashdb", "theporndb", "pmvstash", "javstash", "fansdb",  # Stash-boxes
    "iafd", "imdb", "wikidata",  # Unique IDs
    "twitter", "instagram", "onlyfans", "fansly",  # Social (unique usernames)
}

# Lower confidence (may have duplicates or ambiguity)
MEDIUM_CONFIDENCE_SITES = {
    "pornhub", "xvideos", "freeones", "babepedia", "boobpedia",
    "indexxx", "afdb", "data18",
}


class URLNormalizer:
    """
    Normalizes URLs from various adult industry sites into canonical identifiers.

    Usage:
        normalizer = URLNormalizer()
        result = normalizer.normalize("https://www.iafd.com/person.rme/perfid=aaliyahhadid/Aaliyah-Hadid.htm")
        # result.site = "iafd"
        # result.normalized_id = "aaliyahhadid"
    """

    def __init__(self):
        # Compile regex patterns for performance
        self._patterns = self._compile_patterns()

    def _compile_patterns(self) -> dict:
        """Compile URL matching patterns."""
        return {
            # Stash-boxes (UUID-based)
            "stashdb": re.compile(r"stashdb\.org/performers/([a-f0-9-]{36})", re.I),
            "pmvstash": re.compile(r"pmvstash\.org/performers/([a-f0-9-]{36})", re.I),
            "javstash": re.compile(r"javstash\.org/performers/([a-f0-9-]{36})", re.I),
            "fansdb": re.compile(r"fansdb\.cc/performers/([a-f0-9-]{36})", re.I),

            # ThePornDB (slug-based)
            "theporndb": re.compile(r"theporndb\.net/performers/([^/?#]+)", re.I),

            # IAFD - multiple URL formats
            "iafd": re.compile(r"iafd\.com/person\.rme/perfid=([^/]+)", re.I),

            # IMDb
            "imdb": re.compile(r"imdb\.com/name/(nm\d+)", re.I),

            # Wikidata
            "wikidata": re.compile(r"wikidata\.org/wiki/(Q\d+)", re.I),

            # Wikipedia (various languages)
            "wikipedia": re.compile(r"(\w+)\.wikipedia\.org/wiki/([^?#]+)", re.I),

            # Social media (note: x.com pattern must not match inside other domains like indexxx.com)
            # Use (?:^|://) to ensure we're matching at domain position
            "twitter": re.compile(r"(?:twitter\.com|(?:^|[/\.])x\.com)/([^/?#]+)", re.I),
            "instagram": re.compile(r"instagram\.com/([^/?#]+)", re.I),
            "onlyfans": re.compile(r"onlyfans\.com/([^/?#]+)", re.I),
            "fansly": re.compile(r"fansly\.com/([^/?#]+)", re.I),
            "tiktok": re.compile(r"tiktok\.com/@([^/?#]+)", re.I),
            "threads": re.compile(r"threads\.(?:com|net)/@([^/?#]+)", re.I),

            # Reference sites
            "pornhub": re.compile(r"pornhub\.com/pornstar/([^/?#]+)", re.I),
            "xvideos": re.compile(r"xvideos\.com/pornstars?/([^/?#]+)", re.I),
            "freeones": re.compile(r"freeones\.com/([^/?#]+)", re.I),
            "babepedia": re.compile(r"babepedia\.com/babe/([^/?#]+)", re.I),
            "boobpedia": re.compile(r"boobpedia\.com/boobs/([^/?#]+)", re.I),
            "indexxx": re.compile(r"indexxx\.com/m/([^/?#]+)", re.I),

            # Additional databases
            "afdb": re.compile(r"adultfilmdatabase\.com/actor/([^/?#]+)", re.I),
            "data18": re.compile(r"data18\.com/name/([^/?#]+)", re.I),
            "dbnaked": re.compile(r"dbnaked\.com/models/general/\w/([^/?#]+)", re.I),
            "thenude": re.compile(r"thenude\.com/([^_]+)_(\d+)\.htm", re.I),

            # JAV-specific
            "dmm_fanza": re.compile(r"(?:dmm\.co\.jp|fanza\.com)/.*article=actress/id=(\d+)", re.I),
            "r18dev": re.compile(r"r18\.dev/idols/([^/?#]+)", re.I),
            "minnano_av": re.compile(r"minnano-av\.com/actress(\d+)\.html", re.I),
            "xcity": re.compile(r"xcity\.jp/idol/detail/(\d+)", re.I),

            # Tube sites (lower confidence - same slug doesn't always mean same person)
            "pornpics": re.compile(r"pornpics\.com/pornstars/([^/?#]+)", re.I),
            "elitebabes": re.compile(r"elitebabes\.com/model/([^/?#]+)", re.I),
            "javdatabase": re.compile(r"javdatabase\.com/idols/([^/?#]+)", re.I),
        }

    def normalize(self, url: str) -> Optional[NormalizedURL]:
        """
        Normalize a URL into a canonical (site, id) tuple.

        Args:
            url: The URL to normalize

        Returns:
            NormalizedURL if recognized, None if unrecognized
        """
        if not url:
            return None

        # Clean up URL
        url = url.strip()
        url = unquote(url)  # Decode %20 etc.

        # Try each pattern
        for site, pattern in self._patterns.items():
            match = pattern.search(url)
            if match:
                # Extract the ID (last group, or combine groups for wikipedia/thenude)
                if site == "wikipedia":
                    lang, title = match.groups()
                    normalized_id = f"{lang}:{title}".lower()
                elif site == "thenude":
                    name, num = match.groups()
                    normalized_id = f"{name}_{num}".lower()
                else:
                    normalized_id = match.group(1).lower()

                # Clean up the ID
                normalized_id = self._clean_id(normalized_id, site)

                # Determine confidence
                confidence = self._get_confidence(site)

                return NormalizedURL(
                    site=site,
                    normalized_id=normalized_id,
                    original_url=url,
                    confidence=confidence,
                )

        # Unrecognized URL
        return None

    def _clean_id(self, id_str: str, site: str) -> str:
        """Clean up extracted ID."""
        # Remove trailing slashes
        id_str = id_str.rstrip("/")

        # Site-specific cleaning
        if site in ("twitter", "instagram", "onlyfans", "fansly", "tiktok"):
            # Remove @ prefix if present
            id_str = id_str.lstrip("@")
            # Social handles are case-insensitive
            id_str = id_str.lower()

        if site in ("babepedia", "boobpedia"):
            # These use underscores for spaces
            # Keep as-is for matching
            pass

        if site in ("freeones", "pornhub", "indexxx"):
            # These use hyphens for spaces
            # Keep as-is for matching
            pass

        return id_str

    def _get_confidence(self, site: str) -> float:
        """Get confidence level for a site."""
        if site in HIGH_CONFIDENCE_SITES:
            return 1.0
        elif site in MEDIUM_CONFIDENCE_SITES:
            return 0.8
        else:
            return 0.6

    def normalize_batch(self, urls: list[str]) -> list[NormalizedURL]:
        """Normalize multiple URLs, filtering out unrecognized ones."""
        results = []
        for url in urls:
            normalized = self.normalize(url)
            if normalized:
                results.append(normalized)
        return results

    def group_by_site(self, urls: list[str]) -> dict[str, list[NormalizedURL]]:
        """Group normalized URLs by site."""
        by_site: dict[str, list[NormalizedURL]] = {}
        for url in urls:
            normalized = self.normalize(url)
            if normalized:
                if normalized.site not in by_site:
                    by_site[normalized.site] = []
                by_site[normalized.site].append(normalized)
        return by_site


# Convenience function
def normalize_url(url: str) -> Optional[NormalizedURL]:
    """Normalize a single URL using default normalizer."""
    return URLNormalizer().normalize(url)


# Self-test when run directly
if __name__ == "__main__":
    normalizer = URLNormalizer()

    test_urls = [
        # Stash-boxes
        "https://stashdb.org/performers/019bef93-b467-73eb-a04b-ac44fdaa7a04",
        "https://theporndb.net/performers/aaliyah-hadid",
        "https://pmvstash.org/performers/abc12345-1234-5678-9abc-def012345678",
        "https://javstash.org/performers/abc12345-1234-5678-9abc-def012345678",

        # Reference sites
        "https://www.iafd.com/person.rme/perfid=aaliyahhadid/Aaliyah-Hadid.htm",
        "https://www.imdb.com/name/nm1234567/",
        "https://www.wikidata.org/wiki/Q12345678",
        "https://en.wikipedia.org/wiki/Aaliyah_Hadid",

        # Social
        "https://twitter.com/AaliyahHadid",
        "https://x.com/aaliyahhadid",
        "https://www.instagram.com/aaliyahhadid/",
        "https://onlyfans.com/aaliyahhadid",

        # Tube/reference sites
        "https://www.pornhub.com/pornstar/aaliyah-hadid",
        "https://www.freeones.com/aaliyah-hadid",
        "https://www.babepedia.com/babe/Aaliyah_Hadid",
        "https://www.boobpedia.com/boobs/Aaliyah_Hadid",
        "https://www.indexxx.com/m/Aaliyah-Hadid/",

        # JAV
        "https://r18.dev/idols/hatano-yui",
        "https://www.dmm.co.jp/digital/videoa/-/list/=/article=actress/id=12345/",

        # Invalid/unrecognized
        "https://example.com/random/path",
        "not a url",
        "",
    ]

    print("URL Normalization Test Results")
    print("=" * 80)

    for url in test_urls:
        result = normalizer.normalize(url)
        if result:
            print(f"✓ {result.site:12} | {result.normalized_id:30} | conf={result.confidence}")
        else:
            print(f"✗ Unrecognized: {url[:60]}")

    print("\n" + "=" * 80)
    print("Batch grouping test:")
    grouped = normalizer.group_by_site(test_urls)
    for site, urls in sorted(grouped.items()):
        print(f"  {site}: {len(urls)} URLs")

"""Tests for IAFD scraper."""
import pytest
from unittest.mock import MagicMock, patch


class TestIAFDScraper:
    """Test IAFD scraper implements BaseScraper."""

    def test_is_base_scraper(self):
        """IAFDScraper inherits from BaseScraper."""
        from iafd_client import IAFDScraper
        from base_scraper import BaseScraper

        assert issubclass(IAFDScraper, BaseScraper)

    def test_has_correct_source_name(self):
        """IAFDScraper has correct source_name."""
        from iafd_client import IAFDScraper

        assert IAFDScraper.source_name == "iafd"
        assert IAFDScraper.source_type == "reference_site"

    def test_base_url(self):
        """IAFDScraper has correct base URL."""
        from iafd_client import IAFDScraper

        assert IAFDScraper.BASE_URL == "https://www.iafd.com"

    def test_query_performers_returns_empty(self):
        """query_performers returns empty (IAFD doesn't have pagination API)."""
        from iafd_client import IAFDScraper

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            scraper = IAFDScraper(flaresolverr_url="http://test:8191")

        total, performers = scraper.query_performers(page=1, per_page=25)
        assert total == 0
        assert performers == []

    def test_parse_performer_page(self):
        """_parse_performer_page extracts performer data correctly."""
        from iafd_client import IAFDScraper
        from base_scraper import ScrapedPerformer

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            scraper = IAFDScraper(flaresolverr_url="http://test:8191")

        # Use actual IAFD HTML structure (p tags, div for AKA)
        html = '''
        <html>
        <h1>Angela White</h1>
        <img src="https://www.iafd.com/graphics/headshots/angelawhite.jpg" />
        <p class="bioheading">AKA</p><div class="biodata">Angela<br/>Angie</div>
        <p class="bioheading">Nationality</p><p class="biodata">Australian</p>
        <p class="bioheading">Birthday</p><p class="biodata"><a href="/calendar.asp">March 4, 1985</a></p>
        <p class="bioheading">Ethnicity</p><p class="biodata">Caucasian</p>
        </html>
        '''

        url = "https://www.iafd.com/person.rme/perfid=angelawhite/Angela-White.htm"
        result = scraper._parse_performer_page(url, html)

        assert isinstance(result, ScrapedPerformer)
        assert result.id == "angelawhite"
        assert result.name == "Angela White"
        assert len(result.image_urls) > 0
        assert "Angela" in result.aliases
        assert "Angie" in result.aliases
        assert result.country == "Australian"
        assert result.birth_date == "1985"
        assert result.ethnicity == "Caucasian"
        assert result.stash_ids.get("iafd") == "angelawhite"

    def test_lookup_by_url(self):
        """lookup_by_url fetches and parses performer page."""
        from iafd_client import IAFDScraper
        from base_scraper import ScrapedPerformer

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_instance = mock_flare.return_value
            mock_instance.is_available.return_value = True
            mock_response = MagicMock()
            mock_response.solution_status = 200
            mock_response.solution_html = '''
            <html>
            <h1>Test Performer</h1>
            <img src="/graphics/headshots/test.jpg" class="headshot" />
            </html>
            '''
            mock_response.cookies = []
            mock_response.user_agent = "Test"
            mock_instance.get.return_value = mock_response

            scraper = IAFDScraper(flaresolverr_url="http://test:8191")
            result = scraper.lookup_by_url(
                "https://www.iafd.com/person.rme/perfid=testperformer/Test-Performer.htm"
            )

            assert result is not None
            assert isinstance(result, ScrapedPerformer)
            assert result.id == "testperformer"


class TestIAFDSlugMethods:
    """Tests for IAFD slug extraction and conversion."""

    def test_source_type_is_reference_site(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        assert scraper.source_type == "reference_site"

    def test_gender_filter_is_none(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        assert scraper.gender_filter is None

    def test_extract_slug_from_url(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        # Standard URL with perfid
        url = "https://www.iafd.com/person.rme/perfid=miamalkova/Mia-Malkova.htm"
        assert scraper.extract_slug_from_url(url) == "miamalkova"

        # Different format
        url2 = "https://iafd.com/person.rme/perfid=angelawhite/Angela-White.htm"
        assert scraper.extract_slug_from_url(url2) == "angelawhite"

        # Non-matching URL
        assert scraper.extract_slug_from_url("https://babepedia.com/babe/test") is None

    def test_name_to_slug(self):
        from unittest.mock import patch

        with patch('iafd_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from iafd_client import IAFDScraper
            scraper = IAFDScraper()

        assert scraper.name_to_slug("Mia Malkova") == "miamalkova"
        assert scraper.name_to_slug("Angela White") == "angelawhite"
        assert scraper.name_to_slug("Mary-Jane") == "maryjane"

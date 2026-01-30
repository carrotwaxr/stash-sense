"""Tests for Babepedia scraper."""
import pytest
from unittest.mock import MagicMock, patch


class TestBabepediaScraper:
    """Test Babepedia scraper."""

    def test_is_base_scraper(self):
        """BabepediaScraper inherits from BaseScraper."""
        from babepedia_client import BabepediaScraper
        from base_scraper import BaseScraper

        assert issubclass(BabepediaScraper, BaseScraper)

    def test_has_source_name(self):
        """BabepediaScraper has correct source_name."""
        from babepedia_client import BabepediaScraper

        assert BabepediaScraper.source_name == "babepedia"
        assert BabepediaScraper.source_type == "reference_site"

    def test_parse_performer_page(self):
        """Can parse a performer page."""
        from babepedia_client import BabepediaScraper

        # Mock FlareSolverr
        scraper = BabepediaScraper.__new__(BabepediaScraper)
        scraper.rate_limit_delay = 0
        scraper._last_request_time = 0

        # Sample HTML with expected structure
        html = '''
        <title>Test Performer - Babepedia</title>
        <img src="/pics/Test%20Performer_thumb3.jpg">
        <img src="/pics/Test%20Performer2_thumb3.jpg">
        <span class="label">Birthplace:</span> <span class="value">Los Angeles, California, USA</span>
        <a href="/born-in-the-year/1995">1995</a>
        '''

        performer = scraper._parse_performer_page("Test_Performer", html)

        assert performer.id == "Test_Performer"
        assert performer.name == "Test Performer"
        assert performer.gender == "FEMALE"
        assert len(performer.image_urls) >= 2
        assert performer.country == "USA"
        assert performer.birth_date == "1995"

    def test_full_size_image_urls(self):
        """Image URLs are full-size (no _thumb suffix)."""
        from babepedia_client import BabepediaScraper

        scraper = BabepediaScraper.__new__(BabepediaScraper)
        scraper.rate_limit_delay = 0
        scraper._last_request_time = 0

        html = '''
        <title>Test - Babepedia</title>
        <img src="/pics/Test_thumb3.jpg">
        <img src="/pics/Test2_thumb1.jpg">
        '''

        performer = scraper._parse_performer_page("Test", html)

        # All URLs should be full-size (no _thumb)
        for url in performer.image_urls:
            assert "_thumb" not in url

    def test_lookup_by_name(self):
        """lookup_by_name converts name to slug."""
        from babepedia_client import BabepediaScraper

        with patch.object(BabepediaScraper, 'get_performer') as mock_get:
            mock_get.return_value = None

            scraper = BabepediaScraper.__new__(BabepediaScraper)
            scraper.lookup_by_name("Mia Malkova")

            # Should convert spaces to underscores
            mock_get.assert_called_once_with("Mia_Malkova")

    def test_stash_ids_includes_babepedia(self):
        """stash_ids includes babepedia slug."""
        from babepedia_client import BabepediaScraper

        scraper = BabepediaScraper.__new__(BabepediaScraper)
        scraper.rate_limit_delay = 0
        scraper._last_request_time = 0

        html = '<title>Test - Babepedia</title>'
        performer = scraper._parse_performer_page("Test_Performer", html)

        assert performer.stash_ids.get("babepedia") == "Test_Performer"


class TestBabepediaSlugMethods:
    """Tests for Babepedia slug extraction and conversion."""

    def test_source_type_is_reference_site(self):
        from unittest.mock import patch, MagicMock

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        assert scraper.source_type == "reference_site"

    def test_gender_filter_is_female(self):
        from unittest.mock import patch

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        assert scraper.gender_filter == "FEMALE"

    def test_extract_slug_from_url(self):
        from unittest.mock import patch

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        # Standard URL
        assert scraper.extract_slug_from_url("https://www.babepedia.com/babe/Mia_Malkova") == "Mia_Malkova"

        # Without www
        assert scraper.extract_slug_from_url("https://babepedia.com/babe/Angela_White") == "Angela_White"

        # With query string
        assert scraper.extract_slug_from_url("https://www.babepedia.com/babe/Test_Name?ref=123") == "Test_Name"

        # Non-matching URL
        assert scraper.extract_slug_from_url("https://iafd.com/person/test") is None

    def test_name_to_slug(self):
        from unittest.mock import patch

        with patch('babepedia_client.FlareSolverr') as mock_flare:
            mock_flare.return_value.is_available.return_value = True
            from babepedia_client import BabepediaScraper
            scraper = BabepediaScraper()

        assert scraper.name_to_slug("Mia Malkova") == "Mia_Malkova"
        assert scraper.name_to_slug("Angela White") == "Angela_White"
        assert scraper.name_to_slug("Single") == "Single"

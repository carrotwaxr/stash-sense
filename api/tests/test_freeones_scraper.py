"""Tests for FreeOnes scraper."""
import pytest
from unittest.mock import MagicMock, patch


class TestFreeOnesScraper:
    """Test FreeOnes scraper implements BaseScraper."""

    def test_is_base_scraper(self):
        """FreeOnesScraper inherits from BaseScraper."""
        from freeones_client import FreeOnesScraper
        from base_scraper import BaseScraper

        assert issubclass(FreeOnesScraper, BaseScraper)

    def test_has_correct_source_name(self):
        """FreeOnesScraper has correct source_name."""
        from freeones_client import FreeOnesScraper

        assert FreeOnesScraper.source_name == "freeones"
        assert FreeOnesScraper.source_type == "reference_site"

    def test_base_url(self):
        """FreeOnesScraper has correct base URL."""
        from freeones_client import FreeOnesScraper

        assert FreeOnesScraper.BASE_URL == "https://www.freeones.com"

    def test_cdn_domains(self):
        """FreeOnesScraper knows CDN domains."""
        from freeones_client import FreeOnesScraper

        assert "thumbs.freeones.com" in FreeOnesScraper.CDN_DOMAINS
        assert "ch-thumbs.freeones.com" in FreeOnesScraper.CDN_DOMAINS
        assert "img.freeones.com" in FreeOnesScraper.CDN_DOMAINS

    def test_query_performers_returns_empty(self):
        """query_performers returns empty (FreeOnes uses name lookup)."""
        from freeones_client import FreeOnesScraper

        scraper = FreeOnesScraper()
        total, performers = scraper.query_performers(page=1, per_page=25)
        assert total == 0
        assert performers == []

    def test_extract_images(self):
        """_extract_images extracts CDN URLs correctly."""
        from freeones_client import FreeOnesScraper

        scraper = FreeOnesScraper()
        html = '''
        <html>
        <img src="https://thumbs.freeones.com/image1.jpg" />
        <img src="https://ch-thumbs.freeones.com/image2.webp" />
        <img src="https://img.freeones.com/full/image3.png" />
        <img src="https://other.site.com/image4.jpg" />
        </html>
        '''
        image_urls = []
        scraper._extract_images(html, image_urls)

        assert len(image_urls) == 3
        assert any("thumbs.freeones.com" in url for url in image_urls)
        assert any("ch-thumbs.freeones.com" in url for url in image_urls)
        assert any("img.freeones.com" in url for url in image_urls)
        assert not any("other.site.com" in url for url in image_urls)

    def test_extract_images_skips_tiny(self):
        """_extract_images skips tiny thumbnails."""
        from freeones_client import FreeOnesScraper

        scraper = FreeOnesScraper()
        html = '''
        <img src="https://thumbs.freeones.com/tiny/small.jpg" />
        <img src="https://thumbs.freeones.com/small/tiny.jpg" />
        <img src="https://thumbs.freeones.com/full/large.jpg" />
        '''
        image_urls = []
        scraper._extract_images(html, image_urls)

        assert len(image_urls) == 1
        assert "large.jpg" in image_urls[0]

    def test_parse_performer(self):
        """_parse_performer extracts performer data correctly."""
        from freeones_client import FreeOnesScraper
        from base_scraper import ScrapedPerformer

        scraper = FreeOnesScraper()
        html = '''
        <html>
        <h1>Angela White</h1>
        <img src="https://thumbs.freeones.com/image1.jpg" />
        <a href="/angela-white/photos/gallery1">Gallery 1</a>
        </html>
        '''

        result = scraper._parse_performer("angela-white", html)

        assert isinstance(result, ScrapedPerformer)
        assert result.id == "angela-white"
        assert result.name == "Angela White"
        assert result.gender == "FEMALE"
        assert result.stash_ids.get("freeones") == "angela-white"
        assert "FreeOnes" in result.external_urls

    def test_lookup_by_name_creates_slug(self):
        """lookup_by_name converts name to proper slug."""
        from freeones_client import FreeOnesScraper

        scraper = FreeOnesScraper()

        with patch.object(scraper, 'get_performer') as mock_get:
            mock_get.return_value = None
            with patch.object(scraper, '_search_performer') as mock_search:
                mock_search.return_value = None
                scraper.lookup_by_name("Angela White")

        # Should have tried angela-white as slug
        mock_get.assert_called_once_with("angela-white")

    def test_lookup_by_name_handles_special_chars(self):
        """lookup_by_name removes special characters from slug."""
        from freeones_client import FreeOnesScraper

        scraper = FreeOnesScraper()

        with patch.object(scraper, 'get_performer') as mock_get:
            mock_get.return_value = None
            with patch.object(scraper, '_search_performer') as mock_search:
                mock_search.return_value = None
                scraper.lookup_by_name("Name O'Brien (Test)")

        # Should have cleaned the slug
        call_args = mock_get.call_args[0][0]
        assert "'" not in call_args
        assert "(" not in call_args
        assert ")" not in call_args

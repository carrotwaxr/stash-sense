"""Tests for base scraper interface."""
import pytest
from unittest.mock import MagicMock, patch
import time


class TestBaseScraper:
    """Test base scraper behavior."""

    def test_rate_limiting_enforced(self):
        """Rate limiting sleeps between requests."""
        from base_scraper import BaseScraper

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def get_performer(self, performer_id):
                self._rate_limit()
                return None

            def query_performers(self, page=1, per_page=25):
                return (0, [])

            def download_image(self, url, max_retries=3):
                return None

        scraper = TestScraper(rate_limit_delay=0.1)

        start = time.time()
        scraper.get_performer("1")
        scraper.get_performer("2")
        elapsed = time.time() - start

        # Should have delayed at least 0.1s between calls
        assert elapsed >= 0.1

    def test_iter_all_performers_paginates(self):
        """iter_all_performers uses pagination correctly."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)
                self.pages_fetched = []

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                self.pages_fetched.append(page)
                if page == 1:
                    return (50, [
                        ScrapedPerformer(id=f"p{i}", name=f"Performer {i}")
                        for i in range(25)
                    ])
                elif page == 2:
                    return (50, [
                        ScrapedPerformer(id=f"p{i}", name=f"Performer {i}")
                        for i in range(25, 50)
                    ])
                else:
                    return (50, [])

            def download_image(self, url, max_retries=3):
                return None

        scraper = TestScraper()
        performers = list(scraper.iter_all_performers(per_page=25))

        assert len(performers) == 50
        assert scraper.pages_fetched == [1, 2]

    def test_iter_respects_max_performers(self):
        """iter_all_performers stops at max_performers."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                return (100, [
                    ScrapedPerformer(id=f"p{i}", name=f"Performer {i}")
                    for i in range((page-1)*per_page, page*per_page)
                ])

            def download_image(self, url, max_retries=3):
                return None

        scraper = TestScraper()
        performers = list(scraper.iter_all_performers(per_page=25, max_performers=30))

        assert len(performers) == 30

    def test_scraped_performer_dataclass(self):
        """ScrapedPerformer has expected fields."""
        from base_scraper import ScrapedPerformer

        p = ScrapedPerformer(
            id="abc-123",
            name="Test Performer",
            aliases=["Alias 1"],
            image_urls=["https://example.com/img.jpg"],
            gender="FEMALE",
            country="US",
        )

        assert p.id == "abc-123"
        assert p.name == "Test Performer"
        assert p.aliases == ["Alias 1"]
        assert len(p.image_urls) == 1

    def test_iter_performers_after_skips_until_last_id(self):
        """iter_performers_after skips performers until last_id is found."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)
                self.all_performers = [
                    ScrapedPerformer(id=f"p{i}", name=f"Performer {i}")
                    for i in range(10)
                ]

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                return (len(self.all_performers), self.all_performers)

            def download_image(self, url, max_retries=3):
                return None

        scraper = TestScraper()
        # Resume after p5
        performers = list(scraper.iter_performers_after(last_id="p5"))

        # Should only get p6, p7, p8, p9
        assert len(performers) == 4
        assert performers[0].id == "p6"
        assert performers[-1].id == "p9"

    def test_iter_performers_after_with_none_starts_from_beginning(self):
        """iter_performers_after with None starts from beginning."""
        from base_scraper import BaseScraper, ScrapedPerformer

        class TestScraper(BaseScraper):
            source_name = "test"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                return (3, [
                    ScrapedPerformer(id="p1", name="P1"),
                    ScrapedPerformer(id="p2", name="P2"),
                    ScrapedPerformer(id="p3", name="P3"),
                ])

            def download_image(self, url, max_retries=3):
                return None

        scraper = TestScraper()
        performers = list(scraper.iter_performers_after(last_id=None))

        assert len(performers) == 3
        assert performers[0].id == "p1"


class TestReferenceScraperMethods:
    """Tests for reference site scraper methods."""

    def test_base_scraper_has_source_type(self):
        from base_scraper import BaseScraper

        assert hasattr(BaseScraper, 'source_type')
        assert BaseScraper.source_type == "stash_box"

    def test_base_scraper_has_gender_filter(self):
        from base_scraper import BaseScraper

        assert hasattr(BaseScraper, 'gender_filter')
        assert BaseScraper.gender_filter is None

    def test_extract_slug_from_url_raises_not_implemented(self):
        from base_scraper import BaseScraper, ScrapedPerformer

        class MinimalScraper(BaseScraper):
            source_name = "test"

            def query_performers(self, page, per_page):
                return (0, [])

            def get_performer(self, performer_id):
                return None

            def download_image(self, url, max_retries=3):
                return None

        scraper = MinimalScraper()

        import pytest
        with pytest.raises(NotImplementedError):
            scraper.extract_slug_from_url("http://example.com")

    def test_name_to_slug_raises_not_implemented(self):
        from base_scraper import BaseScraper

        class MinimalScraper(BaseScraper):
            source_name = "test"

            def query_performers(self, page, per_page):
                return (0, [])

            def get_performer(self, performer_id):
                return None

            def download_image(self, url, max_retries=3):
                return None

        scraper = MinimalScraper()

        import pytest
        with pytest.raises(NotImplementedError):
            scraper.name_to_slug("Test Name")

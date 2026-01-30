"""Tests for stash-box client wrappers (PMVStash, JAVStash, FansDB)."""
import pytest
from unittest.mock import patch


class TestPMVStashClient:
    """Test PMVStash client wrapper."""

    def test_is_base_scraper(self):
        """PMVStashClient inherits from BaseScraper."""
        from stashbox_clients import PMVStashClient
        from base_scraper import BaseScraper

        assert issubclass(PMVStashClient, BaseScraper)

    def test_has_correct_source_name(self):
        """PMVStashClient has correct source_name."""
        from stashbox_clients import PMVStashClient

        assert PMVStashClient.source_name == "pmvstash"
        assert PMVStashClient.source_type == "stash_box"

    def test_inherits_from_stashdb_client(self):
        """PMVStashClient inherits from StashDBClient."""
        from stashbox_clients import PMVStashClient
        from stashdb_client import StashDBClient

        assert issubclass(PMVStashClient, StashDBClient)

    def test_default_url(self):
        """PMVStashClient uses correct default URL."""
        from stashbox_clients import PMVStashClient

        assert PMVStashClient.DEFAULT_URL == "https://pmvstash.org/graphql"

    def test_query_performers_returns_tuple(self):
        """query_performers returns (total, list) tuple."""
        from stashbox_clients import PMVStashClient
        from base_scraper import ScrapedPerformer

        with patch.object(PMVStashClient, '_query') as mock_query:
            mock_query.return_value = {
                'queryPerformers': {
                    'count': 50,
                    'performers': [
                        {'id': 'p1', 'name': 'Test 1', 'images': [], 'aliases': [], 'urls': []},
                    ]
                }
            }

            client = PMVStashClient(api_key="test-key")
            total, performers = client.query_performers(page=1, per_page=25)

            assert total == 50
            assert len(performers) == 1
            assert all(isinstance(p, ScrapedPerformer) for p in performers)


class TestJAVStashClient:
    """Test JAVStash client wrapper."""

    def test_is_base_scraper(self):
        """JAVStashClient inherits from BaseScraper."""
        from stashbox_clients import JAVStashClient
        from base_scraper import BaseScraper

        assert issubclass(JAVStashClient, BaseScraper)

    def test_has_correct_source_name(self):
        """JAVStashClient has correct source_name."""
        from stashbox_clients import JAVStashClient

        assert JAVStashClient.source_name == "javstash"
        assert JAVStashClient.source_type == "stash_box"

    def test_inherits_from_stashdb_client(self):
        """JAVStashClient inherits from StashDBClient."""
        from stashbox_clients import JAVStashClient
        from stashdb_client import StashDBClient

        assert issubclass(JAVStashClient, StashDBClient)

    def test_default_url(self):
        """JAVStashClient uses correct default URL."""
        from stashbox_clients import JAVStashClient

        assert JAVStashClient.DEFAULT_URL == "https://javstash.org/graphql"


class TestFansDBClient:
    """Test FansDB client wrapper."""

    def test_is_base_scraper(self):
        """FansDBClient inherits from BaseScraper."""
        from stashbox_clients import FansDBClient
        from base_scraper import BaseScraper

        assert issubclass(FansDBClient, BaseScraper)

    def test_has_correct_source_name(self):
        """FansDBClient has correct source_name."""
        from stashbox_clients import FansDBClient

        assert FansDBClient.source_name == "fansdb"
        assert FansDBClient.source_type == "stash_box"

    def test_inherits_from_stashdb_client(self):
        """FansDBClient inherits from StashDBClient."""
        from stashbox_clients import FansDBClient
        from stashdb_client import StashDBClient

        assert issubclass(FansDBClient, StashDBClient)

    def test_default_url(self):
        """FansDBClient uses correct default URL."""
        from stashbox_clients import FansDBClient

        assert FansDBClient.DEFAULT_URL == "https://fansdb.cc/graphql"

"""Tests for StashDB client adapter."""
import pytest
from unittest.mock import MagicMock, patch


class TestStashDBAdapter:
    """Test StashDB client implements BaseScraper."""

    def test_is_base_scraper(self):
        """StashDBClient inherits from BaseScraper."""
        from stashdb_client import StashDBClient
        from base_scraper import BaseScraper

        assert issubclass(StashDBClient, BaseScraper)

    def test_has_source_name(self):
        """StashDBClient has correct source_name."""
        from stashdb_client import StashDBClient

        assert StashDBClient.source_name == "stashdb"
        assert StashDBClient.source_type == "stash_box"

    def test_query_performers_returns_tuple(self):
        """query_performers returns (total, list) tuple."""
        from stashdb_client import StashDBClient
        from base_scraper import ScrapedPerformer

        # Mock the GraphQL query
        with patch.object(StashDBClient, '_query') as mock_query:
            mock_query.return_value = {
                'queryPerformers': {
                    'count': 100,
                    'performers': [
                        {'id': 'p1', 'name': 'Test 1', 'images': [], 'aliases': [], 'urls': []},
                        {'id': 'p2', 'name': 'Test 2', 'images': [], 'aliases': [], 'urls': []},
                    ]
                }
            }

            client = StashDBClient(
                url="https://stashdb.org/graphql",
                api_key="test-key",
            )
            total, performers = client.query_performers(page=1, per_page=25)

            assert total == 100
            assert len(performers) == 2
            assert all(isinstance(p, ScrapedPerformer) for p in performers)

    def test_get_performer_returns_scraped_performer(self):
        """get_performer returns ScrapedPerformer."""
        from stashdb_client import StashDBClient
        from base_scraper import ScrapedPerformer

        with patch.object(StashDBClient, '_query') as mock_query:
            mock_query.return_value = {
                'findPerformer': {
                    'id': 'abc-123',
                    'name': 'Test Performer',
                    'images': [{'url': 'https://example.com/img.jpg'}],
                    'gender': 'FEMALE',
                    'country': 'US',
                    'aliases': ['Alias 1'],
                    'urls': [{'url': 'https://example.com', 'site': {'name': 'Example'}}],
                }
            }

            client = StashDBClient(
                url="https://stashdb.org/graphql",
                api_key="test-key",
            )
            performer = client.get_performer("abc-123")

            assert isinstance(performer, ScrapedPerformer)
            assert performer.id == "abc-123"
            assert performer.name == "Test Performer"
            assert performer.gender == "FEMALE"
            assert performer.country == "US"
            assert performer.aliases == ["Alias 1"]
            assert "https://example.com/img.jpg" in performer.image_urls

    def test_external_urls_parsed_correctly(self):
        """External URLs are grouped by site."""
        from stashdb_client import StashDBClient
        from base_scraper import ScrapedPerformer

        with patch.object(StashDBClient, '_query') as mock_query:
            mock_query.return_value = {
                'findPerformer': {
                    'id': 'abc-123',
                    'name': 'Test',
                    'images': [],
                    'aliases': [],
                    'urls': [
                        {'url': 'https://twitter.com/test', 'site': {'name': 'Twitter'}},
                        {'url': 'https://instagram.com/test', 'site': {'name': 'Instagram'}},
                        {'url': 'https://twitter.com/test2', 'site': {'name': 'Twitter'}},
                    ],
                }
            }

            client = StashDBClient(
                url="https://stashdb.org/graphql",
                api_key="test-key",
            )
            performer = client.get_performer("abc-123")

            assert "Twitter" in performer.external_urls
            assert "Instagram" in performer.external_urls
            assert len(performer.external_urls["Twitter"]) == 2
            assert len(performer.external_urls["Instagram"]) == 1

    def test_stash_ids_includes_self(self):
        """stash_ids includes the stashdb ID."""
        from stashdb_client import StashDBClient

        with patch.object(StashDBClient, '_query') as mock_query:
            mock_query.return_value = {
                'findPerformer': {
                    'id': 'abc-123',
                    'name': 'Test',
                    'images': [],
                    'aliases': [],
                    'urls': [],
                }
            }

            client = StashDBClient(
                url="https://stashdb.org/graphql",
                api_key="test-key",
            )
            performer = client.get_performer("abc-123")

            assert performer.stash_ids.get("stashdb") == "abc-123"

"""Tests for ThePornDB client adapter."""
import pytest
from unittest.mock import MagicMock, patch


class TestThePornDBAdapter:
    """Test ThePornDB client implements BaseScraper."""

    def test_is_base_scraper(self):
        """ThePornDBClient inherits from BaseScraper."""
        from theporndb_client import ThePornDBClient
        from base_scraper import BaseScraper

        assert issubclass(ThePornDBClient, BaseScraper)

    def test_has_source_name(self):
        """ThePornDBClient has correct source_name."""
        from theporndb_client import ThePornDBClient

        assert ThePornDBClient.source_name == "theporndb"
        assert ThePornDBClient.source_type == "stash_box"

    def test_query_performers_returns_tuple(self):
        """query_performers returns (total, list) tuple."""
        from theporndb_client import ThePornDBClient
        from base_scraper import ScrapedPerformer

        with patch.object(ThePornDBClient, '_request') as mock_request:
            mock_request.return_value = {
                'meta': {'total': 100},
                'data': [
                    {'id': 1, 'name': 'Test 1', 'image': 'url1', 'extras': {}},
                    {'id': 2, 'name': 'Test 2', 'image': 'url2', 'extras': {}},
                ]
            }

            client = ThePornDBClient(api_key="test-key")
            total, performers = client.query_performers(page=1, per_page=25)

            assert total == 100
            assert len(performers) == 2
            assert all(isinstance(p, ScrapedPerformer) for p in performers)

    def test_get_performer_returns_scraped_performer(self):
        """get_performer returns ScrapedPerformer."""
        from theporndb_client import ThePornDBClient
        from base_scraper import ScrapedPerformer

        with patch.object(ThePornDBClient, '_request') as mock_request:
            mock_request.return_value = {
                'data': {
                    'id': 123,
                    'name': 'Test Performer',
                    'image': 'https://example.com/img.jpg',
                    'face': 'https://example.com/face.jpg',
                    'extras': {
                        'gender': 'female',
                        'nationality': 'US',
                        'birthday': '1990-01-01',
                    },
                    'aliases': ['Alias 1', 'Alias 2'],
                }
            }

            client = ThePornDBClient(api_key="test-key")
            performer = client.get_performer("123")

            assert isinstance(performer, ScrapedPerformer)
            assert performer.id == "123"
            assert performer.name == "Test Performer"
            assert performer.gender == "FEMALE"
            assert performer.country == "US"
            assert performer.aliases == ["Alias 1", "Alias 2"]
            # Face URL should be first in image_urls
            assert performer.image_urls[0] == "https://example.com/face.jpg"

    def test_stashdb_crossref_extracted(self):
        """StashDB ID is extracted from links."""
        from theporndb_client import ThePornDBClient

        with patch.object(ThePornDBClient, '_request') as mock_request:
            mock_request.return_value = {
                'data': {
                    'id': 123,
                    'name': 'Test',
                    'extras': {
                        'links': {
                            'StashDB': 'https://stashdb.org/performers/abc-123-def',
                            'Twitter': 'https://twitter.com/test',
                        }
                    },
                }
            }

            client = ThePornDBClient(api_key="test-key")
            performer = client.get_performer("123")

            assert performer.stash_ids.get("stashdb") == "abc-123-def"
            assert performer.stash_ids.get("theporndb") == "123"
            assert "StashDB" in performer.external_urls
            assert "Twitter" in performer.external_urls

    def test_handles_missing_extras(self):
        """Handles performers with missing extras gracefully."""
        from theporndb_client import ThePornDBClient

        with patch.object(ThePornDBClient, '_request') as mock_request:
            mock_request.return_value = {
                'data': {
                    'id': 123,
                    'name': 'Test',
                    # No extras field
                }
            }

            client = ThePornDBClient(api_key="test-key")
            performer = client.get_performer("123")

            assert performer.id == "123"
            assert performer.name == "Test"
            assert performer.country is None
            assert performer.gender is None

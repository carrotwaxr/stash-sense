# api/tests/test_upstream_sync_integration.py
"""Integration tests for the full upstream sync flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from recommendations_db import RecommendationsDB


class TestUpstreamSyncIntegration:

    @pytest.fixture
    def rec_db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key", "name": "stashdb"},
        ])
        return stash

    @pytest.mark.asyncio
    async def test_full_flow_first_scan_then_rescan(self, mock_stash, rec_db):
        """Test: first scan creates recs, rescan updates existing."""
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        local_performer = {
            "id": "42", "name": "Jane Doe", "disambiguation": "",
            "alias_list": ["JD"], "gender": "FEMALE", "birthdate": "1990-01-15",
            "death_date": None, "ethnicity": "Caucasian", "country": "US",
            "eye_color": "Brown", "hair_color": "Brown", "height_cm": 165,
            "measurements": "", "fake_tits": "", "career_length": "",
            "tattoos": "", "piercings": "", "details": "", "urls": [],
            "favorite": False, "image_path": None,
            "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
        }
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[local_performer])

        def make_upstream(height, updated):
            return {
                "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
                "aliases": ["JD"], "gender": "FEMALE", "birth_date": "1990-01-15",
                "death_date": None, "ethnicity": "CAUCASIAN", "country": "US",
                "eye_color": "BROWN", "hair_color": "BROWN",
                "height": height,
                "cup_size": None, "band_size": None, "waist_size": None,
                "hip_size": None, "breast_type": None,
                "career_start_year": None, "career_end_year": None,
                "tattoos": [], "piercings": [], "urls": [],
                "is_favorite": False, "deleted": False, "merged_into_id": None,
                "created": "2024-01-01T00:00:00Z", "updated": updated,
            }

        # First scan: height changed from 165 -> 168
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(
                return_value=make_upstream(168, "2026-01-15T10:00:00Z")
            )
            MockSBC.return_value = mock_sbc
            result1 = await analyzer.run(incremental=False)

        assert result1.recommendations_created == 1

        # Second scan: height changed again 168 -> 170
        analyzer2 = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(
                return_value=make_upstream(170, "2026-01-16T10:00:00Z")
            )
            MockSBC.return_value = mock_sbc
            result2 = await analyzer2.run(incremental=False)

        # Should update existing, not create new
        assert result2.recommendations_created == 0
        recs = rec_db.get_recommendations(type="upstream_performer_changes")
        assert len(recs) == 1
        assert any(c["upstream_value"] == 170 for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_soft_dismiss_then_new_changes_creates_new_rec(self, mock_stash, rec_db):
        """Test: soft dismiss, then new upstream changes create new recommendation."""
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        local_performer = {
            "id": "42", "name": "Jane Doe", "disambiguation": "",
            "alias_list": [], "gender": None, "birthdate": None,
            "death_date": None, "ethnicity": None, "country": None,
            "eye_color": None, "hair_color": None, "height_cm": 165,
            "measurements": "", "fake_tits": "", "career_length": "",
            "tattoos": "", "piercings": "", "details": "", "urls": [],
            "favorite": False, "image_path": None,
            "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
        }
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[local_performer])

        def make_upstream(height, updated):
            return {
                "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
                "aliases": [], "gender": None, "birth_date": None, "death_date": None,
                "ethnicity": None, "country": None, "eye_color": None, "hair_color": None,
                "height": height, "cup_size": None, "band_size": None, "waist_size": None,
                "hip_size": None, "breast_type": None, "career_start_year": None,
                "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
                "is_favorite": False, "deleted": False, "merged_into_id": None,
                "created": "2024-01-01T00:00:00Z", "updated": updated,
            }

        # First scan creates recommendation
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(
                return_value=make_upstream(168, "2026-01-15T10:00:00Z")
            )
            MockSBC.return_value = mock_sbc
            await analyzer.run(incremental=False)

        recs = rec_db.get_recommendations(type="upstream_performer_changes", status="pending")
        assert len(recs) == 1

        # Soft dismiss
        rec_db.dismiss_recommendation(recs[0].id, permanent=False)

        # New scan with newer upstream changes
        analyzer2 = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(
                return_value=make_upstream(170, "2026-01-16T10:00:00Z")
            )
            MockSBC.return_value = mock_sbc
            result = await analyzer2.run(incremental=False)

        assert result.recommendations_created == 1

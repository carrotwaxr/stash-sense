"""Tests for UpstreamPerformerAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestUpstreamPerformerAnalyzer:
    """Tests for the upstream performer changes analyzer."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "test-key", "name": "stashdb"},
        ])
        stash.get_performers_for_endpoint = AsyncMock(return_value=[])
        return stash

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def test_analyzer_type(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        assert analyzer.type == "upstream_performer_changes"

    @pytest.mark.asyncio
    async def test_no_performers_no_recommendations(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[])
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(return_value=([], 0))
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_changed_performer(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42", "name": "Jane Doe", "disambiguation": "",
                "alias_list": ["JD"], "gender": "FEMALE", "birthdate": "1990-01-15",
                "death_date": None, "ethnicity": "Caucasian", "country": "US",
                "eye_color": "Brown", "hair_color": "Brown", "height_cm": 165,
                "measurements": "", "fake_tits": "", "career_length": "",
                "tattoos": "", "piercings": "", "details": "", "urls": [],
                "favorite": False, "image_path": "/performer/42/image",
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
            }
        ])
        upstream_performer = {
            "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
            "aliases": ["JD"], "gender": "FEMALE", "birth_date": "1990-01-15",
            "death_date": None, "ethnicity": "CAUCASIAN", "country": "US",
            "eye_color": "BROWN", "hair_color": "BROWN", "height": 168,
            "cup_size": None, "band_size": None, "waist_size": None,
            "hip_size": None, "breast_type": None, "career_start_year": None,
            "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
            "is_favorite": False, "deleted": False, "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(return_value=upstream_performer)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 1
        recs = rec_db.get_recommendations(type="upstream_performer_changes")
        assert len(recs) == 1
        assert recs[0].details["performer_id"] == "42"
        assert any(c["field"] == "height" for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_updates_existing_pending_recommendation(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer
        rec_id = rec_db.create_recommendation(
            type="upstream_performer_changes", target_type="performer",
            target_id="42", details={"changes": [{"field": "height", "upstream_value": 167}]},
            confidence=1.0,
        )
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42", "name": "Jane Doe", "disambiguation": "",
                "alias_list": [], "gender": None, "birthdate": None,
                "death_date": None, "ethnicity": None, "country": None,
                "eye_color": None, "hair_color": None, "height_cm": 165,
                "measurements": "", "fake_tits": "", "career_length": "",
                "tattoos": "", "piercings": "", "details": "", "urls": [],
                "favorite": False, "image_path": None,
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
            }
        ])
        upstream_performer = {
            "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
            "aliases": [], "gender": None, "birth_date": None, "death_date": None,
            "ethnicity": None, "country": None, "eye_color": None, "hair_color": None,
            "height": 170, "cup_size": None, "band_size": None, "waist_size": None,
            "hip_size": None, "breast_type": None, "career_start_year": None,
            "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
            "is_favorite": False, "deleted": False, "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z", "updated": "2026-01-16T10:00:00Z",
        }
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(return_value=upstream_performer)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0  # updated, not created
        recs = rec_db.get_recommendations(type="upstream_performer_changes")
        assert len(recs) == 1
        assert any(c["upstream_value"] == 170 for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_skips_permanently_dismissed(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer
        with rec_db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id, permanent) VALUES (?, ?, ?, ?)",
                ("upstream_performer_changes", "performer", "42", 1),
            )
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42", "name": "Jane Doe", "disambiguation": "",
                "alias_list": [], "gender": None, "birthdate": None,
                "death_date": None, "ethnicity": None, "country": None,
                "eye_color": None, "hair_color": None, "height_cm": 165,
                "measurements": "", "fake_tits": "", "career_length": "",
                "tattoos": "", "piercings": "", "details": "", "urls": [],
                "favorite": False, "image_path": None,
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
            }
        ])
        upstream = {
            "id": "abc-123", "name": "Jane Smith", "disambiguation": "",
            "aliases": [], "gender": None, "birth_date": None, "death_date": None,
            "ethnicity": None, "country": None, "eye_color": None, "hair_color": None,
            "height": 168, "cup_size": None, "band_size": None, "waist_size": None,
            "hip_size": None, "breast_type": None, "career_start_year": None,
            "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
            "is_favorite": False, "deleted": False, "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_skips_deleted_upstream_performers(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42", "name": "Jane Doe", "disambiguation": "",
                "alias_list": [], "gender": None, "birthdate": None,
                "death_date": None, "ethnicity": None, "country": None,
                "eye_color": None, "hair_color": None, "height_cm": 165,
                "measurements": "", "fake_tits": "", "career_length": "",
                "tattoos": "", "piercings": "", "details": "", "urls": [],
                "favorite": False, "image_path": None,
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
            }
        ])
        upstream = {
            "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
            "aliases": [], "gender": None, "birth_date": None, "death_date": None,
            "ethnicity": None, "country": None, "eye_color": None, "hair_color": None,
            "height": 165, "cup_size": None, "band_size": None, "waist_size": None,
            "hip_size": None, "breast_type": None, "career_start_year": None,
            "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
            "is_favorite": False, "deleted": True, "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_performer = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

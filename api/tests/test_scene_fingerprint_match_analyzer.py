"""Tests for SceneFingerprintMatchAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from analyzers.scene_fingerprint_match import SceneFingerprintMatchAnalyzer


def make_mock_db():
    """Create a mock RecommendationsDB."""
    db = MagicMock()
    db.is_dismissed.return_value = False
    db.create_recommendation.return_value = 1
    db.get_user_setting.return_value = None
    db.get_watermark.return_value = None
    db.set_watermark.return_value = None
    return db


def make_mock_stash():
    """Create a mock StashClientUnified."""
    stash = AsyncMock()
    stash.get_stashbox_connections.return_value = [
        {"endpoint": "https://stashdb.org/graphql", "api_key": "key1", "name": "StashDB"}
    ]
    return stash


def make_scene(scene_id, title, fingerprints, stash_ids=None, duration=1800.0, updated_at="2026-01-15T00:00:00Z"):
    """Helper to build a scene dict."""
    return {
        "id": scene_id,
        "title": title,
        "updated_at": updated_at,
        "files": [
            {
                "id": f"file-{scene_id}",
                "duration": duration,
                "fingerprints": fingerprints,
            }
        ],
        "stash_ids": stash_ids or [],
    }


def make_stashbox_match(scene_id, title, fingerprints, studio=None, performers=None, date=None, duration=None):
    """Helper to build a stash-box match result."""
    return {
        "id": scene_id,
        "title": title,
        "date": date,
        "duration": duration,
        "studio": studio,
        "performers": performers or [],
        "urls": [],
        "images": [],
        "fingerprints": fingerprints,
    }


class TestAnalyzerRun:
    """Test the full analyzer run pipeline."""

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_match(self):
        """A scene with matching fingerprints should produce a recommendation."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [
                {"type": "md5", "value": "abc123"},
                {"type": "oshash", "value": "def456"},
            ])],
            1,
        )

        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [
                {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"},
                {"hash": "def456", "algorithm": "OSHASH", "duration": 1800, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
            ],
            studio={"id": "s1", "name": "Studio A"},
            performers=[{"performer": {"id": "p1", "name": "Actor A"}, "as": None}],
            date="2024-01-15",
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 1
        db.create_recommendation.assert_called_once()
        call_kwargs = db.create_recommendation.call_args[1]
        assert call_kwargs["type"] == "scene_fingerprint_match"
        assert call_kwargs["target_type"] == "scene"
        assert "42|" in call_kwargs["target_id"]
        assert "sb-uuid-1" in call_kwargs["target_id"]

    @pytest.mark.asyncio
    async def test_skips_scene_already_linked_to_endpoint(self):
        """Scene with existing stash_id for this endpoint should be skipped."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene",
                [{"type": "md5", "value": "abc123"}],
                stash_ids=[{"endpoint": "https://stashdb.org/graphql", "stash_id": "existing-uuid"}],
            )],
            1,
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0
        mock_sbc.find_scenes_by_fingerprints.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_dismissed_pair(self):
        """Dismissed (scene, stashbox_scene) pair should be skipped."""
        db = make_mock_db()
        db.is_dismissed.return_value = True
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [{"type": "md5", "value": "abc123"}])],
            1,
        )

        stashbox_match = make_stashbox_match(
            "sb-uuid-1", "Matched Scene",
            [{"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 5, "created": "2024-01-01", "updated": "2024-06-01"}],
        )

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[stashbox_match]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0
        db.create_recommendation.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_stashbox_connections_returns_zero(self):
        """If no stash-box endpoints configured, analyzer exits cleanly."""
        db = make_mock_db()
        stash = make_mock_stash()
        stash.get_stashbox_connections.return_value = []

        analyzer = SceneFingerprintMatchAnalyzer(stash, db)
        result = await analyzer.run()

        assert result.items_processed == 0
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_ambiguous_matches_not_high_confidence(self):
        """Multiple matches for same scene from same endpoint = not high confidence."""
        db = make_mock_db()
        stash = make_mock_stash()

        stash.get_scenes_with_fingerprints.return_value = (
            [make_scene("42", "My Scene", [
                {"type": "md5", "value": "abc123"},
                {"type": "oshash", "value": "def456"},
            ])],
            1,
        )

        match_a = make_stashbox_match("sb-1", "Match A", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 3, "created": "2024-01-01", "updated": "2024-06-01"},
            {"hash": "def456", "algorithm": "OSHASH", "duration": 1800, "submissions": 2, "created": "2024-01-01", "updated": "2024-06-01"},
        ])
        match_b = make_stashbox_match("sb-2", "Match B", [
            {"hash": "abc123", "algorithm": "MD5", "duration": 1800, "submissions": 1, "created": "2024-01-01", "updated": "2024-06-01"},
        ])

        with patch("analyzers.scene_fingerprint_match.StashBoxClient") as MockSBC:
            mock_sbc = AsyncMock()
            mock_sbc.find_scenes_by_fingerprints.return_value = [[match_a, match_b]]
            MockSBC.return_value = mock_sbc

            analyzer = SceneFingerprintMatchAnalyzer(stash, db)
            result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 2
        for call in db.create_recommendation.call_args_list:
            details = call[1]["details"]
            assert details["high_confidence"] is False

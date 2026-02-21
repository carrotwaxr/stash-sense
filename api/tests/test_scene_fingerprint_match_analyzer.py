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


class TestAcceptAction:
    """Test the accept fingerprint match action logic."""

    @pytest.mark.asyncio
    async def test_accept_adds_stash_id_to_scene(self):
        """Accepting a match should add the stash_id to the local scene."""
        from recommendations_router import _accept_fingerprint_match

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {
            "id": "42",
            "stash_ids": [
                {"endpoint": "https://other.org/graphql", "stash_id": "other-uuid"},
            ],
        }
        mock_stash.update_scene.return_value = {"id": "42"}

        mock_db = MagicMock()
        mock_db.resolve_recommendation.return_value = True

        await _accept_fingerprint_match(
            stash=mock_stash,
            db=mock_db,
            rec_id=1,
            scene_id="42",
            endpoint="https://stashdb.org/graphql",
            stash_id="sb-uuid-1",
        )

        # Verify stash_ids includes both old and new
        call_kwargs = mock_stash.update_scene.call_args[1]
        stash_ids = call_kwargs["stash_ids"]
        assert len(stash_ids) == 2
        assert {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-uuid-1"} in stash_ids

        # Verify recommendation was resolved
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")


class TestAcceptAllAction:
    """Test the accept-all fingerprint matches action."""

    @pytest.mark.asyncio
    async def test_accepts_only_high_confidence(self):
        """Only high_confidence recommendations should be accepted."""
        from recommendations_router import _accept_all_fingerprint_matches
        from recommendations_db import Recommendation

        high_conf_rec = Recommendation(
            id=1, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="42|https://stashdb.org/graphql|sb-1",
            details={
                "local_scene_id": "42",
                "endpoint": "https://stashdb.org/graphql",
                "stashbox_scene_id": "sb-1",
                "high_confidence": True,
            },
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.67, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )
        low_conf_rec = Recommendation(
            id=2, type="scene_fingerprint_match", status="pending",
            target_type="scene", target_id="43|https://stashdb.org/graphql|sb-2",
            details={
                "local_scene_id": "43",
                "endpoint": "https://stashdb.org/graphql",
                "stashbox_scene_id": "sb-2",
                "high_confidence": False,
            },
            resolution_action=None, resolution_details=None, resolved_at=None,
            confidence=0.33, source_analysis_id=None,
            created_at="2026-01-01", updated_at="2026-01-01",
        )

        mock_db = MagicMock()
        mock_db.get_recommendations.return_value = [high_conf_rec, low_conf_rec]
        mock_db.resolve_recommendation.return_value = True

        mock_stash = AsyncMock()
        mock_stash.get_scene_by_id.return_value = {"id": "42", "stash_ids": []}
        mock_stash.update_scene.return_value = {"id": "42"}

        accepted = await _accept_all_fingerprint_matches(mock_stash, mock_db)

        assert accepted == 1
        mock_db.resolve_recommendation.assert_called_once_with(1, action="accepted")

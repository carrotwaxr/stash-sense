"""Tests for UpstreamTagAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestUpstreamTagAnalyzer:
    """Tests for the upstream tag changes analyzer."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "test-key", "name": "stashdb"},
        ])
        stash.get_tags_for_endpoint = AsyncMock(return_value=[])
        return stash

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def test_analyzer_type(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        assert analyzer.type == "upstream_tag_changes"

    def test_entity_type(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        assert analyzer.entity_type == "tag"

    @pytest.mark.asyncio
    async def test_no_tags_no_recommendations(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[])
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_changed_name(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Blowjob",
                "description": "Oral sex performed on a male",
                "aliases": ["BJ"],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
            }
        ])
        upstream_tag = {
            "id": "tag-uuid-1", "name": "Blowjob",
            "description": "Oral sex performed on a penis",
            "aliases": ["BJ"],
            "category": {"id": "cat-1", "name": "Action", "group": "ACTION"},
            "deleted": False,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_tag = AsyncMock(return_value=upstream_tag)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 1
        recs = rec_db.get_recommendations(type="upstream_tag_changes")
        assert len(recs) == 1
        assert recs[0].details["tag_id"] == "10"
        assert recs[0].details["tag_name"] == "Blowjob"
        assert any(c["field"] == "description" for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_alias_change(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Blowjob",
                "description": "Oral sex",
                "aliases": ["BJ"],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
            }
        ])
        upstream_tag = {
            "id": "tag-uuid-1", "name": "Blowjob",
            "description": "Oral sex",
            "aliases": ["BJ", "Fellatio"],
            "category": None, "deleted": False,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_tag = AsyncMock(return_value=upstream_tag)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 1
        recs = rec_db.get_recommendations(type="upstream_tag_changes")
        changes = recs[0].details["changes"]
        assert any(c["field"] == "aliases" for c in changes)

    @pytest.mark.asyncio
    async def test_no_recommendation_when_in_sync(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Blowjob",
                "description": "Oral sex",
                "aliases": ["BJ"],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
            }
        ])
        upstream_tag = {
            "id": "tag-uuid-1", "name": "Blowjob",
            "description": "Oral sex",
            "aliases": ["BJ"],
            "category": None, "deleted": False,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_tag = AsyncMock(return_value=upstream_tag)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_updates_existing_pending_recommendation(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        rec_db.create_recommendation(
            type="upstream_tag_changes", target_type="tag",
            target_id="10", details={"changes": [{"field": "description", "upstream_value": "Old desc"}]},
            confidence=1.0,
        )
        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Blowjob",
                "description": "Local desc",
                "aliases": [],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
            }
        ])
        upstream_tag = {
            "id": "tag-uuid-1", "name": "Blowjob",
            "description": "New upstream desc",
            "aliases": [],
            "category": None, "deleted": False,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-16T10:00:00Z",
        }
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_tag = AsyncMock(return_value=upstream_tag)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0  # updated, not created
        recs = rec_db.get_recommendations(type="upstream_tag_changes")
        assert len(recs) == 1
        assert any(c["upstream_value"] == "New upstream desc" for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_skips_permanently_dismissed(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        with rec_db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id, permanent) VALUES (?, ?, ?, ?)",
                ("upstream_tag_changes", "tag", "10", 1),
            )
        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Blowjob",
                "description": "Oral sex",
                "aliases": [],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
            }
        ])
        upstream_tag = {
            "id": "tag-uuid-1", "name": "Different Name",
            "description": "Different desc",
            "aliases": ["new alias"],
            "category": None, "deleted": False,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_tag = AsyncMock(return_value=upstream_tag)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_skips_deleted_upstream_tags(self, mock_stash, rec_db):
        from analyzers.upstream_tag import UpstreamTagAnalyzer
        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Blowjob",
                "description": "Oral sex",
                "aliases": [],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
            }
        ])
        upstream_tag = {
            "id": "tag-uuid-1", "name": "Blowjob",
            "description": "Oral sex",
            "aliases": [],
            "category": None, "deleted": True,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_tag = AsyncMock(return_value=upstream_tag)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_3way_diff_skips_user_intentional_override(self, mock_stash, rec_db):
        """If upstream == snapshot but local differs, user overrode intentionally."""
        from analyzers.upstream_tag import UpstreamTagAnalyzer

        # Pre-seed a snapshot
        rec_db.upsert_upstream_snapshot(
            entity_type="tag",
            local_entity_id="10",
            endpoint="https://stashdb.org/graphql",
            stash_box_id="tag-uuid-1",
            upstream_data={"name": "Blowjob", "description": "Upstream desc", "aliases": []},
            upstream_updated_at="2026-01-14T10:00:00Z",
        )

        mock_stash.get_tags_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Blowjob",
                "description": "My custom description",  # User changed locally
                "aliases": [],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "tag-uuid-1"}],
            }
        ])
        upstream_tag = {
            "id": "tag-uuid-1", "name": "Blowjob",
            "description": "Upstream desc",  # Same as snapshot — no upstream change
            "aliases": [],
            "category": None, "deleted": False,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamTagAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_tag = AsyncMock(return_value=upstream_tag)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

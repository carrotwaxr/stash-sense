"""Tests for UpstreamStudioAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestUpstreamStudioAnalyzer:
    """Tests for the upstream studio changes analyzer."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "test-key", "name": "stashdb"},
        ])
        stash.get_studios_for_endpoint = AsyncMock(return_value=[])
        return stash

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def test_analyzer_type(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        assert analyzer.type == "upstream_studio_changes"

    def test_entity_type(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        assert analyzer.entity_type == "studio"

    @pytest.mark.asyncio
    async def test_no_studios_no_recommendations(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            MockSBC.return_value = MagicMock()
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_changed_name(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "10", "name": "Brazzrs", "urls": ["https://brazzers.com"],
             "parent_studio": None,
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}]}
        ])
        upstream = {
            "id": "studio-uuid-1", "name": "Brazzers",
            "urls": [{"url": "https://brazzers.com"}], "parent": None,
            "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 1
        recs = rec_db.get_recommendations(type="upstream_studio_changes")
        assert len(recs) == 1
        assert recs[0].details["studio_id"] == "10"
        assert recs[0].details["studio_name"] == "Brazzrs"
        assert any(c["field"] == "name" for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_url_change(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "10", "name": "Brazzers", "urls": ["https://old.com"],
             "parent_studio": None,
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}]}
        ])
        upstream = {
            "id": "studio-uuid-1", "name": "Brazzers",
            "urls": [{"url": "https://brazzers.com"}], "parent": None,
            "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 1
        changes = rec_db.get_recommendations(type="upstream_studio_changes")[0].details["changes"]
        assert any(c["field"] == "urls" for c in changes)

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_parent_change(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "20", "name": "Brazzers Exxtra", "urls": [],
             "parent_studio": None,
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-2"}]}
        ])
        upstream = {
            "id": "studio-uuid-2", "name": "Brazzers Exxtra",
            "urls": [], "parent": {"id": "parent-uuid-1", "name": "Brazzers"},
            "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 1
        changes = rec_db.get_recommendations(type="upstream_studio_changes")[0].details["changes"]
        assert any(c["field"] == "parent_studio" for c in changes)

    @pytest.mark.asyncio
    async def test_no_recommendation_when_in_sync(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "10", "name": "Brazzers", "urls": ["https://brazzers.com"],
             "parent_studio": None,
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}]}
        ])
        upstream = {
            "id": "studio-uuid-1", "name": "Brazzers",
            "urls": [{"url": "https://brazzers.com"}], "parent": None,
            "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_skips_permanently_dismissed(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        with rec_db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id, permanent) VALUES (?, ?, ?, ?)",
                ("upstream_studio_changes", "studio", "10", 1),
            )
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "10", "name": "Brazzers", "urls": [], "parent_studio": None,
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}]}
        ])
        upstream = {
            "id": "studio-uuid-1", "name": "Different", "urls": [{"url": "https://new.com"}],
            "parent": None, "deleted": False,
            "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_skips_deleted_upstream_studios(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "10", "name": "Brazzers", "urls": [], "parent_studio": None,
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}]}
        ])
        upstream = {
            "id": "studio-uuid-1", "name": "Brazzers", "urls": [], "parent": None,
            "deleted": True, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_no_false_positive_when_parent_ids_match(self, mock_stash, rec_db):
        """Regression: parent_studio should compare stashbox UUIDs, not local ID vs UUID."""
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "20", "name": "Brazzers Exxtra", "urls": [],
             "parent_studio": {
                 "id": "10", "name": "Brazzers",
                 "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "parent-uuid-1"}],
             },
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-2"}]},
        ])
        upstream = {
            "id": "studio-uuid-2", "name": "Brazzers Exxtra",
            "urls": [], "parent": {"id": "parent-uuid-1", "name": "Brazzers"},
            "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        # Should NOT create a recommendation — parent_studio matches via stashbox UUID
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_parent_change_includes_display_names(self, mock_stash, rec_db):
        """Display names should be included in parent_studio change details."""
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "20", "name": "Sub Studio", "urls": [],
             "parent_studio": {
                 "id": "10", "name": "Old Parent",
                 "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "old-parent-uuid"}],
             },
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-2"}]},
        ])
        upstream = {
            "id": "studio-uuid-2", "name": "Sub Studio",
            "urls": [], "parent": {"id": "new-parent-uuid", "name": "New Parent"},
            "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 1
        changes = rec_db.get_recommendations(type="upstream_studio_changes")[0].details["changes"]
        parent_change = next(c for c in changes if c["field"] == "parent_studio")
        assert parent_change["local_display"] == "Old Parent"
        assert parent_change["upstream_display"] == "New Parent"
        assert parent_change["local_value"] == "old-parent-uuid"
        assert parent_change["upstream_value"] == "new-parent-uuid"

    @pytest.mark.asyncio
    async def test_3way_diff_skips_user_intentional_override(self, mock_stash, rec_db):
        from analyzers.upstream_studio import UpstreamStudioAnalyzer
        rec_db.upsert_upstream_snapshot(
            entity_type="studio", local_entity_id="10",
            endpoint="https://stashdb.org/graphql", stash_box_id="studio-uuid-1",
            upstream_data={"name": "Brazzers", "urls": ["https://brazzers.com"], "parent_studio": None},
            upstream_updated_at="2026-01-14T10:00:00Z",
        )
        mock_stash.get_studios_for_endpoint = AsyncMock(return_value=[
            {"id": "10", "name": "My Custom Studio Name", "urls": ["https://brazzers.com"],
             "parent_studio": None,
             "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "studio-uuid-1"}]}
        ])
        upstream = {
            "id": "studio-uuid-1", "name": "Brazzers",
            "urls": [{"url": "https://brazzers.com"}], "parent": None,
            "deleted": False, "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }
        analyzer = UpstreamStudioAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_studio = AsyncMock(return_value=upstream)
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()
        assert result.recommendations_created == 0

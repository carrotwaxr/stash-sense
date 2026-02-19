"""Tests for BaseUpstreamAnalyzer abstract contract and generic logic."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Optional

from analyzers.base_upstream import BaseUpstreamAnalyzer
from analyzers.base import AnalysisResult


class ConcreteUpstreamAnalyzer(BaseUpstreamAnalyzer):
    """Minimal concrete implementation for testing the base class."""

    type = "upstream_test_entity_changes"

    def __init__(self, stash, rec_db, run_id=None, default_fields=None, field_labels=None):
        super().__init__(stash, rec_db, run_id)
        self._default_fields = default_fields or {"name", "description"}
        self._field_labels = field_labels or {"name": "Name", "description": "Description"}

    @property
    def entity_type(self) -> str:
        return "test_entity"

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        return await self.stash.get_test_entities_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client, stashbox_id: str) -> Optional[dict]:
        return await stashbox_client.get_test_entity(stashbox_id)

    def _build_local_data(self, entity: dict) -> dict:
        return {
            "name": entity.get("name"),
            "description": entity.get("description") or "",
        }

    def _normalize_upstream(self, raw_data: dict) -> dict:
        return {
            "name": raw_data.get("name"),
            "description": raw_data.get("description") or "",
        }

    def _get_default_fields(self) -> set[str]:
        return self._default_fields

    def _get_field_labels(self) -> dict[str, str]:
        return self._field_labels

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        changes = []
        for field in sorted(enabled_fields):
            local_val = local_data.get(field)
            upstream_val = upstream_data.get(field)
            if local_val != upstream_val:
                if snapshot is not None:
                    prev = snapshot.get(field)
                    if upstream_val == prev:
                        continue
                else:
                    prev = None
                changes.append({
                    "field": field,
                    "field_label": self._field_labels.get(field, field),
                    "local_value": local_val,
                    "upstream_value": upstream_val,
                    "previous_upstream_value": prev,
                    "merge_type": "simple",
                })
        return changes


class TestBaseUpstreamAnalyzerAbstractContract:
    """Test that the abstract methods are required."""

    def test_cannot_instantiate_base_class(self):
        """BaseUpstreamAnalyzer is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseUpstreamAnalyzer(MagicMock(), MagicMock())

    def test_must_implement_entity_type(self):
        """Subclass missing entity_type should raise TypeError."""

        class MissingEntityType(BaseUpstreamAnalyzer):
            type = "test"

            async def _get_local_entities(self, endpoint):
                return []

            async def _get_upstream_entity(self, sbc, sid):
                return None

            def _build_local_data(self, entity):
                return {}

            def _normalize_upstream(self, raw):
                return {}

            def _get_default_fields(self):
                return set()

            def _get_field_labels(self):
                return {}

            def _diff_fields(self, local, upstream, snapshot, fields):
                return []

        with pytest.raises(TypeError):
            MissingEntityType(MagicMock(), MagicMock())

    def test_must_implement_get_local_entities(self):
        """Subclass missing _get_local_entities should raise TypeError."""

        class MissingLocalEntities(BaseUpstreamAnalyzer):
            type = "test"

            @property
            def entity_type(self):
                return "test"

            async def _get_upstream_entity(self, sbc, sid):
                return None

            def _build_local_data(self, entity):
                return {}

            def _normalize_upstream(self, raw):
                return {}

            def _get_default_fields(self):
                return set()

            def _get_field_labels(self):
                return {}

            def _diff_fields(self, local, upstream, snapshot, fields):
                return []

        with pytest.raises(TypeError):
            MissingLocalEntities(MagicMock(), MagicMock())

    def test_concrete_subclass_can_be_instantiated(self):
        """A fully implemented subclass can be instantiated."""
        analyzer = ConcreteUpstreamAnalyzer(MagicMock(), MagicMock())
        assert analyzer.entity_type == "test_entity"
        assert analyzer.type == "upstream_test_entity_changes"


class TestBaseUpstreamAnalyzerGenericLogic:
    """Test the generic processing logic using ConcreteUpstreamAnalyzer."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://test.box/graphql", "api_key": "key"},
        ])
        stash.get_test_entities_for_endpoint = AsyncMock(return_value=[])
        return stash

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_no_entities_returns_zero(self, mock_stash, rec_db):
        """No local entities means zero processed and created."""
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient"):
            result = await analyzer.run()
        assert result.items_processed == 0
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_changed_entity(self, mock_stash, rec_db):
        """Detects upstream change and creates a recommendation."""
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "Old desc",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Alpha", "description": "New desc",
                "updated": "2026-01-15T10:00:00Z", "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()

        assert result.recommendations_created == 1
        recs = rec_db.get_recommendations(type="upstream_test_entity_changes")
        assert len(recs) == 1
        assert recs[0].details["test_entity_id"] == "10"
        assert any(c["field"] == "description" for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_skips_permanently_dismissed(self, mock_stash, rec_db):
        """Permanently dismissed entities are skipped."""
        with rec_db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id, permanent) VALUES (?, ?, ?, ?)",
                ("upstream_test_entity_changes", "test_entity", "10", 1),
            )
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Changed", "description": "New",
                "updated": "2026-01-15T10:00:00Z", "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()

        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_skips_deleted_upstream_entities(self, mock_stash, rec_db):
        """Deleted upstream entities are skipped."""
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Alpha", "description": "",
                "updated": "2026-01-15T10:00:00Z", "deleted": True, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()

        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_updates_existing_pending_recommendation(self, mock_stash, rec_db):
        """If a pending recommendation exists, it is updated instead of creating a new one."""
        rec_db.create_recommendation(
            type="upstream_test_entity_changes", target_type="test_entity",
            target_id="10", details={"changes": [{"field": "description", "upstream_value": "v1"}]},
            confidence=1.0,
        )
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "Old",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Alpha", "description": "v2",
                "updated": "2026-01-16T10:00:00Z", "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()

        assert result.recommendations_created == 0  # updated, not created
        recs = rec_db.get_recommendations(type="upstream_test_entity_changes")
        assert len(recs) == 1
        assert any(c["upstream_value"] == "v2" for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_auto_resolves_stale_when_no_changes(self, mock_stash, rec_db):
        """When upstream matches local, any stale pending rec is auto-resolved."""
        rec_id = rec_db.create_recommendation(
            type="upstream_test_entity_changes", target_type="test_entity",
            target_id="10", details={"changes": []},
            confidence=1.0,
        )
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "Same",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Alpha", "description": "Same",
                "updated": "2026-01-15T10:00:00Z", "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            await analyzer.run()

        rec = rec_db.get_recommendation(rec_id)
        assert rec.status == "resolved"
        assert rec.resolution_action == "auto_resolved"

    @pytest.mark.asyncio
    async def test_reopens_soft_dismissed_recommendation(self, mock_stash, rec_db):
        """A soft-dismissed recommendation gets reopened when new changes arrive."""
        rec_id = rec_db.create_recommendation(
            type="upstream_test_entity_changes", target_type="test_entity",
            target_id="10", details={"changes": [{"field": "name"}]},
            confidence=1.0,
        )
        rec_db.dismiss_recommendation(rec_id, permanent=False)

        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Beta", "description": "",
                "updated": "2026-01-16T10:00:00Z", "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            result = await analyzer.run()

        assert result.recommendations_created == 1
        rec = rec_db.get_recommendation(rec_id)
        assert rec.status == "pending"

    @pytest.mark.asyncio
    async def test_watermark_set_after_processing(self, mock_stash, rec_db):
        """Watermark is set after successfully processing an endpoint."""
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "Same",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Alpha", "description": "Same",
                "updated": "2026-01-15T10:00:00Z", "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            await analyzer.run()

        wm = rec_db.get_watermark("upstream_test_entity_changes:https://test.box/graphql")
        assert wm is not None
        assert wm["last_cursor"] == "2026-01-15T10:00:00Z"

    @pytest.mark.asyncio
    async def test_incremental_skips_old_entities(self, mock_stash, rec_db):
        """In incremental mode, entities older than watermark are skipped."""
        # Set watermark
        rec_db.set_watermark(
            "upstream_test_entity_changes:https://test.box/graphql",
            last_cursor="2026-01-15T00:00:00Z",
        )
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "Old",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Changed", "description": "New",
                "updated": "2026-01-14T10:00:00Z",  # Before watermark
                "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            result = await analyzer.run(incremental=True)

        assert result.items_processed == 0  # skipped by watermark
        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_endpoint_error_is_captured(self, mock_stash, rec_db):
        """Errors during endpoint processing are captured in the result."""
        mock_stash.get_test_entities_for_endpoint = AsyncMock(
            side_effect=RuntimeError("Connection failed")
        )
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient"):
            result = await analyzer.run()

        assert len(result.errors) == 1
        assert "Connection failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_disabled_endpoint_is_skipped(self, mock_stash, rec_db):
        """Endpoints marked as disabled in settings are skipped."""
        rec_db.upsert_settings(
            "upstream_test_entity_changes",
            config={"endpoints": {"https://test.box/graphql": {"enabled": False}}},
        )
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient"):
            result = await analyzer.run()

        # Should not even call get_test_entities_for_endpoint
        assert result.items_processed == 0

    @pytest.mark.asyncio
    async def test_entity_type_used_in_details(self, mock_stash, rec_db):
        """The entity_type property is used to build recommendation details."""
        mock_stash.get_test_entities_for_endpoint = AsyncMock(return_value=[
            {
                "id": "10", "name": "Alpha", "description": "Old",
                "stash_ids": [{"endpoint": "https://test.box/graphql", "stash_id": "sb-10"}],
            }
        ])
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_test_entity = AsyncMock(return_value={
                "id": "sb-10", "name": "Alpha", "description": "New",
                "updated": "2026-01-15T10:00:00Z", "deleted": False, "merged_into_id": None,
            })
            MockSBC.return_value = mock_sbc
            await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_test_entity_changes")
        assert len(recs) == 1
        assert "test_entity_id" in recs[0].details
        assert "test_entity_name" in recs[0].details

    @pytest.mark.asyncio
    async def test_build_local_lookup(self, mock_stash, rec_db):
        """_build_local_lookup correctly maps stash_box_id to entity."""
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        entities = [
            {"id": "1", "stash_ids": [
                {"endpoint": "https://a.com", "stash_id": "sb-1"},
                {"endpoint": "https://b.com", "stash_id": "sb-2"},
            ]},
            {"id": "2", "stash_ids": [
                {"endpoint": "https://a.com", "stash_id": "sb-3"},
            ]},
        ]
        lookup = analyzer._build_local_lookup(entities, "https://a.com")
        assert len(lookup) == 2
        assert lookup["sb-1"]["id"] == "1"
        assert lookup["sb-3"]["id"] == "2"
        assert "sb-2" not in lookup  # different endpoint

    def test_is_upstream_deleted_default(self, mock_stash, rec_db):
        """Default _is_upstream_deleted checks deleted and merged_into_id."""
        analyzer = ConcreteUpstreamAnalyzer(mock_stash, rec_db)
        assert analyzer._is_upstream_deleted({"deleted": True, "merged_into_id": None}) is True
        assert analyzer._is_upstream_deleted({"deleted": False, "merged_into_id": "xyz"}) == "xyz"
        assert analyzer._is_upstream_deleted({"deleted": False, "merged_into_id": None}) is None
        assert not analyzer._is_upstream_deleted({})

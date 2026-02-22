"""Tests for performer update actions, including auto-merge on name conflict."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from fastapi.testclient import TestClient


@pytest.fixture
def app():
    """Create a test app with recommendations router."""
    from fastapi import FastAPI
    from recommendations_router import router, init_recommendations
    import tempfile
    import os

    app = FastAPI()
    app.include_router(router)

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        init_recommendations(db_path, "http://localhost:9999", "test-key")
        yield app


class TestPerformerNameConflictAutoMerge:
    """When updating a performer name conflicts with an existing performer,
    auto-merge the conflicting performer into the one being updated."""

    def test_auto_merges_on_name_conflict(self, app):
        """Reproduces: Upstream renames 'Miss Kenzie Anne' to 'Kenzie Anne',
        but a local 'Kenzie Anne' already exists. Should merge the conflicting
        performer into the one being updated, then retry the name update.
        """
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()

            # First update_performer call fails with name conflict
            # Second call (after merge) succeeds
            call_count = 0

            async def mock_update_performer(performer_id, **fields):
                nonlocal call_count
                call_count += 1
                if call_count == 1 and "name" in fields and fields["name"] == "Kenzie Anne":
                    raise RuntimeError(
                        'GraphQL error: [{"message": "Name \\"Kenzie Anne\\" already used by '
                        'performer \\"Kenzie Anne\\" (add a disambiguation to make it unique)"}]'
                    )
                return {"id": performer_id}

            mock_stash.update_performer = AsyncMock(side_effect=mock_update_performer)

            # search_performers finds the conflicting performer
            mock_stash.search_performers = AsyncMock(return_value=[
                {
                    "id": "99",
                    "name": "Kenzie Anne",
                    "disambiguation": "",
                    "alias_list": [],
                    "stash_ids": [],
                },
            ])

            # merge_performers succeeds
            mock_stash.merge_performers = AsyncMock(return_value={"id": "50", "name": "Miss Kenzie Anne"})

            # get_performer for fetching current data (lazy-fetch)
            mock_stash.get_performer = AsyncMock(return_value={
                "id": "50",
                "name": "Miss Kenzie Anne",
                "alias_list": [],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-123"}],
            })

            mock_get_stash.return_value = mock_stash

            # Clear entity name cache
            import recommendations_router
            recommendations_router._entity_name_cache_loaded.clear()
            recommendations_router._entity_name_cache.clear()

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Kenzie Anne"},
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            # Should have merged the conflicting performer
            assert data.get("auto_merged") is True
            assert data.get("merged_performer_id") == "99"
            # Should have called merge_performers
            mock_stash.merge_performers.assert_called_once_with(["99"], "50")

    def test_no_merge_on_non_name_conflict_error(self, app):
        """Regular errors should propagate normally, not trigger merge logic."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_performer = AsyncMock(
                side_effect=RuntimeError("Some other GraphQL error")
            )
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"country": "US"},
            })

            assert resp.status_code == 500

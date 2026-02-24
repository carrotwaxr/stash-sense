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


def _make_mock_stash(
    *,
    dest_id="50",
    dest_name="Old Name",
    dest_disambig="",
    conflict_id="99",
    conflict_name="Test",
    conflict_disambig="",
    update_fails_with_name_conflict=True,
    merge_side_effect=None,
):
    """Create a mock StashClientUnified with common name-conflict setup.

    Args:
        dest_id: ID of the performer being updated (destination).
        dest_name: Current name of the destination performer.
        dest_disambig: Disambiguation of the destination performer.
        conflict_id: ID of the conflicting performer found by search.
        conflict_name: Name of the conflicting performer.
        conflict_disambig: Disambiguation of the conflicting performer.
        update_fails_with_name_conflict: If True, first update call raises a
            name conflict error; second call succeeds.
        merge_side_effect: Optional side_effect for merge_performers mock.
    """
    mock_stash = MagicMock()

    if update_fails_with_name_conflict:
        call_count = 0

        async def mock_update(performer_id, **fields):
            nonlocal call_count
            call_count += 1
            if call_count == 1 and "name" in fields:
                raise RuntimeError(
                    f'GraphQL error: Name "{conflict_name}" already used by '
                    f'performer "{conflict_name}"'
                )
            return {"id": performer_id}

        mock_stash.update_performer = AsyncMock(side_effect=mock_update)
    else:
        mock_stash.update_performer = AsyncMock(return_value={"id": dest_id})

    mock_stash.search_performers = AsyncMock(return_value=[
        {
            "id": conflict_id,
            "name": conflict_name,
            "disambiguation": conflict_disambig,
            "alias_list": [],
            "stash_ids": [],
        },
    ])

    if merge_side_effect:
        mock_stash.merge_performers = AsyncMock(side_effect=merge_side_effect)
    else:
        mock_stash.merge_performers = AsyncMock(
            return_value={"id": dest_id, "name": dest_name}
        )

    mock_stash.get_performer = AsyncMock(return_value={
        "id": dest_id,
        "name": dest_name,
        "disambiguation": dest_disambig,
        "alias_list": [],
        "stash_ids": [],
    })

    return mock_stash


def _clear_caches():
    """Clear module-level caches that persist between tests."""
    import recommendations_router
    recommendations_router._entity_name_cache_loaded.clear()
    recommendations_router._entity_name_cache.clear()


class TestPerformerNameConflictAutoMerge:
    """When updating a performer name conflicts with an existing performer,
    auto-merge the conflicting performer into the one being updated."""

    def test_auto_merges_on_name_conflict(self, app):
        """Reproduces: Upstream renames 'Miss Kenzie Anne' to 'Kenzie Anne',
        but a local 'Kenzie Anne' already exists. Should merge the conflicting
        performer into the one being updated, then retry the name update.
        """
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = _make_mock_stash(
                dest_id="50",
                dest_name="Miss Kenzie Anne",
                conflict_id="99",
                conflict_name="Kenzie Anne",
            )
            # Override get_performer to include stash_ids
            mock_stash.get_performer = AsyncMock(return_value={
                "id": "50",
                "name": "Miss Kenzie Anne",
                "alias_list": [],
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "uuid-123"}],
            })
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Kenzie Anne"},
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data.get("auto_merged") is True
            assert data.get("merged_performer_id") == "99"
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

    def test_deleted_conflicting_performer_skips_merge(self, app):
        """If the conflicting performer was deleted, skip merge and proceed."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = _make_mock_stash(
                conflict_name="Test",
                merge_side_effect=RuntimeError("performer not found"),
            )
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Test"},
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data.get("auto_merged") is True

    def test_merge_raises_on_non_deleted_error(self, app):
        """If merge fails for a reason other than 'not found', the error should propagate."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()

            mock_stash.update_performer = AsyncMock(
                side_effect=RuntimeError(
                    'GraphQL error: Name "Test" already used by performer "Test"'
                )
            )
            mock_stash.search_performers = AsyncMock(return_value=[
                {"id": "99", "name": "Test", "disambiguation": "", "alias_list": [], "stash_ids": []},
            ])
            mock_stash.merge_performers = AsyncMock(
                side_effect=RuntimeError("internal server error")
            )
            mock_stash.get_performer = AsyncMock(return_value={
                "id": "50", "name": "Old Name", "alias_list": [], "stash_ids": [],
            })
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Test"},
            })

            assert resp.status_code == 500


class TestDisambiguationMergeSafety:
    """Disambiguation-aware auto-merge: never merge when either performer
    has a disambiguation, since disambiguated performers are explicitly
    marked as distinct people sharing a name."""

    def test_skips_merge_when_both_have_different_disambiguations(self, app):
        """'Hazel Grace (US)' and 'Hazel Grace (Russian)' should NOT merge.
        Returns 409 with disambiguation-specific message."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = _make_mock_stash(
                dest_disambig="US",
                conflict_name="Hazel Grace",
                conflict_disambig="Russian",
            )
            # Override: always fail (no retry expected since merge is skipped)
            mock_stash.update_performer = AsyncMock(
                side_effect=RuntimeError(
                    'GraphQL error: Name "Hazel Grace" already used by performer "Hazel Grace"'
                )
            )
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Hazel Grace"},
            })

            assert resp.status_code == 409
            assert "different disambiguation" in resp.json()["detail"]
            mock_stash.merge_performers.assert_not_called()

    def test_skips_merge_when_only_destination_has_disambiguation(self, app):
        """'Hazel Grace (US)' should not merge with plain 'Hazel Grace'.
        Returns 409 with disambiguation-specific message."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = _make_mock_stash(
                dest_disambig="US",
                conflict_name="Hazel Grace",
                conflict_disambig="",
            )
            mock_stash.update_performer = AsyncMock(
                side_effect=RuntimeError(
                    'GraphQL error: Name "Hazel Grace" already used by performer "Hazel Grace"'
                )
            )
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Hazel Grace"},
            })

            assert resp.status_code == 409
            assert "different disambiguation" in resp.json()["detail"]
            mock_stash.merge_performers.assert_not_called()

    def test_skips_merge_when_only_conflict_has_disambiguation(self, app):
        """Plain 'Hazel Grace' should not auto-merge 'Hazel Grace (Russian)'.
        Returns 409 with disambiguation-specific message."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = _make_mock_stash(
                dest_disambig="",
                conflict_name="Hazel Grace",
                conflict_disambig="Russian",
            )
            mock_stash.update_performer = AsyncMock(
                side_effect=RuntimeError(
                    'GraphQL error: Name "Hazel Grace" already used by performer "Hazel Grace"'
                )
            )
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Hazel Grace"},
            })

            assert resp.status_code == 409
            assert "different disambiguation" in resp.json()["detail"]
            mock_stash.merge_performers.assert_not_called()

    def test_allows_merge_when_neither_has_disambiguation(self, app):
        """When neither performer has disambiguation, merge should proceed."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = _make_mock_stash(
                dest_disambig="",
                conflict_name="Hazel Grace",
                conflict_disambig="",
            )
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Hazel Grace"},
            })

            assert resp.status_code == 200
            data = resp.json()
            assert data["success"] is True
            assert data.get("auto_merged") is True
            mock_stash.merge_performers.assert_called_once()

    def test_uses_disambiguation_from_fields_over_current(self, app):
        """When fields dict includes disambiguation, it should be used for the
        merge safety check (it's what the performer will become).
        Returns 409 with disambiguation-specific message."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = _make_mock_stash(
                dest_name="Hazel Grace",
                dest_disambig="",  # current has no disambig
                conflict_name="Hazel Grace",
                conflict_disambig="",
            )
            # Override: always fail since merge should be skipped
            mock_stash.update_performer = AsyncMock(
                side_effect=RuntimeError(
                    'GraphQL error: Name "Hazel Grace" already used by performer "Hazel Grace"'
                )
            )
            mock_get_stash.return_value = mock_stash
            _clear_caches()

            client = TestClient(app, raise_server_exceptions=False)
            # fields includes disambiguation being set — should block merge
            resp = client.post("/recommendations/actions/update-performer", json={
                "performer_id": "50",
                "fields": {"name": "Hazel Grace", "disambiguation": "US"},
            })

            assert resp.status_code == 409
            assert "different disambiguation" in resp.json()["detail"]
            mock_stash.merge_performers.assert_not_called()

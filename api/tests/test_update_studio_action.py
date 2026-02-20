"""Tests for the update-studio action endpoint."""

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


class TestUpdateStudioEndpoint:
    def test_update_studio_name(self, app):
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "1"})
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "1",
                "fields": {"name": "New Name"},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200
            assert resp.json()["success"] is True
            mock_stash.update_studio.assert_called_once()

    def test_update_studio_url(self, app):
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "1"})
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "1",
                "fields": {"url": "https://new-url.com"},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200

    def test_update_studio_parent_found_locally(self, app):
        """When parent studio exists locally, resolve its ID."""
        with patch("recommendations_router.get_stash_client") as mock_get_stash:
            mock_stash = MagicMock()
            mock_stash.update_studio = AsyncMock(return_value={"id": "20"})
            mock_stash._execute = AsyncMock(return_value={
                "findStudios": {
                    "studios": [{"id": "10", "name": "Parent Studio"}]
                }
            })
            mock_get_stash.return_value = mock_stash

            client = TestClient(app)
            resp = client.post("/recommendations/actions/update-studio", json={
                "studio_id": "20",
                "fields": {"parent_studio": "parent-stashbox-uuid"},
                "endpoint": "https://stashdb.org/graphql",
            })
            assert resp.status_code == 200

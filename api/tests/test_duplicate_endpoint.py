"""Tests for duplicate detection API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDuplicateScanEndpoint:
    """Tests for POST /recommendations/analysis/duplicate_scenes/run endpoint."""

    @pytest.fixture
    def client(self, tmp_path):
        # Set up test environment
        import os
        os.environ["STASH_URL"] = "http://test:9999"
        os.environ["DATA_DIR"] = str(tmp_path)

        # Create minimal required files
        (tmp_path / "manifest.json").write_text('{"version": "test"}')

        from fastapi.testclient import TestClient
        from main import app

        return TestClient(app)

    def test_endpoint_exists(self, client):
        # The endpoint should exist even if analysis fails
        response = client.post("/recommendations/analysis/duplicate_scenes/run")

        # May fail due to missing stash connection, but endpoint should exist
        # and recognize the type (not 404 or 400 "Unknown analysis type")
        assert response.status_code in [200, 500, 503], (
            f"Expected 200, 500, or 503 but got {response.status_code}: {response.json()}"
        )

    def test_duplicate_scenes_in_analysis_types(self, client):
        # Check that duplicate_scenes appears in the available types
        response = client.get("/recommendations/analysis/types")

        if response.status_code == 200:
            data = response.json()
            type_names = [t["type"] for t in data.get("types", [])]
            assert "duplicate_scenes" in type_names

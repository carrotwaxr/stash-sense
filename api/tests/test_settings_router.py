"""Tests for the settings API router."""

import pytest
from fastapi.testclient import TestClient

from recommendations_db import RecommendationsDB
from hardware import HardwareProfile
from settings import init_settings, SETTING_DEFS
from settings_router import router, init_settings_router


@pytest.fixture
def db(tmp_path):
    """Create a fresh RecommendationsDB."""
    return RecommendationsDB(str(tmp_path / "test.db"))


@pytest.fixture
def mock_hardware():
    """Mock hardware profile."""
    import hardware
    original = hardware._hardware_profile
    hardware._hardware_profile = HardwareProfile(
        gpu_available=True, gpu_name="Test GPU", gpu_vram_mb=8192,
        cpu_cores=8, memory_total_mb=32768, memory_available_mb=16384,
        storage_free_mb=500000, tier="gpu-high",
    )
    yield hardware._hardware_profile
    hardware._hardware_profile = original


@pytest.fixture
def client(db, mock_hardware):
    """Create a test client with initialized settings."""
    import settings as settings_mod

    # Initialize settings manager
    original_mgr = settings_mod._settings_manager
    mgr = init_settings(db, "gpu-high")
    init_settings_router()

    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)

    yield client

    settings_mod._settings_manager = original_mgr


class TestGetAllSettings:
    """Test GET /settings."""

    def test_returns_categories(self, client):
        resp = client.get("/settings")
        assert resp.status_code == 200
        data = resp.json()
        assert "hardware_tier" in data
        assert data["hardware_tier"] == "gpu-high"
        assert "categories" in data

    def test_all_settings_present(self, client):
        resp = client.get("/settings")
        data = resp.json()
        all_keys = set()
        for cat in data["categories"].values():
            all_keys.update(cat["settings"].keys())
        for key in SETTING_DEFS:
            assert key in all_keys, f"Missing setting: {key}"


class TestGetSingleSetting:
    """Test GET /settings/{key}."""

    def test_existing_setting(self, client):
        resp = client.get("/settings/embedding_batch_size")
        assert resp.status_code == 200
        data = resp.json()
        assert data["key"] == "embedding_batch_size"
        assert data["value"] == 32  # gpu-high default
        assert data["is_override"] is False

    def test_unknown_setting(self, client):
        resp = client.get("/settings/nonexistent")
        assert resp.status_code == 404


class TestUpdateSetting:
    """Test PUT /settings/{key}."""

    def test_update_valid(self, client):
        resp = client.put("/settings/stash_api_rate", json={"value": 10.0})
        assert resp.status_code == 200
        data = resp.json()
        assert data["value"] == 10.0
        assert data["is_override"] is True

        # Verify persisted
        resp2 = client.get("/settings/stash_api_rate")
        assert resp2.json()["value"] == 10.0
        assert resp2.json()["is_override"] is True

    def test_update_unknown_setting(self, client):
        resp = client.put("/settings/nonexistent", json={"value": 42})
        assert resp.status_code == 404

    def test_update_out_of_range(self, client):
        resp = client.put("/settings/embedding_batch_size", json={"value": 0})
        assert resp.status_code == 422

    def test_update_bool(self, client):
        resp = client.put("/settings/gpu_enabled", json={"value": False})
        assert resp.status_code == 200
        assert resp.json()["value"] is False


class TestBulkUpdate:
    """Test PUT /settings (bulk)."""

    def test_bulk_update(self, client):
        resp = client.put("/settings", json={
            "settings": {
                "stash_api_rate": 10.0,
                "num_frames": 45,
            }
        })
        assert resp.status_code == 200
        assert resp.json()["stored"]["stash_api_rate"] == 10.0
        assert resp.json()["stored"]["num_frames"] == 45

    def test_bulk_update_with_invalid_key(self, client):
        resp = client.put("/settings", json={
            "settings": {
                "stash_api_rate": 10.0,
                "nonexistent": 42,
            }
        })
        assert resp.status_code == 422


class TestResetSetting:
    """Test DELETE /settings/{key}."""

    def test_reset_reverts_to_default(self, client):
        # Set override
        client.put("/settings/embedding_batch_size", json={"value": 64})
        resp = client.get("/settings/embedding_batch_size")
        assert resp.json()["is_override"] is True

        # Reset
        resp = client.delete("/settings/embedding_batch_size")
        assert resp.status_code == 200
        assert resp.json()["value"] == 32  # gpu-high default
        assert resp.json()["is_override"] is False

    def test_reset_unknown_setting(self, client):
        resp = client.delete("/settings/nonexistent")
        assert resp.status_code == 404


class TestSystemInfo:
    """Test GET /system/info."""

    def test_returns_info(self, client):
        resp = client.get("/system/info")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "uptime_seconds" in data
        assert "hardware" in data
        hw = data["hardware"]
        assert hw["gpu_available"] is True
        assert hw["tier"] == "gpu-high"
        assert "summary" in hw

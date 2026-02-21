"""Tests for the model management API router."""

import json
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from model_manager import ModelManager, init_model_manager
from model_router import router, init_model_router


SAMPLE_MANIFEST = {
    "version": 2,
    "repo": "test/repo",
    "release_tag": "models",
    "models": {
        "facenet512": {
            "file": "facenet512.onnx",
            "size": 1000,
            "sha256": "abc123",
            "group": "face_recognition",
            "description": "FaceNet512 face embedding model",
        },
        "arcface": {
            "file": "arcface.onnx",
            "size": 2000,
            "sha256": "def456",
            "group": "face_recognition",
            "description": "ArcFace face embedding model",
        },
    },
}


@pytest.fixture
def manifest_path(tmp_path):
    """Write a sample manifest and return its path."""
    path = tmp_path / "models.json"
    path.write_text(json.dumps(SAMPLE_MANIFEST))
    return path


@pytest.fixture
def models_dir(tmp_path):
    """Create and return a models directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def manager(manifest_path, models_dir):
    """Create a ModelManager with sample manifest."""
    return ModelManager(manifest_path, models_dir)


@pytest.fixture
def client(manager):
    """Create a test client with initialized model router."""
    init_model_router(manager)
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestGetModelStatus:
    """Test GET /models/status."""

    def test_returns_all_models(self, client):
        resp = client.get("/models/status")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        models = data["models"]
        assert "facenet512" in models
        assert "arcface" in models

    def test_model_info_fields(self, client):
        resp = client.get("/models/status")
        model = resp.json()["models"]["facenet512"]
        assert model["status"] == "not_installed"
        assert model["file"] == "facenet512.onnx"
        assert model["size"] == 1000
        assert model["group"] == "face_recognition"
        assert model["description"] == "FaceNet512 face embedding model"

    def test_installed_model_status(self, client, manager, models_dir):
        # Create a fake model file with correct size
        model_file = models_dir / "facenet512.onnx"
        model_file.write_bytes(b"\x00" * 1000)
        resp = client.get("/models/status")
        assert resp.json()["models"]["facenet512"]["status"] == "installed"


class TestDownloadModel:
    """Test POST /models/download/{model_name}."""

    def test_unknown_model_returns_404(self, client):
        resp = client.post("/models/download/nonexistent")
        assert resp.status_code == 404
        assert "Unknown model" in resp.json()["detail"]

    def test_valid_model_returns_download_started(self, client):
        resp = client.post("/models/download/facenet512")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "download_started"
        assert data["model"] == "facenet512"


class TestDownloadAllModels:
    """Test POST /models/download-all."""

    def test_returns_download_started(self, client):
        resp = client.post("/models/download-all")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "download_started"


class TestDownloadProgress:
    """Test GET /models/download-progress."""

    def test_returns_empty_progress(self, client):
        resp = client.get("/models/download-progress")
        assert resp.status_code == 200
        data = resp.json()
        assert "progress" in data
        assert data["progress"] == {}

    def test_returns_progress_after_download_attempt(self, client, manager):
        # Manually inject a progress entry to simulate an active download
        from model_manager import DownloadProgress, DownloadStatus
        manager._progress["facenet512"] = DownloadProgress(
            model_name="facenet512",
            downloaded_bytes=500,
            total_bytes=1000,
            status=DownloadStatus.DOWNLOADING,
        )
        resp = client.get("/models/download-progress")
        data = resp.json()
        progress = data["progress"]
        assert "facenet512" in progress
        assert progress["facenet512"]["downloaded_bytes"] == 500
        assert progress["facenet512"]["total_bytes"] == 1000
        assert progress["facenet512"]["status"] == "downloading"
        assert progress["facenet512"]["percent"] == 50.0

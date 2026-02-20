"""Tests for the model manager."""

import hashlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio

from model_manager import (
    DownloadProgress,
    DownloadStatus,
    ModelManager,
    ModelStatus,
    get_model_manager,
    init_model_manager,
)


# ============================================================================
# Fixtures
# ============================================================================

SAMPLE_MANIFEST = {
    "version": 2,
    "repo": "carrotwaxr/stash-sense",
    "release_tag": "models",
    "models": {
        "test_model_a": {
            "file": "test_model_a.onnx",
            "size": 1024,
            "sha256": "abc123",
            "group": "face_recognition",
            "description": "Test model A",
        },
        "test_model_b": {
            "file": "test_model_b.onnx",
            "size": 2048,
            "sha256": "def456",
            "group": "tattoo_detection",
            "description": "Test model B",
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
    """Create a models directory."""
    d = tmp_path / "models"
    d.mkdir()
    return d


@pytest.fixture
def mgr(manifest_path, models_dir):
    """Create a ModelManager with the sample manifest."""
    return ModelManager(manifest_path, models_dir)


def _make_file(models_dir: Path, filename: str, size: int) -> Path:
    """Create a file with the given size filled with zeros."""
    path = models_dir / filename
    path.write_bytes(b"\x00" * size)
    return path


def _make_file_with_content(models_dir: Path, filename: str, content: bytes) -> Path:
    """Create a file with specific content."""
    path = models_dir / filename
    path.write_bytes(content)
    return path


# ============================================================================
# Status detection tests
# ============================================================================

class TestGetStatus:
    """Test model status detection."""

    def test_all_not_installed_when_empty(self, mgr):
        """When no model files exist, all should be not_installed."""
        status = mgr.get_status()
        assert len(status) == 2
        assert status["test_model_a"]["status"] == "not_installed"
        assert status["test_model_b"]["status"] == "not_installed"

    def test_installed_when_file_matches_size(self, mgr, models_dir):
        """When file exists with matching size, should be installed."""
        _make_file(models_dir, "test_model_a.onnx", 1024)

        status = mgr.get_status()
        assert status["test_model_a"]["status"] == "installed"
        assert status["test_model_b"]["status"] == "not_installed"

    def test_corrupted_when_wrong_size(self, mgr, models_dir):
        """When file exists with wrong size, should be corrupted."""
        _make_file(models_dir, "test_model_a.onnx", 512)  # Wrong size

        status = mgr.get_status()
        assert status["test_model_a"]["status"] == "corrupted"

    def test_status_includes_metadata(self, mgr):
        """Status entries should include file, size, group, description."""
        status = mgr.get_status()
        entry = status["test_model_a"]
        assert entry["file"] == "test_model_a.onnx"
        assert entry["size"] == 1024
        assert entry["group"] == "face_recognition"
        assert entry["description"] == "Test model A"

    def test_all_installed(self, mgr, models_dir):
        """When all model files exist with correct sizes, all are installed."""
        _make_file(models_dir, "test_model_a.onnx", 1024)
        _make_file(models_dir, "test_model_b.onnx", 2048)

        status = mgr.get_status()
        assert status["test_model_a"]["status"] == "installed"
        assert status["test_model_b"]["status"] == "installed"


# ============================================================================
# get_model_path tests
# ============================================================================

class TestGetModelPath:
    """Test get_model_path."""

    def test_returns_none_when_missing(self, mgr):
        """Returns None when model file does not exist."""
        assert mgr.get_model_path("test_model_a") is None

    def test_returns_path_when_installed(self, mgr, models_dir):
        """Returns Path when model file exists with correct size."""
        _make_file(models_dir, "test_model_a.onnx", 1024)
        path = mgr.get_model_path("test_model_a")
        assert path is not None
        assert path == models_dir / "test_model_a.onnx"

    def test_returns_none_when_corrupted(self, mgr, models_dir):
        """Returns None when model file has wrong size."""
        _make_file(models_dir, "test_model_a.onnx", 512)
        assert mgr.get_model_path("test_model_a") is None

    def test_returns_none_for_unknown_model(self, mgr):
        """Returns None for unknown model name."""
        assert mgr.get_model_path("nonexistent") is None


# ============================================================================
# get_download_url tests
# ============================================================================

class TestGetDownloadUrl:
    """Test download URL generation."""

    def test_url_format(self, mgr):
        """URL should follow GitHub release asset format."""
        url = mgr.get_download_url("test_model_a")
        expected = (
            "https://github.com/carrotwaxr/stash-sense/releases/download/"
            "models/test_model_a.onnx"
        )
        assert url == expected

    def test_url_for_second_model(self, mgr):
        """URL uses correct filename for each model."""
        url = mgr.get_download_url("test_model_b")
        assert url.endswith("test_model_b.onnx")

    def test_returns_none_for_unknown(self, mgr):
        """Returns None for unknown model name."""
        assert mgr.get_download_url("nonexistent") is None


# ============================================================================
# download_model tests
# ============================================================================

class TestDownloadModel:
    """Test model downloading with mocked httpx."""

    @pytest.mark.asyncio
    async def test_download_writes_file_and_validates(self, mgr, models_dir, tmp_path):
        """Successful download writes file and validates SHA256."""
        content = b"fake model data for testing"
        expected_hash = hashlib.sha256(content).hexdigest()

        # Update manifest to match our test content
        manifest = dict(SAMPLE_MANIFEST)
        manifest["models"] = {
            "test_model_a": {
                "file": "test_model_a.onnx",
                "size": len(content),
                "sha256": expected_hash,
                "group": "face_recognition",
                "description": "Test model A",
            },
        }
        manifest_path = tmp_path / "manifest_dl.json"
        manifest_path.write_text(json.dumps(manifest))
        mgr = ModelManager(manifest_path, models_dir)

        # Mock httpx streaming response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_bytes(chunk_size=65536):
            yield content

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("model_manager.httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value = AsyncMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=False),
            )
            result = await mgr.download_model("test_model_a")

        assert result == models_dir / "test_model_a.onnx"
        assert result.exists()
        assert result.read_bytes() == content

    @pytest.mark.asyncio
    async def test_download_bad_hash_removes_file(self, mgr, models_dir, tmp_path):
        """Download with SHA256 mismatch raises and removes the file."""
        content = b"wrong content"

        # Manifest with a hash that won't match
        manifest = dict(SAMPLE_MANIFEST)
        manifest["models"] = {
            "test_model_a": {
                "file": "test_model_a.onnx",
                "size": len(content),
                "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "group": "face_recognition",
                "description": "Test model A",
            },
        }
        manifest_path = tmp_path / "manifest_bad.json"
        manifest_path.write_text(json.dumps(manifest))
        mgr = ModelManager(manifest_path, models_dir)

        # Mock httpx streaming response
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_bytes(chunk_size=65536):
            yield content

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("model_manager.httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value = AsyncMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=False),
            )
            with pytest.raises(RuntimeError, match="SHA256 mismatch"):
                await mgr.download_model("test_model_a")

        # File should be removed after failed validation
        assert not (models_dir / "test_model_a.onnx").exists()

    @pytest.mark.asyncio
    async def test_download_unknown_model_raises(self, mgr):
        """Downloading an unknown model raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            await mgr.download_model("nonexistent")

    @pytest.mark.asyncio
    async def test_download_creates_models_dir(self, tmp_path):
        """Download creates the models directory if it doesn't exist."""
        content = b"test data"
        expected_hash = hashlib.sha256(content).hexdigest()

        manifest = {
            "version": 2,
            "repo": "carrotwaxr/stash-sense",
            "release_tag": "models",
            "models": {
                "test_model": {
                    "file": "test.onnx",
                    "size": len(content),
                    "sha256": expected_hash,
                    "group": "face_recognition",
                    "description": "Test",
                },
            },
        }
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps(manifest))

        # Use a non-existent directory
        models_dir = tmp_path / "nonexistent" / "models"
        mgr = ModelManager(manifest_path, models_dir)

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_bytes(chunk_size=65536):
            yield content

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("model_manager.httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value = AsyncMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=False),
            )
            result = await mgr.download_model("test_model")

        assert result.exists()
        assert models_dir.exists()


# ============================================================================
# get_progress tests
# ============================================================================

class TestGetProgress:
    """Test download progress tracking."""

    def test_no_progress_initially(self, mgr):
        """No progress entries when nothing has been downloaded."""
        assert mgr.get_progress() == {}

    @pytest.mark.asyncio
    async def test_progress_during_download(self, mgr, models_dir, tmp_path):
        """Progress is tracked during download."""
        content = b"x" * 1024
        expected_hash = hashlib.sha256(content).hexdigest()

        manifest = dict(SAMPLE_MANIFEST)
        manifest["models"] = {
            "test_model_a": {
                "file": "test_model_a.onnx",
                "size": len(content),
                "sha256": expected_hash,
                "group": "face_recognition",
                "description": "Test model A",
            },
        }
        manifest_path = tmp_path / "manifest_prog.json"
        manifest_path.write_text(json.dumps(manifest))
        mgr = ModelManager(manifest_path, models_dir)

        # Capture progress during download via chunks
        progress_snapshots = []

        async def mock_aiter_bytes(chunk_size=65536):
            # Yield in two chunks
            half = len(content) // 2
            yield content[:half]
            # Capture progress after first chunk
            progress_snapshots.append(dict(mgr.get_progress()))
            yield content[half:]

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.aiter_bytes = mock_aiter_bytes

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("model_manager.httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value = AsyncMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=False),
            )
            await mgr.download_model("test_model_a")

        # Should have captured progress mid-download
        assert len(progress_snapshots) == 1
        mid_progress = progress_snapshots[0]["test_model_a"]
        assert mid_progress["downloaded_bytes"] == 512
        assert mid_progress["total_bytes"] == 1024
        assert mid_progress["status"] == "downloading"
        assert mid_progress["percent"] == pytest.approx(50.0)

        # After download, progress should show complete
        final = mgr.get_progress()["test_model_a"]
        assert final["status"] == "complete"
        assert final["percent"] == 100.0

    @pytest.mark.asyncio
    async def test_progress_shows_failure(self, mgr, models_dir, tmp_path):
        """Progress shows failure status on bad hash."""
        content = b"bad data"

        manifest = dict(SAMPLE_MANIFEST)
        manifest["models"] = {
            "test_model_a": {
                "file": "test_model_a.onnx",
                "size": len(content),
                "sha256": "0" * 64,
                "group": "face_recognition",
                "description": "Test model A",
            },
        }
        manifest_path = tmp_path / "manifest_fail.json"
        manifest_path.write_text(json.dumps(manifest))
        mgr = ModelManager(manifest_path, models_dir)

        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_bytes(chunk_size=65536):
            yield content

        mock_response.aiter_bytes = mock_aiter_bytes

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(return_value=AsyncMock(
            __aenter__=AsyncMock(return_value=mock_response),
            __aexit__=AsyncMock(return_value=False),
        ))

        with patch("model_manager.httpx.AsyncClient") as mock_async_client:
            mock_async_client.return_value = AsyncMock(
                __aenter__=AsyncMock(return_value=mock_client),
                __aexit__=AsyncMock(return_value=False),
            )
            with pytest.raises(RuntimeError):
                await mgr.download_model("test_model_a")

        progress = mgr.get_progress()["test_model_a"]
        assert progress["status"] == "failed"
        assert "SHA256 mismatch" in progress["error"]


# ============================================================================
# download_all tests
# ============================================================================

class TestDownloadAll:
    """Test bulk download functionality."""

    @pytest.mark.asyncio
    async def test_skips_already_installed(self, mgr, models_dir):
        """download_all skips models that are already installed."""
        # Install model A
        _make_file(models_dir, "test_model_a.onnx", 1024)

        # Mock download_model to track calls
        calls = []
        original_download = mgr.download_model

        async def tracked_download(name):
            calls.append(name)
            return models_dir / f"{name}.onnx"

        with patch.object(mgr, "download_model", side_effect=tracked_download):
            results = await mgr.download_all()

        assert results["test_model_a"] == "skipped"
        assert "test_model_a" not in calls
        # Model B should have been attempted
        assert "test_model_b" in calls

    @pytest.mark.asyncio
    async def test_downloads_all_missing(self, mgr, models_dir):
        """download_all downloads all missing models."""
        calls = []

        async def tracked_download(name):
            calls.append(name)
            # Create the file so it appears downloaded
            _make_file(models_dir, f"{name}.onnx", SAMPLE_MANIFEST["models"][name]["size"])
            return models_dir / f"{name}.onnx"

        with patch.object(mgr, "download_model", side_effect=tracked_download):
            results = await mgr.download_all()

        assert "test_model_a" in calls
        assert "test_model_b" in calls
        assert results["test_model_a"] == "downloaded"
        assert results["test_model_b"] == "downloaded"

    @pytest.mark.asyncio
    async def test_handles_download_errors(self, mgr, models_dir):
        """download_all captures errors without stopping."""
        _make_file(models_dir, "test_model_a.onnx", 1024)  # Already installed

        async def failing_download(name):
            raise RuntimeError(f"Network error for {name}")

        with patch.object(mgr, "download_model", side_effect=failing_download):
            results = await mgr.download_all()

        assert results["test_model_a"] == "skipped"
        assert "error" in results["test_model_b"]


# ============================================================================
# DownloadProgress tests
# ============================================================================

class TestDownloadProgress:
    """Test the DownloadProgress dataclass."""

    def test_percent_zero_when_no_total(self):
        p = DownloadProgress(model_name="test", total_bytes=0)
        assert p.percent == 0.0

    def test_percent_calculation(self):
        p = DownloadProgress(model_name="test", downloaded_bytes=500, total_bytes=1000)
        assert p.percent == 50.0

    def test_percent_capped_at_100(self):
        p = DownloadProgress(model_name="test", downloaded_bytes=1500, total_bytes=1000)
        assert p.percent == 100.0

    def test_default_status(self):
        p = DownloadProgress(model_name="test")
        assert p.status == DownloadStatus.DOWNLOADING


# ============================================================================
# Singleton tests
# ============================================================================

class TestSingleton:
    """Test module-level singleton functions."""

    def test_get_before_init_raises(self):
        """get_model_manager raises before init_model_manager."""
        import model_manager as mm
        original = mm._model_manager
        try:
            mm._model_manager = None
            with pytest.raises(RuntimeError, match="not initialized"):
                get_model_manager()
        finally:
            mm._model_manager = original

    def test_init_and_get(self, manifest_path, models_dir):
        """init_model_manager creates a manager, get_model_manager returns it."""
        import model_manager as mm
        original = mm._model_manager
        try:
            mgr = init_model_manager(manifest_path, models_dir)
            assert mgr is get_model_manager()
        finally:
            mm._model_manager = original

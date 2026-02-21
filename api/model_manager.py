"""ONNX model manager.

Manages ONNX model files on the persistent data volume. Downloads models
from GitHub Release assets on demand and validates integrity via SHA256.

Usage:
    # At startup
    init_model_manager(manifest_path=Path("models.json"), models_dir=Path("models"))

    # Later
    mgr = get_model_manager()
    status = mgr.get_status()
    path = mgr.get_model_path("facenet512")
    await mgr.download_model("facenet512")
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


# ============================================================================
# Data types
# ============================================================================

class ModelStatus(str, Enum):
    NOT_INSTALLED = "not_installed"
    INSTALLED = "installed"
    CORRUPTED = "corrupted"


class DownloadStatus(str, Enum):
    DOWNLOADING = "downloading"
    VALIDATING = "validating"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class ModelInfo:
    """Model metadata from the manifest."""
    name: str
    file: str
    size: int
    sha256: str
    group: str
    description: str


@dataclass
class DownloadProgress:
    """Progress tracker for an active or recent download."""
    model_name: str
    downloaded_bytes: int = 0
    total_bytes: int = 0
    status: DownloadStatus = DownloadStatus.DOWNLOADING
    error: Optional[str] = None

    @property
    def percent(self) -> float:
        if self.total_bytes <= 0:
            return 0.0
        return min(100.0, (self.downloaded_bytes / self.total_bytes) * 100.0)


# ============================================================================
# ModelManager
# ============================================================================

class ModelManager:
    """Manages ONNX model files: status checks, downloads, and validation."""

    def __init__(self, manifest_path: Path, models_dir: Path):
        """Initialize with bundled manifest and target models directory.

        Args:
            manifest_path: Path to models.json manifest file.
            models_dir: Directory where model files are stored / downloaded to.
        """
        self._models_dir = models_dir
        self._models: dict[str, ModelInfo] = {}
        self._repo: str = ""
        self._release_tag: str = ""
        self._progress: dict[str, DownloadProgress] = {}

        self._load_manifest(manifest_path)

    def _load_manifest(self, manifest_path: Path) -> None:
        """Load and parse the models.json manifest."""
        with open(manifest_path) as f:
            data = json.load(f)

        self._repo = data["repo"]
        self._release_tag = data["release_tag"]

        for name, info in data["models"].items():
            self._models[name] = ModelInfo(
                name=name,
                file=info["file"],
                size=info["size"],
                sha256=info["sha256"],
                group=info["group"],
                description=info["description"],
            )

    def _model_path(self, model: ModelInfo) -> Path:
        """Get the expected file path for a model."""
        return self._models_dir / model.file

    def _check_model_status(self, model: ModelInfo) -> ModelStatus:
        """Check the status of a single model."""
        path = self._model_path(model)
        if not path.exists():
            return ModelStatus.NOT_INSTALLED

        actual_size = path.stat().st_size
        if actual_size != model.size:
            return ModelStatus.CORRUPTED

        return ModelStatus.INSTALLED

    def get_status(self) -> dict[str, dict]:
        """Get status of all models.

        Returns:
            Dict keyed by model name with status, file, size, group, and
            description for each model.
        """
        result = {}
        for name, model in self._models.items():
            status = self._check_model_status(model)
            result[name] = {
                "status": status.value,
                "file": model.file,
                "size": model.size,
                "group": model.group,
                "description": model.description,
            }
        return result

    def get_model_path(self, model_name: str) -> Optional[Path]:
        """Get path to a model file, or None if not installed/corrupted.

        Args:
            model_name: Name of the model (e.g. "facenet512").

        Returns:
            Path to the model file if installed and valid, None otherwise.
        """
        model = self._models.get(model_name)
        if model is None:
            return None

        status = self._check_model_status(model)
        if status != ModelStatus.INSTALLED:
            return None

        return self._model_path(model)

    def get_download_url(self, model_name: str) -> Optional[str]:
        """Get GitHub Release download URL for a model.

        Args:
            model_name: Name of the model (e.g. "facenet512").

        Returns:
            URL string, or None if model name is unknown.
        """
        model = self._models.get(model_name)
        if model is None:
            return None

        return (
            f"https://github.com/{self._repo}/releases/download/"
            f"{self._release_tag}/{model.file}"
        )

    async def download_model(self, model_name: str) -> Path:
        """Download a model from GitHub Releases.

        Downloads with streaming, tracks progress, and validates SHA256
        after download. Removes the file if validation fails.

        Args:
            model_name: Name of the model to download.

        Returns:
            Path to the downloaded model file.

        Raises:
            ValueError: If model_name is unknown.
            RuntimeError: If download fails or SHA256 validation fails.
        """
        model = self._models.get(model_name)
        if model is None:
            raise ValueError(f"Unknown model: {model_name}")

        url = self.get_download_url(model_name)
        dest = self._model_path(model)
        dest.parent.mkdir(parents=True, exist_ok=True)

        progress = DownloadProgress(
            model_name=model_name,
            total_bytes=model.size,
            status=DownloadStatus.DOWNLOADING,
        )
        self._progress[model_name] = progress

        logger.warning(f"Downloading model {model_name} from {url}")

        try:
            timeout = httpx.Timeout(connect=30.0, read=60.0, write=None, pool=30.0)
            sha256_hasher = hashlib.sha256()

            async with httpx.AsyncClient(follow_redirects=True, timeout=timeout) as client:
                async with client.stream("GET", url) as response:
                    response.raise_for_status()

                    with open(dest, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=65536):
                            f.write(chunk)
                            sha256_hasher.update(chunk)
                            progress.downloaded_bytes += len(chunk)

            # Validate SHA256 (already computed incrementally)
            progress.status = DownloadStatus.VALIDATING
            actual_hash = sha256_hasher.hexdigest()

            if actual_hash != model.sha256:
                dest.unlink(missing_ok=True)
                error_msg = (
                    f"SHA256 mismatch for {model_name}: "
                    f"expected {model.sha256}, got {actual_hash}"
                )
                progress.status = DownloadStatus.FAILED
                progress.error = error_msg
                raise RuntimeError(error_msg)

            progress.status = DownloadStatus.COMPLETE
            logger.warning(f"Model {model_name} downloaded and verified")
            return dest

        except httpx.HTTPStatusError as e:
            error_msg = f"HTTP error downloading {model_name}: {e}"
            progress.status = DownloadStatus.FAILED
            progress.error = error_msg
            dest.unlink(missing_ok=True)
            raise RuntimeError(error_msg) from e

        except Exception:
            if progress.status not in (DownloadStatus.FAILED, DownloadStatus.COMPLETE):
                progress.status = DownloadStatus.FAILED
                progress.error = f"Download failed for {model_name}"
            raise

    async def download_all(self) -> dict[str, str]:
        """Download all missing or corrupted models.

        Returns:
            Dict mapping model name to result string ("downloaded", "skipped",
            or error message).
        """
        results = {}
        for name, model in self._models.items():
            status = self._check_model_status(model)
            if status == ModelStatus.INSTALLED:
                results[name] = "skipped"
                continue

            try:
                await self.download_model(name)
                results[name] = "downloaded"
            except Exception as e:
                results[name] = f"error: {e}"

        return results

    def get_progress(self) -> dict[str, dict]:
        """Get download progress for active and recent downloads.

        Returns:
            Dict keyed by model name with progress details.
        """
        result = {}
        for name, progress in self._progress.items():
            result[name] = {
                "model_name": progress.model_name,
                "downloaded_bytes": progress.downloaded_bytes,
                "total_bytes": progress.total_bytes,
                "status": progress.status.value,
                "error": progress.error,
                "percent": progress.percent,
            }
        return result

    @staticmethod
    def _compute_sha256(path: Path) -> str:
        """Compute SHA256 hash of a file."""
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()


# ============================================================================
# Module-level singleton
# ============================================================================

_model_manager: Optional[ModelManager] = None


def init_model_manager(
    manifest_path: Path,
    models_dir: Path,
) -> ModelManager:
    """Initialize the global model manager. Called once at startup.

    Args:
        manifest_path: Path to models.json manifest file.
        models_dir: Directory where model files are stored.

    Returns:
        The initialized ModelManager instance.
    """
    global _model_manager
    _model_manager = ModelManager(manifest_path, models_dir)
    return _model_manager


def get_model_manager() -> ModelManager:
    """Get the global model manager.

    Must be called after init_model_manager().

    Raises:
        RuntimeError: If init_model_manager() has not been called.
    """
    if _model_manager is None:
        raise RuntimeError(
            "ModelManager not initialized. Call init_model_manager() during startup."
        )
    return _model_manager

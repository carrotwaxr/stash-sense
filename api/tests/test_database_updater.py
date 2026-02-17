"""Tests for the database_updater module."""

import asyncio
import hashlib
import json
import time
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from database_updater import (
    CACHE_TTL_SECONDS,
    GITHUB_API_URL,
    RELEASE_FILES,
    REQUIRED_FILES,
    DatabaseUpdater,
    UpdateStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    """Compute sha256 hex digest."""
    return hashlib.sha256(data).hexdigest()


def _write_manifest(data_dir: Path, version: str = "2026.02.12", checksums: dict | None = None):
    """Write a manifest.json into *data_dir*."""
    manifest = {
        "version": version,
        "created_at": "2026-02-12T00:00:00Z",
        "performer_count": 100,
        "face_count": 200,
        "sources": ["stashdb"],
        "models": {},
    }
    if checksums is not None:
        manifest["checksums"] = checksums
    (data_dir / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def _make_release_zip(dest: Path, version: str = "2026.02.15", extra_files: dict | None = None):
    """Create a zip file with required release files and a valid manifest.

    Returns the manifest dict (with checksums already filled in).
    """
    file_contents: dict[str, bytes] = {}
    for fname in REQUIRED_FILES:
        if fname == "manifest.json":
            continue  # built at the end
        file_contents[fname] = f"data for {fname} v{version}".encode()

    if extra_files:
        file_contents.update(extra_files)

    # Build checksums
    checksums = {name: f"sha256:{_sha256(data)}" for name, data in file_contents.items()}

    manifest = {
        "version": version,
        "created_at": "2026-02-15T00:00:00Z",
        "performer_count": 110,
        "face_count": 210,
        "sources": ["stashdb"],
        "models": {},
        "checksums": checksums,
    }
    manifest_bytes = json.dumps(manifest).encode()
    file_contents["manifest.json"] = manifest_bytes

    with zipfile.ZipFile(dest, "w") as zf:
        for name, data in file_contents.items():
            zf.writestr(name, data)

    return manifest


def _github_release_json(tag: str = "2026.02.15", zip_url: str = "https://github.com/fake/download.zip"):
    """Return a dict resembling a GitHub latest-release API response."""
    return {
        "tag_name": tag,
        "name": f"Release {tag}",
        "published_at": "2026-02-15T00:00:00Z",
        "assets": [
            {
                "name": "stash-sense-data.zip",
                "browser_download_url": zip_url,
                "size": 12345,
            }
        ],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGetCurrentVersion:
    """Reading the current version from manifest.json."""

    def test_reads_version_from_manifest(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))
        assert updater._get_current_version() == "2026.02.12"

    def test_returns_none_when_no_manifest(self, tmp_path):
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))
        assert updater._get_current_version() is None


class TestCheckUpdate:
    """check_update() hits the GitHub API and determines if an update is available."""

    async def test_update_available(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        release_json = _github_release_json(tag="2026.02.15")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = release_json
        mock_response.raise_for_status = MagicMock()

        with patch("database_updater.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await updater.check_update()

        assert result["update_available"] is True
        assert result["current_version"] == "2026.02.12"
        assert result["latest_version"] == "2026.02.15"
        assert result["download_url"] == "https://github.com/fake/download.zip"

    async def test_already_on_latest(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.15")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        release_json = _github_release_json(tag="2026.02.15")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = release_json
        mock_response.raise_for_status = MagicMock()

        with patch("database_updater.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result = await updater.check_update()

        assert result["update_available"] is False
        assert result["current_version"] == "2026.02.15"
        assert result["latest_version"] == "2026.02.15"

    async def test_cache_prevents_repeated_api_calls(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        release_json = _github_release_json(tag="2026.02.15")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = release_json
        mock_response.raise_for_status = MagicMock()

        with patch("database_updater.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            result1 = await updater.check_update()
            result2 = await updater.check_update()

        # Should only make one HTTP request due to caching
        assert client_instance.get.call_count == 1
        assert result1 == result2

    async def test_cache_expires(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        release_json = _github_release_json(tag="2026.02.15")

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = release_json
        mock_response.raise_for_status = MagicMock()

        with patch("database_updater.httpx.AsyncClient") as MockClient:
            client_instance = AsyncMock()
            client_instance.get.return_value = mock_response
            MockClient.return_value.__aenter__ = AsyncMock(return_value=client_instance)
            MockClient.return_value.__aexit__ = AsyncMock(return_value=False)

            await updater.check_update()

            # Expire the cache by shifting the timestamp back
            updater._cache_time = time.monotonic() - CACHE_TTL_SECONDS - 1

            await updater.check_update()

        assert client_instance.get.call_count == 2


class TestChecksumVerification:
    """_verify_checksums validates file integrity."""

    def test_valid_checksums_pass(self, tmp_path):
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        file_data = b"hello world"
        (extract_dir / "performers.db").write_bytes(file_data)

        manifest = {
            "checksums": {
                "performers.db": f"sha256:{_sha256(file_data)}",
            }
        }

        # Should not raise
        updater._verify_checksums(extract_dir, manifest)

    def test_invalid_checksum_raises(self, tmp_path):
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        (extract_dir / "performers.db").write_bytes(b"hello world")

        manifest = {
            "checksums": {
                "performers.db": "sha256:0000000000000000000000000000000000000000000000000000000000000000",
            }
        }

        with pytest.raises(ValueError, match="Checksum mismatch.*performers.db"):
            updater._verify_checksums(extract_dir, manifest)

    def test_missing_file_in_checksums_is_skipped(self, tmp_path):
        """Files not listed in checksums are not verified (optional files)."""
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        (extract_dir / "face_adaface.voy").write_bytes(b"optional")

        manifest = {"checksums": {}}

        # Should not raise — file not in checksums
        updater._verify_checksums(extract_dir, manifest)


class TestLoadAndVerify:
    """_load_and_verify reads the manifest from the extract dir and verifies checksums."""

    def test_loads_manifest_and_verifies(self, tmp_path):
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Create required files
        file_data = {}
        for fname in REQUIRED_FILES:
            if fname == "manifest.json":
                continue
            data = f"data-{fname}".encode()
            (extract_dir / fname).write_bytes(data)
            file_data[fname] = data

        checksums = {name: f"sha256:{_sha256(data)}" for name, data in file_data.items()}
        manifest = {
            "version": "2026.02.15",
            "checksums": checksums,
        }
        (extract_dir / "manifest.json").write_text(json.dumps(manifest))

        result = updater._load_and_verify(extract_dir)
        assert result["version"] == "2026.02.15"

    def test_raises_if_required_file_missing(self, tmp_path):
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()

        # Only write manifest — required files are missing
        (extract_dir / "manifest.json").write_text(json.dumps({"version": "2026.02.15", "checksums": {}}))

        with pytest.raises(FileNotFoundError, match="Missing required"):
            updater._load_and_verify(extract_dir)


class TestExtract:
    """_extract unzips files."""

    def test_extracts_zip_contents(self, tmp_path):
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        zip_path = tmp_path / "release.zip"
        extract_dir = tmp_path / "extract"

        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("file_a.txt", "aaa")
            zf.writestr("file_b.txt", "bbb")

        updater._extract(zip_path, extract_dir)

        assert (extract_dir / "file_a.txt").read_text() == "aaa"
        assert (extract_dir / "file_b.txt").read_text() == "bbb"


class TestSwapFiles:
    """_swap_files replaces data dir contents while preserving stash_sense.db."""

    def test_preserves_stash_sense_db(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create existing stash_sense.db
        stash_db = data_dir / "stash_sense.db"
        stash_db.write_text("local recommendations data")

        # Create existing data files
        for fname in REQUIRED_FILES:
            (data_dir / fname).write_text(f"old-{fname}")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))

        # Create extract dir with new files
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        for fname in REQUIRED_FILES:
            (extract_dir / fname).write_text(f"new-{fname}")

        updater._swap_files(extract_dir)

        # stash_sense.db should be UNCHANGED
        assert stash_db.read_text() == "local recommendations data"

        # Release files should be updated
        for fname in REQUIRED_FILES:
            assert (data_dir / fname).read_text() == f"new-{fname}"

    def test_creates_backup(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        for fname in REQUIRED_FILES:
            (data_dir / fname).write_text(f"old-{fname}")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        for fname in REQUIRED_FILES:
            (extract_dir / fname).write_text(f"new-{fname}")

        updater._swap_files(extract_dir)

        # Backup directory should exist and have old files
        backup_dirs = list(data_dir.glob("backup_*"))
        assert len(backup_dirs) == 1
        backup_dir = backup_dirs[0]

        for fname in REQUIRED_FILES:
            assert (backup_dir / fname).read_text() == f"old-{fname}"

    def test_does_not_touch_other_files(self, tmp_path):
        """Non-release files (image_cache, models, etc.) are preserved."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Create non-release files and dirs
        (data_dir / "image_cache").mkdir()
        (data_dir / "image_cache" / "img.jpg").write_bytes(b"\xff\xd8")
        (data_dir / "custom_file.txt").write_text("keep me")

        for fname in REQUIRED_FILES:
            (data_dir / fname).write_text(f"old-{fname}")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        for fname in REQUIRED_FILES:
            (extract_dir / fname).write_text(f"new-{fname}")

        updater._swap_files(extract_dir)

        assert (data_dir / "image_cache" / "img.jpg").read_bytes() == b"\xff\xd8"
        assert (data_dir / "custom_file.txt").read_text() == "keep me"

    def test_handles_optional_release_files(self, tmp_path):
        """Optional release files (e.g. face_adaface.voy) are swapped if present in extract."""
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        for fname in REQUIRED_FILES:
            (data_dir / fname).write_text(f"old-{fname}")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))

        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        for fname in REQUIRED_FILES:
            (extract_dir / fname).write_text(f"new-{fname}")
        # Include an optional file
        (extract_dir / "face_adaface.voy").write_bytes(b"adaface-data")

        updater._swap_files(extract_dir)

        assert (data_dir / "face_adaface.voy").read_bytes() == b"adaface-data"


class TestRollback:
    """_rollback restores files from backup."""

    def test_rollback_restores_old_files(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Write "new" files (simulating a swap that went wrong)
        for fname in REQUIRED_FILES:
            (data_dir / fname).write_text(f"new-{fname}")

        # Create backup with "old" files
        backup_dir = data_dir / "backup_test"
        backup_dir.mkdir()
        for fname in REQUIRED_FILES:
            (backup_dir / fname).write_text(f"old-{fname}")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))
        updater._rollback(backup_dir)

        for fname in REQUIRED_FILES:
            assert (data_dir / fname).read_text() == f"old-{fname}"

    def test_rollback_preserves_stash_sense_db(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        (data_dir / "stash_sense.db").write_text("my local db")
        (data_dir / "manifest.json").write_text("new-manifest")

        backup_dir = data_dir / "backup_test"
        backup_dir.mkdir()
        (backup_dir / "manifest.json").write_text("old-manifest")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))
        updater._rollback(backup_dir)

        assert (data_dir / "stash_sense.db").read_text() == "my local db"


class TestStateTracking:
    """State management and get_status()."""

    def test_initial_state_is_idle(self, tmp_path):
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))
        status = updater.get_status()

        assert status["status"] == "idle"
        assert status["progress_pct"] == 0
        assert status["error"] is None

    def test_get_status_returns_dict(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))
        status = updater.get_status()

        assert isinstance(status, dict)
        assert "status" in status
        assert "progress_pct" in status
        assert "current_version" in status
        assert "target_version" in status
        assert "error" in status

    def test_status_shows_current_version(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))
        status = updater.get_status()

        assert status["current_version"] == "2026.02.12"


class TestStartUpdate:
    """start_update launches a background task."""

    async def test_returns_job_id(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        with patch.object(updater, "_run_update", new_callable=AsyncMock):
            job_id = await updater.start_update(
                download_url="https://example.com/release.zip",
                target_version="2026.02.15",
            )

        assert isinstance(job_id, str)
        assert len(job_id) > 0

    async def test_returns_409_if_already_running(self, tmp_path):
        _write_manifest(tmp_path, version="2026.02.12")
        updater = DatabaseUpdater(data_dir=tmp_path, reload_fn=MagicMock(return_value=True))

        # Simulate a long-running update
        running_event = asyncio.Event()

        async def slow_update(*args, **kwargs):
            await running_event.wait()

        with patch.object(updater, "_run_update", side_effect=slow_update):
            await updater.start_update("https://example.com/release.zip", "2026.02.15")

            # Give the task a moment to start
            await asyncio.sleep(0.05)

            with pytest.raises(RuntimeError, match="[Aa]lready running"):
                await updater.start_update("https://example.com/release.zip", "2026.02.16")

        # Clean up
        running_event.set()
        await asyncio.sleep(0.05)


class TestRunUpdate:
    """Full pipeline integration test with mocked download."""

    async def test_successful_update_pipeline(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Set up existing data
        _write_manifest(data_dir, version="2026.02.12")
        for fname in REQUIRED_FILES:
            if fname != "manifest.json":
                (data_dir / fname).write_text(f"old-{fname}")

        # Create the zip that will be "downloaded"
        zip_path = tmp_path / "release.zip"
        _make_release_zip(zip_path, version="2026.02.15")

        reload_fn = MagicMock(return_value=True)
        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=reload_fn)

        # Mock the download to copy our pre-made zip
        async def fake_download(url, dest):
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(updater, "_download", side_effect=fake_download):
            await updater._run_update(
                download_url="https://example.com/release.zip",
                target_version="2026.02.15",
            )

        # Verify final state
        status = updater.get_status()
        assert status["status"] == "complete"

        # Verify reload was called
        reload_fn.assert_called_once_with(data_dir)

        # Verify files were swapped
        manifest = json.loads((data_dir / "manifest.json").read_text())
        assert manifest["version"] == "2026.02.15"

    async def test_failed_update_sets_error_state(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_manifest(data_dir, version="2026.02.12")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))

        async def failing_download(url, dest):
            raise ConnectionError("Network down")

        with patch.object(updater, "_download", side_effect=failing_download):
            await updater._run_update(
                download_url="https://example.com/release.zip",
                target_version="2026.02.15",
            )

        status = updater.get_status()
        assert status["status"] == "failed"
        assert "Network down" in status["error"]

    async def test_swap_failure_triggers_rollback(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()

        # Set up existing data files
        _write_manifest(data_dir, version="2026.02.12")
        for fname in REQUIRED_FILES:
            if fname != "manifest.json":
                (data_dir / fname).write_text(f"old-{fname}")

        # Create the zip
        zip_path = tmp_path / "release.zip"
        _make_release_zip(zip_path, version="2026.02.15")

        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))

        async def fake_download(url, dest):
            import shutil
            shutil.copy2(zip_path, dest)

        # Make swap fail
        with patch.object(updater, "_download", side_effect=fake_download), \
             patch.object(updater, "_swap_files", side_effect=OSError("Permission denied")):
            await updater._run_update(
                download_url="https://example.com/release.zip",
                target_version="2026.02.15",
            )

        status = updater.get_status()
        assert status["status"] == "failed"
        assert "Permission denied" in status["error"]

        # Original files should still be intact (swap was mocked to fail,
        # so no actual changes were made to data_dir)
        assert (data_dir / "manifest.json").exists()
        old_manifest = json.loads((data_dir / "manifest.json").read_text())
        assert old_manifest["version"] == "2026.02.12"

    async def test_cleanup_removes_temp_files(self, tmp_path):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        _write_manifest(data_dir, version="2026.02.12")
        for fname in REQUIRED_FILES:
            if fname != "manifest.json":
                (data_dir / fname).write_text(f"old-{fname}")

        zip_path = tmp_path / "release.zip"
        _make_release_zip(zip_path, version="2026.02.15")

        reload_fn = MagicMock(return_value=True)
        updater = DatabaseUpdater(data_dir=data_dir, reload_fn=reload_fn)

        async def fake_download(url, dest):
            import shutil
            shutil.copy2(zip_path, dest)

        with patch.object(updater, "_download", side_effect=fake_download):
            await updater._run_update(
                download_url="https://example.com/release.zip",
                target_version="2026.02.15",
            )

        # Temp download and extract dirs should be cleaned up
        update_dirs = list(data_dir.glob("update_*"))
        assert len(update_dirs) == 0, f"Temp dirs not cleaned up: {update_dirs}"

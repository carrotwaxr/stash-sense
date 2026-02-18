# Sidecar Database Self-Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Enable the sidecar to download new DB releases from GitHub and hot-swap them without a container restart, triggered from the plugin UI.

**Architecture:** New `database_updater.py` module handles download/verify/swap/reload. Three new endpoints on the sidecar (`/database/check-update`, `/database/update`, `/database/update/status`). Plugin backend proxies these, plugin UI shows an update badge on the recommendations dashboard and a progress modal during updates. A FastAPI dependency gates `/identify/*` routes with 503 during the brief reload window.

**Tech Stack:** Python 3.11, FastAPI, httpx (async HTTP), zipfile, hashlib, asyncio.Task, Pydantic. Plugin: vanilla JS via existing `runPluginOperation` proxy pattern.

**Design doc:** `docs/plans/2026-02-17-sidecar-database-self-update-design.md`

---

### Task 1: Extract reload_database() from lifespan

Refactor `main.py` so the DB loading logic that currently lives inside `lifespan()` is extracted into a standalone function that can be called both at startup and during a hot-swap.

**Files:**
- Modify: `api/main.py:168-259`

**Step 1: Extract the reload function**

Add a new function `reload_database()` that encapsulates lines 174-249 of `lifespan()`. It should:
- Accept `data_dir: Path` as parameter
- Set all the same globals: `recognizer`, `db_manifest`, `multi_signal_matcher`, `body_extractor`, `tattoo_detector`, `tattoo_matcher`, `multi_signal_config`
- Call `set_db_version()` if manifest has a version
- Return `True` on success, raise on failure

```python
def reload_database(data_dir: Path) -> bool:
    """Load or reload the face recognition database and multi-signal components.

    Sets global state: recognizer, db_manifest, multi_signal_matcher, etc.
    Called at startup from lifespan() and during hot-swap from database_updater.

    Returns True on success, raises Exception on failure.
    """
    global recognizer, db_manifest
    global multi_signal_matcher, body_extractor, tattoo_detector, tattoo_matcher, multi_signal_config

    print(f"Loading face database from {data_dir}...")

    db_config = DatabaseConfig(data_dir=data_dir)

    # Load manifest
    if db_config.manifest_json_path.exists():
        with open(db_config.manifest_json_path) as f:
            db_manifest = json.load(f)

    recognizer = FaceRecognizer(db_config)
    print("Face database loaded successfully!")

    # Initialize multi-signal components
    multi_signal_config = MultiSignalConfig.from_env()

    body_extractor = None
    if multi_signal_config.enable_body:
        print("Initializing body proportion extractor...")
        body_extractor = BodyProportionExtractor()

    enable_tattoo = multi_signal_config.enable_tattoo
    tattoo_enabled = (
        enable_tattoo == "true"
        or (enable_tattoo == "auto"
            and db_config.tattoo_index_path.exists()
            and db_config.tattoo_json_path.exists())
    )

    tattoo_detector = None
    tattoo_matcher = None
    if tattoo_enabled:
        print("Initializing tattoo detector...")
        tattoo_detector = TattooDetector()
        if recognizer.tattoo_index is not None and recognizer.tattoo_mapping is not None:
            from tattoo_matcher import TattooMatcher
            tattoo_matcher = TattooMatcher(
                tattoo_index=recognizer.tattoo_index,
                tattoo_mapping=recognizer.tattoo_mapping,
            )
            print(f"Tattoo embedding matching ready: {len(recognizer.tattoo_index)} embeddings loaded")

    multi_signal_matcher = None
    if recognizer.db_reader and (body_extractor or tattoo_detector):
        print("Initializing multi-signal matcher...")
        multi_signal_matcher = MultiSignalMatcher(
            face_recognizer=recognizer,
            db_reader=recognizer.db_reader,
            body_extractor=body_extractor,
            tattoo_detector=tattoo_detector,
            tattoo_matcher=tattoo_matcher,
        )
        tattoo_count = len(multi_signal_matcher.performers_with_tattoo_embeddings)
        print(f"Multi-signal ready: {len(multi_signal_matcher.body_data)} body, "
              f"{tattoo_count} performers with tattoo embeddings")

    if db_manifest.get("version"):
        set_db_version(db_manifest["version"])
        print(f"Face recognition DB version: {db_manifest['version']}")

    return True
```

**Step 2: Simplify lifespan() to call reload_database()**

Replace the body of `lifespan()` to call the new function:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the recognizer and initialize recommendations on startup."""
    data_dir = Path(DATA_DIR)

    try:
        reload_database(data_dir)
    except Exception as e:
        print(f"Warning: Failed to load face database: {e}")
        print("API will start but /identify will not work until database is available")

    # Initialize recommendations database
    rec_db_path = data_dir / "stash_sense.db"
    print(f"Initializing recommendations database at {rec_db_path}...")
    init_recommendations(
        db_path=str(rec_db_path),
        stash_url=STASH_URL,
        stash_api_key=STASH_API_KEY,
    )
    print("Recommendations database initialized!")

    if STASH_URL:
        print(f"Stash connection configured: {STASH_URL}")
    else:
        print("Warning: STASH_URL not set - recommendations analysis will not work")

    yield
    recognizer = None
```

**Step 3: Verify sidecar still starts correctly**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -c "from main import reload_database; print('Import OK')"`

**Step 4: Commit**

```bash
git add api/main.py
git commit -m "refactor: extract reload_database() from lifespan for hot-swap support"
```

---

### Task 2: Create database_updater.py core module

The core update logic: GitHub check, download, extract, verify, swap, reload.

**Files:**
- Create: `api/database_updater.py`
- Test: `api/tests/test_database_updater.py`

**Step 1: Write tests for the updater**

```python
"""Tests for database_updater module."""
import hashlib
import json
import zipfile
from pathlib import Path
from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import pytest_asyncio

from database_updater import (
    DatabaseUpdater,
    UpdateStatus,
    UpdateState,
)


@pytest.fixture
def data_dir(tmp_path):
    """Create a data directory with mock DB files."""
    # Create minimal current DB files
    for name in ["performers.db", "face_facenet.voy", "face_arcface.voy",
                  "faces.json", "performers.json"]:
        (tmp_path / name).write_text(f"current-{name}")

    manifest = {
        "version": "2026.02.12",
        "performer_count": 100,
        "face_count": 200,
        "sources": ["stashdb.org"],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))

    # stash_sense.db should be preserved across updates
    (tmp_path / "stash_sense.db").write_text("recommendations-data")
    return tmp_path


@pytest.fixture
def updater(data_dir):
    return DatabaseUpdater(data_dir=data_dir, reload_fn=MagicMock(return_value=True))


def _make_release_zip(tmp_path, version="2026.02.14"):
    """Create a mock release zip with proper checksums."""
    files = {
        "performers.db": b"new-performers",
        "face_facenet.voy": b"new-facenet",
        "face_arcface.voy": b"new-arcface",
        "faces.json": b'{"0": "stashdb.org:abc"}',
        "performers.json": b'{"stashdb.org:abc": {"name": "Test"}}',
    }
    checksums = {}
    for name, content in files.items():
        checksums[name] = f"sha256:{hashlib.sha256(content).hexdigest()}"

    manifest = {
        "version": version,
        "performer_count": 150,
        "face_count": 300,
        "sources": ["stashdb.org"],
        "checksums": checksums,
    }
    manifest_bytes = json.dumps(manifest).encode()
    checksums["manifest.json"] = f"sha256:{hashlib.sha256(manifest_bytes).hexdigest()}"
    # Rebuild manifest with its own checksum excluded (manifest doesn't checksum itself)
    # Actually: manifest checksums all OTHER files, not itself. So we're fine.

    zip_path = tmp_path / "release.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for name, content in files.items():
            zf.writestr(name, content)
        zf.writestr("manifest.json", manifest_bytes)

    return zip_path, manifest


class TestCheckUpdate:
    """Tests for checking GitHub for updates."""

    @pytest.mark.asyncio
    async def test_update_available(self, updater):
        mock_response = {
            "tag_name": "v2026.02.14",
            "name": "Database Release v2026.02.14",
            "published_at": "2026-02-14T19:30:00Z",
            "assets": [{
                "name": "stash-sense-data-v2026.02.14.zip",
                "browser_download_url": "https://github.com/carrotwaxr/stash-sense-data/releases/download/v2026.02.14/stash-sense-data-v2026.02.14.zip",
                "size": 900_000_000,
            }],
        }
        with patch("database_updater.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)

            result = await updater.check_update()

        assert result["update_available"] is True
        assert result["current_version"] == "2026.02.12"
        assert result["latest_version"] == "2026.02.14"

    @pytest.mark.asyncio
    async def test_no_update_available(self, updater):
        mock_response = {
            "tag_name": "v2026.02.12",
            "name": "Database Release v2026.02.12",
            "published_at": "2026-02-12T19:30:00Z",
            "assets": [{"name": "stash-sense-data-v2026.02.12.zip", "browser_download_url": "https://example.com/dl.zip", "size": 900_000_000}],
        }
        with patch("database_updater.httpx.AsyncClient") as mock_client:
            mock_client.return_value.__aenter__ = AsyncMock(return_value=mock_client.return_value)
            mock_client.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_resp = AsyncMock()
            mock_resp.json.return_value = mock_response
            mock_resp.raise_for_status = MagicMock()
            mock_client.return_value.get = AsyncMock(return_value=mock_resp)

            result = await updater.check_update()

        assert result["update_available"] is False


class TestVerifyChecksums:
    """Tests for checksum verification."""

    def test_valid_checksums(self, updater, tmp_path):
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        content = b"test content"
        (extract_dir / "test.bin").write_bytes(content)
        expected = f"sha256:{hashlib.sha256(content).hexdigest()}"

        manifest = {"checksums": {"test.bin": expected}}
        (extract_dir / "manifest.json").write_text(json.dumps(manifest))

        # Should not raise
        updater._verify_checksums(extract_dir, manifest)

    def test_invalid_checksum_raises(self, updater, tmp_path):
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        (extract_dir / "test.bin").write_bytes(b"actual content")

        manifest = {"checksums": {"test.bin": "sha256:0000000000000000000000000000000000000000000000000000000000000000"}}
        (extract_dir / "manifest.json").write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Checksum mismatch"):
            updater._verify_checksums(extract_dir, manifest)


class TestSwapFiles:
    """Tests for the file swap operation."""

    def test_swap_preserves_stash_sense_db(self, updater, data_dir, tmp_path):
        """stash_sense.db must never be overwritten during a swap."""
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        for name in ["performers.db", "face_facenet.voy", "face_arcface.voy",
                      "faces.json", "performers.json", "manifest.json"]:
            (extract_dir / name).write_text(f"new-{name}")

        updater._swap_files(extract_dir)

        assert (data_dir / "stash_sense.db").read_text() == "recommendations-data"
        assert (data_dir / "performers.db").read_text() == "new-performers.db"

    def test_swap_creates_backup(self, updater, data_dir, tmp_path):
        extract_dir = tmp_path / "extracted"
        extract_dir.mkdir()
        for name in ["performers.db", "face_facenet.voy", "face_arcface.voy",
                      "faces.json", "performers.json", "manifest.json"]:
            (extract_dir / name).write_text(f"new-{name}")

        updater._swap_files(extract_dir)

        backup_dir = data_dir / "backup"
        assert backup_dir.exists()
        assert (backup_dir / "performers.db").read_text() == "current-performers.db"


class TestUpdateState:
    """Tests for update state tracking."""

    def test_initial_state_is_idle(self, updater):
        assert updater.state.status == UpdateStatus.IDLE

    def test_status_transitions(self, updater):
        updater.state.status = UpdateStatus.DOWNLOADING
        updater.state.progress_pct = 50
        assert updater.state.status == UpdateStatus.DOWNLOADING
        assert updater.state.progress_pct == 50

    def test_to_dict(self, updater):
        updater.state.status = UpdateStatus.DOWNLOADING
        updater.state.target_version = "2026.02.14"
        d = updater.get_status()
        assert d["status"] == "downloading"
        assert d["target_version"] == "2026.02.14"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense && .venv/bin/python -m pytest api/tests/test_database_updater.py -v`
Expected: ImportError — `database_updater` module doesn't exist yet.

**Step 3: Implement database_updater.py**

```python
"""Database self-update: check GitHub, download, verify, swap, reload."""
import asyncio
import hashlib
import json
import shutil
import time
import zipfile
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import httpx


GITHUB_REPO = "carrotwaxr/stash-sense-data"
GITHUB_API_URL = f"https://api.github.com/repos/{GITHUB_REPO}/releases/latest"
CACHE_TTL_SECONDS = 600  # 10 minutes

# Files that are part of the release archive (DB data).
# stash_sense.db is NOT in this list — it's the local recommendations DB.
RELEASE_FILES = {
    "performers.db",
    "face_facenet.voy",
    "face_arcface.voy",
    "faces.json",
    "performers.json",
    "manifest.json",
    # Optional files (included if present in release):
    "face_adaface.voy",
    "tattoo_embeddings.voy",
    "tattoo_embeddings.json",
}

# Files that must exist in every release
REQUIRED_FILES = {
    "performers.db",
    "face_facenet.voy",
    "face_arcface.voy",
    "faces.json",
    "performers.json",
    "manifest.json",
}


class UpdateStatus(str, Enum):
    IDLE = "idle"
    DOWNLOADING = "downloading"
    EXTRACTING = "extracting"
    VERIFYING = "verifying"
    SWAPPING = "swapping"
    RELOADING = "reloading"
    COMPLETE = "complete"
    FAILED = "failed"


class UpdateState:
    """Mutable state for tracking update progress."""

    def __init__(self):
        self.status: UpdateStatus = UpdateStatus.IDLE
        self.progress_pct: int = 0
        self.current_version: Optional[str] = None
        self.target_version: Optional[str] = None
        self.error: Optional[str] = None


class DatabaseUpdater:
    """Manages database updates from GitHub releases."""

    def __init__(self, data_dir: Path, reload_fn: Callable[[Path], bool]):
        """
        Args:
            data_dir: Path to the data directory (e.g. /data)
            reload_fn: Function to call to reload DB globals (reload_database from main.py)
        """
        self.data_dir = Path(data_dir)
        self.reload_fn = reload_fn
        self.state = UpdateState()
        self._update_task: Optional[asyncio.Task] = None
        self._cached_check: Optional[dict] = None
        self._cache_time: float = 0

    def _get_current_version(self) -> str:
        manifest_path = self.data_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                return json.load(f).get("version", "unknown")
        return "unknown"

    async def check_update(self) -> dict:
        """Check GitHub for the latest release, compare with current version."""
        now = time.time()
        if self._cached_check and (now - self._cache_time) < CACHE_TTL_SECONDS:
            return self._cached_check

        current = self._get_current_version()

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    GITHUB_API_URL,
                    headers={"Accept": "application/vnd.github+json"},
                    timeout=15.0,
                )
                resp.raise_for_status()
                release = resp.json()
        except Exception as e:
            return {
                "current_version": current,
                "update_available": False,
                "error": str(e),
            }

        tag = release.get("tag_name", "").lstrip("v")
        assets = release.get("assets", [])
        zip_asset = next((a for a in assets if a["name"].endswith(".zip")), None)

        result = {
            "current_version": current,
            "latest_version": tag,
            "update_available": tag != current and tag > current,
            "release_name": release.get("name", ""),
            "published_at": release.get("published_at", ""),
        }

        if zip_asset:
            result["download_url"] = zip_asset["browser_download_url"]
            result["download_size_mb"] = round(zip_asset["size"] / 1_000_000)

        self._cached_check = result
        self._cache_time = now
        return result

    def get_status(self) -> dict:
        """Return current update status for the polling endpoint."""
        return {
            "status": self.state.status.value,
            "progress_pct": self.state.progress_pct,
            "current_version": self._get_current_version(),
            "target_version": self.state.target_version,
            "error": self.state.error,
        }

    async def start_update(self, download_url: str, target_version: str) -> str:
        """Start the update pipeline in a background task. Returns job ID."""
        if self._update_task and not self._update_task.done():
            raise RuntimeError("Update already in progress")

        job_id = f"update-{target_version}-{int(time.time())}"
        self.state = UpdateState()
        self.state.target_version = target_version
        self.state.current_version = self._get_current_version()

        self._update_task = asyncio.create_task(
            self._run_update(download_url, target_version)
        )
        return job_id

    async def _run_update(self, download_url: str, target_version: str):
        """Execute the full download → extract → verify → swap → reload pipeline."""
        staging_dir = self.data_dir / "staging"
        backup_dir = self.data_dir / "backup"

        try:
            # Clean up any leftover staging/backup from a previous failed run
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            staging_dir.mkdir(parents=True)

            # Step 1: Download
            self.state.status = UpdateStatus.DOWNLOADING
            zip_path = staging_dir / "archive.zip"
            await self._download(download_url, zip_path)

            # Step 2: Extract
            self.state.status = UpdateStatus.EXTRACTING
            self.state.progress_pct = 0
            extract_dir = staging_dir / "extracted"
            extract_dir.mkdir()
            self._extract(zip_path, extract_dir)

            # Step 3: Verify
            self.state.status = UpdateStatus.VERIFYING
            manifest = self._load_and_verify(extract_dir)

            # Step 4: Swap
            self.state.status = UpdateStatus.SWAPPING
            self._swap_files(extract_dir)

            # Step 5: Reload
            self.state.status = UpdateStatus.RELOADING
            self.reload_fn(self.data_dir)

            # Step 6: Cleanup
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if backup_dir.exists():
                shutil.rmtree(backup_dir)

            self.state.status = UpdateStatus.COMPLETE
            self.state.progress_pct = 100

            # Invalidate cache so next check shows current version
            self._cached_check = None

            print(f"[database_updater] Update complete: {self.state.current_version} -> {target_version}")

        except Exception as e:
            self.state.status = UpdateStatus.FAILED
            self.state.error = str(e)
            print(f"[database_updater] Update failed: {e}")

            # Attempt rollback if backup exists
            if backup_dir.exists():
                try:
                    self._rollback(backup_dir)
                    self.reload_fn(self.data_dir)
                    print("[database_updater] Rollback successful")
                except Exception as rb_err:
                    print(f"[database_updater] Rollback also failed: {rb_err}")
                    self.state.error += f" (rollback failed: {rb_err})"

            # Clean up staging
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    async def _download(self, url: str, dest: Path):
        """Stream-download the zip with progress tracking."""
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url, timeout=600.0) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0

                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=1_048_576):  # 1 MB
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            self.state.progress_pct = int(downloaded / total * 100)

        print(f"[database_updater] Downloaded {downloaded / 1_000_000:.0f} MB to {dest}")

    def _extract(self, zip_path: Path, extract_dir: Path):
        """Extract zip archive."""
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(extract_dir)
        print(f"[database_updater] Extracted to {extract_dir}: {list(extract_dir.iterdir())}")

    def _load_and_verify(self, extract_dir: Path) -> dict:
        """Load manifest and verify all checksums."""
        manifest_path = extract_dir / "manifest.json"
        if not manifest_path.exists():
            raise ValueError("manifest.json missing from release archive")

        with open(manifest_path) as f:
            manifest = json.load(f)

        # Verify required files exist
        for name in REQUIRED_FILES:
            if not (extract_dir / name).exists():
                raise ValueError(f"Required file missing: {name}")

        self._verify_checksums(extract_dir, manifest)
        return manifest

    def _verify_checksums(self, extract_dir: Path, manifest: dict):
        """Verify SHA-256 checksums for all files listed in manifest."""
        checksums = manifest.get("checksums", {})
        for filename, expected in checksums.items():
            file_path = extract_dir / filename
            if not file_path.exists():
                raise ValueError(f"Checksum listed for {filename} but file missing")

            actual_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
            expected_hash = expected.replace("sha256:", "")
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Checksum mismatch for {filename}: "
                    f"expected {expected_hash[:16]}..., got {actual_hash[:16]}..."
                )

        print(f"[database_updater] All checksums verified ({len(checksums)} files)")

    def _swap_files(self, extract_dir: Path):
        """Move old DB files to backup, move new files into place."""
        backup_dir = self.data_dir / "backup"
        backup_dir.mkdir(exist_ok=True)

        # Backup current files (only release files, not stash_sense.db)
        for path in self.data_dir.iterdir():
            if path.name in RELEASE_FILES and path.is_file():
                shutil.move(str(path), str(backup_dir / path.name))

        # Move new files into place
        for path in extract_dir.iterdir():
            if path.is_file() and path.name in RELEASE_FILES:
                shutil.move(str(path), str(self.data_dir / path.name))

    def _rollback(self, backup_dir: Path):
        """Restore files from backup directory."""
        # Remove any new files that were moved in
        for path in self.data_dir.iterdir():
            if path.name in RELEASE_FILES and path.is_file():
                path.unlink()

        # Restore from backup
        for path in backup_dir.iterdir():
            if path.is_file():
                shutil.move(str(path), str(self.data_dir / path.name))

        shutil.rmtree(backup_dir, ignore_errors=True)
        print("[database_updater] Rolled back to previous version")
```

**Step 4: Run tests**

Run: `cd /home/carrot/code/stash-sense && .venv/bin/python -m pytest api/tests/test_database_updater.py -v`
Expected: All tests pass.

**Step 5: Commit**

```bash
git add api/database_updater.py api/tests/test_database_updater.py
git commit -m "feat: add database_updater module for GitHub release download and hot-swap"
```

---

### Task 3: Add API endpoints and request gating

Wire up the three new endpoints and the 503 gating dependency.

**Files:**
- Modify: `api/main.py`

**Step 1: Add the update_in_progress flag and dependency**

At the top of `main.py` (near the other globals around line 157):

```python
from database_updater import DatabaseUpdater

# Database updater (initialized in lifespan)
db_updater: Optional[DatabaseUpdater] = None
```

Add a dependency function after the app creation (after line 276):

```python
from fastapi import Depends

async def require_db_available():
    """Return 503 if a database update is currently swapping files."""
    if db_updater and db_updater.state.status in (
        UpdateStatus.SWAPPING, UpdateStatus.RELOADING,
    ):
        raise HTTPException(
            status_code=503,
            detail="Database update in progress",
            headers={"Retry-After": "10"},
        )
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")
```

**Step 2: Apply the dependency to identify endpoints**

Update each `/identify*` endpoint signature to use the dependency. For example:

```python
@app.post("/identify", response_model=IdentifyResponse)
async def identify_performers(request: IdentifyRequest, _=Depends(require_db_available)):
```

Apply to: `/identify`, `/identify/url`, `/identify/image`, `/identify/scene`, `/identify/gallery`.

**Step 3: Initialize updater in lifespan and add endpoints**

In `lifespan()`, after `reload_database()`:

```python
    global db_updater
    db_updater = DatabaseUpdater(
        data_dir=data_dir,
        reload_fn=reload_database,
    )
```

Add three new endpoints:

```python
from database_updater import DatabaseUpdater, UpdateStatus


class CheckUpdateResponse(BaseModel):
    current_version: str
    latest_version: Optional[str] = None
    update_available: bool
    release_name: Optional[str] = None
    download_url: Optional[str] = None
    download_size_mb: Optional[int] = None
    published_at: Optional[str] = None
    error: Optional[str] = None


class StartUpdateResponse(BaseModel):
    job_id: str
    status: str


class UpdateStatusResponse(BaseModel):
    status: str
    progress_pct: int = 0
    current_version: Optional[str] = None
    target_version: Optional[str] = None
    error: Optional[str] = None


@app.get("/database/check-update", response_model=CheckUpdateResponse)
async def check_database_update():
    """Check GitHub for a newer database release."""
    if db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    result = await db_updater.check_update()
    return CheckUpdateResponse(**result)


@app.post("/database/update", response_model=StartUpdateResponse)
async def start_database_update():
    """Trigger a database update. Downloads, verifies, and swaps the DB."""
    if db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")

    if db_updater._update_task and not db_updater._update_task.done():
        raise HTTPException(status_code=409, detail="Update already in progress")

    # Check for available update
    check = await db_updater.check_update()
    if not check.get("update_available"):
        raise HTTPException(status_code=400, detail="Already on latest version")

    job_id = await db_updater.start_update(
        download_url=check["download_url"],
        target_version=check["latest_version"],
    )
    return StartUpdateResponse(job_id=job_id, status="started")


@app.get("/database/update/status", response_model=UpdateStatusResponse)
async def get_update_status():
    """Get the current status of a database update."""
    if db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    return UpdateStatusResponse(**db_updater.get_status())
```

**Step 4: Verify it compiles**

Run: `cd /home/carrot/code/stash-sense/api && ../.venv/bin/python -c "from main import app; print('OK')"`

**Step 5: Commit**

```bash
git add api/main.py
git commit -m "feat: add /database/check-update, /database/update, /database/update/status endpoints with 503 gating"
```

---

### Task 4: Add plugin backend proxy modes

**Files:**
- Modify: `plugin/stash_sense_backend.py`

**Step 1: Add three new modes**

In `handle_recommendations()` function (or in the main mode dispatch at the top), add these cases. Since these are `db_` prefixed (not `rec_` or `fp_`), add them to the top-level `main()` dispatch (around line 42):

```python
    elif mode == "db_check_update":
        result = sidecar_get(sidecar_url, "/database/check-update")
    elif mode == "db_update":
        result = sidecar_post(sidecar_url, "/database/update", timeout=10)
    elif mode == "db_update_status":
        result = sidecar_get(sidecar_url, "/database/update/status")
```

Insert these before the `else: unknown mode` branch around line 48.

**Step 2: Commit**

```bash
git add plugin/stash_sense_backend.py
git commit -m "feat: add db_check_update, db_update, db_update_status proxy modes"
```

---

### Task 5: Add plugin UI for update badge and progress

**Files:**
- Modify: `plugin/stash-sense-recommendations.js` (update badge in dashboard)
- Modify: `plugin/stash-sense.css` (styles for update badge and modal)

**Step 1: Add API methods to RecommendationsAPI**

In the `RecommendationsAPI` object (at the top of `stash-sense-recommendations.js`), add:

```javascript
    async checkUpdate() {
      return SS.runPluginOperation('db_check_update', {});
    },

    async startUpdate() {
      return SS.runPluginOperation('db_update', {});
    },

    async getUpdateStatus() {
      return SS.runPluginOperation('db_update_status', {});
    },
```

**Step 2: Add update check to dashboard render**

In `renderDashboard()`, add a call to check for updates alongside the existing `Promise.all` (around line 253). Add `RecommendationsAPI.checkUpdate()` to the array:

```javascript
const [counts, sidecarStatus, analysisTypes, fpStatus, updateInfo] = await Promise.all([
    RecommendationsAPI.getCounts(),
    RecommendationsAPI.getSidecarStatus(),
    RecommendationsAPI.getAnalysisTypes(),
    RecommendationsAPI.getFingerprintStatus(),
    RecommendationsAPI.checkUpdate(),
]);
```

**Step 3: Render update badge next to DB Version**

Replace the DB Version stat block (around line 300-303) with a version that includes an update badge:

```javascript
<div class="ss-fp-stat">
  <span class="ss-fp-stat-value">${fpStatus.current_db_version || 'N/A'}</span>
  <span class="ss-fp-stat-label">DB Version</span>
  ${updateInfo.update_available ? `
    <div class="ss-update-badge" id="ss-update-badge">
      <span class="ss-update-badge-text">v${updateInfo.latest_version} available</span>
      <button class="ss-update-btn" id="ss-update-btn">Update</button>
    </div>
  ` : ''}
</div>
```

**Step 4: Add update button click handler**

After the dashboard HTML is rendered (where other event listeners are attached), add:

```javascript
const updateBtn = document.getElementById('ss-update-btn');
if (updateBtn) {
    updateBtn.addEventListener('click', async () => {
        if (!confirm(`Download ~${updateInfo.download_size_mb || '???'} MB and update database?\n\nFace recognition will be briefly unavailable during the swap.`)) {
            return;
        }

        updateBtn.disabled = true;
        updateBtn.textContent = 'Starting...';

        const startResult = await RecommendationsAPI.startUpdate();
        if (startResult.error) {
            alert(`Update failed to start: ${startResult.error}`);
            updateBtn.disabled = false;
            updateBtn.textContent = 'Update';
            return;
        }

        // Replace badge with progress display
        const badge = document.getElementById('ss-update-badge');
        badge.innerHTML = `<div class="ss-update-progress"><div class="ss-spinner ss-spinner-small"></div><span id="ss-update-progress-text">Starting download...</span></div>`;

        // Poll for progress
        const pollInterval = setInterval(async () => {
            const status = await RecommendationsAPI.getUpdateStatus();
            const textEl = document.getElementById('ss-update-progress-text');
            if (!textEl) { clearInterval(pollInterval); return; }

            const labels = {
                downloading: `Downloading... ${status.progress_pct || 0}%`,
                extracting: 'Extracting...',
                verifying: 'Verifying checksums...',
                swapping: 'Swapping database...',
                reloading: 'Reloading...',
                complete: `Updated to v${status.target_version}!`,
                failed: `Failed: ${status.error}`,
            };
            textEl.textContent = labels[status.status] || status.status;

            if (status.status === 'complete' || status.status === 'failed') {
                clearInterval(pollInterval);
                if (status.status === 'complete') {
                    badge.innerHTML = `<span class="ss-update-complete">Updated to v${status.target_version}</span>`;
                    // Refresh the dashboard after a brief pause
                    setTimeout(() => renderDashboard(container), 2000);
                }
            }
        }, 2000);
    });
}
```

**Step 5: Add CSS styles**

Append to `plugin/stash-sense.css`:

```css
/* Database update badge */
.ss-update-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-top: 4px;
}
.ss-update-badge-text {
  color: #f0ad4e;
  font-size: 11px;
  font-weight: 500;
}
.ss-update-btn {
  background: #f0ad4e;
  color: #1a1a2e;
  border: none;
  border-radius: 4px;
  padding: 2px 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
}
.ss-update-btn:hover {
  background: #ec971f;
}
.ss-update-btn:disabled {
  opacity: 0.6;
  cursor: not-allowed;
}
.ss-update-progress {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: #ccc;
}
.ss-spinner-small {
  width: 12px;
  height: 12px;
  border-width: 2px;
}
.ss-update-complete {
  color: #5cb85c;
  font-size: 11px;
  font-weight: 500;
}
```

**Step 6: Commit**

```bash
git add plugin/stash-sense-recommendations.js plugin/stash-sense.css
git commit -m "feat: add DB update badge and progress UI to recommendations dashboard"
```

---

### Task 6: Deploy and test end-to-end

**Files:** None new — deployment and verification.

**Step 1: Deploy plugin to Stash**

```bash
scp plugin/* root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/
```

**Step 2: Rebuild and push Docker image**

```bash
cd /home/carrot/code/stash-sense
docker build -t carrotwaxr/stash-sense:latest .
docker push carrotwaxr/stash-sense:latest
```

Then on unRAID, pull and restart the stash-sense container.

**Step 3: Verify check-update endpoint**

From local machine:
```bash
curl http://10.0.0.4:6960/database/check-update | jq .
```

Expected: Shows `current_version` and `latest_version` with `update_available: true` if v2026.02.14 has been published.

**Step 4: Test from plugin UI**

1. Open Stash UI → navigate to Recommendations page
2. Verify the "DB Version" stat shows an update badge with "v2026.02.14 available" and an "Update" button
3. Click "Update" → confirm the dialog
4. Watch progress: downloading → extracting → verifying → swapping → reloading → complete
5. Dashboard refreshes showing new version

**Step 5: Verify face recognition works after swap**

Navigate to any scene → click Identify. Should work normally with the new DB.

**Step 6: Commit any final tweaks**

```bash
git add -A
git commit -m "chore: final tweaks from e2e testing"
```

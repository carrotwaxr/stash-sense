# Sidecar Database Self-Update

**Date:** 2026-02-17
**Status:** Proposed

## Overview

The sidecar should be able to check for new published database releases on GitHub (`carrotwaxr/stash-sense-data`), download them, and swap the database in place without a container restart. The plugin UI triggers the check and update, and shows when an update is available.

## Key Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Discovery & trigger | Plugin-driven | Plugin checks for updates on demand, user clicks to trigger. Sidecar is passive. |
| Swap strategy | Brief unavailability (503) | 5-10s downtime during reload. Simple, safe, no doubled memory. |
| GitHub auth | None (public repo) | Unauthenticated GitHub API; 60 req/hr rate limit is plenty. |
| Download strategy | Stage → verify → swap | Download to `/data/staging/`, verify checksums, then move files. Rollback on failure. |
| Proxy pattern | Existing `runPluginOperation` | Consistent with all other plugin→sidecar calls. Bypasses CSP. |

## Sidecar API

### `GET /database/check-update`

Hits `https://api.github.com/repos/carrotwaxr/stash-sense-data/releases/latest` and compares the release tag against the current `manifest.json` version.

**Response:**

```json
{
  "current_version": "2026.02.12",
  "latest_version": "2026.02.14",
  "update_available": true,
  "release_name": "Database Release v2026.02.14",
  "download_url": "https://github.com/carrotwaxr/stash-sense-data/releases/download/v2026.02.14/stash-sense-data-v2026.02.14.zip",
  "download_size_mb": 850,
  "published_at": "2026-02-14T19:30:00Z"
}
```

If versions match: `update_available: false`, remaining fields omitted.

Caches the GitHub response for 10 minutes to avoid hammering the API on repeated checks.

### `POST /database/update`

Triggers the download-stage-swap pipeline. Returns immediately with a job ID. The work runs in a background `asyncio.Task`.

**Response:**

```json
{
  "job_id": "update-2026.02.14-1708200000",
  "status": "started"
}
```

**Error cases:**
- Returns `409 Conflict` if an update is already in progress.
- Returns `400 Bad Request` if no update is available (already on latest).

### `GET /database/update/status`

Plugin polls this to track progress.

**Response:**

```json
{
  "status": "downloading",
  "progress_pct": 45,
  "current_version": "2026.02.12",
  "target_version": "2026.02.14",
  "error": null
}
```

**Status values:** `idle` | `downloading` | `extracting` | `verifying` | `swapping` | `complete` | `failed`

When `status: "failed"`, `error` contains a human-readable message and the old database is still active (rollback succeeded or swap was never attempted).

## Update Pipeline (Sidecar)

All logic lives in a new `api/database_updater.py` module.

### Step 1: Download

- Stream the zip from `download_url` to `/data/staging/archive.zip`
- Track bytes downloaded for progress reporting
- Timeout: 10 minutes (zip is ~850 MB)

### Step 2: Extract

- Unzip to `/data/staging/extracted/`
- Verify expected files exist (at minimum: `performers.db`, `face_facenet.voy`, `face_arcface.voy`, `faces.json`, `performers.json`, `manifest.json`)

### Step 3: Verify

- Read `manifest.json` from extracted files
- Verify SHA-256 checksums for every file listed in `manifest.checksums`
- If any checksum fails, abort and report which file was corrupt

### Step 4: Swap

This is the critical section. Order of operations:

1. Set global `update_in_progress = True` — all face recognition endpoints return 503
2. Wait for any in-flight requests to drain (sleep 1-2s, or use a request counter)
3. Move current files from `/data/` to `/data/backup/` (atomic rename on same filesystem)
4. Move extracted files from `/data/staging/extracted/` to `/data/`
5. Preserve `stash_sense.db` — this is the recommendations database, NOT part of the release archive

### Step 5: Reload

Reload all global state in the same order as `lifespan()`:

1. Create new `DatabaseConfig` from data dir
2. Create new `FaceRecognizer` (loads Voyager indices + JSON mappings)
3. Load new `db_manifest` from `manifest.json`
4. Rebuild `MultiSignalMatcher` if applicable (re-reads body proportions, tattoo mappings)
5. Replace global references (`recognizer`, `db_manifest`, `multi_signal_matcher`, `tattoo_matcher`)
6. Clear `update_in_progress` flag — endpoints resume serving

### Step 6: Cleanup

- Delete `/data/backup/` (old DB files)
- Delete `/data/staging/` (zip + extracted temp)
- Log the version transition

### Error Handling & Rollback

| Scenario | Handling |
|---|---|
| Download interrupted | Delete partial zip, report `failed` |
| Checksum mismatch | Delete staging, report which file failed |
| Swap fails (move error) | Restore from `/data/backup/`, clear `update_in_progress` |
| Reload fails (corrupt index) | Restore from `/data/backup/`, reload old DB, clear flag |
| Disk full | Check available space before download (need ~2.5x DB size for staging + backup) |
| GitHub API unreachable | `check-update` returns current version with `update_available: false` and an `error` field |

## Request Gating

A lightweight mechanism to return 503 during the swap window:

```python
# In database_updater.py
update_in_progress: bool = False

# Dependency for face recognition endpoints
async def require_db_available():
    if update_in_progress:
        raise HTTPException(
            status_code=503,
            detail="Database update in progress",
            headers={"Retry-After": "10"},
        )
```

Applied as a FastAPI dependency on `/identify/*` routes only. Other endpoints (`/health`, `/database/*`, `/recommendations/*`) remain available during the swap.

## Plugin Backend (Python)

Two new modes in `stash_sense_backend.py`:

```python
elif mode == "db_check_update":
    return sidecar_get("/database/check-update")

elif mode == "db_update":
    return sidecar_post("/database/update")

elif mode == "db_update_status":
    return sidecar_get("/database/update/status")
```

## Plugin UI (JavaScript)

### Update Badge

On the **Recommendations** dashboard (or Stash Sense settings panel), add next to the existing "DB Version" stat:

- On page load, call `db_check_update` via `runPluginOperation`
- If `update_available`, show a small badge: **"v2026.02.14 available"** with an **"Update"** button
- If no update, show the current version with a subtle "Up to date" label

### Update Flow

1. User clicks **"Update Database"**
2. Confirmation: "Download ~850 MB and update? Face recognition will be briefly unavailable."
3. On confirm, call `db_update` mode
4. Show a progress modal polling `db_update_status` every 2 seconds:
   - "Downloading... 45%"
   - "Extracting..."
   - "Verifying checksums..."
   - "Reloading database..."
   - "Complete! Now running v2026.02.14"
5. On failure, show error message and "Database unchanged" reassurance

### Placement

The update check runs when the user navigates to the Recommendations page (piggyback on the existing `fpStatus` fetch). No background polling — only checked when the user is looking.

## Disk Space Considerations

On unRAID, `/data` maps to `/mnt/nvme_cache/appdata/stash-sense/data/`. Current DB is ~1 GB. During update, peak usage is ~2.5 GB (current + staging zip + extracted). After cleanup, back to ~1 GB.

The pre-download space check should verify at least `download_size_mb * 2.5` free space on the filesystem before starting.

## File Layout During Update

```
/data/
  performers.db            ← current (live)
  face_facenet.voy
  face_arcface.voy
  faces.json
  performers.json
  manifest.json
  tattoo_embeddings.voy    ← optional
  tattoo_embeddings.json   ← optional
  face_adaface.voy         ← optional
  stash_sense.db           ← recommendations DB, never touched
  staging/
    archive.zip            ← downloaded release
    extracted/
      performers.db        ← new version
      face_facenet.voy
      ...
  backup/                  ← old files during swap (deleted after success)
    performers.db
    face_facenet.voy
    ...
```

## Future Extensions

- **Release notes display**: Show the GitHub release body (stats, changelog) in the update modal
- **Selective signal updates**: If only tattoo data changed, skip reloading face indices
- **Fingerprint invalidation**: After a DB update with a new version, mark stale fingerprints for regeneration (already handled by manifest version → `db_version` column)
- **Periodic background check**: Optional setting to auto-check every N hours and show a passive notification

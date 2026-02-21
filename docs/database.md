# Database & Updates

## Where the Data Comes From

The face recognition database is built and published in the [stash-sense-data](https://github.com/carrotwaxr/stash-sense-data) repository. It contains performer metadata and face embeddings sourced from multiple stash-box endpoints:

| Source | Type | Performers |
|--------|------|------------|
| [StashDB](https://stashdb.org) | Primary | ~100,000+ |
| [FansDB](https://fansdb.cc) | Supplementary | Varies |
| [ThePornDB](https://theporndb.net) | Supplementary | ~10,000 |
| [PMVStash](https://pmvstash.org) | Supplementary | ~6,500 |
| [JAVStash](https://javstash.org) | Supplementary | ~21,700 |

Additional sources include Babepedia, Freeones, IAFD, and other public performer databases via hybrid scraping.

**Current stats (v2026.02.18):** 108,001 performers, 366,794 face embeddings.

The database is built using a private training pipeline that:

1. Downloads performer images from each source
2. Detects and aligns faces using RetinaFace with 5-point similarity transform
3. Generates embeddings using FaceNet512 and ArcFace models with flip-averaging
4. Builds Voyager vector indices for fast cosine-distance search

Database releases are published as zip files on the [stash-sense-data releases page](https://github.com/carrotwaxr/stash-sense-data/releases).

---

## Database Files

The data directory contains:

| File | Size | Purpose |
|------|------|---------|
| `performers.db` | ~210 MB | SQLite database with performer metadata and stash-box IDs |
| `face_facenet.voy` | ~550 MB | FaceNet512 embedding Voyager index |
| `face_arcface.voy` | ~550 MB | ArcFace embedding Voyager index |
| `face_adaface.voy` | ~550 MB | AdaFace IR-101 embedding Voyager index |
| `tattoo_embeddings.voy` | varies | Tattoo embedding Voyager index |
| `faces.json` | ~15 MB | Face-to-performer mapping |
| `performers.json` | ~10 MB | Performer lookup data |
| `tattoo_embeddings.json` | varies | Tattoo-to-performer mapping |
| `manifest.json` | <1 KB | Version, checksums, and build metadata |

---

## Two-Database Design

Stash Sense uses two separate databases:

- **`performers.db`** + Voyager indices — The distributable face recognition data. Updated via stash-sense-data releases either in-app (Settings tab) or manually. Contains no user-specific information.
- **`stash_sense.db`** (read-write, user-local, schema version 9) — Your recommendation history, dismissed items, analysis watermarks, upstream snapshots, scene fingerprints, operation queue, and settings overrides.

This separation means database updates never touch your personal data, and the distributable database contains nothing specific to your library.

### stash_sense.db Tables

| Table | Purpose |
|-------|---------|
| `recommendations` | Core recommendations (face matches, upstream changes, duplicates) |
| `dismissed_recommendations` | Permanently or temporarily dismissed items |
| `analysis_watermarks` | Tracks last-completed timestamps and cursors for incremental analysis runs. Includes a `logic_version` column to detect when comparison logic changes and a full re-analysis is needed. |
| `upstream_snapshots` | Cached upstream performer data for 3-way diff engine |
| `scene_fingerprints` | Per-scene face fingerprints for duplicate detection |
| `duplicate_candidates` | Candidate duplicate scene pairs from fingerprint matching |
| `job_queue` | Persistent operation queue with priority, status (`queued`/`running`/`completed`/`failed`/`cancelled`), cursor-based resumption, and progress tracking (`items_processed`/`items_total`) |
| `job_schedules` | Configurable recurring job schedules with enable/disable toggle, interval (hours), priority, and next-run tracking |
| `user_settings` | User setting overrides (key-value with JSON-encoded values) |

---

## Checking for Updates

### Via the Plugin UI

1. Open the **Settings** tab in the Stash Sense plugin
2. The **Database** section shows your current version and checks for updates automatically
3. If an update is available, click **Update**
4. Progress is shown in real-time (download, extract, verify, swap, reload)

### Via the API

**Check for updates:**

```bash
curl http://localhost:6960/database/check-update
```

```json
{
  "current_version": "2026.02.12",
  "latest_version": "2026.02.18",
  "update_available": true,
  "release_name": "Database Release v2026.02.18",
  "download_size_mb": 1478,
  "published_at": "2026-02-18T01:09:34Z"
}
```

**Start an update:**

```bash
curl -X POST http://localhost:6960/database/update
```

**Check update progress:**

```bash
curl http://localhost:6960/database/update/status
```

---

## Manual Database Update

If you prefer to update manually (e.g., on a system without internet access):

1. Download the latest release zip from [stash-sense-data releases](https://github.com/carrotwaxr/stash-sense-data/releases/latest)
2. Stop the container: `docker stop stash-sense`
3. Extract the zip into your data directory, replacing existing files
4. Start the container: `docker start stash-sense`

The `stash_sense.db` file in your data directory is your personal data — do not delete it during a manual update.

---

## Update Safety

- A timestamped backup is created before every update
- SHA-256 checksums are verified against `manifest.json`
- On failure, the sidecar automatically rolls back to the backup
- Face identification endpoints return 503 briefly during the swap (~5-10 seconds)
- All other API endpoints remain available during updates
- `stash_sense.db` (your recommendations, settings, history) is never touched

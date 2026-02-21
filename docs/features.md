# Features

## Face Recognition

Stash Sense identifies performers in your scenes by analyzing sprite sheets — the thumbnail strips that Stash generates for timeline scrubbing. No video decoding required.

**How it works:**

1. Click **Identify Performers** on any scene page in Stash
2. The sidecar extracts frames from the sprite sheet and detects faces using RetinaFace
3. Each face is aligned and embedded using two models (FaceNet512 + ArcFace) with flip-averaging for stability
4. Embeddings are searched against Voyager vector indices containing 366,000+ face references
5. Results are clustered by person — the same performer appearing across multiple frames is grouped together
6. Matched performers are shown with confidence scores and one-click tagging

**Matching modes:**

- **Clustered frequency matching** (default) — Groups detected faces by person using cosine distance, then frequency-matches within each cluster. Multi-frame appearances boost confidence.
- **Tagged-performer boost** — Performers already tagged on the scene receive a small distance bonus (+0.03), reducing false negatives for known cast.

**Performance:** On a GTX 1080 (8GB VRAM), a typical scene (60 frames) processes in ~5 seconds.

---

## Gallery & Image Identification

Extends face recognition to gallery images, which are typically higher quality and better-framed than video frames.

- Single image or full gallery identification
- Results grouped by performer across images with best/average distance
- Fingerprint caching avoids re-processing on subsequent requests

---

## Duplicate Scene Detection

Finds duplicate scenes that Stash's built-in phash matching misses — different intros/outros, trimming, aspect ratio changes, or watermarks.

**Detection signals:**

| Signal | Max Confidence | How it works |
|--------|---------------|--------------|
| Stash-box ID match | 100% | Authoritative — same scene on stash-box |
| Face fingerprint similarity | 85% | Same performers in same ratios (robust to trimming) |
| Metadata overlap | 60% | Same studio + same performers |

Signals combine with diminishing returns (`primary + secondary x 0.3`) to prevent false confidence inflation.

**Scalability:** Uses candidate generation (SQL joins + inverted indices) to produce O(n) candidate pairs instead of O(n^2) brute-force comparisons. Handles 15,000+ scene libraries.

---

## Upstream Performer Sync

Detects metadata changes on stash-box endpoints (StashDB, FansDB, etc.) and presents per-field merge controls to keep your local Stash library current.

**3-way diff engine:** Compares three states for each field:

- **Upstream** — current stash-box value
- **Local** — current value in your Stash
- **Snapshot** — last-seen upstream value (stored locally)

This distinguishes intentional local differences from actual upstream changes. If you've deliberately set a different name locally, it won't keep suggesting the upstream name on every sync.

**Per-field merge controls:**

| Field type | Options |
|------------|---------|
| Name fields | Keep local / Accept upstream / Demote to alias / Add as alias / Custom |
| Aliases | Union checkboxes (pick which aliases to keep) |
| Simple fields | Keep local / Accept upstream / Custom |

**Dismissal:** Soft dismiss resurfaces if upstream changes again. Permanent dismiss skips entirely.

---

## Upstream Studio Sync

Same pattern as performer sync, applied to studios. Detects changes on stash-box endpoints and presents merge controls to keep your local studio metadata current.

**Tracked fields:**

- **Name** — studio display name
- **URL** — studio website
- **Parent studio** — hierarchical studio relationships

**Parent studio comparison** uses stash-box IDs rather than name matching, so renamed parent studios are still correctly identified. The 3-way diff engine (upstream / local / snapshot) works identically to performer sync — intentional local differences are preserved across syncs.

**Merge controls** mirror the performer sync UI: keep local, accept upstream, or dismiss (soft or permanent).

---

## Upstream Scene Sync

Detects metadata changes for scenes linked to stash-box endpoints and provides per-field merge controls for updating your local library.

**Tracked field categories:**

| Category | Fields |
|----------|--------|
| Core metadata | Title, details, date, director, code, URLs |
| Studio assignment | Studio linked to the scene |
| Performer lineup | Added/removed performers, alias changes |
| Tags | Tag set additions and removals |

**3-way diff with snapshots:** Like performer and studio sync, scene sync stores a snapshot of the last-seen upstream state. This distinguishes genuine upstream edits from intentional local differences — if you have customized a scene title locally, it will not be flagged on every sync cycle.

**Per-field merge controls** allow you to accept, reject, or customize each change independently. Performer lineup changes show which performers were added or removed upstream, and alias changes are surfaced when an upstream performer's credited name differs from the local alias.

---

## Recommendations Dashboard

A unified plugin page showing all actionable suggestions:

- **Duplicate scenes** — with confidence scores and signal breakdown
- **Unidentified scenes** — scenes with sprite sheets but no face recognition results
- **Missing stash-box links** — performers in your library without stash-box IDs
- **Upstream updates** — per-field changes from stash-box endpoints

Each recommendation type runs as a background analyzer with incremental watermarking — only items modified since the last run are re-processed, keeping subsequent runs fast.

---

## Operation Queue

An async job queue with resource-aware scheduling that coordinates all background work — analysis runs, sync operations, and database updates.

**Priority levels:**

| Priority | Use case |
|----------|----------|
| Critical | Database updates, rollback operations |
| High | User-initiated analysis (e.g., "Identify Performers" click) |
| Normal | Scheduled analysis runs |
| Low | Background housekeeping |

**Resource slots** prevent contention by ensuring only compatible jobs run concurrently:

| Slot | What uses it | Why limited |
|------|-------------|-------------|
| `GPU` | Face detection, embedding inference | VRAM is finite |
| `CPU_HEAVY` | Clustering, fingerprint computation | Avoid saturating cores |
| `NETWORK` | Stash-box API calls, database downloads | Rate limits, bandwidth |
| `LIGHT` | Metadata reads, cache checks | Runs alongside anything |

Jobs declare which resource slot they need. The scheduler runs all compatible jobs in parallel but serializes jobs that compete for the same resource.

**Persistence:** The queue survives sidecar restarts. In-progress jobs are re-queued on startup.

**Configurable schedules:** Recurring analysis jobs (upstream sync, duplicate detection) can be scheduled via the plugin UI with cron-style intervals. Manual triggers are always available.

**Monitoring:** The plugin UI shows active, queued, and completed jobs with status, duration, and error details.

---

## Model Management

On-demand download and lifecycle management for optional ONNX models that are not bundled in the Docker image.

**Supported optional models:**

| Model | Purpose | Size |
|-------|---------|------|
| Tattoo detection | Identifies tattoos for performer matching | ~50 MB |

**Validation:** Each model download is verified against a SHA256 checksum before activation. If a model file becomes corrupted (disk error, partial write), the sidecar detects the mismatch and marks the model as `corrupted`.

**Status tracking:**

| Status | Meaning |
|--------|---------|
| `not_installed` | Model available for download but not present locally |
| `installed` | Model downloaded, verified, and ready for use |
| `corrupted` | Checksum mismatch — re-download required |

**Download progress UI:** The plugin Settings tab shows real-time download progress for each model with a progress bar. Models can be installed or removed at any time without restarting the sidecar.

---

## Settings System

The sidecar auto-detects your hardware at startup (GPU model, VRAM, CPU cores, RAM) and classifies it into a performance tier:

| Tier | Criteria | Effect |
|------|----------|--------|
| `gpu-high` | CUDA + 4GB+ VRAM | Full batch sizes, 8 concurrent workers |
| `gpu-low` | CUDA + <4GB VRAM | Reduced batch sizes, 6 concurrent workers |
| `cpu` | No CUDA | Minimal batching, 2 concurrent workers |

All settings can be adjusted in the plugin's **Settings** tab. Changes are stored in the local database and override tier defaults. Reset any setting to revert to the auto-detected default.

See [Settings Reference](settings-system.md) for the full list of configurable options.

---

## Database Self-Update

The face recognition database is distributed separately from the Docker image, allowing updates without rebuilding or restarting the container.

**Update process:**

1. Check for updates via the Settings UI or the `/database/check-update` API
2. The sidecar compares your local `manifest.json` version against the latest [stash-sense-data](https://github.com/carrotwaxr/stash-sense-data/releases) release
3. Click Update — the sidecar downloads, verifies checksums, swaps files, and reloads
4. A backup of the previous database is created automatically
5. On failure, the sidecar rolls back to the backup

During an update, face identification endpoints return 503 for a brief period (5-10 seconds). All other endpoints remain available. Your recommendation history and settings are never affected by database updates.

See [Database & Updates](database.md) for details on the data source and update procedures.

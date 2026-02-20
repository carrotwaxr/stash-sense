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

## Recommendations Dashboard

A unified plugin page showing all actionable suggestions:

- **Duplicate scenes** — with confidence scores and signal breakdown
- **Unidentified scenes** — scenes with sprite sheets but no face recognition results
- **Missing stash-box links** — performers in your library without stash-box IDs
- **Upstream updates** — per-field changes from stash-box endpoints

Each recommendation type runs as a background analyzer with incremental watermarking — only items modified since the last run are re-processed, keeping subsequent runs fast.

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

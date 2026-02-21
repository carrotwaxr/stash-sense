# Stash Sense Architecture

Reference for how the system works and why key decisions were made.

---

## System Overview

Stash Sense is a two-component system: a **sidecar API** (Python/FastAPI) that runs face recognition and analysis, and a **plugin** (JS/CSS/Python) injected into the Stash web UI.

**Deployment model:** The sidecar runs as a Docker container alongside Stash (default port 6960). The plugin backend proxies all requests from Stash to the sidecar, bypassing browser CSP restrictions. The sidecar URL is configurable in Stash plugin settings.

**Two-database design:**
- `performers.db` — Read-only, distributable. Contains performer metadata, face references, and stash-box IDs sourced from the private trainer repo. Updated via GitHub Releases.
- `stash_sense.db` — Read-write, user-local. Contains recommendations, dismissed items, analysis watermarks, upstream snapshots, and scene fingerprints. Persists across face DB updates.

This separation means face DB updates never touch user state, and the distributable database contains no user-specific data.

**Docker image:** Two-stage build on `nvidia/cuda:12.4.0-runtime-ubuntu22.04`. ONNX models (FaceNet512 + ArcFace, ~220MB) are baked into the image. Data files (Voyager indices ~1.1GB, performers.db ~210MB) are volume-mounted from NVMe for independent updates. PyTorch, TensorFlow, and DeepFace are stripped — only needed by the trainer's offline scripts.

---

## Face Recognition

**Models:** RetinaFace (buffalo_sc, ONNX GPU) for detection, FaceNet512 + ArcFace for embeddings. Both embedding models use flip-averaging (original + horizontally flipped face, averaged) for more stable representations.

**Pipeline (3-phase batch):**
1. **Extract** — ffmpeg seeks N frames from scene, 8 concurrent workers
2. **Detect** — RetinaFace processes all frames, InsightFace `norm_crop` aligns faces via 5-point similarity transform
3. **Embed + Match** — All faces batched through both ONNX models in 2 calls (1 per model), then searched against Voyager indices (cosine space)

This replaced a per-face sequential pipeline: 53.6s → 4.8s on 1080p/60 frames/64 faces (11x speedup, embedding step 138x faster via GPU batching).

**Matching:** `clustered_frequency_matching` is the default. Faces are clustered by person using cosine distance (threshold 0.6) on concatenated FaceNet+ArcFace embeddings, then frequency-matched within each cluster. Multi-frame appearances boost confidence. Tagged-performer boost (+0.03) applied when scene already has performers tagged.

**Tuned defaults** (from 20-scene stratified benchmark): fusion weights 0.5/0.5 FaceNet/ArcFace, max_distance 0.5, num_frames 60, min_unique_frames 2. Baseline: ~42% scene accuracy, ~33% precision. Biggest improvement levers are num_frames and database coverage.

---

## Recommendations Engine

**BaseAnalyzer pattern:** Each recommendation type (duplicate scenes, missing stash-box links, upstream changes, etc.) is a pluggable analyzer with a consistent interface for running, progress tracking, and creating recommendations. The scheduler runs analyzers as background jobs.

**Incremental watermarking:** Most analyzers track a `last_watermark` timestamp or cursor and only process items modified since the last run. This keeps incremental runs fast (~seconds) versus full scans (~73 min for 7K+ performers at 5 req/s rate limit).

**Polymorphic recommendations:** A single `recommendations` table handles all entity types (duplicate_performer, duplicate_scene, unidentified_scene, missing_stashbox_link, stashbox_updates) with type-specific JSON payloads. Users review via a dashboard or contextual actions on affected pages.

---

## Duplicate Scene Detection

Stash's built-in phash matching fails with different intros/outros, trimming, aspect ratio changes, or watermarks. Stash Sense uses multi-signal detection with a confidence hierarchy.

**Signal hierarchy with caps:**
- Stash-box ID match = 100% (authoritative)
- Face fingerprint similarity = up to 85%
- Metadata overlap = up to 60%
- No single signal other than stash-box ID reaches 100%

**Face fingerprints:** Per-scene records of which performers appeared, how many times detected, and proportion of total faces. Same performers appearing in the same ratios is robust to trimming and length differences.

**Diminishing returns scoring:** `primary + secondary × 0.3` — the second signal adds less than the first, preventing false confidence inflation.

**Scalable candidate generation:** Direct O(n²) comparison crashes on 15K+ scenes. Instead, a two-phase approach:
1. **Candidate generation** — SQL joins and inverted indices produce O(n) candidate pairs from 3 sources: stash-box ID grouping, face fingerprint self-join (shared performers), and metadata intersection (same studio AND performer). Yields 10K-50K candidates vs 112M brute-force comparisons.
2. **Sequential scoring** — Cursor-based pagination (`id > last_id LIMIT 100`) iterates candidates, scores each, writes recommendations immediately. Constant memory throughout.

---

## Multi-Signal Identification

Face recognition alone achieves ~50-60% accuracy. Additional biometric signals improve identification when faces are unclear or absent.

**Late fusion architecture:** Each signal is searched independently; results are combined via multiplicative scoring. Missing signals are handled gracefully (no tattoo visible → neutral multiplier).

**Signals:**
- **Face** (primary) — Voyager index search, top-K candidates
- **Body proportions** — MediaPipe pose estimation extracts shoulder-hip ratio, leg-torso ratio, arm-span-height ratio. Compared against database values with tolerance ~0.15. Returns penalty multiplier (1.0 compatible, 0.3 severe mismatch).
- **Tattoo presence** — YOLO-based detection. Query shows tattoos but candidate has none → 0.7x penalty. Matching tattoo locations → 1.15x boost.

**Fusion:** `final_score = face_score × body_multiplier × tattoo_multiplier`

Tattoo embedding matching (semantic similarity of tattoo designs via EfficientNet-B0 1280-dim vectors) exists in the trainer but is not yet deployed to the sidecar.

---

## Upstream Performer Sync

Detects stash-box field changes and presents per-field merge controls to sync local Stash against upstream updates.

**3-way diff engine:** Compares upstream (current stash-box state) vs local (current Stash state) vs snapshot (last-seen upstream state stored in `upstream_snapshots` table). This distinguishes intentional local differences from actual upstream changes.

**Merge controls:** Name fields get 5 options (keep/accept/demote-to-alias/add-as-alias/custom). Aliases use union checkboxes. Simple fields use 3-option radio (keep/accept/custom). Per-field monitoring can be disabled per endpoint via `upstream_field_config`.

**Field mapping complexity:** Stash-box uses separate fields (cup_size, band_size, waist_size, hip_size, career_start_year, career_end_year) that Stash combines into compound strings (measurements "38F-24-35", career_length "2015-2023"). Translation handled in `recommendations_router.py:update_performer_fields()`.

**Dismissal model:** Soft dismiss (permanent=0) resurfaces if upstream changes again. Permanent dismiss skips entirely.

---

## Upstream Studio & Tag Sync

Both follow the same `BaseUpstreamAnalyzer` pattern as performer sync: 3-way diff with stored snapshots, incremental watermarking, and per-field enable/disable via `upstream_field_config`.

**Studios:** Compared fields are name, URL, and parent studio. Parent studio comparison resolves the parent's stash-box ID for the current endpoint rather than comparing local numeric IDs, which would always differ from upstream UUIDs. Falls back to local ID only when the parent isn't linked to the endpoint.

**Tags:** Compared fields are name, description, and aliases. Tag comparison is set-based (aliases are compared as unordered sets, not ordered lists).

---

## Upstream Scene Sync

Detects changes in scene metadata, studio assignment, performer lineup, and tags between local Stash and stash-box endpoints. Uses the same `BaseUpstreamAnalyzer` infrastructure but with relational diffing rather than simple field comparison.

**Relational diffing:** Scene changes go beyond flat field comparison. Performer changes detect added/removed performers with alias detection (performer "as" credits). Tag changes are set-based (added/removed). Studio changes resolve stash-box IDs for comparison, same as the studio analyzer. Simple fields (title, date, URL, details) use standard 3-way diff.

**Change detection:** A scene is flagged only when at least one category has actual differences — simple field changes, a studio change, performer additions/removals/alias changes, or tag additions/removals. All entity IDs (performers, studios, tags) are resolved to stash-box IDs scoped to the current endpoint to avoid cross-endpoint mismatches.

---

## Logic Versioning

Each upstream analyzer has a `logic_version` class attribute (e.g., performer analyzer v2 removed the favorite field from comparison, studio analyzer v2 switched to stash-box IDs for parent studio). The version is stored in the `analysis_watermarks.logic_version` column.

When the version changes, the next analysis run detects the mismatch and clears all stale snapshots and watermarks for that entity type, forcing a full re-analysis. Recommendations generated under old logic that no longer produce differences are automatically resolved. This acts as a migration mechanism — when comparison logic changes (new fields, different normalization, ID resolution fixes), bumping `logic_version` ensures all entities are re-evaluated cleanly without manual intervention.

---

## Gallery & Image Identification

Extends face recognition from scenes (frame extraction) to gallery images (typically higher quality, better-framed).

**Endpoints:** `/identify/image` (single image) and `/identify/gallery` (all images in a gallery).

**Aggregation:** Unlike scenes, gallery images are independent — no temporal clustering needed. Results are grouped by performer across images: best distance, average distance, image count. Filtering: 2+ appearances OR single match with distance < 0.4.

**Fingerprint caching:** `image_fingerprints` and `image_fingerprint_faces` tables store per-image results, avoiding re-processing on subsequent requests.

---

## Database Self-Update

The sidecar checks for new face recognition database releases on GitHub and performs hot-swap updates without container restart.

**Pipeline:** Download → extract → verify checksums against `manifest.json` → swap → reload global state. Files are staged in `/data/staging/`, old files backed up to `/data/backup/`.

**Availability during update:** Brief 503 responses (5-10s) on `/identify/*` endpoints only. An `update_in_progress` flag gates these endpoints. Non-identification endpoints remain available.

**Reload order:** New DatabaseConfig → new FaceRecognizer (loads Voyager indices) → load manifest → rebuild MultiSignalMatcher → replace global references → clear flag.

**Safety:** Disk space check (2.5x current DB size) before download. Rollback from backup if reload fails. `stash_sense.db` (recommendations) is never touched during updates.

---

## Operation Queue

Persistent job queue backed by SQLite (`job_queue` and `job_schedules` tables) that manages all background analysis and maintenance work.

**Job lifecycle:** PENDING -> QUEUED -> RUNNING -> COMPLETED/FAILED/CANCELLED. Jobs are submitted with a type, priority, and optional cursor. The `QueueManager` singleton orchestrates dispatch.

**Resource-aware scheduling:** Each job type declares a resource requirement. Slots limit concurrency per resource type: GPU (1), CPU_HEAVY (1), NETWORK (2), LIGHT (3). A job is dispatched only when a slot is available for its resource type, preventing GPU contention or network flooding.

**Priority levels:** CRITICAL (0), HIGH (10), NORMAL (50), LOW (100). Lower values run first. Most analyzers default to NORMAL; database self-update uses HIGH; StashBox queries use LOW.

**Incremental job support:** Jobs can persist a cursor (e.g., last-processed ID or timestamp) so they resume from where they left off on subsequent runs. Combined with the watermark system, this keeps incremental analysis fast.

**Schedules:** `job_schedules` table stores cron-like schedules per job type, allowing automatic periodic runs without external schedulers.

---

## Resource Management

`ResourceManager` handles lazy loading and idle unloading of heavy resources to free GPU memory and RAM when not in use.

**Resources managed:** Face recognition data (Voyager indices + metadata), tattoo detection models, body proportion indices. Each is registered with a loader and unloader function.

**Lifecycle:** Resources are loaded on first `require()` call and cached. A background task periodically calls `check_idle()`. Resources unused for 1800 seconds (30 minutes) are unloaded — their unloader runs, data references are cleared, and `gc.collect()` reclaims memory. Thread-safe via internal locking.

---

## Model Management

Optional ONNX models (tattoo detection) are not baked into the Docker image. They are downloaded on-demand from GitHub Releases.

**Download pipeline:** `ModelManager` tracks each model's expected SHA256 hash. Downloads are triggered via the `/models/download/{model_name}` endpoint, run in the background, and validated against the hash before being placed in the models directory. The `/models/status` endpoint reports per-model installation state.

**Capabilities API:** The `/capabilities` endpoint inspects which data files and model files are present on disk and reports which features are available (upstream sync, duplicate detection, identification, tattoo signal). The plugin UI uses this to show/hide feature-specific controls and offer model download prompts for missing capabilities.

---

## Plugin Distribution

The plugin is distributed via a Stash-compatible package index hosted on GitHub Pages.

**Build:** `build_plugin_index.sh` extracts metadata from `plugin/stash-sense.yml`, creates a zip of the plugin directory, generates an `index.yml` with version info (plugin version + git commit hash), and writes both to an output directory.

**Deployment:** A GitHub Actions docs workflow triggers on version tag pushes and plugin file changes. It runs the build script and deploys the output to the `gh-pages` branch via `ghp-import`. The resulting index is available at `https://carrotwaxr.github.io/stash-sense/plugin/index.yml`.

**Usage:** Users add this URL as a plugin source in Stash settings (Settings > Plugins > Available Plugins). Stash checks the index for updates and installs/updates the plugin automatically.

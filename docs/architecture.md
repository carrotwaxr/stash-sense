# Stash Sense Architecture

Reference for how the system works and why key decisions were made.

---

## System Overview

Stash Sense is a two-component system: a **sidecar API** (Python/FastAPI) that runs face recognition and analysis, and a **plugin** (JS/CSS/Python) injected into the Stash web UI.

**Deployment model:** The sidecar runs as a Docker container on unRAID (port 6960) alongside Stash. The plugin backend proxies all requests from Stash to the sidecar, bypassing browser CSP restrictions. The sidecar URL is configurable in Stash plugin settings.

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

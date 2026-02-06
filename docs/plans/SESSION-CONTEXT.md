# Stash Sense: Session Context

**Last Updated:** 2026-02-06
**Status:** Phase 2 & 3 Complete, Multi-Signal Implemented, Benchmark Framework Complete

---

## Project Overview

**Stash Sense** is a two-repo project for AI-powered Stash library organization:

| Repository | Purpose | Visibility |
|------------|---------|------------|
| **stash-sense** (this repo) | Read-only sidecar API + Stash plugin | Public |
| **stash-sense-trainer** | Database builder + enrichment scrapers | Private |

The split was completed on 2026-01-30 to keep scraping code private while enabling open-sourcing the inference/UI components.

---

## Current State

### What's Working

| Component | Status | Location |
|-----------|--------|----------|
| **Face Recognition API** | Production-ready | `/api/` |
| **Stash Plugin** | Production-ready | `/plugin/` |
| **Database Reader** | Working (read-only) | `/api/database_reader.py` |
| **Recommendations Engine** | **Complete** (~2,700 lines) | `/api/recommendations_*.py`, `/api/analyzers/` |
| **Duplicate Detection** | **Complete** | `/api/duplicate_detection/`, `/api/analyzers/duplicate_scenes.py` |
| **Recommendations Dashboard** | **Complete** | `/plugin/stash-sense-recommendations.js` |

### What's Building (in trainer repo)

The trainer Docker is currently running with **5 new sources enabled**, bug fixes, and improved filtering. It's actively adding faces to the database. The trainer now supports a `db optimize` command that can:
- Prune performers with zero faces from the database and index
- Filter by minimum face count (`--min-faces N`) to reduce noise
- This should significantly reduce false positives in stash-sense identification

Sources include:
- Multi-source scraping (13+ stash-boxes + reference sites)
- Face detection and embedding generation
- Quality filtering and trust levels

---

## Architecture After Split

```
User's System
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Stash                              Stash Sense Sidecar                 │
│  ┌──────────────┐                  ┌────────────────────────────────┐  │
│  │  Plugin UI   │  ───HTTP───►     │  FastAPI (read-only)           │  │
│  │  (JavaScript)│                  │  ├─ /identify/scene            │  │
│  │              │  ◄──JSON───      │  ├─ /identify                  │  │
│  └──────────────┘                  │  ├─ /recommendations           │  │
│                                    │  └─ /health                    │  │
│                                    │                                │  │
│                                    │  Database Volume (read-only)   │  │
│                                    │  ├─ performers.db              │  │
│                                    │  ├─ face_*.voy                 │  │
│                                    │  └─ manifest.json              │  │
│                                    └────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘

Maintainer's System (private)
┌─────────────────────────────────────────────────────────────────────────┐
│  Stash Sense Trainer                                                    │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  Database Builder                                                   │ │
│  │  ├─ Scrape StashDB, ThePornDB, JAVStash, etc.                      │ │
│  │  ├─ Scrape reference sites (Babepedia, IAFD, FreeOnes, etc.)       │ │
│  │  ├─ Face detection + embedding generation                          │ │
│  │  └─ Quality filtering + trust scoring                              │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                            │                                            │
│                            ▼                                            │
│                    Publish to releases                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Plan Documents Status

### Completed/Stable Designs

| Document | Purpose | Status |
|----------|---------|--------|
| [face-recognition-system-design.md](docs/plans/2026-01-24-face-recognition-system-design.md) | Core face recognition pipeline | **Phase 3 Complete** |
| [project-status-and-vision.md](docs/plans/2026-01-26-project-status-and-vision.md) | Overall vision and roadmap | **Comprehensive** (needs date update) |
| [recommendations-engine-design.md](docs/plans/2026-01-28-recommendations-engine-design.md) | Recommendations engine architecture | **Design Complete** |
| [duplicate-scene-detection-design.md](docs/plans/2026-01-30-duplicate-scene-detection-design.md) | Multi-signal duplicate detection | **Design Complete** |

### Implemented

| Document | Purpose | Status |
|----------|---------|--------|
| [duplicate-scene-detection-implementation.md](docs/plans/2026-01-30-duplicate-scene-detection-implementation.md) | TDD implementation plan | **Implemented** |
| [multi-signal-implementation-design.md](docs/plans/2026-02-04-multi-signal-implementation-design.md) | Multi-signal implementation | **Implemented** |
| [identification-benchmark-design.md](docs/plans/2026-02-04-identification-benchmark-design.md) | Benchmark framework design | **Implemented** |
| [identification-benchmark-implementation.md](docs/plans/2026-02-04-identification-benchmark-implementation.md) | Benchmark implementation plan | **Implemented** |

### Future Vision

| Document | Purpose | Status |
|----------|---------|--------|
| [multi-signal-performer-identification-design.md](docs/plans/2026-02-02-multi-signal-performer-identification-design.md) | Body proportions, tattoos, multi-signal fusion | **Design Complete** |
| [performer-identity-graph.md](docs/plans/2026-01-27-performer-identity-graph.md) | Cross-stash-box linking | Design in progress |
| [scene-tagging-strategy.md](docs/plans/2026-01-26-scene-tagging-strategy.md) | CLIP/YOLO scene tagging | Conceptual |
| [skier-ai-ecosystem-analysis.md](docs/plans/2026-01-27-skier-ai-ecosystem-analysis.md) | Haven VLM integration analysis | Research |

---

## Known Issues to Address

### Resolved (2026-01-31)

- [x] **docker-compose.yml** - Updated to use `stash-sense` image/container names
- [x] **docs/stash-box-endpoints.md** - Added note about trainer repo
- [x] **duplicate-scene-detection-implementation.md** - Updated all paths
- [x] **Deleted deprecated files:**
  - `docker-compose.builder.yml`
  - `Dockerfile.builder`
  - `scripts/` directory (all scripts were obsolete)

### Remaining

1. **project-status-and-vision.md** - Could use refresh to reflect split (low priority)
2. **Test Docker build** - Not yet verified after the repo split

### Recently Completed (2026-02-06)

1. **Identification precision improvement (+75%)** - False positive filtering: `min_appearances=2`, `min_unique_frames=2`, `min_confidence=0.35`, stronger hybrid boost (0.15→0.25), cluster-based max performers. Precision 17.8%→31.1%, FP ratio 81.4%→68.9%.
2. **Benchmark framework** - Full benchmark suite for measuring identification accuracy: scene selector with stratified sampling, test executor, analyzer for metrics/failure patterns, reporter for summaries/CSV, CLI interface, integration tests.
3. **Multi-signal implementation** - Body proportions (MediaPipe) + tattoo detection (YOLO) integrated into API. Current finding: multi-signal provides 0% benefit with current DB — needs more face diversity first.
4. **Retry logic** - Added retry for intermittent scene processing failures during identification.
5. **Trainer improvements doc** - `docs/trainer-improvements.md` with actionable recommendations: 29.1% of performers have only 1 face (biggest FP source), resolution impacts (480p=12.5% vs 1080p=37.7% accuracy).

### Previously Completed (2026-02-02)

1. **Fresh run mode** - Added `fresh_run` parameter to re-evaluate all performers without restarting from scratch. Respects existing face counts, enables backfilling when raising limits or adding sources. Exposed in web UI.
2. **Lowered face detection threshold** - Changed `min_detection_confidence` from 0.8 to 0.7. Diagnostic showed 0.8 rejected 90% of valid faces (scores 0.70-0.79). With 0.7, pass rate improved from ~10% to 100%.
3. **Test infrastructure** - Added pytest with unit tests for face limits, fresh_run behavior, and API endpoints. Created diagnostic script for debugging face detection issues.
4. **Name-based performer merge** - `merge_by_name.py` merges duplicate performers across stash-box endpoints by exact name match (skips ambiguous cases where stashdb has multiple performers with same name)
5. **Multi-signal identification design** - Complete design for body proportions, tattoo recognition, and score fusion architecture
6. **Database integrity** - All checks passing: 197,720 faces, 56,281 performers, zero gaps

### Pending (Next Steps)

1. **Wait for trainer to finish building** - 5 new sources enabled, bugs fixed, actively adding faces. Once complete, run `db optimize --min-faces 2` to prune low-face-count performers.
2. **Re-benchmark with optimized DB** - After receiving the new optimized database, re-run benchmarks to measure improvement. Target: precision >40%, FP ratio <60%.
3. **Investigate multi-signal ineffectiveness** - Lower priority. More faces per performer should help before tuning body/tattoo signals.

### Previously Completed (2026-01-31)

1. **Fingerprint persistence** - `/identify/scene` now saves fingerprints to stash_sense.db
2. **DB version tracking** - Fingerprints store which face DB version was used
3. **Fingerprint generation API** - New endpoints for bulk fingerprint generation
4. **Rate limiting** - Centralized rate limiter with priority queue

---

## Rate Limiting

All Stash API calls go through a centralized rate limiter with priority queue support.

### Configuration

```bash
# .env
STASH_RATE_LIMIT=5.0  # requests per second (default: 5.0)
```

### Priority Levels

| Priority | Value | Use Case |
|----------|-------|----------|
| CRITICAL | 0 | User-initiated mutations (merge, delete) |
| HIGH | 10 | Interactive requests |
| NORMAL | 50 | Background analysis (default) |
| LOW | 100 | Bulk/batch operations |

### Metrics Endpoint

```bash
curl http://localhost:5000/health/rate-limiter
```

Returns: `total_requests`, `avg_wait_time`, `queue_depth`, etc.

---

## Scene Fingerprints

Fingerprints are per-scene summaries of which performers appear (via face recognition).
They're stored in `stash_sense.db` and used for duplicate scene detection.

### How Fingerprints Get Generated

1. **Manual:** Click "Identify Performers" on a scene → fingerprint saved automatically
2. **Bulk:** Call `POST /recommendations/fingerprints/generate` → processes all scenes

### Fingerprint API Endpoints

```bash
# Check coverage and status
GET /recommendations/fingerprints/status

# Start bulk generation (runs in background)
POST /recommendations/fingerprints/generate
  {"refresh_outdated": true, "num_frames": 12}

# Check progress
GET /recommendations/fingerprints/progress

# Stop generation gracefully
POST /recommendations/fingerprints/stop

# Mark fingerprints for refresh (when face DB updates)
POST /recommendations/fingerprints/refresh-all?confirm=true
```

### DB Version Tracking

Each fingerprint stores which face DB version was used (`db_version` column).
When you update to a new face recognition database:
1. `needs_refresh_count` in status will show how many are outdated
2. Run `POST /fingerprints/generate?refresh_outdated=true` to regenerate

---

## Roadmap (Stash Sense Sidecar)

### Phase 1: Core ✅
- [x] Face recognition working end-to-end
- [x] Plugin UI for identification
- [x] Split trainer to separate repo
- [x] Clean up stale references to old repo name
- [ ] Test Docker build works after split

### Phase 2: Recommendations Engine ✅
- [x] Database schema for stash_sense.db (v2 with migrations)
- [x] Duplicate performer analyzer
- [x] Duplicate scene files analyzer
- [x] Plugin UI for recommendations dashboard (dashboard, list, detail views)
- [x] 25+ API endpoints for recommendations management

### Phase 3: Duplicate Scene Detection ✅
- [x] Scene fingerprint storage (scene_fingerprints, scene_fingerprint_faces tables)
- [x] Multi-signal duplicate scoring (stash-box ID, face similarity, metadata heuristics)
- [x] DuplicateScenesAnalyzer implementation
- [x] API endpoints
- [ ] Wire up fingerprint generation from `/identify/scene` (not yet connected)

### Phase 4: Multi-Signal Identification (Implemented, Needs Better Data)
- [x] Body proportion extraction via pose estimation (MediaPipe)
- [x] Tattoo detection (YOLO-based)
- [x] Multi-signal fusion architecture
- [ ] Tattoo embedding and matching (deferred — needs discriminative data first)
- [ ] Semi-supervised training UI in stash-sense-trainer
- **Note:** Multi-signal currently provides 0% accuracy gain. Root cause is insufficient face diversity (29.1% of performers have only 1 face). Trainer improvements needed before signal tuning.

### Phase 5: Advanced Features
- [ ] Cross-stash-box linking (performer identity graph)
- [ ] Scene similarity search
- [ ] Background scanning mode

### Future Vision
- Scene tagging (CLIP/YOLO)
- Breast characteristics model (if needed after Phase 4)
- Birthmark detection (high value, high effort)

---

## Deployment (Test/Dev)

The plugin is deployed to a test Stash instance on Unraid for development.

| Setting | Value |
|---------|-------|
| **Unraid Server** | `10.0.0.4` |
| **SSH User** | `root` |
| **SSH Key** | `~/.ssh/id_ed25519` (default) |
| **Stash Appdata** | `/mnt/nvme_cache/appdata/stash` |
| **Plugin Path** | `/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense` |

### Deploy Plugin Changes
```bash
# From local machine
scp plugin/* root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/
```

### Quick SSH Access
```bash
ssh root@10.0.0.4
```

**Note:** This plugin is not yet published formally - it's deployed manually for testing.

---

## Quick Reference

### Run Sidecar Locally
```bash
cd /home/carrot/code/stash-sense
docker compose up
```

### Run Tests
```bash
cd /home/carrot/code/stash-sense/api
python -m pytest tests/ -v
```

### Key Files

| File | Purpose |
|------|---------|
| `/api/main.py` | FastAPI application |
| `/api/recognizer.py` | Face recognition engine |
| `/api/database_reader.py` | Read-only SQLite layer |
| `/api/recommendations_db.py` | Recommendations SQLite schema (v2) |
| `/api/recommendations_router.py` | 25+ recommendation API endpoints |
| `/api/analyzers/duplicate_performer.py` | Duplicate performer analyzer |
| `/api/analyzers/duplicate_scene_files.py` | Duplicate scene files analyzer |
| `/api/analyzers/duplicate_scenes.py` | Multi-signal duplicate scene analyzer |
| `/api/duplicate_detection/` | Models and scoring for duplicate detection |
| `/api/fingerprint_generator.py` | Bulk fingerprint generation with checkpointing |
| `/api/rate_limiter.py` | Centralized rate limiter with priority queue |
| `/api/benchmark/` | Identification benchmark framework |
| `/plugin/stash-sense.js` | Main plugin UI |
| `/plugin/stash-sense-recommendations.js` | Recommendations dashboard UI |
| `/plugin/stash_sense_backend.py` | Python proxy for CSP bypass |
| `/docs/trainer-improvements.md` | Actionable recommendations for trainer |

---

## Identification Benchmark Framework (2026-02-05)

Full benchmark suite for measuring and improving identification accuracy.

### Components
- `api/benchmark/scene_selector.py` - Stratified scene sampling from Stash library
- `api/benchmark/test_executor.py` - Runs identification against known performers
- `api/benchmark/analyzer.py` - Metrics computation and failure pattern analysis
- `api/benchmark/reporter.py` - Summary reports and CSV exports
- `api/benchmark/cli.py` - CLI interface for running benchmarks
- `api/benchmark/runner.py` - Main benchmark runner with iterative tuning

### Key Results
| Metric | Before Tuning | After Tuning |
|--------|--------------|--------------|
| Precision | 17.8% | **31.1%** (+75%) |
| FP Ratio | 81.4% | **68.9%** (-12.5%) |
| Accuracy | 38.6% | 33.6% (-5%) |

### Running Benchmarks
```bash
cd /home/carrot/code/stash-sense/api
python -m benchmark.cli --num-scenes 50 --output-dir benchmark_results/
```

---

## Multi-Signal Identification (2026-02-04)

Added body proportion filtering and tattoo presence signals to improve performer identification.

### New Files
- `api/body_proportions.py` - MediaPipe-based body ratio extraction
- `api/tattoo_detector.py` - YOLO-based tattoo detection
- `api/signal_scoring.py` - Scoring functions for body and tattoo signals
- `api/multi_signal_matcher.py` - Combined multi-signal identification

### API Changes
- `/identify` endpoint now accepts `use_multi_signal`, `use_body`, `use_tattoo` flags
- Response includes `signals_used`, `body_detected`, `tattoos_detected` fields

### Configuration
- `ENABLE_BODY_SIGNAL=true/false` - Enable body proportion filtering
- `ENABLE_TATTOO_SIGNAL=true/false` - Enable tattoo presence signal
- `FACE_CANDIDATES=20` - Number of face candidates to consider for re-ranking

---

*This document provides session continuity across Claude conversations.*

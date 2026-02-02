# Stash Sense: Session Context

**Last Updated:** 2026-02-02
**Status:** Phase 2 & 3 Complete, Multi-Signal Design Complete

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

The trainer Docker is currently running, building the next enriched database with:
- Multi-source scraping (13 stash-boxes + reference sites)
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

### Recently Completed (2026-02-02)

1. **Name-based performer merge** - `merge_by_name.py` merges duplicate performers across stash-box endpoints by exact name match (skips ambiguous cases where stashdb has multiple performers with same name)
2. **Safer merge logic** - Fixed merge scripts to skip 27 ambiguous names, merged 626 performers with 1904 faces
3. **Multi-signal identification design** - Complete design for body proportions, tattoo recognition, and score fusion architecture
4. **Database integrity** - All checks passing: 197,720 faces, 56,281 performers, zero gaps

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

### Phase 4: Multi-Signal Identification (Design Complete)
- [ ] Body proportion extraction via pose estimation (MediaPipe)
- [ ] Tattoo detection (YOLO-based)
- [ ] Tattoo embedding and matching
- [ ] Multi-signal fusion architecture
- [ ] Semi-supervised training UI in stash-sense-trainer

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
| `/plugin/stash-sense.js` | Main plugin UI |
| `/plugin/stash-sense-recommendations.js` | Recommendations dashboard UI |
| `/plugin/stash_sense_backend.py` | Python proxy for CSP bypass |

---

*This document provides session continuity across Claude conversations.*

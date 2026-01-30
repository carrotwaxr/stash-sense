# Stash Sense: Session Context

**Purpose:** Quick orientation for Claude sessions working on this project.
**Last Updated:** 2026-01-30

---

## Quick Reference

| What | Command |
|------|---------|
| Run sidecar (live DB) | `cd api && DATA_DIR=./data STASH_URL=http://localhost:9999 STASH_API_KEY=xxx uvicorn main:app --reload` |
| Run sidecar (backup) | `cd api && DATA_DIR=./data-backup-20260129-complete STASH_URL=http://localhost:9999 STASH_API_KEY=xxx uvicorn main:app --reload` |
| Check metadata refresh | `cat api/data/*_refresh_progress.json \| python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Progress: {d[\"stats\"][\"processed\"]:,}/{d[\"stats\"][\"total\"]:,} ({d[\"stats\"][\"processed\"]/d[\"stats\"][\"total\"]*100:.1f}%)')"` |
| Resume metadata refresh | `cd api && python metadata_refresh.py --database ./data/performers.db --all --resume` |
| Enrichment status | `cd api && python enrichment_builder.py --status` |
| Run enrichment | `cd api && python enrichment_builder.py --sources stashdb` |
| Current database stats | `cat api/data/manifest.json` |
| Stash plugin | `plugin/` directory (symlink to Stash plugins folder) |

## Current State

- **Database Build:** ✅ COMPLETE - 75K performers, 70K faces (avg 1.16 faces/performer with faces)
- **SQLite Migration:** ✅ Complete - `performers.db` (schema v5)
- **Metadata Refresh:** 🔄 Running (updates URLs, aliases, identity graph fields)
- **Multi-Source Enrichment:** ✅ Infrastructure complete (Tasks 1-10)
- **Recommendations System:** ✅ Complete - duplicate performers, duplicate scene files, and duplicate scenes analyzers
- **Plugin:** ✅ Complete - Face recognition + recommendations dashboard
- **Backup Available:** `api/data/performers_backup_20260129_*.db` - pre-enrichment snapshot

### Active Work (2026-01-30)

**8. Duplicate Scene Detection** ✅ COMPLETE
- Multi-signal duplicate detection: stash-box IDs (100%), face fingerprints (85%), metadata (60%)
- New analyzer: `DuplicateScenesAnalyzer` with configurable min_confidence
- API: `POST /recommendations/analysis/duplicate_scenes/run`
- Schema v2: scene_fingerprints and scene_fingerprint_faces tables
- New module: `duplicate_detection/` with models and scoring functions
- 37 new tests, 172 total passing
- **Reference:** [duplicate-scene-detection-design.md](2026-01-30-duplicate-scene-detection-design.md)

### Previous Work (2026-01-29)

**1. Missing Performers Recovery** ✅ COMPLETE
- Found and added 3,223 performers missing from DB due to StashDB pagination bug
- Fixed: `stashdb_client.py` now uses NAME sort (stable)
- Database now has 75,251 performers

**2. Multi-Source Enrichment Infrastructure** ✅ COMPLETE
- `base_scraper.py` - Unified scraper interface with rate limiting
- `enrichment_config.py` - YAML config + CLI overrides
- `quality_filters.py` - Face quality validation (size, angle, confidence)
- `write_queue.py` - Async queue for serialized DB writes
- `enrichment_coordinator.py` - Runs scrapers concurrently
- `enrichment_builder.py` - CLI to run enrichment
- Schema v5 with per-source face tracking and scrape progress
- 93 tests passing

**4. All Scrapers Implemented** ✅ COMPLETE
- Stash-boxes: StashDB, ThePornDB, PMVStash, JAVStash, FansDB
- Reference sites: Babepedia, IAFD, FreeOnes
- See "Run Enrichment" section below for commands

**3. Face Detection Tuning** ✅ COMPLETE
- Tested 1080p and 480p scenes with comprehensive methodology
- **Results:** 70% accuracy on 1080p (optimal scenes), 40% on 480p
- **Key findings:**
  - Performers with 3+ faces in DB match reliably
  - Burst sampling tested and disproven (worse than even sampling)
  - 30-40 frames is sweet spot for speed/accuracy
  - Scene content (lighting, angles) matters more than resolution
- **Reference:** [face-detection-tuning.md](face-detection-tuning.md) for full methodology and results

**3. V2 Matching Logic** ✅ DONE
- `recognize_image()` now uses V2 matching with adaptive health detection
- Replaced sprite-based scene identification with ffmpeg full-resolution frames
- Plugin backend updated to use new endpoint parameters
- Added burst mode to frame extractor (but even sampling preferred)

**5. Frequency-Based Matching Mode** ✅ DONE
- Added `matching_mode` parameter to `/identify/scene` endpoint
- Two modes: `cluster` (original) and `frequency` (new, default)
- Frequency mode counts performer appearances across all face matches
- More robust when face clustering fails (common issue)
- Both achieve 70% accuracy on test set, but complement each other
- Test script: `api/test_matching_modes.py`
- **Reference:** [face-detection-tuning.md](face-detection-tuning.md) for comparison results

**6. Screenshot Processing Enhancement** ✅ DONE (2026-01-30)
- Scene identification now includes Stash screenshot (cover image) for face detection
- **Finding:** Stash serves thumbnails via `paths.screenshot` (e.g., 750x422 for 1080p video)
- **Fix:** Screenshots smaller than 80% of video width are upscaled using Lanczos interpolation
- Upscaling enables face detection on thumbnail screenshots (faces too small before)
- Verbose timing logs added to `/identify/scene` endpoint for debugging

**7. Face Enrichment Integration** ✅ COMPLETE (2026-01-29)
- `FaceValidator` - Trust-level based validation (high/medium/low)
- `IndexManager` - Voyager index loading/saving/querying
- `FaceProcessor` - Image → face detection → embedding pipeline
- `EnrichmentCoordinator` - Full integration with face processing
- CLI: `--enable-faces` flag for enrichment_builder.py
- 108 tests passing

---

## What This Project Does

**Stash Sense** is an AI-powered sidecar for Stash with two main features:

### 1. Face Recognition
- Identifies performers in scenes using face embeddings
- Pre-built database of StashDB performers (~58K with faces)
- Click "Identify Performers" on scene → get matches with StashDB IDs

### 2. Recommendations Engine
- Analyzes Stash library for curation opportunities
- **Duplicate Performers:** Finds performers that may be duplicates (same name, shared scenes, StashDB IDs)
- **Duplicate Scene Files:** Finds scenes with multiple files, suggests which to keep/delete
- **Duplicate Scenes:** Finds scenes that may be duplicates using stash-box IDs, face fingerprints, and metadata heuristics

**Architecture:**
```
User's Stash                          Stash Sense Sidecar
┌─────────────┐                      ┌─────────────────────────┐
│ Plugin UI   │ ──via Python proxy─► │ FastAPI                 │
│ (JavaScript)│                      │ ├─ /identify (faces)    │
│             │ ◄──JSON responses──  │ ├─ /recommendations     │
└─────────────┘                      │ └─ /health              │
                                     │                         │
                                     │ DATA_DIR volume:        │
                                     │ ├─ face_*.voy (indices) │
                                     │ ├─ performers.db        │
                                     │ ├─ stash_sense.db       │
                                     │ └─ manifest.json        │
                                     └─────────────────────────┘
```

**Key Insight:** Plugin JS cannot directly call sidecar due to CSP. All API calls route through `stash_sense_backend.py` which proxies to the sidecar.

---

## Roadmap

### Phase 1: Core Face Recognition ✅ COMPLETE
- Face detection (RetinaFace), embeddings (FaceNet512 + ArcFace)
- Database builder for StashDB
- FastAPI sidecar with /identify endpoints
- Stash plugin with "Identify Performers" UI

### Phase 2: Full StashDB Database ✅ COMPLETE
- 58K performers with 103K faces embedded
- SQLite migration complete (schema v3)
- Metadata refresh tool for URL/alias updates

### Phase 2.5: Recommendations System ✅ COMPLETE
- Recommendations database (stash_sense.db)
- Duplicate performer analyzer
- Duplicate scene files analyzer
- Duplicate scenes analyzer (stash-box IDs, face fingerprints, metadata heuristics)
- Plugin dashboard UI with resolve/dismiss actions

### Phase 3: Multi-Source Enrichment 🔄 IN PROGRESS
- ✅ Infrastructure complete (config, queue, coordinator, CLI)
- ✅ StashDB adapter updated to unified interface
- ✅ All stash-box scrapers: ThePornDB, PMVStash, JAVStash, FansDB
- ✅ Reference site scrapers: Babepedia, IAFD, FreeOnes
- ⏳ Face detection integration (download images, extract faces)
- ⏳ Run full enrichment pass (waiting for metadata refresh)
- **Goal:** Increase faces per performer from 1.16 to 5-10
- **Docs:** [multi-source-enrichment-design.md](2026-01-29-multi-source-enrichment-design.md), [scraper-orchestration-implementation.md](2026-01-29-scraper-orchestration-implementation.md)

### Phase 5: Testing & Polish
- Accuracy validation across sources
- Additional recommendation analyzers
- Performance optimization

### Phase 6: Release & Distribution
- Package database for GitHub Releases
- Docker image / Unraid template
- User documentation

---

## What Can Be Worked On Now

### Plugin/Sidecar Improvements

| Task | Description |
|------|-------------|
| **Face recognition UX** | Improve the identify modal, add "link performer" action |
| **More analyzers** | Missing performers, untagged scenes, similar scenes |
| **Bulk actions** | Process multiple recommendations at once |
| **Settings UI** | Better plugin configuration page |
| **Error handling** | Improve error messages and recovery |

### Run Enrichment (after metadata refresh completes)

All scrapers are implemented and tested. Run with:

```bash
cd api

# Run with face processing (slower, requires GPU)
python enrichment_builder.py --sources stashdb --enable-faces

# Run with face limits
python enrichment_builder.py --sources stashdb,babepedia --enable-faces --max-faces-total 12

# Stash-boxes (have API keys in .env)
python enrichment_builder.py --sources theporndb --enable-faces
python enrichment_builder.py --sources pmvstash --enable-faces
python enrichment_builder.py --sources javstash --enable-faces
# python enrichment_builder.py --sources fansdb --enable-faces  # Need FANSDB_API_KEY

# Reference sites (need FlareSolverr at FLARESOLVERR_URL)
python enrichment_builder.py --sources babepedia --enable-faces
python enrichment_builder.py --sources iafd --enable-faces
python enrichment_builder.py --sources freeones --enable-faces

# Multiple sources at once
python enrichment_builder.py --sources theporndb,babepedia,freeones --enable-faces

# Metadata-only (no face processing)
python enrichment_builder.py --sources freeones

# Dry run (no DB writes)
python enrichment_builder.py --sources freeones --dry-run
```

| Source | Type | File | Status | Notes |
|--------|------|------|--------|-------|
| StashDB | stash_box | `stashdb_client.py` | ✅ Ready | ~100K performers |
| ThePornDB | stash_box | `theporndb_client.py` | ✅ Ready | ~10K performers, StashDB cross-refs |
| PMVStash | stash_box | `stashbox_clients.py` | ✅ Ready | ~6.5K performers |
| JAVStash | stash_box | `stashbox_clients.py` | ✅ Ready | ~21.7K performers |
| FansDB | stash_box | `stashbox_clients.py` | ⚠️ Not in Stash | Register at fansdb.cc |
| Babepedia | reference_site | `babepedia_client.py` | ✅ Ready | FlareSolverr, female only |
| IAFD | reference_site | `iafd_client.py` | ✅ Ready | FlareSolverr |
| FreeOnes | reference_site | `freeones_client.py` | ✅ Ready | Gallery drilling |

### Still Pending

| Task | Description |
|------|-------------|
| **Identity graph** | Cross-database performer linking |
| **Run full enrichment** | After metadata refresh completes |

---

## Key Files & Directories

### API Sidecar (`api/`)

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app - face recognition + recommendations |
| `recommendations_router.py` | `/recommendations` endpoints |
| `recommendations_db.py` | SQLite database for recommendations |
| `analyzers/` | Analyzer implementations (duplicate_performer, duplicate_scene_files, duplicate_scenes) |
| `duplicate_detection/` | Models and scoring for duplicate scene detection |
| `database_builder.py` | Builds face embedding database |
| `metadata_refresh.py` | Updates performer metadata from StashDB |
| `database.py` | SQLite layer for performer metadata (schema v5) |
| `stashdb_client.py` | GraphQL client for StashDB (inherits BaseScraper) |
| `theporndb_client.py` | REST client for ThePornDB |
| `base_scraper.py` | Unified scraper interface with rate limiting |
| `enrichment_config.py` | YAML config loader for multi-source |
| `enrichment_coordinator.py` | Orchestrates concurrent scrapers |
| `enrichment_builder.py` | CLI entry point for enrichment |
| `quality_filters.py` | Face quality validation |
| `write_queue.py` | Async queue for serialized writes |
| `sources.yaml` | Source configuration (rate limits, face limits) |
| `face_validator.py` | Trust-level based face validation |
| `index_manager.py` | Voyager index loading/saving/querying |
| `face_processor.py` | Image → face → embedding pipeline |
| `recognizer.py` | Voyager index queries |
| `embeddings.py` | Face detection and embedding |
| `data/` | Live database (don't commit) |
| `data-backup-*/` | Point-in-time backups |

### Stash Plugin (`plugin/`)

| File | Purpose |
|------|---------|
| `stash-sense.yml` | Plugin manifest and settings |
| `stash-sense-core.js` | Shared utilities, settings, API client |
| `stash-sense-recommendations.js` | Recommendations dashboard UI |
| `stash-sense.js` | Face recognition UI (identify modal) |
| `stash-sense.css` | All plugin styles |
| `stash_sense_backend.py` | Python proxy for CSP bypass |

### Documentation (`docs/plans/`)

| Document | When To Read |
|----------|--------------|
| `SESSION-CONTEXT.md` | Start of every session (this file) |
| [face-detection-tuning.md](face-detection-tuning.md) | Tuning face detection, clustering, resolution handling |
| [face-recognition-system-design.md](2026-01-24-face-recognition-system-design.md) | Face recognition details |
| [performer-identity-graph.md](2026-01-27-performer-identity-graph.md) | Multi-source linking strategy |
| [data-sources-catalog.md](2026-01-27-data-sources-catalog.md) | All data sources with rate limits |
| [duplicate-scene-detection-design.md](2026-01-30-duplicate-scene-detection-design.md) | Multi-signal duplicate scene detection |
| [duplicate-scene-detection-implementation.md](2026-01-30-duplicate-scene-detection-implementation.md) | TDD implementation plan for duplicate scenes |

---

## Important Context

### Two Databases

| Database | Purpose | Location |
|----------|---------|----------|
| `performers.db` | Face recognition metadata (read-only at runtime) | `DATA_DIR/performers.db` |
| `stash_sense.db` | Recommendations (read-write) | `DATA_DIR/stash_sense.db` |

### CSP Bypass Pattern

Browser JS cannot call sidecar directly due to Content Security Policy. Pattern:
1. JS calls `runPluginOperation(mode, args)` → Stash GraphQL
2. Stash executes `stash_sense_backend.py` with args
3. Python makes HTTP request to sidecar
4. Response flows back through GraphQL

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `DATA_DIR` | Path to database files | `./data` |
| `STASH_URL` | Stash GraphQL endpoint | `http://localhost:9999` |
| `STASH_API_KEY` | Stash API key | `eyJ...` |
| `STASHDB_API_KEY` | StashDB API key | In `.env` ✅ |
| `THEPORNDB_API_KEY` | ThePornDB API key | In `.env` ✅ |
| `PMVSTASH_API_KEY` | PMVStash API key | In `.env` ✅ |
| `JAVSTASH_API_KEY` | JAVStash API key | In `.env` ✅ |
| `FANSDB_API_KEY` | FansDB API key | **Missing** |
| `FLARESOLVERR_URL` | FlareSolverr URL | `http://10.0.0.4:8191` |

### Rate Limits

| Source | Safe Rate | Notes |
|--------|-----------|-------|
| StashDB | 240/min | 0.25s between requests |
| ThePornDB | 240/min | REST API |
| Reference sites | 30-60/min | Be respectful |

---

## Starting a New Session

### Quick Status Check

```bash
# Check if sidecar is running
curl -s http://localhost:5000/health | python -c "import json,sys; d=json.load(sys.stdin); print(f'Status: {d[\"status\"]}, Performers: {d[\"performer_count\"]:,}')"

# Metadata refresh progress (if running)
cat api/data/*_refresh_progress.json | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'Processed: {d[\"stats\"][\"processed\"]:,}/{d[\"stats\"][\"total\"]:,}')"

# Database stats
cat api/data/manifest.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'Performers: {d[\"performer_count\"]:,}, Faces: {d[\"face_count\"]:,}')"

# Git status
git status --short
```

### Before Writing Code

1. **Read this document** for current state
2. **Check the roadmap** - what phase are we in?
3. **Read relevant detailed docs** before implementing

---

*This document should be updated as the project progresses.*

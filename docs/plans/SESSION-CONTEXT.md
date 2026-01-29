# Stash Sense: Session Context

**Purpose:** Quick orientation for Claude sessions working on this project.
**Last Updated:** 2026-01-29

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
- **Recommendations System:** ✅ Complete - duplicate performers & duplicate scene files analyzers
- **Plugin:** ✅ Complete - Face recognition + recommendations dashboard
- **Backup Available:** `api/data/performers_backup_20260129_*.db` - pre-enrichment snapshot

### Active Work (2026-01-29)

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
- 52 tests passing

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
- Plugin dashboard UI with resolve/dismiss actions

### Phase 3: Multi-Source Enrichment 🔄 IN PROGRESS
- ✅ Infrastructure complete (config, queue, coordinator, CLI)
- ✅ StashDB adapter updated to unified interface
- ⏳ Add ThePornDB, PMVStash, JAVStash, FansDB scrapers
- ⏳ Face detection integration (download images, extract faces)
- ⏳ Reference site scrapers (Babepedia, IAFD, FreeOnes)
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

### Multi-Source Support (Phase 3)

| Task | Description | Docs |
|------|-------------|------|
| **ThePornDB client** | Already have `theporndb_client.py` - test and integrate | [data-sources-catalog.md](2026-01-27-data-sources-catalog.md) |
| **Stash-box unified client** | PMVStash/JAVStash/FansDB (same GraphQL schema) | [performer-identity-graph.md](2026-01-27-performer-identity-graph.md) |
| **Identity graph** | Cross-database performer linking | [performer-identity-graph.md](2026-01-27-performer-identity-graph.md) |

### Reference Site Enrichment (Phase 4)

| Task | Description |
|------|-------------|
| **Babepedia scraper** | HTML scraping, needs FlareSolverr |
| **IAFD scraper** | Structured data, good for aliases |
| **FreeOnes scraper** | Extensive metadata |

---

## Key Files & Directories

### API Sidecar (`api/`)

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app - face recognition + recommendations |
| `recommendations_router.py` | `/recommendations` endpoints |
| `recommendations_db.py` | SQLite database for recommendations |
| `recommendations_analyzers.py` | Analyzer implementations |
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

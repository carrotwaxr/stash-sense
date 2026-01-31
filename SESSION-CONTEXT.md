# Stash Sense: Session Context

**Last Updated:** 2026-01-31
**Status:** Post-Split Audit Complete

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
| **Recommendations Engine** | Designed, partially implemented | `/api/recommendations_*.py` |
| **Duplicate Detection** | Designed, not implemented | See plans |

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

### Ready for Implementation

| Document | Purpose | Notes |
|----------|---------|-------|
| [duplicate-scene-detection-implementation.md](docs/plans/2026-01-30-duplicate-scene-detection-implementation.md) | TDD implementation plan | 7 tasks, paths updated |

### Future Vision (Not Yet Designed)

| Document | Purpose | Status |
|----------|---------|--------|
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

---

## Roadmap (Stash Sense Sidecar)

### Phase 1: Current Focus
- [x] Face recognition working end-to-end
- [x] Plugin UI for identification
- [x] Split trainer to separate repo
- [x] Clean up stale references to old repo name
- [ ] Test Docker build works after split

### Phase 2: Recommendations Engine
- [ ] Implement database schema for stash_sense.db
- [ ] Duplicate performer analyzer
- [ ] Duplicate scene files analyzer
- [ ] Plugin UI for recommendations dashboard

### Phase 3: Duplicate Scene Detection
- [ ] Scene fingerprint storage
- [ ] Multi-signal duplicate scoring
- [ ] DuplicateScenesAnalyzer implementation
- [ ] API endpoints

### Phase 4: Advanced Features
- [ ] Cross-stash-box linking (performer identity graph)
- [ ] Scene similarity search
- [ ] Background scanning mode

### Future Vision
- Scene tagging (CLIP/YOLO)
- Body recognition (Person Re-ID)
- Crowd-sourced face database

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
| `/api/recommendations_*.py` | Recommendations engine |
| `/plugin/stash-sense.js` | Main plugin UI |
| `/plugin/stash_sense_backend.py` | Python proxy for CSP bypass |

---

*This document provides session continuity across Claude conversations.*

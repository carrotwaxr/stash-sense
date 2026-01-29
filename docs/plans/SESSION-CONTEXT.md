# Stash Sense: Session Context

**Purpose:** Quick orientation for Claude sessions working on this project.
**Last Updated:** 2026-01-28

---

## Quick Reference

| What | Command |
|------|---------|
| Run face recognition API | `cd api && DATABASE_PATH=./data-snapshot-20260127-1653 uvicorn main:app --reload` |
| Resume database build | `cd api && python database_builder.py --resume --output ./data --max-performers 1000000 --rate-limit 0.25` |
| Test scraping sources | `cd api && python test_scrape_sources.py "Performer Name"` |
| Test URL enrichment | `cd api && python test_url_enrichment.py <stashdb-uuid>` |
| Current database stats | `cat api/data/manifest.json` |
| Stash plugin | `plugin/` directory |

## Current State

- **Database Build:** ~93% complete (~96k faces embedded, nearly done!)
- **SQLite Migration:** ✅ Complete - now using `performers.db` (schema v3)
- **New Fields:** `scene_count`, `stashdb_updated_at` for incremental sync
- **Working:** Face recognition API, Stash plugin, database builder
- **Snapshot Available:** `api/data-snapshot-20260127-1653/` - usable for testing

---

## What This Project Does

**Stash Sense** is a face recognition sidecar for Stash that identifies performers in scenes.

**Core Flow:**
1. User clicks "Identify Performers" on a scene in Stash
2. Plugin sends scene sprite sheet to our FastAPI sidecar
3. Sidecar detects faces, generates embeddings, queries pre-built database
4. Returns matches with StashDB IDs - user can link performers

**Key Insight:** We distribute a pre-built embedding database (~1GB), not model weights. Users download the database, run the sidecar, and get instant performer identification without training anything.

**Future Scope:** The architecture supports additional AI capabilities (scene tagging, visual similarity, OCR) but this document focuses on performer recognition - the current priority. See [project-status-and-vision.md](2026-01-26-project-status-and-vision.md) for the broader vision.

## Architecture

```
User's Stash                          Face Recognition Sidecar
┌─────────────┐                      ┌─────────────────────────┐
│ Plugin UI   │ ──POST /identify──►  │ FastAPI                 │
│ (JavaScript)│                      │ ├─ RetinaFace (detect)  │
│             │ ◄──JSON matches────  │ ├─ FaceNet512 (embed)   │
└─────────────┘                      │ └─ Voyager (search)     │
                                     │                         │
                                     │ /data volume:           │
                                     │ ├─ face_*.voy (indices) │
                                     │ ├─ performers.db (SQLite)│
                                     │ ├─ faces.json (idx map) │
                                     │ └─ manifest.json        │
                                     └─────────────────────────┘
```

**Detailed design:** [face-recognition-system-design.md](2026-01-24-face-recognition-system-design.md)

---

## Roadmap

### Phase 1: Core Face Recognition ✅ COMPLETE
- Face detection (RetinaFace), embeddings (FaceNet512 + ArcFace)
- Database builder for StashDB
- FastAPI sidecar with /identify endpoints
- Stash plugin with "Identify Performers" UI

### Phase 2: Full StashDB Database 🔨 NEARLY COMPLETE
- Building complete StashDB database (~103k performers)
- Currently ~93% complete (~96k faces), finishing up
- SQLite migration complete (schema v3 with identity fields)

### Phase 3: Multi-Stash-Box Support ⏳ NEXT
- Add ThePornDB, PMVStash, JAVStash, FansDB
- Performer identity graph (cross-database linking)
- Concurrent scraping architecture
- **Docs:** [performer-identity-graph.md](2026-01-27-performer-identity-graph.md), [concurrent-scraping-architecture.md](2026-01-26-concurrent-scraping-architecture.md)

### Phase 4: Embedding Enrichment
- Reference site scraping (Babepedia, IAFD, FreeOnes, etc.)
- Increase embeddings per performer (current avg: 1.04 → goal: 5-10)
- **Docs:** [reference-site-enrichment.md](2026-01-26-reference-site-enrichment.md), [data-sources-catalog.md](2026-01-27-data-sources-catalog.md)

### Phase 5: Testing & Tooling
- Accuracy validation across sources
- Curation UI for uncertain matches
- Database management tooling

### Phase 6: Release & Distribution
- Package database for GitHub Releases
- Unraid Docker template
- GPU passthrough documentation
- User-facing documentation

---

## What Can Be Worked On Now

### While Database Build Runs (Use 65% Snapshot)

| Task | Description | Docs |
|------|-------------|------|
| **ThePornDB client** | Build API client, test cross-linking via `extras.links.StashDB` | [data-sources-catalog.md](2026-01-27-data-sources-catalog.md) |
| **Stash-box unified client** | Refactor for PMVStash/JAVStash/FansDB (same GraphQL schema) | [performer-identity-graph.md](2026-01-27-performer-identity-graph.md) |
| **URL normalization library** | ✅ Basic impl in `url_normalizer.py` - parse IAFD, Twitter, IMDb URLs into canonical IDs | [performer-identity-graph.md](2026-01-27-performer-identity-graph.md) |
| **Reference site scrapers** | Babepedia, IAFD, FreeOnes - patterns documented | [data-sources-catalog.md](2026-01-27-data-sources-catalog.md) |
| **Identity graph schema** | ✅ Foundation in `database.py` - has aliases, URLs, merged_ids tables | [performer-identity-graph.md](2026-01-27-performer-identity-graph.md) |
| **Accuracy testing** | Validate recognition against 65% snapshot | [face-recognition-system-design.md](2026-01-24-face-recognition-system-design.md) |

### Blocked Until Full Database

| Task | Why Blocked |
|------|-------------|
| Final accuracy benchmarks | Need complete performer coverage |
| Database packaging | Need final checksums and version |
| Multi-source merge testing | Need full StashDB as baseline |

## What NOT To Work On Yet

- Scene tagging (CLIP/YOLO) - future phase, not current priority
- Recommendations - out of scope for now
- Public release prep - scraping and testing come first

---

## Key Files & Directories

### API Sidecar (`api/`)

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, `/identify`, `/identify/scene`, `/health` endpoints |
| `database_builder.py` | Builds embedding database from StashDB |
| `database.py` | SQLite database layer (schema v3) |
| `stashdb_client.py` | GraphQL client for StashDB API |
| `theporndb_client.py` | REST client for ThePornDB API |
| `recognizer.py` | Queries Voyager indices for face matches |
| `embeddings.py` | RetinaFace detection, FaceNet/ArcFace embedding generation |
| `config.py` | Configuration dataclasses |
| `data/` | Live database files (don't commit) |
| `data-snapshot-*/` | Point-in-time database copies for testing |

### Stash Plugin (`plugin/`)

| File | Purpose |
|------|---------|
| `face-recognition.js` | UI integration, "Identify Performers" button |
| `face-recognition.yml` | Plugin manifest and settings |
| `face_recognition_backend.py` | Python backend for CSP bypass |
| `face-recognition.css` | Modal styling |

### Documentation (`docs/plans/`)

| Document | When To Read |
|----------|--------------|
| [face-recognition-system-design.md](2026-01-24-face-recognition-system-design.md) | API design, data formats, pipeline details |
| [performer-identity-graph.md](2026-01-27-performer-identity-graph.md) | Cross-database linking, **URL-first resolution strategy** |
| [data-sources-catalog.md](2026-01-27-data-sources-catalog.md) | All data sources, rate limits, URL patterns |
| [concurrent-scraping-architecture.md](2026-01-26-concurrent-scraping-architecture.md) | Multi-source scraping design |
| [reference-site-enrichment.md](2026-01-26-reference-site-enrichment.md) | Reference site scraping, **smart enrichment strategies** |
| [project-status-and-vision.md](2026-01-26-project-status-and-vision.md) | Full vision, **accuracy enhancement strategies** |

---

## Important Context

### Technical Decisions

| Decision | Rationale |
|----------|-----------|
| **CPU for embeddings (dev)** | Current dev GPU (RTX 5060 Ti/Blackwell) lacks TensorFlow support. Production will support GPU embeddings where available. Hardware flexibility is a goal. |
| **Dual embedding models** | FaceNet512 + ArcFace ensemble improves accuracy over single model |
| **Voyager for indices** | Fast cosine similarity search, compact E4M3 storage |
| **SQLite for metadata** | Migrated from JSON to SQLite (`performers.db`). Schema v3 includes identity fields (aliases, URLs, tattoos, piercings) and sync fields (`scene_count`, `stashdb_updated_at`). |

### Rate Limits

| Source | Safe Rate | Notes |
|--------|-----------|-------|
| StashDB | 240/min | Running 24h+ stable |
| ThePornDB | 240/min | REST API, not GraphQL |
| PMVStash | 300/min | Same schema as StashDB |
| JAVStash | 300/min | ~1 image per performer (limitation) |
| Reference sites | 30-60/min | HTML scraping, be respectful |

### Gotchas

- **~50% of StashDB performers have no detectable faces** - Missing images or faces not detected
- **StashDB avg 1.04 faces/performer** - Need reference sites to improve accuracy
- **ThePornDB only ~3.3% have explicit StashDB links** - Most need face/name matching
- **JAVStash single image per performer** - Major accuracy limitation, need supplemental sources
- **FlareSolverr needed for some sites** - Babepedia, IAFD, Indexxx require it for bulk scraping

---

## Starting a New Session

### Quick Status Check

```bash
# Database build progress (if running)
ps aux | grep database_builder

# Current database stats
cat api/data/manifest.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'Performers: {d[\"performer_count\"]:,}, Faces: {d[\"face_count\"]:,}')"

# Git status
git status --short
```

### Before Writing Code

1. **Read this document** for current state and priorities
2. **Check the roadmap** - what phase are we in?
3. **Check "What To Work On"** - is the task blocked or actionable?
4. **Read relevant detailed doc** before implementing (links throughout)

### Key Commands

```bash
# Run API sidecar (against snapshot for testing)
cd api && DATABASE_PATH=./data-snapshot-20260127-1653 uvicorn main:app --reload

# Resume database build
cd api && python database_builder.py --resume --output ./data --max-performers 1000000 --rate-limit 0.25

# Test a scraper
cd api && python test_scrape_sources.py "Angela White"

# Test URL enrichment
cd api && python test_url_enrichment.py <stashdb-uuid>
```

---

*This document should be updated as the project progresses.*

# Stash Sense Trainer: Design Document

**Date:** 2026-01-30
**Status:** Approved

---

## Overview

Extract the enrichment/database building functionality from `stash-face-recognition` into a separate, private repository called `stash-sense-trainer`. This service:

- Runs as a Docker container on unRAID
- Provides a React web UI to monitor progress, configure settings, and trigger builds
- Produces versioned performer databases consumed by the public sidecar

**Why separate?**
- Trainer code (scrapers, enrichment) should remain private
- Sidecar + plugin can be open-sourced
- Scraping activity stays isolated from public distribution

---

## Repository Structure

### stash-sense-trainer (private)

```
stash-sense-trainer/
├── api/
│   ├── main.py                   # FastAPI app (dashboard API + serves React)
│   ├── enrichment_builder.py     # CLI entry point (existing)
│   ├── enrichment_coordinator.py # Orchestrates scrapers
│   ├── enrichment_config.py      # YAML config loader
│   ├── sources.yaml              # Source configuration
│   ├── database.py               # Full read-write version
│   ├── embeddings.py             # Face detection/embedding (copied from sidecar)
│   ├── models.py                 # Data models (copied from sidecar)
│   ├── base_scraper.py           # Scraper interface
│   ├── stashdb_client.py         # StashDB scraper
│   ├── theporndb_client.py       # ThePornDB scraper
│   ├── babepedia_client.py       # Babepedia scraper
│   ├── iafd_client.py            # IAFD scraper
│   ├── freeones_client.py        # FreeOnes scraper
│   ├── boobpedia_client.py       # Boobpedia scraper
│   ├── thenude_client.py         # TheNude scraper
│   ├── afdb_client.py            # AFDB scraper
│   ├── pornpics_client.py        # PornPics scraper
│   ├── elitebabes_client.py      # EliteBabes scraper
│   ├── javdatabase_client.py     # JavDatabase scraper
│   ├── indexxx_client.py         # Indexxx scraper (currently blocked)
│   ├── stashbox_clients.py       # PMVStash, JAVStash, FansDB
│   ├── face_processor.py         # Image → face → embedding pipeline
│   ├── face_validator.py         # Trust-level validation
│   ├── quality_filters.py        # Face quality checks
│   ├── index_manager.py          # Voyager index saving
│   ├── write_queue.py            # Async DB write queue
│   ├── flaresolverr_client.py    # FlareSolverr wrapper
│   └── url_normalizer.py         # URL parsing utilities
├── web/                          # React dashboard
│   ├── src/
│   │   ├── App.jsx
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── Runs.jsx
│   │   │   ├── RunDetail.jsx
│   │   │   ├── Config.jsx
│   │   │   └── Versions.jsx
│   │   └── components/
│   │       ├── BuildForm.jsx
│   │       ├── ProgressCard.jsx
│   │       ├── SourceCard.jsx
│   │       └── RunHistoryTable.jsx
│   ├── package.json
│   ├── vite.config.js
│   └── tailwind.config.js
├── Dockerfile
├── docker-compose.yml            # Optional: trainer + FlareSolverr stack
├── unraid-template.xml
├── requirements.txt
└── README.md
```

### stash-face-recognition (public) - changes

- Remove all `*_client.py` scraper files
- Remove `enrichment_*.py` files
- Remove `face_processor.py`, `face_validator.py`, `quality_filters.py`
- Remove `index_manager.py`, `write_queue.py`
- Replace `database.py` with `database_reader.py` (read-only subset)
- Keep: `main.py`, `recognizer.py`, `embeddings.py`, `matching.py`, `recommendations_*.py`, `analyzers/`, plugin files

### Shared Files

These files exist in both repos and should be kept in sync:

| File | Notes |
|------|-------|
| `embeddings.py` | Face detection + embedding generation. Stable, rarely changes. |
| `models.py` | Data classes. Stable. |

Add comment at top of each: `# Shared with stash-sense / stash-sense-trainer - keep in sync if modified`

**Trainer is source of truth** - if changes needed, update trainer first, then copy to sidecar.

### database.py Split

**Trainer keeps full `database.py`:**
- Schema migrations
- All write operations (add_performer, add_face, update_scrape_progress, etc.)
- All read operations

**Sidecar gets `database_reader.py`:**
- Read-only operations only
- get_performer, get_performers_with_faces, search_by_name, etc.
- No schema migration code
- ~20% of original file size

---

## FastAPI Backend

### Endpoints

```
GET  /api/status              # Current build status
GET  /api/config              # Current sources.yaml as JSON
PUT  /api/config              # Update config
POST /api/build/start         # Start build with params
POST /api/build/stop          # Graceful shutdown
GET  /api/runs                # List past runs
GET  /api/runs/{id}           # Run details with diff
GET  /api/versions            # List DB versions
GET  /api/logs                # SSE stream of live logs
GET  /                        # Serves React SPA
```

### Build Parameters

```json
{
  "sources": ["stashdb", "theporndb", "pmvstash", "javstash", "babepedia", "iafd", "freeones", "boobpedia", "thenude", "afdb", "pornpics", "elitebabes"],
  "enable_faces": true,
  "max_faces_per_source": 5,
  "max_faces_total": 12,
  "reference_mode": "url"
}
```

### State Management

- Build runs as background asyncio task
- Progress updates stored in memory during run
- On completion, persisted to `runs.json` with full stats
- Config changes write directly to `sources.yaml`
- DB versions discovered by scanning `DATA_DIR` for `performers_*.db` files

### Run History Schema

```json
{
  "id": "run_20260130_143022",
  "started_at": "2026-01-30T14:30:22Z",
  "completed_at": "2026-01-30T16:45:10Z",
  "status": "completed",
  "params": {
    "sources": ["stashdb", "babepedia"],
    "enable_faces": true,
    "max_faces_total": 12
  },
  "stats": {
    "performers_processed": 75000,
    "faces_added": 12500,
    "errors": 23,
    "by_source": {
      "stashdb": {"performers": 70000, "faces": 10000},
      "babepedia": {"performers": 5000, "faces": 2500}
    }
  },
  "diff_from_previous": {
    "new_performers": 1234,
    "new_faces": 5678
  },
  "output_db": "performers_20260130_164510.db"
}
```

---

## React Dashboard

### Tech Stack

- Vite + React
- Tailwind CSS
- React Query (API state management)
- React Router (simple routing)

### Pages

**Dashboard (`/`)**
- Status card: "Idle" or "Running: stashdb (2,340/75,000)"
- Quick stats: total performers, total faces, last run date
- Recent runs list (last 5)
- "Start Build" button → opens modal with build form

**Runs (`/runs`)**
- Full run history table
- Columns: date, duration, sources, performers added, faces added, status
- Click row → run detail page

**Run Detail (`/runs/{id}`)**
- Run summary (duration, sources, params)
- Per-source breakdown table
- Diff from previous run
- Error log (collapsible)

**Config (`/config`)**
- Source cards (one per source)
  - Enable/disable toggle
  - Rate limit slider
  - Trust level dropdown
- Global settings section
- Save button

**Versions (`/versions`)**
- Table of `performers_*.db` files
- Columns: filename, date, size, performer count, face count

### Build Form Fields

| Field | Type | Default |
|-------|------|---------|
| Sources | Checkbox group | All enabled |
| Enable Faces | Toggle | true |
| Max Faces Per Source | Number input | 5 |
| Max Faces Total | Number input | 12 |
| Reference Mode | Radio (url/name) | url |

---

## Docker

### Dockerfile (Layered)

```dockerfile
# Layer 1: CUDA base + system deps
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04 AS base
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    nodejs npm \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    curl && rm -rf /var/lib/apt/lists/*

# Layer 2: ML dependencies (rarely changes)
FROM base AS ml-deps
WORKDIR /app
RUN python3.11 -m venv /app/venv
ENV PATH="/app/venv/bin:$PATH"
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir \
    deepface onnxruntime-gpu voyager tensorflow

# Layer 3: App dependencies (changes occasionally)
FROM ml-deps AS app-deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY web/package.json web/package-lock.json ./web/
RUN cd web && npm ci

# Layer 4: Build React app
FROM app-deps AS web-build
COPY web/ ./web/
RUN cd web && npm run build

# Layer 5: Runtime (changes frequently)
FROM ml-deps AS runtime
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=web-build /app/web/dist ./web/dist
COPY api/ ./api/

ENV DATA_DIR=/data
ENV PYTHONUNBUFFERED=1
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/status || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STASHDB_API_KEY` | Yes | - | StashDB API key |
| `THEPORNDB_API_KEY` | No | - | ThePornDB API key |
| `PMVSTASH_API_KEY` | No | - | PMVStash API key |
| `JAVSTASH_API_KEY` | No | - | JAVStash API key |
| `FANSDB_API_KEY` | No | - | FansDB API key |
| `FLARESOLVERR_URL` | No | http://localhost:8191 | FlareSolverr endpoint |
| `CHROME_CDP_URL` | No | http://localhost:9222 | Chrome DevTools for Indexxx |
| `DATA_DIR` | No | /data | Output directory |

### Volumes

| Mount | Purpose |
|-------|---------|
| `/data` | Database output, run history, config persistence |

### GPU Access

Uses NVIDIA Container Toolkit (`--gpus all`). Works in:
- Ubuntu VM with GPU passthrough (dev environment)
- unRAID with NVIDIA plugin (production)

---

## unRAID Template

```xml
<?xml version="1.0"?>
<Container version="2">
  <Name>stash-sense-trainer</Name>
  <Repository>ghcr.io/yourusername/stash-sense-trainer</Repository>
  <Registry>https://ghcr.io</Registry>
  <Network>bridge</Network>
  <Privileged>false</Privileged>
  <Support/>
  <Project/>
  <Overview>Database builder for Stash Sense face recognition. Scrapes performer images from StashDB and reference sites, generates face embeddings.</Overview>
  <Category>Tools:</Category>
  <WebUI>http://[IP]:[PORT:8080]/</WebUI>
  <Icon/>
  <ExtraParams>--gpus all</ExtraParams>
  <DateInstalled/>
  <Config Name="Web UI Port" Target="8080" Default="8080" Mode="tcp" Description="Dashboard port" Type="Port" Display="always" Required="true">8080</Config>
  <Config Name="Data Directory" Target="/data" Default="/mnt/user/appdata/stash-sense-trainer" Mode="rw" Description="Database output and config" Type="Path" Display="always" Required="true">/mnt/user/appdata/stash-sense-trainer</Config>
  <Config Name="StashDB API Key" Target="STASHDB_API_KEY" Default="" Description="Required. Get from stashdb.org account settings." Type="Variable" Display="always" Required="true"/>
  <Config Name="ThePornDB API Key" Target="THEPORNDB_API_KEY" Default="" Description="Optional. For ThePornDB enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="PMVStash API Key" Target="PMVSTASH_API_KEY" Default="" Description="Optional. For PMVStash enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="JAVStash API Key" Target="JAVSTASH_API_KEY" Default="" Description="Optional. For JAVStash enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="FansDB API Key" Target="FANSDB_API_KEY" Default="" Description="Optional. For FansDB enrichment." Type="Variable" Display="always" Required="false"/>
  <Config Name="FlareSolverr URL" Target="FLARESOLVERR_URL" Default="http://localhost:8191" Description="Optional. Required for Babepedia, IAFD, FreeOnes scrapers." Type="Variable" Display="always" Required="false">http://localhost:8191</Config>
  <Config Name="Chrome CDP URL" Target="CHROME_CDP_URL" Default="http://localhost:9222" Description="Optional. Required for Indexxx scraper." Type="Variable" Display="always" Required="false">http://localhost:9222</Config>
</Container>
```

---

## Database Versioning

**Simple approach for now:**

- Each completed build produces `performers_YYYYMMDD_HHMMSS.db`
- Files accumulate in `DATA_DIR`
- UI lists all versions with metadata (size, performer count, face count)
- Manual process: copy desired version to sidecar's data volume, restart sidecar

**Future enhancement (not in scope):**
- API-driven version switching
- Hot-reload without restart
- Automatic sync between trainer and sidecar

---

## Dependencies

### External Services

| Service | Required For | Notes |
|---------|--------------|-------|
| FlareSolverr | Babepedia, IAFD, FreeOnes | Cloudflare bypass |
| Chrome (CDP) | Indexxx | Currently blocked, future use |

### Python Dependencies (trainer-specific)

```
fastapi
uvicorn
aiofiles
httpx
beautifulsoup4
lxml
pyyaml
python-dotenv
voyager
deepface
tensorflow
onnxruntime-gpu
torch
torchvision
nodriver  # for Chrome CDP
```

---

## Implementation Phases

### Phase 1: Repo Setup & File Migration
- Create private `stash-sense-trainer` repo
- Copy enrichment files, scrapers, shared files
- Split `database.py` → full version stays, create `database_reader.py` for sidecar
- Verify CLI still works: `python enrichment_builder.py --status`
- Clean up public repo (remove trainer files)

### Phase 2: FastAPI Backend
- New `main.py` with API endpoints
- Background task runner wrapping existing `enrichment_coordinator`
- Run history persistence to `runs.json`
- Config read/write endpoints for `sources.yaml`
- SSE endpoint for log streaming

### Phase 3: React Dashboard
- Vite + React + Tailwind setup
- Dashboard page with status and build form
- Runs page with history table
- Run detail page with per-source breakdown
- Config page with source cards
- Versions page listing DB files

### Phase 4: Docker & unRAID
- Layered Dockerfile
- Build and test locally in Ubuntu VM
- Create unRAID template XML
- Document external dependencies (FlareSolverr, Chrome)
- Test GPU passthrough in both environments

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| How much code is shared? | Minimal - only `embeddings.py` and `models.py`. `database.py` gets split. |
| Stash connection needed? | No - trainer only scrapes external sources, never talks to user's Stash. |
| GPU approach? | NVIDIA Container Toolkit (`--gpus all`), works in VM and unRAID. |
| FlareSolverr bundled? | No - external dependency, document as prerequisite. |
| DB versioning? | Simple timestamped files, manual copy for now. |
| Python venv in Docker? | Optional but fine to keep for cleanliness. |

---

*This document approved 2026-01-30. Ready for implementation.*

# Stash Face Recognition System Design

**Date:** 2026-01-24
**Status:** Draft
**Author:** Collaborative design session

## Overview

A two-component system for identifying performers in Stash scenes using face recognition, with a shareable pre-built database keyed by StashDB IDs.

### Primary Use Case

Given a scene (or performer image) where the performer is unknown or unlinked, identify them by matching against a database of known performers from StashDB, returning their StashDB ID for easy linking.

### Design Principles

- **Local-first:** All processing happens on user's hardware, no cloud dependencies
- **Shareable database:** Pre-built database uses universal StashDB IDs, works for any Stash installation
- **Extensible:** Architecture supports future AI capabilities (scene similarity, object detection)
- **Privacy-friendly:** Users download database files; no images sent to external services

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER'S SYSTEM                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────┐              ┌─────────────────────────────┐  │
│  │  Stash Docker   │              │  Face Recognition Sidecar   │  │
│  │                 │    HTTP      │  (Docker)                   │  │
│  │  ┌───────────┐  │   ──────►    │                             │  │
│  │  │  Plugin   │  │  POST        │  ┌───────────────────────┐  │  │
│  │  │  (JS/UI)  │  │  /identify   │  │  FastAPI Backend      │  │  │
│  │  └───────────┘  │              │  │  - /identify          │  │  │
│  │                 │   ◄──────    │  │  - /health            │  │  │
│  │                 │   JSON       │  │  - /database/info     │  │  │
│  └────────┬────────┘  results     │  └───────────────────────┘  │  │
│           │                       │                             │  │
│           │ fetch                 │  ┌───────────────────────┐  │  │
│           │ images                │  │  ML Models            │  │  │
│           ▼                       │  │  - YOLOv8 (detect)    │  │  │
│  ┌─────────────────┐              │  │  - FaceNet512 (embed) │  │  │
│  │  Scene Files    │              │  │  - ArcFace (embed)    │  │  │
│  │  Screenshots    │◄─────────────│  └───────────────────────┘  │  │
│  │  Sprites        │   fetch      │                             │  │
│  └─────────────────┘              │  ┌───────────────────────┐  │  │
│                                   │  │  /data (volume)       │  │  │
│                                   │  │  - face_facenet.voy   │  │  │
│                                   │  │  - face_arcface.voy   │  │  │
│                                   │  │  - performers.json    │  │  │
│                                   │  │  - manifest.json      │  │  │
│                                   │  └───────────────────────┘  │  │
│                                   │                       GPU   │  │
│                                   └─────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Component 1: Face Recognition Sidecar (Docker)

### Purpose

Standalone Docker container that handles all ML inference and face matching.

### API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/identify` | POST | Identify faces in an image (by URL) |
| `/health` | GET | Health check, GPU status, database version |
| `/database/info` | GET | Current database stats and version |
| `/database/reload` | POST | Reload database files without restart |

### `/identify` Request

```json
{
  "image_url": "http://stash:9999/scene/123/screenshot",
  "sprite_url": "http://stash:9999/scene/123/sprite",
  "vtt_url": "http://stash:9999/scene/123/vtt",
  "max_results": 5,
  "min_confidence": 0.5
}
```

Either `image_url` (single image) or `sprite_url` + `vtt_url` (multi-frame) should be provided.

### `/identify` Response

```json
{
  "performers_detected": 3,
  "matches": [
    {
      "stashdb_id": "50459d16-787c-47c9-8ce9-a4cac9404324",
      "name": "Aaliyah Hadid",
      "country": "US",
      "confidence": 0.91,
      "appearances": 12,
      "image_url": "https://stashdb.org/images/b0aef39d-...",
      "stashdb_url": "https://stashdb.org/performers/50459d16-..."
    },
    {
      "stashdb_id": "9ac606f4-a784-4849-a2fa-5c8b04831e7e",
      "name": "Aaliyah Love",
      "country": "US",
      "confidence": 0.84,
      "appearances": 8,
      "image_url": "https://stashdb.org/images/...",
      "stashdb_url": "https://stashdb.org/performers/9ac606f4-..."
    },
    {
      "unknown": true,
      "appearances": 5,
      "face_thumbnail": "base64..."
    }
  ]
}
```

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `STASH_URL` | Yes | - | URL to Stash instance |
| `STASH_API_KEY` | Yes | - | Stash API key for image fetching |
| `DATABASE_PATH` | No | `/data` | Path to database files |
| `LOG_LEVEL` | No | `info` | Logging verbosity |
| `WORKERS` | No | `1` | Number of API workers |

### Volume Mounts

```
/data/                      # Database files (read-only)
  ├── face_facenet.voy      # FaceNet embedding index
  ├── face_arcface.voy      # ArcFace embedding index
  ├── performers.json       # Performer metadata
  ├── faces.json            # Index-to-ID mapping
  └── manifest.json         # Version and checksums
```

### Resource Requirements

| Mode | RAM | VRAM | Speed |
|------|-----|------|-------|
| CPU-only | 4GB | - | ~2-3 sec/image |
| GPU | 4GB | 4GB+ | ~200ms/image |

### Dockerfile

```dockerfile
FROM nvidia/cuda:12.8-cudnn9-runtime-ubuntu22.04

WORKDIR /app

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-venv python3-pip \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## Component 2: Stash Plugin (JavaScript)

### Purpose

Lightweight UI integration that sends requests to the sidecar and displays results.

### UI Integration Points

1. **Scene Page** - "Identify Performers" button
2. **Performer Page** - "Find on StashDB" button (for unlinked performers)
3. **Image Page** - "Identify" button

### Scene Identification Workflow

1. User clicks "Identify Performers" on scene page
2. Plugin sends sprite URL + VTT URL to sidecar
3. Sidecar extracts frames, detects faces, clusters by person, matches against database
4. Plugin displays modal with results:
   - Detected face thumbnail (from scene)
   - Best match reference image (from StashDB)
   - Name, confidence, appearances count
   - "Add to Scene" button (if performer in user's Stash)
   - "View on StashDB" link (always)

### Local Performer Lookup

When displaying results, plugin queries user's Stash to check if performer exists:

```graphql
query FindByStashDBId($stashdb_id: String!) {
  findPerformers(performer_filter: {
    stash_id_endpoint: {
      endpoint: "https://stashdb.org/graphql"
      stash_id: $stashdb_id
      modifier: EQUALS
    }
  }) {
    performers { id name }
  }
}
```

- If found: Show "Add to Scene" button
- If not found: Show "Not in Library" with StashDB link

### Plugin Settings

| Setting | Type | Description |
|---------|------|-------------|
| `sidecarUrl` | String | URL to sidecar API (default: `http://localhost:8000`) |
| `minConfidence` | Number | Minimum confidence threshold (default: 0.5) |
| `maxResults` | Number | Max results per face (default: 5) |

---

## Component 3: Pre-built Database

### Primary Key Design

**Universal ID Format:** `{stashbox_endpoint}:{performer_uuid}`

Example: `stashdb.org:50459d16-787c-47c9-8ce9-a4cac9404324`

This ensures:
- Database is shareable across all Stash installations
- Extensible to other stash-box instances (PMVStash, FansDB)
- Users match via their existing StashDB links

### File Structure

```
stash-face-db-2026.01.1/
  ├── face_facenet.voy      # Voyager index (FaceNet512 embeddings)
  ├── face_arcface.voy      # Voyager index (ArcFace embeddings)
  ├── performers.json       # Performer metadata
  ├── faces.json            # Face index to performer ID mapping
  └── manifest.json         # Version, checksums, build info
```

### performers.json

```json
{
  "stashdb.org:50459d16-787c-47c9-8ce9-a4cac9404324": {
    "name": "Aaliyah Hadid",
    "country": "US",
    "image_url": "https://stashdb.org/images/b0aef39d-...",
    "face_count": 12
  }
}
```

### faces.json

Maps Voyager index position to performer ID:

```json
[
  "stashdb.org:50459d16-787c-47c9-8ce9-a4cac9404324",
  "stashdb.org:50459d16-787c-47c9-8ce9-a4cac9404324",
  "stashdb.org:9ac606f4-a784-4849-a2fa-5c8b04831e7e",
  ...
]
```

(Same performer appears multiple times if they have multiple face images indexed)

### manifest.json

```json
{
  "version": "2026.01.1",
  "created_at": "2026-01-24T12:00:00Z",
  "performer_count": 45000,
  "face_count": 180000,
  "sources": ["stashdb.org"],
  "models": {
    "detector": "yolov8",
    "facenet_dim": 512,
    "arcface_dim": 512
  },
  "checksums": {
    "face_facenet.voy": "sha256:abc123...",
    "face_arcface.voy": "sha256:def456...",
    "performers.json": "sha256:789ghi...",
    "faces.json": "sha256:jkl012..."
  }
}
```

### Distribution

- GitHub Releases on the project repository
- Separate from Docker image (database updates independently)
- Users download `.tar.gz`, extract to data volume
- Sidecar auto-detects new database on startup or via `/database/reload`

---

## Database Builder Pipeline

### Purpose

Generate the shareable database files from StashDB (run by maintainers, not end users).

### Pipeline Stages

```
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Fetch   │───►│ Download │───►│  Detect  │───►│  Embed   │───►│  Index   │
│          │    │          │    │          │    │          │    │          │
│ Query    │    │ Images   │    │ Faces    │    │ Generate │    │ Build    │
│ StashDB  │    │ (cached) │    │ (YOLOv8) │    │ vectors  │    │ Voyager  │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
```

### Configuration

```yaml
# builder_config.yaml
stashdb:
  url: "https://stashdb.org/graphql"
  api_key: "${STASHDB_API_KEY}"
  rate_limit_delay: 0.5  # seconds between requests (EASILY CONFIGURABLE)
  max_images_per_performer: 10

output:
  directory: "./output"
  version: "2026.01.1"

cache:
  image_directory: "./cache/images"
  embeddings_directory: "./cache/embeddings"

processing:
  batch_size: 100
  workers: 4
  gpu: true
```

### Incremental Updates

- Track last-processed performer timestamp
- Query StashDB for performers updated since last build
- Merge new embeddings into existing index
- Avoids re-processing entire database each release

### Build Estimates (Full StashDB)

| Metric | Estimate |
|--------|----------|
| Performers | ~100,000 |
| Images | ~500,000 |
| Build time (GPU) | 8-12 hours |
| face_facenet.voy | ~500MB |
| face_arcface.voy | ~500MB |
| performers.json | ~20MB |
| Total download | ~1GB compressed |

---

## Future Extensibility

### Additional Data Sources (Documented for Later)

| Source | Type | Integration Notes |
|--------|------|-------------------|
| PMVStash | stash-box API | Same architecture, different endpoint |
| FansDB | stash-box API | Same architecture, different endpoint |
| Boobpedia | Web scraper | Wiki-style, HTML parsing needed |
| IAFD | Web scraper | Industry database, headshots |
| Babepedia | Web scraper | Gallery-style, many images |
| User contributions | Embeddings | Submit embeddings (not images) for missing performers |

### Additional AI Capabilities (Same Sidecar)

| Capability | Endpoint | Model | Use Case |
|------------|----------|-------|----------|
| Scene similarity | `/similar` | CLIP / DINOv2 | Find scenes from same studio/series |
| Object detection | `/detect-objects` | YOLOv8 / CLIP | Auto-tag: pool, outdoor, car |
| Scene classification | `/classify` | CLIP zero-shot | Indoor/outdoor, lighting style |
| OCR | `/extract-text` | PaddleOCR | Extract text from title cards |

Plugin updates to call new endpoints as they become available.

---

## Implementation Roadmap

### Phase 1: Database Builder
- [ ] Refactor current code to use StashDB ID as primary key
- [ ] Add configurable rate limiting
- [ ] Build full StashDB database
- [ ] Package and publish first release on GitHub

### Phase 2: Sidecar Container
- [ ] FastAPI service with `/identify`, `/health`, `/database/info`
- [ ] Voyager index loading and querying
- [ ] Sprite sheet + VTT parsing for multi-frame extraction
- [ ] Face clustering (same person across frames)
- [ ] Dockerfile with GPU support
- [ ] Docker Compose for local testing

### Phase 3: Stash Plugin
- [ ] JavaScript plugin scaffolding
- [ ] Scene page "Identify Performers" button
- [ ] Results modal with side-by-side comparison
- [ ] Local performer lookup via GraphQL
- [ ] "Add to Scene" functionality
- [ ] Settings page for sidecar URL

### Phase 4: Unraid Deployment
- [ ] Unraid Docker template XML
- [ ] GPU passthrough documentation
- [ ] Community Applications submission (optional)

### Phase 5: Refinement
- [ ] Performer page "Find on StashDB" feature
- [ ] On-demand frame capture from video player
- [ ] Confidence threshold tuning UI
- [ ] Performance optimization
- [ ] Incremental database updates

---

## Open Questions

1. **Database hosting:** GitHub Releases has size limits. May need alternative for 1GB+ files.
2. **Update notifications:** How to notify users of new database versions?
3. **Confidence calibration:** What threshold indicates "confident match" vs "needs review"?
4. **Unknown performers:** Workflow for faces that don't match anyone in database?

---

## Appendix: Technology Choices

| Component | Technology | Rationale |
|-----------|------------|-----------|
| Face detection | YOLOv8 | Fast, accurate, well-maintained |
| Face embeddings | FaceNet512 + ArcFace | Ensemble improves accuracy, same as stashface |
| Vector index | Voyager | Fast similarity search, compact storage |
| Sidecar API | FastAPI | Async, fast, good OpenAPI docs |
| Plugin | JavaScript | Native Stash plugin support |
| Containerization | Docker + nvidia-container-toolkit | GPU support, portable |

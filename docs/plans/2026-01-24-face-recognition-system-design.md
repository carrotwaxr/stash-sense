# Stash Face Recognition System Design

**Date:** 2026-01-24
**Status:** Phase 3 Complete - End-to-End Working
**Author:** Collaborative design session
**Last Updated:** 2026-01-25

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

## Current Status (2026-01-25)

### What's Working ✅

The full end-to-end pipeline is operational:

1. **Database** - 10k performers built (4,948 with faces, 8,491 total face embeddings)
2. **API Sidecar** - Running on Ubuntu VM, fetches sprites from Stash, detects faces, returns matches
3. **Stash Plugin** - "Identify Performers" button on scene pages, shows results modal, "Add to Scene" works

### Current Limitations

- Only ~10% of StashDB performers indexed (10k of 100k+)
- Detection accuracy will improve significantly as database grows
- Running manually (not yet containerized for Unraid)

### What's Next

| Task | Status | Can Run Concurrently? |
|------|--------|----------------------|
| **Full database build (100k performers)** | 🔨 In progress | N/A - running now |
| Unraid Docker template | Not started | ✅ Yes |
| GPU passthrough docs | Not started | ✅ Yes |
| Performance benchmarking | Not started | ✅ Yes |
| Database packaging/distribution | Not started | ⏳ After build completes |
| GitHub release prep | Not started | ✅ Yes |

### Concurrent Work While Database Builds

These tasks don't depend on the full database and can be done now:

1. **Unraid Docker Template** - Create XML template for Unraid Community Applications
2. **GPU Passthrough Documentation** - Document NVIDIA GPU setup for Unraid users
3. **Performance Benchmarking** - Test inference speed, memory usage with current 10k DB
4. **GitHub Repo Setup** - README, LICENSE, contributing guidelines, release workflow
5. **Plugin Polish** - UI improvements, error handling, loading states

---

## Implementation Progress

### Phase 1: Database Builder ✅ COMPLETE

**Status:** Validated and working

#### What's Built

| Component | File | Status |
|-----------|------|--------|
| Face detection | `embeddings.py` | ✅ RetinaFace (InsightFace) on GPU |
| Face embeddings | `embeddings.py` | ✅ FaceNet512 + ArcFace on CPU |
| Database builder | `database_builder.py` | ✅ Builds from StashDB |
| Face recognizer | `recognizer.py` | ✅ Queries Voyager indices |
| StashDB client | `stashdb_client.py` | ✅ GraphQL API with rate limiting |
| Local Stash client | `stash_client.py` | ✅ For testing against local library |
| Accuracy testing | `test_accuracy.py` | ✅ Full pipeline validation |

#### Technology Stack (Updated)

| Component | Technology | Device | Notes |
|-----------|------------|--------|-------|
| Face detection | **RetinaFace** (InsightFace) | GPU (ONNX Runtime) | Changed from YOLO - higher quality |
| Face embeddings | FaceNet512 + ArcFace | CPU (TensorFlow) | TensorFlow 2.20 lacks Blackwell GPU support |
| Vector search | Voyager | CPU | Fast cosine similarity |
| GPU | RTX 5060 Ti (Blackwell) | CUDA 12.8 | PyTorch nightly cu128 |

#### GPU Compatibility Notes

The RTX 5060 Ti uses NVIDIA Blackwell architecture (compute capability 12.0 / sm_120):
- **TensorFlow 2.20:** No pre-built CUDA kernels for sm_120; forced to CPU mode
- **PyTorch nightly cu128:** Works on GPU
- **ONNX Runtime GPU:** Works on GPU via CUDAExecutionProvider
- **Solution:** Use InsightFace (ONNX) for GPU detection, TensorFlow for CPU embeddings

### Validation Results

**Test Configuration:**
- 50 performers from local Stash with known StashDB links
- Database built from those same StashDB performers (3 images each)
- 100 test cases: 50 profile images + 50 scene screenshots

**Results:**

| Metric | Profile Images | Scene Screenshots | Combined |
|--------|----------------|-------------------|----------|
| Face detection | 100% | 86% | 93% |
| **Top-1 accuracy** | **86%** | 20% | 53% |
| Top-3 accuracy | 86% | 22% | 54% |
| Top-5 accuracy | 86% | 22% | 54% |

**Analysis:**
- **Profile-to-profile matching: 86% accuracy** - The core technology works well
- **Scene screenshots: 20% accuracy** - Expected; scenes contain multiple performers
- Scene failures are a workflow issue, not a model issue - need to match ALL faces, not just the first
- Clear score separation: correct matches avg 0.360, incorrect avg 0.693

**Conclusion:** Pipeline validated. Ready for full database build.

### Phase 2: FastAPI Sidecar ✅ COMPLETE

**Status:** Working end-to-end with Stash plugin

#### What's Built

| Component | File | Status |
|-----------|------|--------|
| FastAPI application | `main.py` | ✅ All endpoints implemented |
| Image identification | `/identify` | ✅ URL or base64 input |
| Scene identification | `/identify/scene` | ✅ Sprite sheet + face clustering |
| Health check | `/health` | ✅ Database status |
| Database info | `/database/info` | ✅ Version and stats |
| Sprite parser | `sprite_parser.py` | ✅ VTT + sprite extraction |
| Face clustering | `main.py` | ✅ Group same person across frames |
| Dockerfile | `Dockerfile` | ✅ Multi-stage with CUDA 12.4 |
| Docker Compose | `docker-compose.yml` | ✅ GPU support, volumes |

#### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check + database status |
| `/database/info` | GET | Database version, performer/face counts |
| `/identify` | POST | Identify faces in single image (URL or base64) |
| `/identify/url` | POST | Convenience endpoint with query params |
| `/identify/scene` | POST | Full scene analysis with sprite sheet |

#### Scene Identification Pipeline

The `/identify/scene` endpoint handles multi-performer scene identification:

1. **Fetch** sprite sheet + VTT from user's Stash instance
2. **Extract** frames evenly across the scene (configurable max)
3. **Detect** all faces in each frame using RetinaFace
4. **Cluster** faces by person (greedy clustering with embedding distance)
5. **Match** each person-cluster against the database
6. **Aggregate** scores across frames (consistent matches rank higher)
7. **Return** ranked matches per detected person

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
│           ▼                       │  │  - RetinaFace (detect)    │  │  │
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

**Current format (Phase 1-2, StashDB-only):**
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

**Future format (Phase 2.5+, multi-stash-box):**
```json
{
  "performer_00001": {
    "name": "Aaliyah Hadid",
    "country": "US",
    "stash_ids": {
      "stashdb.org": "50459d16-787c-47c9-8ce9-a4cac9404324",
      "theporndb.net": "78a2b3c4-...",
      "pmvstash.org": null,
      "fansdb.cc": null
    },
    "image_url": "https://stashdb.org/images/b0aef39d-...",
    "face_count": 12
  }
}
```

### faces.json

Maps Voyager index position to performer ID:

**Current format (Phase 1-2):**
```json
[
  "stashdb.org:50459d16-787c-47c9-8ce9-a4cac9404324",
  "stashdb.org:50459d16-787c-47c9-8ce9-a4cac9404324",
  "stashdb.org:9ac606f4-a784-4849-a2fa-5c8b04831e7e",
  ...
]
```

**Future format (Phase 2.5+):**
```json
[
  "performer_00001",
  "performer_00001",
  "performer_00002",
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
  "sources": ["stashdb.org", "theporndb.net"],
  "source_stats": {
    "stashdb.org": {"performers": 45000, "faces": 180000},
    "theporndb.net": {"performers": 12000, "faces": 48000, "cross_matched": 8500}
  },
  "models": {
    "detector": "retinaface",
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
│ StashDB  │    │ (cached) │    │(RetinaFace)│    │ vectors  │    │ Voyager  │
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

### Planned Use Cases

#### 1. Identify Unmatched Performers
**Problem:** Performers in local Stash without stash-box IDs
**Solution:** Use face recognition to match against the database using:
- Profile images attached to the performer
- Scene screenshots from their associated scenes
- Gallery images linked to the performer

This enables bulk identification of existing performers who haven't been linked to StashDB.

#### 2. Cross-Stash-Box Completionist
**Problem:** Performer has StashDB ID but not PMVStash/FansDB/ThePornDB/etc IDs
**Solution:** Given a performer with one stash-box ID:
1. Get their face embedding
2. Query databases built from other configured stash-box endpoints
3. Return matching IDs across all endpoints

Enables "completionist" workflow: find all stash-box IDs for a performer in one click.

##### Multi-Stash-Box Architecture

The database supports unified cross-referencing across multiple stash-box endpoints:

**Unified Database Model:**
```json
{
  "performer_001": {
    "name": "Jane Doe",
    "stash_ids": {
      "stashdb.org": "abc123-...",
      "theporndb.net": "def456-...",
      "fansdb.cc": null
    },
    "image_url": "https://stashdb.org/images/...",
    "face_count": 5
  }
}
```

**Build Process for Additional Stash-Boxes:**

When adding a new stash-box (e.g., ThePornDB) to an existing database:

```
For each performer in ThePornDB:
  1. Generate face embedding from their profile image
  2. Query existing database for matches (distance < threshold)
  3. If match found:
     → Add theporndb.net:<uuid> to existing performer's stash_ids
     → Optionally add new face embeddings to improve recognition
  4. If no match:
     → Create new performer entry with only theporndb.net:<uuid>
     → Add their face embeddings to the index
  5. Rebuild Voyager indices with any new embeddings
```

**Benefits:**
- No duplicate embeddings for the same person
- Single query finds matches across all stash-boxes
- Enables cross-referencing ("find all IDs for this face")
- Efficient storage and lookup

**API Response with Multi-Stash-Box:**
```json
{
  "matches": [{
    "name": "Jane Doe",
    "confidence": 0.92,
    "stash_ids": {
      "stashdb.org": "abc123-...",
      "theporndb.net": "def456-...",
      "pmvstash.org": null
    }
  }]
}
```

**Supported Stash-Box Endpoints:**
| Endpoint | GraphQL URL | Status |
|----------|-------------|--------|
| StashDB | https://stashdb.org/graphql | Primary source |
| ThePornDB | https://theporndb.net/graphql | Planned |
| PMVStash | https://pmvstash.org/graphql | Planned |
| FansDB | https://fansdb.cc/graphql | Planned |

#### 3. Identify All Performers in Scene
**Problem:** Scene has multiple performers, need to identify all of them
**Solution:**
1. Extract multiple frames from scene (via sprite sheet or video sampling)
2. Detect ALL faces across frames
3. Cluster faces by person (same person appears in multiple frames)
4. Match each person-cluster against the database
5. Return list of matches, one per person detected

This is the primary scene identification workflow.

#### 4. Find Similar Scenes
**Problem:** User wants to find scenes from the same studio, series, or shoot
**Solution:** Use visual similarity models (CLIP, DINOv2) to compare:
- Shoot location (same room/set)
- Lighting patterns
- Cinematography style
- Color grading
- Furniture and props

Returns scenes in library that are visually similar, likely from same studio/series.

#### 5. Identify Duplicate Scenes
**Problem:** Library has duplicate scenes (same content, different files)
**Solution:** Combine multiple signals:
- Face recognition: same performers in same positions
- Visual similarity: matching backgrounds, furniture, lighting
- Audio fingerprinting (future): same soundtrack
- Scene structure: similar shot sequences

Higher confidence when multiple signals agree. Helps deduplicate libraries.

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
| Duplicate detection | `/duplicates` | Multi-modal | Identify same content across files |

Plugin updates to call new endpoints as they become available.

---

## Implementation Roadmap

### Phase 1: Database Builder ✅
- [x] Refactor current code to use StashDB ID as primary key
- [x] Add configurable rate limiting (`STASHDB_RATE_LIMIT` env var, `--rate-limit` CLI)
- [x] Implement face detection with RetinaFace (GPU)
- [x] Implement embeddings with FaceNet512 + ArcFace (CPU)
- [x] Create recognizer for querying database
- [x] Build accuracy testing framework
- [x] Validate pipeline (86% accuracy on profile matching)
- [x] Incremental/resume support (auto-save every 100, graceful Ctrl+C)
- [x] Build 10k test database - **Complete (4,948 performers, 8,491 faces)**
- [ ] Build full StashDB database (~100k performers) - **In progress**
- [ ] Package and publish first release on GitHub

#### Current Database Build Status

**Completed:** 10,000 performers (test run) ✅
- 4,948 performers with detectable faces
- 8,491 total face embeddings indexed
- End-to-end testing successful

**Next:** Full 100k build

**Command:**
```bash
python database_builder.py --max-performers 100000 --max-images 5 --rate-limit 0.5 --output ./data --resume
```

**Features:**
- `--resume` flag to continue from where you left off
- Auto-saves every 100 performers
- Graceful Ctrl+C handling (saves progress before exit)
- Image caching to avoid re-downloading
- Progress tracking in `progress.json`

### Phase 2: Sidecar Container ✅ COMPLETE
- [x] FastAPI service with `/identify`, `/health`, `/database/info`
- [x] Voyager index loading and querying
- [x] Sprite sheet + VTT parsing for multi-frame extraction
- [x] Face clustering (same person across frames)
- [x] Dockerfile with GPU support (CUDA 12.4)
- [x] Docker Compose for local testing
- [x] Integration testing with 10k database
- [ ] Performance benchmarking

### Phase 3: Stash Plugin ✅ COMPLETE
- [x] JavaScript plugin scaffolding (`plugin/face-recognition.js`)
- [x] YAML manifest with settings (`plugin/face-recognition.yml`)
- [x] Python backend for CSP bypass (`plugin/face_recognition_backend.py`)
- [x] Scene page "Identify Performers" button
- [x] Results modal with match details
- [x] Local performer lookup via GraphQL
- [x] "Add to Scene" functionality
- [x] CSS styling (`plugin/face-recognition.css`)
- [x] End-to-end testing with 10k database

### Phase 4: Full Database Build 🔨 IN PROGRESS
- [ ] Build full StashDB database (~100k performers)
- [ ] Package database for distribution (GitHub Releases or alternative)
- [ ] Create database download/update documentation

### Phase 5: Unraid Deployment
- [ ] Unraid Docker template XML
- [ ] GPU passthrough documentation
- [ ] Community Applications submission (optional)

### Phase 6: Multi-Stash-Box Support (Future)
- [ ] Refactor data model for unified cross-referencing
- [ ] Add merge logic to database builder (match existing vs create new)
- [ ] Build ThePornDB database with cross-references to StashDB
- [ ] Add PMVStash support
- [ ] Add FansDB support
- [ ] API endpoint to query by any stash-box ID

### Phase 7: Refinement (Future)
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
| Face detection | **RetinaFace** (InsightFace) | State-of-the-art accuracy, GPU via ONNX Runtime |
| Face embeddings | FaceNet512 + ArcFace | Ensemble improves accuracy, same as stashface |
| Vector index | Voyager | Fast similarity search, compact E4M3 storage |
| Sidecar API | FastAPI | Async, fast, good OpenAPI docs |
| Plugin | JavaScript | Native Stash plugin support |
| Containerization | Docker + nvidia-container-toolkit | GPU support, portable |

### Face Detector Comparison (Evaluated)

| Detector | Framework | GPU Support | Quality | Status |
|----------|-----------|-------------|---------|--------|
| RetinaFace (InsightFace) | ONNX Runtime | ✅ Yes | Excellent | **Selected** |
| YOLO face (ultralytics) | PyTorch | ✅ Yes | Good | Tested, lower confidence |
| MTCNN (facenet-pytorch) | PyTorch | ✅ Yes | Good | Not tested |
| RetinaFace (DeepFace) | TensorFlow | ❌ No (sm_120) | Excellent | Incompatible |
| OpenCV | CPU only | ❌ No | Basic | Fallback only |

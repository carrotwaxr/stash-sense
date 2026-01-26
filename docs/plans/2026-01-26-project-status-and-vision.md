# Stash Sense: Project Status and Vision

**Date:** 2026-01-26
**Status:** Phase 4 (Full Database Build) - In Progress

---

## Executive Summary

This project aims to solve common Stash library organization problems through AI-powered face recognition and visual analysis. The core face recognition system is **working end-to-end** with a 10k performer test database, and the full 100k+ StashDB database build is currently running (estimated completion: ~2 days).

---

## Current State

### What's Working Today

| Component | Status | Details |
|-----------|--------|---------|
| **Face Detection** | Production-ready | RetinaFace on GPU via ONNX Runtime |
| **Face Embeddings** | Production-ready | FaceNet512 + ArcFace ensemble |
| **Database Builder** | Production-ready | Scrapes StashDB, handles rate limiting, supports resume |
| **FastAPI Sidecar** | Production-ready | All endpoints implemented, GPU support |
| **Stash Plugin** | Production-ready | "Identify Performers" button, results modal, "Add to Scene" |
| **Test Database** | Complete | 4,948 performers, 8,491 face embeddings |

### Validation Results

- **86% accuracy** on profile-to-profile matching
- Clear score separation between correct and incorrect matches
- Pipeline validated and ready for full-scale deployment

### What's In Progress

**Full StashDB Database Build** - Running now, ~2 days remaining
- Target: ~100,000 performers
- Expected: ~50,000 with detectable faces, ~150,000+ face embeddings
- Auto-saving progress every 100 performers
- Resume-capable if interrupted

---

## The Problems We're Solving

### Phase 1: Performer Identification (Current Focus)

These are the performer-centric problems we're actively solving:

| Problem | Solution | Status |
|---------|----------|--------|
| Scenes with unknown performers | Face recognition against StashDB database | Working |
| Performers unlinked to StashDB | Match existing performer images to database | Working |
| Multi-performer scenes | Detect all faces, cluster by person, match each | Working |
| Performer deduplication | Match face embeddings to identify same person | Planned |

### Phase 2: Cross-Stash-Box Completionist (Near-term)

| Problem | Solution | Status |
|---------|----------|--------|
| Performer linked to StashDB but not ThePornDB | Query unified database with all stash-box IDs | Designed |
| Performer linked to ThePornDB but not StashDB | Same as above - single face lookup, all IDs returned | Designed |
| Inconsistent performer data across endpoints | Show all available stash-box IDs for easy linking | Designed |

**Multi-Stash-Box Support Architecture:**
- Unified performer model with all stash-box IDs
- When adding new stash-box: match against existing embeddings first
- Single query returns matches across all endpoints
- No duplicate embeddings for same person

**Planned Endpoints:**
- StashDB (primary, building now)
- ThePornDB
- PMVStash
- FansDB

---

## Future Vision (Un-spec'd)

These capabilities are on the roadmap but not yet designed in detail. They represent the ultimate vision for the system.

### Scene Organization

| Capability | Use Case | Approach |
|------------|----------|----------|
| **Scene similarity** | Find scenes from same studio/series/shoot | Visual embeddings (CLIP/DINOv2) on backgrounds, furniture, lighting |
| **Duplicate detection** | Identify same content across different files | Multi-signal: face positions + visual similarity + perceptual hashing |
| **Orphan studio detection** | Find scenes that belong with a different studio | Cluster by visual similarity, identify outliers |
| **Series grouping** | Group "My Bang Van" scenes correctly | Combine visual similarity with metadata patterns |

### Automatic Tagging

| Capability | Use Case | Approach |
|------------|----------|----------|
| **Object detection** | Auto-tag: pool, outdoor, car, boat | YOLOv8 or CLIP zero-shot |
| **Scene classification** | Indoor/outdoor, lighting style, POV | CLIP zero-shot classification |
| **Position/act detection** | Auto-tag: threesome, doggy style, anal | Specialized models (requires careful evaluation) |
| **Text extraction** | Read studio watermarks, title cards | OCR (PaddleOCR) |

### Scene Matching

| Capability | Use Case | Approach |
|------------|----------|----------|
| **Match to known scenes** | phash failed, identify by frame inspection | Visual fingerprinting across multiple frames |
| **Partial match detection** | Scene is a clip from a longer known scene | Temporal visual similarity |

### Library Health

| Capability | Use Case | Approach |
|------------|----------|----------|
| **Recommended Actions page** | Surface problems and suggestions | Background analysis + plugin UI |
| **Performer merge suggestions** | "These 3 performers appear to be the same person" | Face similarity clustering |
| **Missing metadata detection** | Scenes with faces that match DB but aren't tagged | Automatic background scanning |

---

## Architecture Overview

```
User's System
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  Stash                              Face Recognition Sidecar            │
│  ┌──────────────┐                  ┌────────────────────────────────┐  │
│  │  Plugin UI   │  ───HTTP───►     │  FastAPI                       │  │
│  │  (JavaScript)│                  │  ├─ /identify/scene            │  │
│  │              │  ◄──JSON───      │  ├─ /identify                  │  │
│  └──────────────┘                  │  ├─ /health                    │  │
│         │                          │  └─ /database/info             │  │
│         │ fetch                    │                                │  │
│         ▼                          │  ML Models                     │  │
│  ┌──────────────┐                  │  ├─ RetinaFace (detect, GPU)   │  │
│  │ Scene Files  │◄────fetch────    │  ├─ FaceNet512 (embed, CPU)    │  │
│  │ Sprites      │                  │  └─ ArcFace (embed, CPU)       │  │
│  └──────────────┘                  │                                │  │
│                                    │  Database Volume               │  │
│                                    │  ├─ face_facenet.voy           │  │
│                                    │  ├─ face_arcface.voy           │  │
│                                    │  ├─ performers.json            │  │
│                                    │  └─ manifest.json              │  │
│                                    └────────────────────────────────┘  │
│                                                               [GPU]    │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Model

### For End Users

1. **Docker sidecar** - Runs alongside Stash, requires NVIDIA GPU (or CPU fallback)
2. **Pre-built database** - Download from GitHub Releases, mount as volume
3. **Stash plugin** - Install via plugin manager, configure sidecar URL
4. **Periodic updates** - New database releases as StashDB grows

### For Database Building (Maintainers)

1. **Dedicated Docker service** - Runs on maintainer hardware
2. **Scheduled builds** - Weekly/monthly rebuilds with incremental updates
3. **Publishing** - Push to GitHub Releases (or alternative for large files)
4. **Multi-source merging** - Cross-reference across stash-box endpoints

---

## Near-term Roadmap

| Task | Dependency | Priority |
|------|------------|----------|
| **Complete full database build** | None | P0 - In progress |
| **Unraid Docker template** | None | P1 |
| **GPU passthrough docs** | None | P1 |
| **GitHub release packaging** | Database build | P1 |
| **Database distribution solution** | Release packaging | P1 |
| **ThePornDB integration** | Full database | P2 |
| **PMVStash/FansDB integration** | Multi-stash-box architecture | P2 |
| **Performer page "Find on StashDB"** | None | P3 |
| **Background scanning mode** | Plugin polish | P3 |

---

## Long-term Roadmap (Unordered)

These are future capabilities that need design work:

- [ ] Scene visual similarity search
- [ ] Automatic scene tagging (objects, locations)
- [ ] Position/act detection and tagging
- [ ] Duplicate scene detection
- [ ] Scene matching to known database
- [ ] Studio organization assistance
- [ ] Performer merge/dedup workflow
- [ ] Recommended Actions dashboard
- [ ] User contribution mechanism (embedding-only submissions)
- [ ] Additional data sources (Boobpedia, IAFD, Babepedia)

---

## Technical Notes

### Why CPU for Embeddings?

The RTX 5060 Ti uses NVIDIA Blackwell architecture (sm_120). TensorFlow 2.20 doesn't have pre-built CUDA kernels for this architecture yet. The solution:
- **GPU:** RetinaFace via ONNX Runtime (works with sm_120)
- **CPU:** FaceNet512 + ArcFace via TensorFlow (fast enough, ~100ms per face)

### Database Sizing Estimates

| Metric | 10k DB (current) | 100k DB (building) |
|--------|------------------|-------------------|
| Performers | 4,948 w/ faces | ~50,000 w/ faces |
| Face embeddings | 8,491 | ~150,000 |
| face_facenet.voy | ~15MB | ~500MB |
| Total download | ~30MB | ~1GB compressed |

### Confidence Interpretation

| Score Range | Interpretation |
|-------------|----------------|
| < 0.4 | High confidence match |
| 0.4 - 0.6 | Likely match, verify visually |
| 0.6 - 0.8 | Possible match, needs review |
| > 0.8 | Unlikely match |

(Lower = better, these are cosine distances)

---

## Open Questions

1. **Database hosting:** GitHub Releases has size limits (~2GB per file). Need alternative for 1GB+ database files?
2. **Update notifications:** How to notify users when new database versions are available?
3. **User contributions:** Can users submit embeddings for missing performers without sharing images?
4. **Background scanning:** How to implement background analysis without impacting Stash performance?
5. **Tagging models:** Which models are appropriate for position/act detection? Need careful evaluation.

---

## Getting Involved

Once the full database build completes:
1. Database will be packaged and released
2. Documentation for self-hosting will be published
3. Unraid template will be submitted to Community Applications
4. Feedback and contributions welcome via GitHub

---

*This document will be updated as the project progresses.*

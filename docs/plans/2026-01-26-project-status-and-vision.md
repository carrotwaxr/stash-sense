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
| **Faceless performers (OnlyFans/FansDB)** | Body recognition via Person Re-ID models | Future (see Body Recognition section) |

### Phase 2: Cross-Stash-Box Completionist (Near-term)

| Problem | Solution | Status |
|---------|----------|--------|
| Performer linked to StashDB but not ThePornDB | Query unified database with all stash-box IDs | Designed |
| Performer linked to ThePornDB but not StashDB | Same as above - single face lookup, all IDs returned | Designed |
| Inconsistent performer data across endpoints | Show all available stash-box IDs for easy linking | Designed |
| Performer exists only in JAVStash/other DB | Create standalone entry with that DB as primary source | Designed |
| Limited face embeddings per performer | Accumulate from multiple sources via identity graph | Designed |

**Multi-Stash-Box Support Architecture:**
- Unified performer model with all stash-box IDs AND external URLs (twitter, IAFD, etc.)
- **Performer Identity Graph** - links each performer across all sources (see [performer-identity-graph.md](2026-01-27-performer-identity-graph.md))
- When adding new stash-box: match by explicit IDs first, then name/metadata, then face embeddings
- Single query returns matches across all endpoints
- Embeddings accumulate from all sources (StashDB average: 1.04 faces/performer → goal: 5-10+)
- Concurrent scraping with single writer queue (see [concurrent-scraping-architecture.md](2026-01-26-concurrent-scraping-architecture.md))

**Planned Endpoints:**
- StashDB (primary, building now)
- ThePornDB
- PMVStash
- FansDB
- JAVStash

**Handling Performers Not in StashDB:**

The system must support performers who exist in secondary databases (ThePornDB, JAVStash, etc.) but not in StashDB. This is especially critical for JAVStash, where many performers have no Western database presence.

*Data Model Changes:*
1. `stashdb_id` becomes **nullable/optional** (not required)
2. Add `primary_source` field (enum: `stashdb`, `tpdb`, `javstash`, `pmvstash`, `fansdb`)
3. `stash_ids` JSONB continues to store all cross-database IDs
4. Add `external_urls` JSONB to store non-stashbox links (twitter, IAFD, babepedia, etc.)
5. Face embeddings become the **universal identifier** across all sources
6. See [Performer Identity Graph](2026-01-27-performer-identity-graph.md) for full data model

*Ingestion Flow for Secondary Databases:*
1. For each performer in the secondary DB:
   - Try to match by explicit IDs in `stash_ids` → if found, add new IDs to existing entry
   - Try to match by **face embeddings** against entire database → if found, link to existing entry
   - If no match found → create new entry with secondary DB as `primary_source`
2. Face recognition serves as the primary **deduplication mechanism** across databases
3. Even with different names/conventions (e.g., Japanese vs romanized), faces match

*Reconciliation When StashDB Adds a Performer Later:*
- During re-sync, match new StashDB performers against existing embeddings
- If matched to a standalone entry (e.g., JAVStash-only performer):
  - Add the new `stashdb_id` to their `stash_ids`
  - Optionally promote `primary_source` to `stashdb` if preferred for metadata

---

## Future Vision (Un-spec'd)

These capabilities are on the roadmap but not yet designed in detail. They represent the ultimate vision for the system.

### Crowd-Sourced Face Database (Cloud Service)

**The Problem:** Stash-box endpoints like JAVStash have limited images per performer (~1 image each), resulting in lower face recognition accuracy. Additionally, promotional images may not represent how performers appear in actual scenes.

**The Solution:** A public cloud service (similar to stash-box architecture) that accepts user-contributed face data from their libraries, allowing us to crowd-source better face embeddings from real scene content.

**How It Works:**

1. **User has a scene identified via stash-box** (fingerprint match to StashDB/JAVStash/etc.)
2. **User's Stash plugin extracts face data** from that scene:
   - VTT-style face track data (timestamps + bounding boxes)
   - Face embeddings (not raw images - privacy preserving)
   - Associated performer stash-box IDs from the scene metadata
3. **Plugin submits to the cloud service** with proof of identification:
   - Scene fingerprint (phash/oshash)
   - Stash-box scene ID (proves legitimate match)
   - Face embeddings with performer attribution
4. **Cloud service aggregates contributions:**
   - Multiple users submitting same scene = higher confidence
   - Voting/consensus mechanism for performer attribution
   - Quality filtering (embedding similarity clustering)
5. **Improved models published back to community**

**Why This Helps JAV (and others):**
- JAVStash has 21k+ performers but only 1 promotional image each
- Users' actual video libraries contain hundreds of frames per performer
- Scene-extracted faces show real appearance, not just promo shots
- Crowd-sourcing bypasses the single-image limitation entirely

**Privacy Considerations:**
- Only embeddings transmitted, never raw images or video
- Embeddings are one-way (can't reconstruct faces)
- Scene fingerprints already public via stash-box
- Users opt-in to contribution

**Technical Requirements:**
- Cloud service with stash-box-like API
- Submission authentication (prevent spam/poisoning)
- Embedding quality validation
- Consensus algorithm for multi-user submissions
- Periodic model rebuilds incorporating contributions

**Open Questions:**
- Hosting costs for cloud service?
- Authentication model (stash-box API keys? separate accounts?)
- How to handle conflicting attributions?
- Minimum submission threshold before including in model?
- How to bootstrap JAV performers with no existing embeddings?

### Body Recognition (Person Re-ID)

**The Problem:** In the modern age of OnlyFans, many performers in StashDB and especially FansDB don't show their faces in content. Face recognition alone cannot identify these performers, creating a significant gap in coverage.

**The Solution:** Add body recognition capabilities using Person Re-Identification (Re-ID) models, which are specifically designed to match people across different viewpoints based on body appearance.

**Why This Matters:**
- OnlyFans/FansDB performers often hide faces for privacy
- Multi-person scenes where faces aren't always visible
- Body confirmation can increase confidence in face matches
- Distinctive features (tattoos, body type) are identifying signals

**Technical Approaches:**

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Person Re-ID Models** | Models like OSNet, AGW, TransReID designed for matching people across surveillance cameras | Purpose-built for body matching, robust to pose/viewpoint changes, well-researched | May need domain adaptation for adult content |
| **Body Pose + Proportions** | Extract skeletal landmarks (MediaPipe/OpenPose), encode body proportions | Interpretable, could encode height/body type | Requires consistent poses, less distinctive alone |
| **Tattoo Detection** | Identify and encode distinctive body markings | Highly distinctive when present, could be separate matching signal | Not everyone has them, requires training data, location varies |
| **Whole-Body Appearance** | CNN-based appearance vectors for full body crops | Similar workflow to face embeddings | Less mature than face recognition for this domain |

**Recommended Approach: Person Re-ID**

Person Re-ID models are the most promising because:
1. **Purpose-built**: Designed specifically for matching people across viewpoints
2. **Pose-invariant**: Handle different poses, angles, partial occlusion
3. **Lightweight options**: OSNet is efficient, TransReID is SOTA
4. **Active research area**: Continuous improvements in the field

**Implementation Plan:**

```
Pipeline Addition:
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ Person Detection │────►│ Body Crop        │────►│ Re-ID Embedding │
│ (YOLO/similar)   │     │ (torso + legs)   │     │ (OSNet/TransReID)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

1. **Detection**: Use existing person detectors (YOLO) to find people in frames
2. **Cropping**: Extract body regions (excluding or including face region)
3. **Embedding**: Generate body embeddings using Re-ID model
4. **Matching**: Compare against known performer body embeddings

**Database Changes:**
- Add `body_embeddings` alongside `face_embeddings`
- Store body embedding type (model used)
- Track which images contributed body vs face embeddings

**Matching Strategy:**
1. Try face matching first (higher accuracy when available)
2. If no face detected or low confidence, try body matching
3. If both available, combine scores for higher confidence
4. Body-only matches go to review queue (lower confidence tier)

**Challenges:**
- **Clothing variation**: Models may over-fit to clothing rather than body shape
- **Training data**: Need body images reliably linked to known performers
- **Multi-person disambiguation**: Track which body belongs to which person in scenes
- **Privacy sensitivity**: Body embeddings may be more sensitive than face embeddings

**Model Candidates:**
| Model | Speed | Accuracy | Notes |
|-------|-------|----------|-------|
| OSNet | Fast | Good | Lightweight, good for real-time |
| AGW | Medium | Better | Strong baseline |
| TransReID | Slow | SOTA | Transformer-based, highest accuracy |

**Integration with Existing System:**
- Parallel to face pipeline, not replacing it
- Same Voyager index pattern for body embeddings
- Body embeddings accumulated from same sources (StashDB, reference sites)
- Plugin UI shows body match confidence separately

**Open Questions:**
- How to handle clothing variation across images?
- Should we fine-tune Re-ID models on adult content?
- Separate confidence thresholds for body vs face matches?
- How to combine face + body scores optimally?
- Privacy implications of storing body embeddings?

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

- [ ] Performer identity graph implementation (see [performer-identity-graph.md](2026-01-27-performer-identity-graph.md))
  - [ ] Update data model with `external_urls` field
  - [ ] Implement explicit ID linking (ThePornDB stashdb_id field)
  - [ ] Implement URL parsing for external links
  - [ ] Implement name/metadata fuzzy matching
  - [ ] Implement confidence scoring for multi-signal matches
- [ ] Scene visual similarity search
- [ ] Scene tagging system (see [scene-tagging-strategy.md](2026-01-26-scene-tagging-strategy.md))
  - [ ] CLIP integration (zero-shot classification) - settings, locations, scene types
  - [ ] YOLOv8 integration (object detection) - furniture, props, vehicles
  - [ ] PaddleOCR integration (watermark/studio detection)
  - [ ] DINOv2 integration (visual similarity)
  - [ ] Position/act detection - **consider VLM approach** (see Haven VLM Connector analysis in scene-tagging-strategy.md). VLMs excel here but are 100x slower than CLIP. Hybrid approach recommended: CLIP/YOLO for most tags, optional VLM for action tags only.
- [ ] Duplicate scene detection
- [ ] Scene matching to known database
- [ ] Studio organization assistance
- [ ] Performer merge/dedup workflow
- [ ] Recommended Actions dashboard
- [ ] **Crowd-sourced cloud service** (see Future Vision section)
  - [ ] Cloud service architecture design
  - [ ] Submission API (VTT + embeddings + stash-box IDs)
  - [ ] Consensus/voting algorithm for attributions
  - [ ] Plugin integration for automated submissions
- [ ] JAVStash integration (requires standalone performer support, unique naming challenges)
  - [ ] Scene frame extraction to supplement single-image limitation
  - [ ] Japanese name handling (romanization, ordering)
- [ ] Reference site enrichment engine (see [reference-site-enrichment.md](2026-01-26-reference-site-enrichment.md))
- [ ] Individual scrapers: Babepedia, IAFD, Boobpedia, FreeOnes
- [ ] **Body recognition (Person Re-ID)** (see Future Vision section)
  - [ ] Evaluate Re-ID models (OSNet lightweight vs TransReID SOTA)
  - [ ] Add person detection + body cropping to pipeline
  - [ ] Build body embedding database alongside face embeddings
  - [ ] Implement body matching with separate confidence thresholds
  - [ ] Combined face + body scoring for higher confidence matches

---

## Accuracy Enhancement Strategies

These strategies can improve face recognition accuracy beyond the basic pipeline.

### Multi-Detector Fusion

The current 50% "no face detected" rate may partly be detection failures. Running multiple detectors and taking the union could recover missed faces:

```python
def detect_faces_multi(image):
    """Run multiple detectors, union results."""
    all_faces = []

    # Primary detector (current)
    all_faces.extend(retinaface_detect(image))

    # Secondary detectors
    all_faces.extend(mtcnn_detect(image))
    all_faces.extend(yolo_face_detect(image))

    # Dedupe overlapping detections (IoU > 0.5)
    return non_max_suppression(all_faces)
```

**Trade-off:** Slower processing, but potentially 10-20% more performers with usable embeddings. Consider for database building (offline) but not real-time inference.

**Implementation:** Could use lower detection thresholds during database building and higher thresholds during query time. More false positives in the database are acceptable if they get filtered at query time.

### Negative Mining

Track rejected matches to improve the system over time:

```python
class NegativeMiningTracker:
    """Track when users reject suggested matches."""

    def record_rejection(self, query_performer_id, rejected_match_id, embedding_distance):
        """User said 'this is NOT the same person'."""
        # Store the rejection
        self.rejections.append({
            'query': query_performer_id,
            'rejected': rejected_match_id,
            'distance': embedding_distance,
            'timestamp': now()
        })

    def analyze_rejections(self):
        """Find systematic errors."""
        # If performer X is frequently rejected when matched to Y,
        # they might have similar appearances but are different people

        # If rejections cluster at certain distance thresholds,
        # we should adjust our confidence thresholds

        # If certain performers are frequently false-positive sources,
        # flag them for review (possible duplicate entries in source DB)
```

**Uses:**
1. Adjust confidence thresholds based on real rejection data
2. Identify performers who look similar but aren't the same person
3. Find systematic issues (duplicate entries in stash-boxes, etc.)
4. Build "definitely not the same person" pairs to avoid suggesting

### Scene Frame Extraction (Crowd-Sourced)

Expanding on the crowd-sourced concept from earlier: users with fingerprint-matched scenes can contribute embeddings automatically.

**Why This Helps:**
- JAVStash has 21k performers but only 1 image each → single promo shot
- Users have actual scenes with hundreds of frames per performer
- Scene frames show real appearance vs promotional images
- Diversity: different angles, lighting, makeup across a scene

**Privacy-Preserving Flow:**
```
User's Stash has scene matched via fingerprint to StashDB
    ↓
Plugin extracts faces from scene (local processing)
    ↓
Plugin generates embeddings (local processing)
    ↓
Only embeddings + stash-box scene ID sent to cloud
    ↓
Cloud aggregates: multiple users submitting = higher confidence
    ↓
Embeddings incorporated into next database release
```

**Trust Signals:**
- Scene fingerprint proves legitimate match
- Multiple users submitting same scene = consensus
- Embedding similarity to existing performer embeddings = validation
- User reputation score over time

**Open Questions:**
- Minimum submissions before incorporating?
- How to handle conflicting attributions?
- Authentication/anti-spam for submissions?

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
5. **Tagging models:** CLIP zero-shot for proof of concept, then identify gaps needing fine-tuning. See [scene-tagging-strategy.md](2026-01-26-scene-tagging-strategy.md). **Note:** Haven VLM Connector exists for action/position tagging via VLMs - recommend hybrid approach (CLIP/YOLO for speed, optional VLM for action tags where it excels).
6. **Metadata precedence:** When a performer exists in multiple databases, which source's metadata (name, aliases, bio) should be preferred? Options: always prefer StashDB if available, or let user configure preference order.
7. **JAVStash naming:** How to handle Japanese name ordering (family name first vs given name first) and romanization variations?
8. **Reference site scraping:** ToS considerations for scraping Babepedia, IAFD, etc.? Rate limit tuning needed per site.
9. **Identity graph storage:** Should `external_urls` be stored in the Voyager index metadata, separate JSON, or a SQLite database for querying? Current JSON-based approach may not scale.
10. **Low embedding count:** Current StashDB average is 1.04 faces/performer. Target 5-10+ faces for reliable matching. Need to prioritize ThePornDB integration and reference site scraping to accumulate embeddings.
11. **Body recognition for faceless performers:** Many OnlyFans/FansDB performers don't show faces. Person Re-ID models (OSNet, TransReID) could enable body-based matching. Need to evaluate accuracy on adult content and determine confidence thresholds.

---

## Getting Involved

Once the full database build completes:
1. Database will be packaged and released
2. Documentation for self-hosting will be published
3. Unraid template will be submitted to Community Applications
4. Feedback and contributions welcome via GitHub

---

*This document will be updated as the project progresses.*

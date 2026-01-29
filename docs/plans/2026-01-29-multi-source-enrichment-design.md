# Multi-Source Enrichment Design

**Date:** 2026-01-29
**Status:** Designed

---

## Overview

### Problem Statement

The current database has 58K performers with 103K faces (avg 1.77 faces/performer). Many performers have only 1 face from StashDB, limiting recognition accuracy. The database builder only processes one source sequentially.

### Goals

1. **Enrich face diversity** - Increase average faces per performer to 5-10 from multiple sources
2. **Parallel scraping** - Run multiple sources concurrently, respecting per-domain rate limits
3. **Per-source limits** - Prevent any single source from dominating a performer's embeddings
4. **Quality control** - Filter out poor images (small faces, extreme angles, scene screenshots)
5. **Configurable trust** - Different validation strategies per source based on reliability
6. **Collaborative tuning** - Workflow to validate new sources before enabling them

---

## Architecture

### Two-Phase Approach

```
Phase A: Stash-Boxes (Authoritative)          Phase B: Reference Sites (Enrichment)
┌─────────────────────────────────────┐       ┌─────────────────────────────────────┐
│  ┌─────────┐ ┌─────────┐ ┌────────┐ │       │  ┌──────────┐ ┌──────┐ ┌─────────┐ │
│  │ StashDB │ │ThePornDB│ │PMVStash│ │       │  │Babepedia │ │ IAFD │ │FreeOnes │ │
│  └────┬────┘ └────┬────┘ └───┬────┘ │       │  └────┬─────┘ └──┬───┘ └────┬────┘ │
│       │           │          │      │       │       │          │          │      │
│  ┌────┴───┐  ┌────┴───┐ ┌────┴───┐  │       │  ┌────┴───┐ ┌────┴───┐ ┌────┴───┐  │
│  │JAVStash│  │ FansDB │ │  ...   │  │       │  │Boobpedia│ │PornPics│ │  ...   │  │
│  └────┬───┘  └────┬───┘ └────┬───┘  │       │  └────┬────┘ └────┬───┘ └────┬───┘  │
│       └──────────┬┴──────────┘      │       │       └──────────┬┴──────────┘      │
│                  ▼                  │       │                  ▼                  │
│         ┌──────────────┐            │       │         ┌──────────────┐            │
│         │ Write Queue  │            │       │         │ Write Queue  │            │
│         └──────┬───────┘            │       │         └──────┬───────┘            │
│                ▼                    │       │                ▼                    │
│         ┌──────────────┐            │       │         ┌──────────────┐            │
│         │   Database   │            │       │         │   Database   │            │
│         └──────────────┘            │       │         └──────────────┘            │
└─────────────────────────────────────┘       └─────────────────────────────────────┘
         Creates performers                           Enriches existing only
```

**Phase A (Stash-boxes):**
- Run concurrently - each has its own rate limit and domain
- Can create new performers
- Authoritative for metadata
- Uses identity graph linking (see `performer-identity-graph.md`)

**Phase B (Reference sites):**
- Run after Phase A completes
- Only add faces to existing performers
- Never create new performers
- Trust levels determine validation requirements

### Write Queue

Each phase uses a single-writer queue to prevent race conditions:

```python
@dataclass
class WriteMessage:
    operation: WriteOperation  # CREATE_PERFORMER, ADD_EMBEDDING, ADD_STASH_ID
    source: str
    performer_data: Optional[ScrapedPerformer] = None
    performer_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    embedding_type: Optional[str] = None
    image_url: Optional[str] = None
```

Scrapers run in parallel, writing to queue. Single consumer processes writes sequentially, handling deduplication and merging.

---

## Source Configuration

### Config File Structure

```yaml
# sources.yaml

global:
  max_faces_per_performer: 20  # Overall cap across all sources
  default_rate_limit: 60       # requests/min if not specified

quality_filters:
  min_face_size: 80            # pixels
  min_image_size: 400          # pixels (shorter dimension)
  min_detection_confidence: 0.8
  max_face_angle: 45           # degrees from frontal
  prefer_single_face: true     # skip multi-face images for high-trust sources

stash_boxes:
  stashdb:
    enabled: true
    url: "https://stashdb.org/graphql"
    rate_limit: 240            # requests/min
    max_faces: 5               # per-source limit
    priority: 1                # metadata priority (lower = higher priority)

  theporndb:
    enabled: true
    url: "https://api.theporndb.net"
    rate_limit: 240
    max_faces: 5
    priority: 2

  pmvstash:
    enabled: true
    url: "https://pmvstash.org/graphql"
    rate_limit: 300
    max_faces: 3
    priority: 3

  javstash:
    enabled: true
    url: "https://javstash.org/graphql"
    rate_limit: 300
    max_faces: 3
    priority: 4

  fansdb:
    enabled: false             # not yet tested
    url: "https://fansdb.cc/graphql"
    rate_limit: 240
    max_faces: 3
    priority: 5

reference_sites:
  babepedia:
    enabled: true
    trust_level: high          # add without validation
    rate_limit: 60
    max_faces: 5
    gender_filter: female      # only use for female performers
    needs_flaresolverr: true

  iafd:
    enabled: true
    trust_level: high
    rate_limit: 120
    max_faces: 3
    needs_flaresolverr: true

  freeones:
    enabled: true
    trust_level: medium        # require face match
    rate_limit: 30
    max_faces: 5
    gender_filter: female
    needs_flaresolverr: false

  boobpedia:
    enabled: true
    trust_level: medium
    rate_limit: 60
    max_faces: 3
    gender_filter: female
    needs_flaresolverr: false

  pornpics:
    enabled: false             # scene galleries, needs clustering
    trust_level: low           # require clustering validation
    rate_limit: 60
    max_faces: 5
    needs_flaresolverr: false

  elitebabes:
    enabled: false
    trust_level: low
    rate_limit: 60
    max_faces: 5
    gender_filter: female
    needs_flaresolverr: false

  javdatabase:
    enabled: false
    trust_level: medium
    rate_limit: 60
    max_faces: 3
    needs_flaresolverr: false
```

### CLI Overrides

```bash
# Run with specific sources only
python enrichment_builder.py --sources stashdb,theporndb

# Run all enabled sources
python enrichment_builder.py --sources all

# Disable a source for this run
python enrichment_builder.py --disable-source pornpics

# Override per-source limit
python enrichment_builder.py --source-max-faces babepedia=10

# Override global max
python enrichment_builder.py --max-faces-total 30

# Dry run (no database writes)
python enrichment_builder.py --dry-run

# Test mode (process N performers only)
python enrichment_builder.py --test-performers 100
```

---

## Trust Levels & Validation Strategies

### Trust Level Definitions

| Trust Level | Validation Strategy | Use Case |
|-------------|---------------------|----------|
| **high** | Add faces without validation | Professional headshot sources (Babepedia, IAFD) |
| **medium** | Require face match to existing embeddings | Mixed content sources (FreeOnes, Boobpedia) |
| **low** | Require clustering across multiple images | Scene galleries (PornPics, EliteBabes) |

### Validation Logic

```python
def validate_face_for_enrichment(
    face_embedding: np.ndarray,
    performer: Performer,
    source_config: SourceConfig,
    image_metadata: ImageMetadata
) -> ValidationResult:
    """Determine if a detected face should be added to a performer."""

    # Quality filters apply to all sources
    if not passes_quality_filters(face_embedding, image_metadata):
        return ValidationResult.REJECTED_QUALITY

    # Check per-source and total face limits
    if performer.face_count >= config.global.max_faces_per_performer:
        return ValidationResult.REJECTED_TOTAL_LIMIT

    source_face_count = performer.faces_from_source(source_config.name)
    if source_face_count >= source_config.max_faces:
        return ValidationResult.REJECTED_SOURCE_LIMIT

    # Trust-level specific validation
    if source_config.trust_level == "high":
        return ValidationResult.ACCEPTED

    elif source_config.trust_level == "medium":
        # Require match to existing embeddings
        if performer.face_count == 0:
            return ValidationResult.REJECTED_NO_EXISTING_FACES

        match_distance = find_best_match(face_embedding, performer.embeddings)

        # Tunable threshold - stricter when performer has few faces
        threshold = get_match_threshold(performer.face_count)

        if match_distance < threshold:
            return ValidationResult.ACCEPTED
        else:
            return ValidationResult.REJECTED_NO_MATCH

    elif source_config.trust_level == "low":
        # Require clustering validation
        return ValidationResult.REQUIRES_CLUSTERING


def get_match_threshold(existing_face_count: int) -> float:
    """Stricter threshold when fewer faces to match against."""
    if existing_face_count == 1:
        return 0.35  # Very strict - single face could be wrong
    elif existing_face_count <= 3:
        return 0.40  # Moderately strict
    else:
        return 0.45  # More lenient with good existing data
```

### Clustering Strategy (for low-trust sources)

```python
def cluster_validation(
    performer_name: str,
    images: list[Image],
    existing_embeddings: list[np.ndarray]
) -> list[np.ndarray]:
    """
    For low-trust sources, detect faces across all images,
    cluster them, and return only the dominant cluster.
    """
    all_faces = []
    for image in images:
        faces = detect_faces(image)
        for face in faces:
            if passes_quality_filters(face):
                embedding = generate_embedding(face)
                all_faces.append(embedding)

    if not all_faces:
        return []

    # Cluster faces by similarity
    clusters = cluster_embeddings(all_faces, threshold=0.4)

    # Find dominant cluster (most faces)
    dominant_cluster = max(clusters, key=len)

    # If we have existing embeddings, verify dominant cluster matches
    if existing_embeddings:
        cluster_representative = np.mean(dominant_cluster, axis=0)
        match_distance = find_best_match(cluster_representative, existing_embeddings)

        if match_distance > 0.45:
            # Dominant cluster doesn't match existing - suspicious
            log_suspicious_cluster(performer_name, dominant_cluster, existing_embeddings)
            return []

    return dominant_cluster
```

---

## Quality Filters

### Filter Parameters

| Filter | Value | Rationale |
|--------|-------|-----------|
| `min_face_size` | 80px | Filters scene screenshots, keeps waist-up portraits |
| `min_image_size` | 400px | Filters thumbnails, keeps standard web images |
| `min_detection_confidence` | 0.8 | RetinaFace confidence threshold |
| `max_face_angle` | 45° | Filters extreme profiles, keeps 3/4 views |
| `prefer_single_face` | true | For high-trust sources, skip multi-face images |

### Implementation

```python
@dataclass
class QualityFilters:
    min_face_size: int = 80
    min_image_size: int = 400
    min_detection_confidence: float = 0.8
    max_face_angle: float = 45.0
    prefer_single_face: bool = True


def passes_quality_filters(
    face: DetectedFace,
    image: Image,
    filters: QualityFilters,
    source_trust: str
) -> bool:
    """Check if a detected face passes quality requirements."""

    # Image resolution check
    min_dimension = min(image.width, image.height)
    if min_dimension < filters.min_image_size:
        return False

    # Face size check
    face_height = face.bbox[3] - face.bbox[1]
    if face_height < filters.min_face_size:
        return False

    # Detection confidence check
    if face.confidence < filters.min_detection_confidence:
        return False

    # Face angle check (if available from detector)
    if hasattr(face, 'yaw') and abs(face.yaw) > filters.max_face_angle:
        return False

    # Single face preference for high-trust sources
    if source_trust == "high" and filters.prefer_single_face:
        if face.total_faces_in_image > 1:
            return False

    return True
```

### Face Size Reference

| Scenario | Typical Face Size | Passes 80px? |
|----------|-------------------|--------------|
| Close-up headshot (400×600) | 200-300px | ✓ |
| Waist-up portrait (400×600) | 90-150px | ✓ |
| Full body (400×600) | 60-90px | Marginal |
| Scene screenshot (1920×1080) | 50-100px | Marginal |
| Group/wide shot | <50px | ✗ |

---

## Per-Source and Total Face Limits

### Limit Logic

```python
def should_add_face(
    performer: Performer,
    source: str,
    source_config: SourceConfig,
    global_config: GlobalConfig
) -> tuple[bool, str]:
    """Check if we should add another face from this source."""

    # Check global limit
    if performer.face_count >= global_config.max_faces_per_performer:
        return False, f"Total limit reached ({global_config.max_faces_per_performer})"

    # Check per-source limit
    source_count = performer.faces_by_source.get(source, 0)
    if source_count >= source_config.max_faces:
        return False, f"Source limit reached for {source} ({source_config.max_faces})"

    return True, "OK"
```

### Example Scenario

Config:
- `max_faces_per_performer: 20`
- StashDB: `max_faces: 5`
- Babepedia: `max_faces: 5`
- FreeOnes: `max_faces: 5`
- IAFD: `max_faces: 3`
- Boobpedia: `max_faces: 3`

Result for a performer found on all sources:
- StashDB: 5 faces
- Babepedia: 5 faces
- FreeOnes: 5 faces
- IAFD: 3 faces
- Boobpedia: 2 faces (stopped at total=20)

**Total: 20 faces from 5 sources** - diverse coverage, no single source dominates.

---

## Source Validation Workflow

### Collaborative Process

```
┌─────────────────────────────────────────────────────────────────┐
│                    Source Validation Workflow                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. IDENTIFY SOURCE                                              │
│     Claude: "Let's evaluate PornPics as a source.               │
│              Check 5 female and 5 male performers.               │
│              Questions:                                          │
│              - Are images headshots or scene galleries?          │
│              - Do male performers show their faces?              │
│              - Are there multiple people in images?"             │
│                                                                  │
│  2. MANUAL INSPECTION                                            │
│     User browses source, reports observations                    │
│                                                                  │
│  3. BUILD TEST TOOLING                                           │
│     Claude creates test script:                                  │
│     - Scrape sample performers                                   │
│     - Run face detection                                         │
│     - Compare to existing DB embeddings                          │
│     - Generate quality report                                    │
│                                                                  │
│  4. REVIEW RESULTS                                               │
│     - Match rate to existing performers                          │
│     - False positive analysis                                    │
│     - Quality distribution (face sizes, angles)                  │
│     - Gender-specific findings                                   │
│                                                                  │
│  5. CONFIGURE & ENABLE                                           │
│     - Set trust level                                            │
│     - Set gender filter if needed                                │
│     - Set per-source limits                                      │
│     - Enable in config                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Test Script Template

```python
# test_source_<name>.py

async def test_source(
    source_name: str,
    performer_ids: list[str],
    dry_run: bool = True
):
    """
    Test a source against known performers without inserting.

    Outputs:
    - images_found: Total images scraped
    - faces_detected: Faces passing quality filters
    - match_rate: % matching existing DB embeddings
    - false_positives: Faces matching wrong performer
    - quality_distribution: Face sizes, angles, confidence
    """

    scraper = get_scraper(source_name)
    results = SourceTestResults()

    for performer_id in performer_ids:
        performer = db.get_performer(performer_id)

        # Scrape images
        images = await scraper.get_images(performer.name)
        results.images_found += len(images)

        for image in images:
            faces = detect_faces(image)

            for face in faces:
                if not passes_quality_filters(face):
                    results.quality_rejected += 1
                    continue

                results.faces_detected += 1
                embedding = generate_embedding(face)

                # Check against this performer's existing faces
                own_match = find_best_match(embedding, performer.embeddings)

                # Check against entire database
                db_matches = search_database(embedding, threshold=0.5)

                if own_match < 0.45:
                    results.correct_matches += 1
                elif db_matches and db_matches[0].performer_id != performer.id:
                    results.false_positives += 1
                    results.log_false_positive(performer, db_matches[0])
                else:
                    results.no_match += 1

    return results.generate_report()
```

---

## Implementation Plan

### Phase 1: Infrastructure (Foundation)

1. **Source configuration system**
   - YAML config loader
   - CLI argument parser with overrides
   - Config validation

2. **Quality filters module**
   - Face size filter
   - Image resolution filter
   - Face angle estimation (if available from RetinaFace)
   - Single-face preference

3. **Write queue implementation**
   - AsyncIO queue
   - Message types
   - Single-writer consumer

### Phase 2: Stash-Box Scrapers

1. **Refactor existing StashDB scraper**
   - Extract to BaseScraper interface
   - Add per-source face tracking
   - Integrate with write queue

2. **Implement additional stash-box scrapers**
   - ThePornDB (REST API)
   - PMVStash (GraphQL)
   - JAVStash (GraphQL)
   - FansDB (GraphQL)

3. **Scraper coordinator**
   - Parallel execution
   - Per-source rate limiting
   - Progress tracking and resume

### Phase 3: Reference Site Scrapers

1. **FlareSolverr integration**
   - Docker compose addition
   - Fallback request handler

2. **Implement reference site scrapers**
   - Babepedia
   - IAFD
   - FreeOnes
   - Boobpedia

3. **Trust-level validation**
   - High: direct add
   - Medium: face matching
   - Low: clustering

### Phase 4: Testing & Tuning

1. **Source validation tooling**
   - Test script template
   - Quality report generation
   - False positive analysis

2. **Collaborative tuning**
   - Evaluate each source
   - Set appropriate trust levels
   - Configure gender filters

3. **Threshold tuning**
   - Match thresholds per face count
   - Quality filter values
   - Clustering parameters

---

## Database Schema Updates

### New Fields Needed

```sql
-- Track face source for per-source limits
ALTER TABLE faces ADD COLUMN source TEXT;

-- Track source-specific face counts (denormalized for performance)
ALTER TABLE performers ADD COLUMN faces_by_source TEXT;  -- JSON: {"stashdb": 5, "babepedia": 3}

-- Scrape progress per source
CREATE TABLE IF NOT EXISTS scrape_progress (
    source TEXT PRIMARY KEY,
    last_processed_id TEXT,
    last_processed_time TEXT,
    performers_processed INTEGER DEFAULT 0,
    faces_added INTEGER DEFAULT 0,
    errors INTEGER DEFAULT 0
);
```

### Migration

```python
def migrate_existing_faces():
    """Mark existing faces as from StashDB."""
    db.execute("""
        UPDATE faces
        SET source = 'stashdb'
        WHERE source IS NULL
    """)
```

---

## Monitoring & Observability

### Progress Logging

```
[2026-01-29 14:32:15] Enrichment Progress:
  Sources active: stashdb, theporndb, babepedia
  Performers processed: 12,450 / 58,000 (21.5%)

  Per-source stats:
    stashdb:    8,230 faces added, 42 errors, queue: 156
    theporndb:  3,120 faces added, 18 errors, queue: 89
    babepedia:  2,890 faces added, 7 errors,  queue: 45

  Quality filter rejections:
    face_too_small: 4,521
    image_too_small: 892
    low_confidence: 1,203
    extreme_angle: 567
    multi_face: 2,341

  Write queue size: 290
  Estimated completion: 4.2 hours
```

### Resume Capability

```bash
# Resume after interruption
python enrichment_builder.py --resume

# Check progress
python enrichment_builder.py --status
```

---

## Open Questions

1. **Face angle detection**: Does RetinaFace provide yaw/pitch, or do we need a separate model?

2. **Embedding deduplication**: If Babepedia has the same image as StashDB, should we detect and skip?

3. **Performer gender**: Where do we get gender reliably? StashDB has it, but we need it for gender filters.

4. **Stash-box priority ordering**: If StashDB and ThePornDB have conflicting metadata, which wins?

5. **Clustering performance**: For low-trust sources, clustering across many images may be slow. Batch size limits?

---

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Avg faces per performer | 1.77 | 5-10 |
| Performers with 1 face | ~40% | <10% |
| Performers with 5+ faces | ~5% | >60% |
| Recognition accuracy (test set) | 70% | 85%+ |
| Sources integrated | 1 | 5+ stash-boxes, 4+ reference |

---

*This document will be updated as implementation progresses.*

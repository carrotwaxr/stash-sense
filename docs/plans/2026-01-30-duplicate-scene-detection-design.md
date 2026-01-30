# Duplicate Scene Detection Design

**Date:** 2026-01-30
**Status:** Design Complete

---

## Overview

Multi-signal duplicate scene detection that improves on Stash's phash-only approach by combining:
- Stash-box ID matching (authoritative)
- Face fingerprint similarity (who appears and in what proportions)
- Metadata heuristics (performers, studio, date, duration)

Results in a 0-100% confidence score with human-readable reasoning shown in the UI.

---

## Why This Is Better Than Stash's Phash

Stash's current duplicate detection:
- Takes 25 screenshots at fixed time intervals (skipping first/last 5%)
- Creates a 5x5 sprite grid (160x160 per frame)
- Generates a single 64-bit perceptual hash
- Compares via Hamming distance

**Problems with fixed-point sampling:**

| Scenario | Why Phash Fails |
|----------|-----------------|
| Different intros/outros | All 25 sample points hit different frames |
| Trimmed versions | Samples are offset, no alignment |
| Different aspect ratios | 4:3 vs 16:9 produces different crops |
| Watermarks/logos | Text overlays affect the hash |
| Color grading differences | Re-releases with different correction |

**Our approach is robust to these issues** because:
- Face proportions are ratio-based, not position-based
- Face detection is resolution-agnostic
- Same people appear in same ratios regardless of trimming
- Faces are detected, not pixel-hashed

---

## Signal Hierarchy & Weighting

| Signal | Weight | Rationale |
|--------|--------|-----------|
| **Stash-box ID match** | 100% (authoritative) | Same endpoint + same stash_id = same scene by definition |
| **Face signature match** | Up to 85% | Who appears and in what proportion is highly identifying |
| **Metadata heuristics** | Up to 60% | Circumstantial - requires base signal to contribute |

### Combined Scoring

```python
if stash_id match on same endpoint:
    confidence = 100%

else:
    face_score = calculate_face_similarity(scene_a, scene_b)  # 0-85
    meta_score = calculate_metadata_similarity(scene_a, scene_b)  # 0-60

    # Combine with diminishing returns
    primary = max(face_score, meta_score)
    secondary = min(face_score, meta_score)
    confidence = primary + (secondary * 0.3)

    # Cap at 95% without stash-box confirmation
    confidence = min(confidence, 95%)
```

---

## Metadata Heuristics Scoring

Metadata scoring requires a **base signal** (studio or performers). Duration and date are **confirmation multipliers**, not standalone factors.

```python
def metadata_score(scene_a, scene_b):
    # No metadata = can't score
    if no_useful_metadata(scene_a) or no_useful_metadata(scene_b):
        return 0, "Insufficient metadata"

    # BASE SIGNALS (need at least one)
    base = 0
    reasons = []

    if same_studio(scene_a, scene_b):
        base += 20
        reasons.append("Same studio")

    performer_overlap = jaccard_performers(scene_a, scene_b)
    if performer_overlap == 1.0:
        base += 20
        reasons.append("Exact performer match")
    elif performer_overlap >= 0.5:
        base += 12
        reasons.append(f"Performers overlap ({performer_overlap:.0%})")

    if base == 0:
        return 0, "No studio or performer match"

    # CONFIRMATION MULTIPLIERS (only if base > 0)
    multiplier = 1.0

    if same_date(scene_a, scene_b):
        multiplier += 0.5
        reasons.append("Same release date")
    elif within_days(scene_a, scene_b, 7):
        multiplier += 0.3
        reasons.append("Release dates within 7 days")

    if duration_diff_seconds(scene_a, scene_b) <= 5:
        multiplier += 0.5
        reasons.append("Duration within 5s")
    elif duration_diff_seconds(scene_a, scene_b) <= 30:
        multiplier += 0.3
        reasons.append("Duration within 30s")

    score = min(base * multiplier, 60)  # Cap at 60
    return score, " + ".join(reasons)
```

### Example Scenarios

| Scenario | Base | Multiplier | Final |
|----------|------|------------|-------|
| Same studio + same date + duration within 5s | 20 | 2.0 | **40%** |
| Exact performers + same studio + same date + duration within 5s | 40 | 2.0 | **60%** (capped) |
| Same performers only | 20 | 1.0 | **20%** |
| No studio, no performer overlap | 0 | - | **0%** |

---

## Face Signature Matching

### Scene Fingerprint Structure

```python
@dataclass
class SceneFingerprint:
    faces: dict[str, FaceAppearance]  # performer_id -> metrics
    total_faces_detected: int
    frames_analyzed: int

@dataclass
class FaceAppearance:
    performer_id: str
    face_count: int           # How many times detected
    avg_confidence: float     # Average match confidence
    proportion: float         # face_count / total_faces_detected
```

### Similarity Calculation

```python
def face_signature_similarity(fp_a: SceneFingerprint, fp_b: SceneFingerprint) -> tuple[float, str]:
    all_performers = set(fp_a.faces.keys()) | set(fp_b.faces.keys())
    all_performers.discard("unknown")

    if not all_performers:
        return 0, "No identified performers in either scene"

    # Compare proportions (robust to different frame counts/durations)
    proportion_diffs = []
    matches = []

    for performer_id in all_performers:
        prop_a = fp_a.faces.get(performer_id, FaceAppearance(...)).proportion
        prop_b = fp_b.faces.get(performer_id, FaceAppearance(...)).proportion

        diff = abs(prop_a - prop_b)
        proportion_diffs.append(diff)

        if prop_a > 0.1 and prop_b > 0.1:  # Both have meaningful presence
            matches.append(performer_id)

    avg_diff = sum(proportion_diffs) / len(proportion_diffs)

    # Convert to similarity score (0-85 range)
    similarity = max(0, 85 * (1 - avg_diff * 2))

    reason = f"{len(matches)} shared performers, {avg_diff:.1%} avg proportion difference"
    return similarity, reason
```

---

## UI Presentation

### Confidence Thresholds

| Range | Color | Label | Suggested Action |
|-------|-------|-------|------------------|
| 90-100% | Green | "Confirmed" / "High confidence" | Auto-suggest merge |
| 70-89% | Yellow | "Likely duplicate" | Review recommended |
| 50-69% | Orange | "Possible duplicate" | Manual verification |
| < 50% | Gray | "Low confidence" | Show but don't highlight |

### Example Display

```
+-------------------------------------------------------------+
| 87% - High confidence duplicate                              |
+-------------------------------------------------------------+
| Scene A: "Studio Y - Some Title" (1080p, 32:15)             |
| Scene B: "random-filename.mp4" (720p, 32:18)                |
+-------------------------------------------------------------+
| * Face analysis: 2 shared performers, 3% avg proportion     |
|   difference                                                |
| * Metadata: Same studio + Duration within 5s                |
+-------------------------------------------------------------+
| Breakdown: Face 78% + Metadata 30% (x0.3) = 87%             |
+-------------------------------------------------------------+
```

---

## Integration Architecture

### New Analyzer

```python
# api/analyzers/duplicate_scenes.py
class DuplicateScenesAnalyzer(BaseAnalyzer):
    """
    Detects duplicate scenes using multi-signal analysis.
    """

    type = "duplicate_scenes"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        # Phase 1: Build/update fingerprints for new scenes
        await self._update_fingerprints(incremental)

        # Phase 2: Compare fingerprints to find duplicates
        duplicates = await self._find_duplicates()

        # Phase 3: Create recommendations
        created = 0
        for pair in duplicates:
            if pair.confidence >= self.config.min_confidence:
                self.rec_db.create_recommendation(
                    type=self.type,
                    target_type="scene",
                    target_id=str(pair.scene_a_id),
                    confidence=pair.confidence / 100,
                    details={
                        "scene_b_id": pair.scene_b_id,
                        "confidence": pair.confidence,
                        "reasoning": pair.reasoning,
                        "signal_breakdown": asdict(pair.signal_breakdown)
                    }
                )
                created += 1

        return AnalysisResult(
            items_processed=self.fingerprint_count,
            recommendations_created=created
        )
```

### Scheduler Integration

```python
DEFAULT_SCHEDULES = {
    # ... existing ...
    "duplicate_scenes": {
        "interval_hours": 24,     # Run daily by default
        "incremental": True,      # Only fingerprint new/modified scenes
        "batch_size": 50,         # Scenes per batch (throttling)
        "batch_delay_sec": 5,     # Delay between batches
    },
}
```

### Environment Variables

```bash
# Duplicate detection throttling
DEDUP_ENABLED=true                    # Enable/disable duplicate scanner
DEDUP_BATCH_SIZE=50                   # Scenes to fingerprint per batch
DEDUP_BATCH_DELAY_SEC=5               # Pause between batches
DEDUP_MIN_CONFIDENCE=50               # Only surface duplicates above this %
DEDUP_SCAN_INTERVAL_HOURS=24          # How often to run full comparison

# Frame extraction throttling
FRAME_EXTRACTION_CONCURRENT=2         # Parallel ffmpeg processes
FRAME_EXTRACTION_TIMEOUT_SEC=30       # Per-frame timeout
```

### Database Schema (stash_sense.db)

```sql
-- Scene fingerprints for duplicate detection
CREATE TABLE scene_fingerprints (
    id INTEGER PRIMARY KEY,
    stash_scene_id INTEGER UNIQUE NOT NULL,
    total_faces INTEGER NOT NULL,
    frames_analyzed INTEGER NOT NULL,
    fingerprint_status TEXT DEFAULT 'complete',  -- 'complete', 'no_stream', 'extraction_failed'
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE scene_fingerprint_faces (
    id INTEGER PRIMARY KEY,
    fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
    performer_id TEXT NOT NULL,
    face_count INTEGER NOT NULL,
    avg_confidence REAL NOT NULL,
    proportion REAL NOT NULL,
    UNIQUE(fingerprint_id, performer_id)
);

CREATE INDEX idx_fingerprint_scene ON scene_fingerprints(stash_scene_id);
CREATE INDEX idx_fingerprint_status ON scene_fingerprints(fingerprint_status);
```

### Two-Phase Operation

| Phase | When | Resource Usage | User Impact |
|-------|------|----------------|-------------|
| **Fingerprint Generation** | Incrementally as scenes added/modified | High (ffmpeg, GPU) | Throttled via batch_size + delay |
| **Duplicate Comparison** | Scheduled (daily default) | Low (DB queries only) | Minimal |

---

## Error Handling

### Fingerprint Generation Failures

| Scenario | Handling | Stored State |
|----------|----------|--------------|
| Scene has no stream URL | Skip, log warning | `fingerprint_status = 'no_stream'` |
| ffmpeg extraction timeout | Retry once, then skip | `fingerprint_status = 'extraction_failed'` |
| No faces detected | Store empty fingerprint | `total_faces = 0` (valid state) |
| All faces are "unknown" | Store with unknowns | Valid for comparison |
| Stash unreachable | Pause analyzer, retry later | Existing scheduler retry logic |

### Comparison Edge Cases

```python
def calculate_duplicate_confidence(scene_a, scene_b) -> DuplicateMatch | None:
    # Same scene ID guard
    if scene_a.id == scene_b.id:
        return None

    fp_a = get_fingerprint(scene_a)
    fp_b = get_fingerprint(scene_b)

    # One or both missing fingerprints
    if fp_a is None or fp_b is None:
        # Can still check stash-box IDs and metadata
        stashbox = check_stashbox_match(scene_a, scene_b)
        if stashbox.matched:
            return DuplicateMatch(confidence=100, ...)

        meta_score, meta_reason = metadata_score(scene_a, scene_b)
        if meta_score >= 40:
            return DuplicateMatch(
                confidence=min(meta_score, 50),
                reasoning=["No face fingerprint available", meta_reason]
            )
        return None

    # Both scenes have zero faces
    if fp_a.total_faces == 0 and fp_b.total_faces == 0:
        meta_score, meta_reason = metadata_score(scene_a, scene_b)
        return DuplicateMatch(
            confidence=min(meta_score, 50),
            reasoning=["No faces detected in either scene", meta_reason]
        ) if meta_score >= 40 else None

    # Asymmetric: one has faces, other doesn't
    if fp_a.total_faces == 0 or fp_b.total_faces == 0:
        stashbox = check_stashbox_match(scene_a, scene_b)
        if stashbox.matched:
            return DuplicateMatch(
                confidence=100,
                reasoning=[f"Identical {stashbox.endpoint_name} ID (face detection asymmetric)"]
            )
        return None

    # Normal case
    return _calculate_full_confidence(scene_a, scene_b, fp_a, fp_b)
```

### Performance Safeguards

```python
async def _find_duplicates(self) -> list[DuplicateMatch]:
    fingerprints = await self.get_all_fingerprints()

    # Safety: limit O(n^2) comparisons
    if len(fingerprints) > 10000:
        logger.warning(f"Large library ({len(fingerprints)} scenes), limiting comparison")
        fingerprints = sorted(fingerprints, key=lambda f: f.updated_at, reverse=True)[:10000]

    duplicates = []
    comparisons = 0

    for i, fp_a in enumerate(fingerprints):
        for fp_b in fingerprints[i+1:]:
            comparisons += 1

            # Yield periodically
            if comparisons % 1000 == 0:
                await asyncio.sleep(0)

            match = calculate_duplicate_confidence(fp_a.scene, fp_b.scene)
            if match and match.confidence >= self.config.min_confidence:
                duplicates.append(match)

    return duplicates
```

### User Controls

| Setting | Default | Purpose |
|---------|---------|---------|
| `enabled` | `true` | Master toggle |
| `min_confidence` | `50` | Threshold for surfacing duplicates |
| `scan_new_scenes_only` | `false` | Only fingerprint new scenes |
| `exclude_organized` | `false` | Skip organized scenes |
| `max_comparisons_per_run` | `50000` | Safety limit |

---

## Stash Schema Reference

### Stash-Box ID Storage

```sql
-- From Stash's schema
CREATE TABLE scene_stash_ids (
  scene_id INTEGER,
  endpoint VARCHAR(255),    -- stash-box URL
  stash_id VARCHAR(36),     -- UUID on that endpoint
  updated_at DATETIME,
  FOREIGN KEY(scene_id) REFERENCES scenes(id) ON DELETE CASCADE
);
```

Each scene can have multiple stash_ids (one per configured endpoint). Two scenes with the same `(endpoint, stash_id)` pair are definitionally the same scene.

### Scene Metadata Available

From Stash GraphQL:
- `performers` - list of performer IDs
- `studio` - studio ID (with parent hierarchy)
- `date` - release date
- `files[].duration` - video duration in seconds

---

## Future Enhancements

- **Audio fingerprinting** - More robust to visual re-encoding
- **Clustering optimization** - Use locality-sensitive hashing to reduce O(n^2) comparisons
- **Cross-library detection** - Compare against known scene database (like stash-box fingerprints)
- **Partial match detection** - Identify clips that are subsets of longer scenes

---

*Design validated through brainstorming session 2026-01-30.*

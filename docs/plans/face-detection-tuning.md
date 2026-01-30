# Face Detection & Clustering Tuning Guide

**Purpose:** Reference for tuning face detection, recognition, and clustering parameters.
**Last Updated:** 2026-01-29

---

## Pipeline Overview

```
Video → Frame Extraction → Face Detection → Size/Confidence Filter → Embedding → Clustering → Match Merge → Deduplication
```

| Stage | What It Does | Key Parameters |
|-------|--------------|----------------|
| Frame Extraction | Extract N frames from video via ffmpeg | `num_frames`, `start_offset_pct`, `end_offset_pct` |
| Face Detection | RetinaFace finds faces in each frame | `min_face_confidence` |
| Size Filter | Remove small/distant faces | `min_face_size` |
| Embedding | Generate 1024-dim vector per face | FaceNet512 (512) + ArcFace (512) |
| Clustering | Group same-person faces | `cluster_threshold` (L2 distance) |
| Match Merge | Merge clusters with same best match | Automatic |
| Deduplication | Each performer → one person only | Automatic |

---

## Tuning Parameters

### Frame Extraction (`/identify/scene`)

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `num_frames` | 20 | 1-100 | More frames = more faces to cluster, slower |
| `start_offset_pct` | 0.05 | 0.0-0.5 | Skip intro (avoid logos, titles) |
| `end_offset_pct` | 0.95 | 0.5-1.0 | Skip outro (avoid credits) |

**How frame sampling works (current implementation):**

```
Video timeline:
|----[===========================================]----|
     5%                                          95%
     ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓    ↓
     Single frame extracted at each evenly-spaced point
```

For `num_frames=40` on a 30-minute video:
1. Calculate usable range: 5% to 95% = 1.5min to 28.5min = 27 minutes
2. Divide into 39 intervals (40-1)
3. Extract **1 single frame** at each of the 40 timestamps
4. Each extraction is an independent ffmpeg seek + decode

The frames are NOT sequential - they are spread across the video. This means:
- Good temporal coverage across the whole video
- But only one chance per sample point to catch a good face angle
- If that single frame has a bad angle, we miss that time region

### Face Detection

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `min_face_confidence` | 0.5 | 0.1-1.0 | Higher = fewer false positives, may miss faces |
| `min_face_size` | 80 | 20-200 | Minimum face width/height in pixels |

### Clustering

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `cluster_threshold` | 0.9 | 0.5-2.0 | L2 distance on 1024-dim vector. Higher = more aggressive grouping |

---

## Video Resolution & Face Size

**Key Question:** How big are faces at different video resolutions?

### Typical Face Sizes by Resolution

Assuming a **medium shot** (face is ~15-25% of frame height):

| Resolution | Frame Size | Face Height Range | Usable with 80px min? |
|------------|------------|-------------------|----------------------|
| 240p | 426×240 | 36-60 px | No - most faces filtered |
| 360p | 640×360 | 54-90 px | Marginal - close-ups only |
| 480p | 854×480 | 72-120 px | Yes - medium shots work |
| 720p | 1280×720 | 108-180 px | Yes - most shots work |
| 1080p | 1920×1080 | 162-270 px | Yes - distant shots work |
| 4K | 3840×2160 | 324-540 px | Yes - all shots work |

### Face Size Guidelines

| Shot Type | Face as % of Height | At 480p | At 720p | At 1080p |
|-----------|---------------------|---------|---------|----------|
| Extreme close-up | 80-100% | 384-480 px | 576-720 px | 864-1080 px |
| Close-up | 40-60% | 192-288 px | 288-432 px | 432-648 px |
| Medium shot | 15-25% | 72-120 px | 108-180 px | 162-270 px |
| Wide shot | 5-10% | 24-48 px | 36-72 px | 54-108 px |

### Recommended `min_face_size` by Content Quality

| Content Quality | Recommended min_face_size | Rationale |
|-----------------|---------------------------|-----------|
| 240p-360p (low) | 40-50 px | Accept lower quality, more false positives |
| 480p (SD) | 60-80 px | Balance quality and coverage |
| 720p (HD) | 80-100 px | Good quality, filter distant faces |
| 1080p+ (FHD) | 100-120 px | High quality only, very reliable |

---

## Clustering Behavior

### How Clustering Works

1. First face creates cluster #1
2. Each subsequent face:
   - Compute L2 distance to all cluster centroids
   - If distance < threshold → add to nearest cluster
   - If distance >= threshold → create new cluster
3. After initial clustering → merge clusters with same best match

### Threshold Guidelines

| Threshold | Behavior | Best For |
|-----------|----------|----------|
| 0.5 | Strict - requires very similar faces | Same lighting/angle |
| 0.7 | Moderate - tolerates some variation | Well-lit professional content |
| 0.9 | Loose - groups varied appearances | Varied angles, lighting changes |
| 1.2+ | Very loose - aggressive grouping | When getting too many clusters |

### Why Faces Fragment Into Multiple Clusters

| Cause | Solution |
|-------|----------|
| Different lighting (bright vs dark) | Increase threshold |
| Different angles (profile vs frontal) | Increase threshold or add pose filtering |
| Different expressions | Increase threshold |
| Glasses on/off | Increase threshold |
| Makeup changes | Increase threshold |
| Actually different people | Decrease threshold |

---

## Current Issues & Experiments

### Issue: Too Many "Unique Persons"

**Symptom:** Scene with 2 performers → 16+ unique persons detected

**Cause:** Embedding vectors vary too much across frames (lighting, angle, expression)

**Observed behavior:**
- Increasing `cluster_threshold` from 0.9 → 2.0 often has NO effect
- Faces are either very similar (cluster) or very different (don't cluster at any threshold)
- Most singleton faces match different random performers at 60-75% confidence
- The correct performers are usually in the top 3-5 matches overall

**Solutions being explored:**
1. ✗ Increase `cluster_threshold` - doesn't help, faces are too different
2. Frame count filtering - require >=2 frames per person (WORKS)
3. Confidence threshold filtering - remove matches <70% (partial help)
4. Frequency-based matching - count performer appearances across all faces (promising)

### Issue: Clustering Fails Completely on Some Scenes

**Symptom:** N faces → N clusters (zero clustering)

**Example:** Scene 18854 - 27 faces → 27 clusters, all singletons

**Possible causes:**
- Rapid position changes in scene
- Extreme lighting variation
- Face detection picking up non-frontal views
- Fast-paced editing with many unique angles

**Current workaround:** Use frequency-based matching instead of clustering

```python
# Frequency-based matching approach
from collections import Counter

performer_counts = Counter()
for person in persons:
    for match in person.get("all_matches", []):
        performer_counts[match.get("name")] += 1

# Top performers by frequency are likely the actual performers
for name, count in performer_counts.most_common(5):
    print(f"{count}x {name}")
```

### Issue: Low-Resolution Content Fails

**Symptom:** No faces detected in 240p/360p content

**Cause:** `min_face_size=80` filters out all faces

**Solutions:**
1. Lower `min_face_size` for low-res content
2. Auto-detect resolution and adjust threshold
3. Accept lower quality matches for low-res

### Issue: Performers With Few Reference Faces Match Poorly

**Symptom:** Performer in scene but doesn't appear as best match

**Example:** Staci Silverstone (1 face in DB) appears at 55-65% confidence as secondary match

**Cause:** With only 1 reference embedding, match quality depends heavily on angle/lighting alignment

**Solutions:**
1. Long-term: Enrich database with more faces per performer
2. Short-term: Lower confidence threshold for known low-coverage performers
3. Use all_matches not just best_match when performer has low face count

---

## Potential Improvements

### 1. Adaptive min_face_size

```python
def get_min_face_size(video_height: int) -> int:
    """Scale min_face_size with video resolution."""
    if video_height <= 360:
        return 40  # Low-res: accept smaller faces
    elif video_height <= 480:
        return 60  # SD: moderate threshold
    elif video_height <= 720:
        return 80  # HD: standard threshold
    else:
        return 100  # FHD+: higher quality
```

### 2. Pose-Based Filtering

Use RetinaFace landmarks to estimate face pose:
- `left_eye`, `right_eye`, `nose`, `mouth_left`, `mouth_right`
- Calculate eye distance ratio and nose position
- Filter faces that are too profile (eyes too close together)

### 3. Face Quality Scoring

Score each face by:
- Size (bigger = better)
- Detection confidence
- Pose (frontal = better)
- Sharpness/blur detection

Only cluster the top N faces per frame.

### 4. Hierarchical Clustering

Instead of greedy single-pass:
1. Use DBSCAN or hierarchical clustering
2. Allow variable cluster sizes
3. Post-process to merge small clusters

### 5. Improved Frame Sampling (Burst Mode)

**Current behavior:** Single frame at each evenly-spaced timestamp

```python
# How it works now (frame_extractor.py):
interval = (end_sec - start_sec) / (num_frames - 1)
timestamps = [start_sec + i * interval for i in range(num_frames)]
# e.g., 40 frames = 40 single-frame extractions at 40 evenly-spaced points
```

**Problem:** A single frame at each point may catch a bad angle (profile, blinking, motion blur). If that frame doesn't cluster well, we miss that time region entirely.

**Proposed improvement:** Burst sampling - grab N frames around each sample point

```python
# Proposed burst mode:
def calculate_burst_timestamps(duration_sec, num_sample_points=10, frames_per_burst=4, burst_spread_sec=0.5):
    """
    Instead of 40 single frames, grab 10 sample points × 4 frames each.
    Frames within a burst are spread ±0.5 seconds around the sample point.
    """
    sample_points = calculate_evenly_spaced_points(duration_sec, num_sample_points)
    timestamps = []
    for point in sample_points:
        # Grab frames at: point-0.5s, point-0.17s, point+0.17s, point+0.5s
        offsets = np.linspace(-burst_spread_sec, burst_spread_sec, frames_per_burst)
        for offset in offsets:
            timestamps.append(max(0, min(duration_sec, point + offset)))
    return timestamps
```

**Hypothesized benefits:**
- More likely to catch at least one good angle per time region
- Nearby frames have similar lighting → better clustering
- Same total frame count, but distributed differently

**Tradeoffs:**
- Less temporal coverage (10 points instead of 40)
- May miss performers who only appear briefly

**Status:** ✗ TESTED - Burst mode performed WORSE than even sampling

### Burst Mode Test Results (2026-01-29)

| Scene ID | Even Mode | Burst Mode | Notes |
|----------|-----------|------------|-------|
| 13938 | 2/2 (100%) | 2/2 (100%) | Same |
| 30835 | 2/2 (100%) | 1/2 (50%) | Lost Damion Dayski |
| 16342 | 2/2 (100%) | 1/2 (50%) | Lost Jess Carter |
| 26367 | 1/2 (50%) | 1/2 (50%) | Same |
| **Total** | **7/8 (88%)** | **5/8 (62%)** | **-2 performers** |

**Why burst mode failed:**
1. Fewer sample points = less temporal coverage
2. If a performer isn't on-screen at one of the 10 sample points, we miss them
3. The problem isn't face quality per frame - it's temporal coverage
4. Even mode's 40 single frames catches more performer instances

**Conclusion:** For this use case, temporal coverage matters more than frame quality at each point. Keep using even sampling

### 6. Multi-Modal Recognition (Future)

Face embeddings alone have limitations:
- Requires clear frontal face shots
- Fails when face is small, blurry, or at extreme angles
- Some scenes have few good face opportunities

**Planned additional models:**
- **Body type recognition** - Silhouette, proportions, posture
- **Tattoo detection** - Location and pattern matching for identifiable marks
- **Clothing/accessories** - When known from scene metadata

**Benefit:** When face detection fails, body-based features can still identify performers. Multi-modal fusion combines signals for higher confidence.

---

## Test Suite Methodology

### Finding Good Test Scenes

The quality of test results depends heavily on performer data in our face database. Use this approach:

**Step 1: Identify performers with sufficient face data**
```sql
-- Find performers with 3+ faces in database
SELECT p.canonical_name, p.face_count, s.stashbox_performer_id
FROM performers p
JOIN stashbox_ids s ON p.id = s.performer_id
WHERE s.endpoint = 'stashdb' AND p.face_count >= 3
ORDER BY p.face_count DESC;
```

**Step 2: Find scenes where ALL performers have good coverage**
```graphql
# Query Stash for scenes with these performers at target resolution
query {
  findScenes(scene_filter: {}, filter: { per_page: 500 }) {
    scenes {
      id
      files { width height }
      performers { stash_ids { stash_id endpoint } }
    }
  }
}
```

**Step 3: Filter to scenes where every performer has 3+ faces**
- Cross-reference scene performers against database
- Only use scenes where ALL performers have sufficient embeddings
- Sort by total face count (more = better)

### Why Face Count Matters

| Performer Face Count | Expected Match Quality |
|---------------------|----------------------|
| 0 faces | Cannot match - performer missing from DB |
| 1 face | Poor - single reference point, angle-dependent |
| 2-3 faces | Moderate - some variation covered |
| 4+ faces | Good - multiple angles/lighting captured |
| 10+ faces | Excellent - robust to most variations |

### Current Test Results (2026-01-29)

**Optimal test scenes (all performers have 3+ faces):**

| Scene ID | Performers | Total Faces | Match Rate |
|----------|-----------|-------------|------------|
| 13938 | Presley Hart (4f), James Deen (4f) | 8 | 2/2 (100%) |
| 30835 | Damion Dayski (3f), Shrooms Q (5f) | 8 | 2/2 (100%) |
| 16342 | Danny D (4f), Jess Carter (3f) | 7 | 2/2 (100%) |
| 26367 | Kaira Love (3f), Bella Angel (5f) | 8 | 1/2 (50%) |

**Settings used:** 40 frames, min_face_size=80, filter to >=2 frames per person

**Overall accuracy on optimal 1080p scenes:** 70%

### 480p Test Results (2026-01-29)

**Optimal test scenes (all performers have 3+ faces):**

| Scene ID | Performers | Total Faces | Match Rate | Notes |
|----------|-----------|-------------|------------|-------|
| 1414 | Xander Corvus (4f), Nia Nacci (3f) | 7 | 2/2 (100%) | Good clustering (5f, 3f) |
| 377 | Danny D (4f), Barbie Sins (3f) | 7 | 1/2 (50%) | Danny D not found |
| 1551 | Danny D (4f), Krissy Lynn (4f) | 8 | 1/2 (50%) | Only 1 multi-frame cluster |
| 3283 | Ryan Madison (3f), Naomi Woods (5f) | 8 | 0/2 (0%) | 0 clustering, not in ANY matches |

**Settings used:** 40 frames, min_face_size=50, filter to >=2 frames per person

**Overall accuracy on optimal 480p scenes:** 40%

**480p-specific findings:**
1. Lower min_face_size (50 vs 80) doesn't significantly improve results
2. Some 480p scenes work perfectly (1414: 100%), others fail completely (3283: 0%)
3. When performers aren't found, they often don't appear in ANY match - face quality too low
4. Scene content matters more than resolution - well-lit close-ups at 480p can work great

### Matching Mode Comparison (2026-01-29)

Two matching modes are available via `matching_mode` parameter:

| Mode | Description | Strengths | Weaknesses |
|------|-------------|-----------|------------|
| `cluster` | Cluster faces by embedding similarity, then match | Better for single-frame excellent matches | Fails when clustering fails |
| `frequency` | Count performer appearances across all face matches | More robust when clustering fails | May miss performers with few appearances |

**Test Results (10 scenes, 20 performers expected):**

| Scene ID | Performers | Cluster | Frequency | Notes |
|----------|------------|---------|-----------|-------|
| 13938 | Presley Hart, James Deen | 2/2 | 2/2 | Both 100% |
| 30835 | Damion Dayski, Shrooms Q | 1/2 | 2/2 | **Frequency found Damion** |
| 16342 | Danny D, Jess Carter | 2/2 | 2/2 | Both 100% |
| 26367 | Kaira Love, Bella Angel | 1/2 | 1/2 | Kaira Love missed both |
| 16134 | Khloe Kapri, GI Joey | 2/2 | 2/2 | Both 100% |
| 4978 | Alberto Blanco, Victoria Pure | 1/2 | 1/2 | Alberto missed both |
| 30834 | Damion Dayski, Myra Moans | 1/2 | 1/2 | Damion missed both |
| 7387 | Evan Stone, Emma Mae | 1/2 | 1/2 | Different performers found |
| 18830 | Katrina Jade, Danny D | 2/2 | 2/2 | Both 100% |
| 6969 | Evan Stone, Kenna James | 1/2 | 0/2 | **Cluster found Evan** |

**Overall:** Both modes achieve **70% accuracy** (14/20 performers)

**Recommendation:** Use `frequency` mode (default) as it's more robust when clustering fails.

### Screenshot Processing (2026-01-30)

The `/identify/scene` endpoint now processes Stash screenshots (cover images) in addition to video frames.

**Issue:** Stash serves thumbnails via `paths.screenshot`, not full-resolution images:
- 1080p video → 750x422 screenshot thumbnail
- Faces in thumbnails are too small for detection (38x45px vs 40px minimum)

**Solution:** Screenshots smaller than 80% of video width are upscaled using Lanczos interpolation before face detection:
```
Screenshot upscaled: 750x422 -> 1920x1080
Screenshot face: 97x119px -> [performer name]
```

This allows cover images to contribute to performer identification, which can help when video frames don't capture good face angles.

### Key Findings

1. **Frame count sweet spot:** 30-40 frames balances speed and sample diversity
2. **Frame filtering is critical:** Filter to persons with >=2 detected frames eliminates most false positives
3. **Clustering often fails:** Even with good performers, face embedding distances can be too large for clustering
4. **Frequency-based matching:** Now implemented - counts performer appearances across all face matches
5. **Male performers harder to match:** Often have fewer DB faces and more variation over career
6. **Career span matters:** Performers with long careers may look different from their DB photos
7. **Screenshot processing:** Stash serves thumbnails via `paths.screenshot` (not full resolution). Screenshots are upscaled to video resolution before face detection to enable matching on cover images.

### Testing Script Template

```python
import requests

def test_scene(scene_id, expected_performers, num_frames=40):
    resp = requests.post(
        "http://localhost:5000/identify/scene",
        json={
            "scene_id": str(scene_id),
            "num_frames": num_frames,
            "top_k": 5,
            "max_distance": 0.5,
            "min_face_size": 80,
        },
        timeout=300
    )
    data = resp.json()

    # Filter to multi-frame persons
    persons = [p for p in data.get("persons", []) if p.get("frame_count", 0) >= 2]
    found = [p.get("best_match", {}).get("name") for p in persons]

    matched = [e for e in expected_performers if any(e.lower() in f.lower() for f in found)]
    return len(matched), len(expected_performers), found
```

---

## Current Issues & Experiments

### `/identify/scene` (main endpoint)

```json
{
  "scene_id": "12345",
  "num_frames": 20,
  "top_k": 3,
  "max_distance": 0.6,
  "min_face_size": 80
}
```

### `/identify/scene/v2` (experimental, more options)

```json
{
  "scene_id": "12345",
  "num_frames": 40,
  "start_offset_pct": 0.05,
  "end_offset_pct": 0.95,
  "min_face_size": 40,
  "min_face_confidence": 0.5,
  "top_k": 3,
  "max_distance": 0.7,
  "cluster_threshold": 0.9
}
```

---

*This document should be updated as tuning experiments yield results.*

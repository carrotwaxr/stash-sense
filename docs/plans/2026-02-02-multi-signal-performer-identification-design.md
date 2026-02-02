# Multi-Signal Performer Identification Design

**Date:** 2026-02-02
**Status:** Brainstorming in progress

---

## Problem Statement

Current face-only identification achieves ~50% accuracy with significant noise:
- Confidence scores are 60-70% even on correct matches
- False positives have similar confidence scores (hard to filter)
- Misses performers entirely when face not visible (POV, angles, masks)
- At 150k+ performers, some faces are similar enough to cause confusion

**Goal:** Improve identification accuracy by combining multiple biometric signals beyond face recognition.

**Target content:** Primarily professional studio content and JAV, with reasonable support for amateur/OnlyFans.

---

## Signal Overview

### High Signal Strength

| Signal | Maturity | Applicability | Notes |
|--------|----------|---------------|-------|
| **Face Recognition** | Production (current) | High when visible | Already implemented, 60-70% confidence |
| **Tattoo Recognition** | Research-grade | High for tattooed performers | Nearly as unique as face for distinctive tattoos |
| **Birthmarks/Moles** | Research-grade | High when detectable | Very unique but requires high-res, consistent visibility |

### Medium Signal Strength

| Signal | Maturity | Applicability | Notes |
|--------|----------|---------------|-------|
| **Body Proportions** | Production (pose estimation) | Broad | Shoulder-hip ratio, leg-torso ratio; useful for filtering |
| **Breast Characteristics** | Would need custom | Female performers | Size, shape, natural vs augmented; changes over career |
| **Genital Recognition** | Would need custom | Uncensored content | Not applicable to JAV (censored) |
| **Person ReID** | Production but wrong domain | Limited | Trained on clothed pedestrians; needs fine-tuning |

### Low Signal Strength

| Signal | Notes |
|--------|-------|
| **Skin Tone** | Low discriminating power, useful only for coarse filtering |
| **Gait Analysis** | Requires walking motion; not common in adult content |
| **Build Classification** | Too coarse (slim/athletic/curvy) |

---

## Detailed Signal Analysis

### Tattoo Recognition

**How it works:**
1. **Detection** - Find tattoos in image (bounding boxes)
2. **Matching** - Compare detected tattoos to database via embeddings

**Existing resources:**
- NIST Tatt-C benchmark (law enforcement)
- YOLO/RetinaNet can be fine-tuned for detection
- CNN embeddings for matching (similar approach to face recognition)
- Keypoint matching (SIFT/SURF) works well since tattoos are rigid patterns

**Challenges:**

| Challenge | Impact | Mitigation |
|-----------|--------|------------|
| Viewing angle | Same tattoo looks different from angles | Multi-view training, store multiple reference angles |
| Partial visibility | Tattoo may be cropped/obscured | Partial matching techniques |
| Lighting/color | Appearance varies | Grayscale or normalized embeddings |
| New tattoos | Performers add over career | Temporal awareness in database |
| Tattoo removal | Rare but happens | Track "tattoo history" per performer |
| No tattoos | Many performers have none | Signal unavailable for them |

**Signal strength:** HIGH for performers with distinctive tattoos (sleeve, back piece). Lower for common small tattoos (stars, hearts).

---

### Person Re-Identification (ReID)

**How it works:**
ReID recognizes same person across camera views using body shape, proportions, clothing, gait.

**Existing models:**
- OSNet - Lightweight, good accuracy
- TransReID - Transformer-based, SOTA
- FastReID - Meta's production system
- Torchreid - Library with many pretrained models

**Problem:** All trained on clothed pedestrians (Market-1501, DukeMTMC, CUHK03). They rely heavily on clothing appearance which doesn't apply to adult content.

**Potential:** Architecture is relevant but would need fine-tuning/retraining on minimal-clothing data focused on body proportions.

**Signal strength:** MEDIUM with significant adaptation work.

---

### Body Shape / Anthropometrics

**Measurable metrics:**

| Metric | Extraction Method | Discriminating Power |
|--------|-------------------|---------------------|
| Height (relative) | Keypoints + perspective | Low alone, good for filtering |
| Body proportions | Shoulder-hip ratio, leg-torso | Medium - narrows candidates |
| Build classification | Model classification | Low - too coarse |

**Pose estimation models:**
- OpenPose - Classic, well-documented
- MediaPipe - Google's, runs on device
- MMPose - Comprehensive library
- ViTPose - SOTA accuracy

These provide body keypoints from which ratios can be computed.

**Signal strength:** LOW to MEDIUM. Won't identify alone but can eliminate impossible candidates.

---

### Breast Characteristics

**Measurable:**
- Size relative to body frame
- Shape (round, teardrop, etc.)
- Position/spacing
- Natural vs augmented appearance

**Augmentation handling:**
Typically happens once or twice per career. Can model temporally:
```
Performer X:
  - Pre-2019: Natural, B cup
  - 2019+: Augmented, D cup
```
This actually helps temporal matching - B cup footage is likely pre-2019.

**Signal strength:** MEDIUM. Not unique enough alone, very useful for candidate filtering when combined with face confidence.

---

### Birthmarks / Moles / Scars

**Appeal:** Genuinely unique identifiers, nearly like fingerprints.

**Challenges:**
- Need high-res images to detect small marks
- Makeup, lighting, angle affect visibility
- Location mapping complex (relative body position encoding needed)
- Must detect consistently in both training and query images

**Approach:**
1. High-res body region extraction
2. Mole/birthmark detection model (needs training)
3. Relative position encoding
4. Pattern matching

**Signal strength:** HIGH when detectable, but LOW availability due to quality/visibility requirements.

---

### Genital Recognition

**Technical feasibility:** Possible with custom training.

**Challenges:**
- No existing training data/models
- JAV is censored (signal unavailable)
- Appearance varies with angle, lighting, state
- Not always clearly visible

**Signal strength:** MEDIUM where visible and uncensored. Useful for candidate filtering.

---

## Multi-Signal Combination (Bayesian Approach)

Weak individual signals become powerful when combined because errors are independent.

**Example scenario:**
```
Initial: Face = 70% Performer A, 65% Performer B

Tattoo signal:
- A has sleeve, none visible in scene
- B has no tattoos
→ Reduce A, boost B

Body proportions:
- Scene: athletic build
- A: athletic, B: curvier
→ Boost A back up

Breast characteristics:
- Scene: natural appearance, dated 2023
- A: augmented since 2020
- B: natural
→ Strong boost for B

Final: 85% confidence Performer B
```

If face and tattoo recognition both point to same person, unlikely both wrong in same direction.

---

## Detailed Signal Implementation

### Tattoo Recognition

**Why highest-value addition:**
- Distinctive tattoos nearly as unique as faces
- Many performers have visible tattoos (especially amateur/OnlyFans)
- Works when face is obscured
- Existing research to build on

**Two-stage approach:**

1. **Detection** - Find tattoos in image (bounding boxes)
   - Fine-tune YOLOv8 on tattoo data
   - Output: bounding boxes with confidence scores

2. **Matching** - Compare detected tattoos to database
   - CNN embeddings (ResNet/EfficientNet) for vector similarity
   - Alternative: Keypoint matching (SIFT/ORB) for partial matches
   - Store in Voyager index like face embeddings

**Body regions to track:**
- Upper arm (L/R), Forearm (L/R)
- Back (upper/lower), Chest, Ribs/side
- Thigh (L/R), Lower leg (L/R)
- Neck, Hand/wrist

**Handling angle variation:**
Store 2-3 reference angles per tattoo when available. Angle-invariant embeddings are harder to train.

**Schema:**
```sql
CREATE TABLE performer_tattoos (
    id INTEGER PRIMARY KEY,
    performer_id INTEGER REFERENCES performers(id),
    body_region TEXT,
    embedding BLOB,
    image_url TEXT,
    first_seen_date TEXT,
    removed_date TEXT
);
```

---

### Body Proportions (Quick Win)

**Why quick win:** Pretrained models exist, can implement in days.

**Implementation:**
1. Extract keypoints via MediaPipe or MMPose
2. Compute ratios (shoulder-hip, leg-torso, etc.)
3. Store as performer attributes
4. Use at inference to eliminate impossible candidates

```python
def compute_body_ratios(landmarks):
    return {
        'shoulder_hip_ratio': shoulder_width / hip_width,
        'leg_torso_ratio': leg_length / torso_length,
        'shoulder_torso_ratio': shoulder_width / torso_length,
    }
```

**Limitation:** Needs full/mostly-full body visible. Won't help for close-ups.

---

### Breast Characteristics (Medium Priority)

**What to capture:**
- Relative size (ratio to torso width)
- Shape classification
- Natural vs augmented probability

**Temporal handling:** Augmentation typically happens once or twice per career.
```sql
CREATE TABLE performer_breast_history (
    performer_id INTEGER,
    size_category TEXT,
    natural_probability REAL,
    effective_from DATE,
    effective_until DATE
);
```

**Training data shortcut:** Stash-boxes sometimes list measurements - can bootstrap from metadata.

---

## Training Approach: Using vs Training Models

### Current System (Face Recognition)
Uses **pretrained** FaceNet/ArcFace models. No training needed - just extract embeddings.

### Adding New Signals - Three Scenarios

| Scenario | Example | Manual Work |
|----------|---------|-------------|
| **Pretrained exists** | Body pose (MediaPipe) | Zero - just integrate |
| **Fine-tuning** | Tattoo detection (YOLO base) | ~1000 labeled images |
| **From scratch** | Breast classifier | ~5000+ labeled images |

### Semi-Supervised Learning Loop

For signals requiring training (tattoos, breast characteristics):

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Manual label       Train        Model predicts            │
│   500-1000 images → rough model → 5-10k more images         │
│        ▲                              │                     │
│        │                              ▼                     │
│        │                    Review predictions              │
│        │                    (5-10x faster than labeling)    │
│        │                              │                     │
│        │         ◄────────────────────┘                     │
│        │         Add corrections to training set            │
│                                                             │
│   Stop when: Model accuracy 95%+, corrections become rare   │
└─────────────────────────────────────────────────────────────┘
```

**Typical iteration timeline:**

| Round | Training Set | Your Time | Accuracy |
|-------|--------------|-----------|----------|
| 1 | 500-1000 (manual) | 10-15 hrs | ~70% |
| 2 | +3000 (reviewed) | 5-8 hrs | ~85% |
| 3 | +5000 (reviewed) | 3-5 hrs | ~92% |
| 4 | +5000 (reviewed) | 1-2 hrs | ~96% |

**Tooling:** Build review UI into stash-sense-trainer web interface:
- Show image with model's predicted boxes/labels
- Click to confirm, drag to adjust, delete false positives
- Keyboard shortcuts for speed

---

## Implementation Phases

### Phase 1: Zero Labeling Required
1. **Body proportions** - Integrate MediaPipe pose estimation
2. **Tattoo presence detection** - Use YOLO pretrained (imperfect but catches obvious tattoos)
3. **Multi-signal fusion architecture** - Combine face + body + rough tattoo

### Phase 2: Light Labeling (~15-20 hours)
4. **Fine-tune tattoo detector** - Label 1000 images, semi-supervised to 10k+
5. **Tattoo embedding/matching** - Store and search tattoo vectors

### Phase 3: Evaluate and Extend
6. **Breast characteristics** - Only if Phase 1-2 don't reach acceptable accuracy
7. **Birthmarks** - High value but high effort, evaluate ROI

### Phase 4: Training UI
8. **Build labeling interface** into stash-sense-trainer web UI
9. **Semi-supervised review workflow** - Model suggests, human reviews
10. **Iterative training pipeline** - Retrain as corrections accumulate

---

## Multi-Signal Fusion Architecture

### Approach: Late Fusion with Fallback Paths

Search each signal independently, then combine results. Handles missing signals gracefully.

```
                           Query Scene
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        Face Detection   Tattoo Detection   Body Pose
              │                │                │
              ▼                ▼                ▼
        Face Embedding   Tattoo Embedding   Body Ratios
              │                │                │
              ▼                ▼                ▼
        Search Voyager   Search Voyager    Filter Function
        (face index)     (tattoo index)    (ratio tolerance)
              │                │                │
              ▼                ▼                ▼
        Top N + scores   Top N + scores    Valid candidate set
              │                │                │
              └────────────────┼────────────────┘
                               ▼
                        Score Fusion
                    (weighted combination)
                               │
                               ▼
                     Final ranked results
                     with combined confidence
```

### Score Fusion Logic

```python
def fuse_scores(face_results, tattoo_results, body_valid_set, weights):
    """
    Combine results from multiple signals.

    Each result is: {performer_id: confidence_score}
    body_valid_set is: set of performer_ids that pass body ratio filter
    """
    all_candidates = set(face_results.keys()) | set(tattoo_results.keys())

    final_scores = {}
    for performer_id in all_candidates:
        face_score = face_results.get(performer_id, 0)
        tattoo_score = tattoo_results.get(performer_id, 0)

        # Body proportions as multiplicative filter
        if body_valid_set and performer_id not in body_valid_set:
            body_penalty = 0.5  # Reduce confidence but don't eliminate
        else:
            body_penalty = 1.0

        # Weighted combination
        combined = (
            weights['face'] * face_score +
            weights['tattoo'] * tattoo_score
        ) * body_penalty

        # Boost if multiple signals agree
        signals_present = (face_score > 0) + (tattoo_score > 0)
        if signals_present > 1:
            combined *= 1.2  # 20% boost for multi-signal agreement

        final_scores[performer_id] = combined

    return sorted(final_scores.items(), key=lambda x: -x[1])
```

### Dynamic Weights Based on Visibility

```python
def compute_weights(detections):
    """Adjust weights based on what's visible in the scene."""
    weights = {'face': 0.0, 'tattoo': 0.0}

    if detections['face_found']:
        if detections['face_quality'] > 0.8:
            weights['face'] = 1.0
        else:
            weights['face'] = 0.5  # Partial/blurry

    if detections['tattoos_found']:
        weights['tattoo'] = min(1.0, detections['tattoo_coverage'] * 2)

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v/total for k, v in weights.items()}

    return weights
```

### Handling "No Face Visible"

When face isn't detected, fall back to tattoo-primary identification:
- If tattoos found: Search tattoo index, use body ratios to filter
- If neither found: Return "insufficient_signals" with low-confidence body-only candidates

---

## Database Schema Changes

### Trainer Database (performers.db)

```sql
-- Body proportions (add to existing performers table)
ALTER TABLE performers ADD COLUMN body_ratios JSON;
-- {"shoulder_hip_ratio": 1.2, "leg_torso_ratio": 1.8, ...}

-- Tattoo catalog
CREATE TABLE performer_tattoos (
    id INTEGER PRIMARY KEY,
    performer_id INTEGER REFERENCES performers(id),
    body_region TEXT NOT NULL,  -- 'left_arm', 'back', 'chest', etc.
    embedding BLOB NOT NULL,    -- 512-d vector
    image_hash TEXT,            -- Avoid duplicate entries
    confidence REAL,            -- Detection confidence
    first_seen_date TEXT
);
CREATE INDEX idx_tattoos_performer ON performer_tattoos(performer_id);
```

### New Index Files

- `tattoo_embeddings.voy` - Voyager index for tattoo similarity search

### Export to Inference (stash-sense)

Same schema, loaded read-only. Export script updated to include:
- `performers.json` with body_ratios field
- `tattoos.json` mapping tattoo index → performer universal ID
- `tattoo_embeddings.voy` copied alongside face indices

---

## Summary

### What We're Building

A multi-signal performer identification system that combines:
1. **Face recognition** (existing) - Primary signal when face visible
2. **Body proportions** (Phase 1) - Pretrained pose estimation, zero labeling
3. **Tattoo recognition** (Phase 1-2) - Detection + embedding matching
4. **Breast characteristics** (Phase 3 if needed) - Custom classifier

### Expected Improvement

| Scenario | Current | With Multi-Signal |
|----------|---------|-------------------|
| Clear face, no tattoos | 60-70% confidence | 70-80% (body ratios filter false positives) |
| Clear face + visible tattoos | 60-70% confidence | 85-95% (multi-signal boost) |
| Obscured face + visible tattoos | ~0% (miss) | 70-85% (tattoo-primary) |
| POV / no face or tattoos | ~0% (miss) | Low confidence body-only candidates |

### Implementation Phases

| Phase | Signals Added | Manual Work | Expected Gain |
|-------|---------------|-------------|---------------|
| 1 | Body proportions, rough tattoo detection | Zero | Filter false positives |
| 2 | Fine-tuned tattoo detection + matching | ~15-20 hrs labeling | Major accuracy boost |
| 3 | Breast characteristics (if needed) | ~20-30 hrs labeling | Incremental |
| 4 | Training UI in stash-sense-trainer | Development time | Enables iteration |

---

## Current Face Recognition Issues (2026-02-02)

Investigation revealed baseline face coverage issues that should be addressed before adding new signals.

### Issue: Missing StashDB Faces

3,417 performers (6.5%) have stashdb IDs but **zero stashdb faces**. Their faces come only from other sources (pmvstash, babepedia, etc.).

**Example:** Mia Khalifa
- Created from stashdb on Jan 29
- 0 faces from stashdb
- 5 faces from pmvstash (Jan 31)
- 2 faces from babepedia (Feb 1)

**Root cause:** StashDB scrape likely ran without face processing, or face processing failed silently.

### Issue: Obsolete Code Confusion

`database_builder.py` was an obsolete standalone script that used `max_images_per_performer` (hard image cap) instead of the modern `EnrichmentCoordinator` which uses face count targets. **Deleted** to avoid confusion.

### Immediate Actions Needed

1. **Run stashdb-only build** with face processing to backfill missing faces
2. **Increase `max_faces_total`** default from 12 to 20 for better coverage
3. **Add backfill mode** that only processes performers below target face count

---

## Next Steps

### Phase 0: Fix Baseline Face Coverage
1. [ ] Run stashdb build to backfill 3,417 performers missing stashdb faces
2. [ ] Update UI defaults: `max_faces_total` 12 → 20
3. [ ] Add "backfill incomplete performers" option to build form

### Phase 1: Multi-Signal Foundation
4. [ ] Implement body proportion extraction in trainer
5. [ ] Add body_ratios to performer records during scraping
6. [ ] Integrate YOLO for rough tattoo detection
7. [ ] Build tattoo embedding pipeline
8. [ ] Update stash-sense inference to use multi-signal fusion

### Phase 2: Training Infrastructure
9. [ ] Build labeling UI for semi-supervised training
10. [ ] Iterate on tattoo model accuracy

---

*Design complete. Ready for implementation planning.*

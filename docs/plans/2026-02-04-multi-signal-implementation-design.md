# Multi-Signal Performer Identification Implementation

**Date:** 2026-02-04
**Status:** Ready for implementation
**Builds on:** 2026-02-02-multi-signal-performer-identification-design.md

---

## Overview

Add body proportion filtering and tattoo presence signals to the existing face recognition pipeline. This is Phase 1 of multi-signal identification - tattoo embedding matching will come later once the trainer generates that data.

---

## Scope

**Implementing:**
1. Body proportion extraction (MediaPipe) at inference time
2. Body proportion filtering against candidate performers
3. Tattoo detection (YOLO) at inference time
4. Tattoo presence signal (boost/penalty based on visibility consistency)
5. Multi-signal fusion to re-rank face candidates

**Not implementing (yet):**
- Tattoo embedding matching (needs trainer work first)

---

## Architecture

```
                        Query Image
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
   Face Detection      Body Pose Est.      Tattoo Detection
   (existing YOLO)     (MediaPipe)         (YOLO tattoo)
         │                   │                   │
         ▼                   ▼                   ▼
   Face Embeddings     Body Ratios         Tattoo Presence
         │            shoulder_hip: 1.4    has_tattoos: true
         │            leg_torso: 1.3       locations: ["left arm"]
         │                   │                   │
         ▼                   │                   │
   Search Voyager            │                   │
   (face index)              │                   │
         │                   │                   │
         ▼                   ▼                   ▼
   ┌─────────────────────────────────────────────────┐
   │              MultiSignalMatcher                 │
   │                                                 │
   │  1. Get face candidates (top 20)                │
   │  2. Load body_proportions for each candidate    │
   │  3. Load tattoo info for each candidate         │
   │  4. Score each candidate:                       │
   │     - face_score (from Voyager distance)        │
   │     - body_penalty (ratio mismatch)             │
   │     - tattoo_adjustment (presence mismatch)     │
   │  5. Combined score → re-rank                    │
   └─────────────────────────────────────────────────┘
         │
         ▼
   Final ranked results
```

---

## Components

### 1. BodyProportionExtractor

**File:** `api/body_proportions.py`

Uses MediaPipe Pose to extract body ratios from query images.

```python
class BodyProportionExtractor:
    def extract(self, image: np.ndarray) -> Optional[BodyRatios]:
        """
        Returns BodyRatios with:
        - shoulder_hip_ratio: float
        - leg_torso_ratio: float
        - arm_span_height_ratio: float
        - confidence: float

        Returns None if pose not detected.
        """
```

**Filtering logic:**
- Compare shoulder-hip ratio (most discriminating)
- Tolerance ~0.15 for compatible, >0.3 for severe mismatch
- Returns penalty multiplier: 1.0 (compatible) to 0.3 (severe mismatch)

---

### 2. TattooDetector

**File:** `api/tattoo_detector.py`

Port of trainer's YOLO-based tattoo detector.

```python
class TattooDetector:
    def detect(self, image: np.ndarray) -> TattooResult:
        """
        Returns TattooResult with:
        - has_tattoos: bool
        - detections: list of {bbox, confidence, location_hint}
        - locations: set of detected regions
        """
```

**Signal logic:**
- Query shows tattoos, candidate has none → 0.7x penalty
- Query shows no tattoos, candidate has tattoos → 0.95x (slight penalty, tattoos could be hidden)
- Both have tattoos with matching locations → 1.15x boost
- Otherwise → 1.0x neutral

---

### 3. MultiSignalMatcher

**File:** `api/multi_signal_matcher.py`

Combines face recognition with body and tattoo signals.

```python
class MultiSignalMatcher:
    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        db_reader: PerformerDatabaseReader,
        body_extractor: BodyProportionExtractor,
        tattoo_detector: TattooDetector,
    ):
        # Preload body + tattoo data into memory
        self.body_data = self._load_body_data(db_reader)
        self.tattoo_data = self._load_tattoo_data(db_reader)

    def identify(
        self,
        image: np.ndarray,
        top_k: int = 5,
        use_body: bool = True,
        use_tattoo: bool = True,
    ) -> list[MultiSignalMatch]:
        """Full multi-signal identification pipeline."""
```

**Fusion formula:**
```
final_score = face_score × body_multiplier × tattoo_multiplier
```

---

### 4. Database Reader Updates

**Modify:** `api/database_reader.py`

New methods:
- `get_all_body_proportions()` → dict keyed by universal_id
- `get_all_tattoo_info()` → dict with has_tattoos, locations per performer

**Memory:** ~10MB for preloaded data (acceptable)

---

### 5. API Integration

**Modify:** `api/main.py`

Update `/identify` endpoint:
```python
class IdentifyRequest(BaseModel):
    use_multi_signal: bool = True
    use_body: bool = True
    use_tattoo: bool = True
```

Response additions:
```python
signals_used: list[str]
body_ratios_detected: Optional[dict]
tattoos_detected: Optional[dict]
```

---

## Dependencies

Add to `requirements.txt`:
```
mediapipe>=0.10.9
```

Tattoo model: Use same YOLOv5 model as trainer (download on first use or copy).

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `api/body_proportions.py` | Create |
| `api/tattoo_detector.py` | Create (port from trainer) |
| `api/multi_signal_matcher.py` | Create |
| `api/database_reader.py` | Modify (add bulk load methods) |
| `api/main.py` | Modify (init + endpoint updates) |
| `api/config.py` | Modify (add tattoo model path) |
| `requirements.txt` | Modify (add mediapipe) |

---

## Testing Strategy

1. Unit tests for each component
2. Integration test with known performer images
3. A/B comparison: face-only vs multi-signal on same test set

---

## Future Enhancements

When trainer implements tattoo embeddings:
1. Load `tattoo_embeddings.voy` index
2. Add tattoo embedding search to MultiSignalMatcher
3. Use tattoo similarity instead of just presence matching

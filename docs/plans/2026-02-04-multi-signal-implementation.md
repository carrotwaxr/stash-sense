# Multi-Signal Performer Identification Implementation

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add body proportion filtering and tattoo presence signals to improve performer identification accuracy.

**Architecture:** Extract body ratios and detect tattoos in query images at inference time, then re-rank face recognition candidates using multiplicative score fusion. Preload performer body/tattoo data into memory for fast lookups.

**Tech Stack:** MediaPipe (body pose), YOLOv5 (tattoo detection), SQLite (performer data), FastAPI (existing API)

---

## Prerequisites

Before starting:
1. Ensure the new database (2026.02.04) is in `api/data/` with body_proportions and tattoo_detections tables
2. Have the trainer's tattoo model available or be able to download it

---

## Task 1: Add MediaPipe Dependency

**Files:**
- Modify: `/home/carrot/code/stash-sense/requirements.txt`

**Step 1: Add mediapipe to requirements**

Add after the "Face recognition" section:

```
# Body pose estimation
mediapipe>=0.10.9
```

**Step 2: Install locally to verify**

Run: `pip install mediapipe>=0.10.9`
Expected: Success with mediapipe installed

**Step 3: Commit**

```bash
git add requirements.txt
git commit -m "feat: add mediapipe dependency for body proportion extraction"
```

---

## Task 2: Port BodyProportionExtractor from Trainer

**Files:**
- Create: `/home/carrot/code/stash-sense/api/body_proportions.py`
- Test: `/home/carrot/code/stash-sense/api/tests/test_body_proportions.py`

**Step 1: Write the failing test**

```python
"""Tests for body proportion extraction."""
import pytest
import numpy as np

from body_proportions import BodyProportionExtractor, BodyProportions


class TestBodyProportions:
    """Test body proportion dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        bp = BodyProportions(
            shoulder_hip_ratio=1.5,
            leg_torso_ratio=1.3,
            arm_span_height_ratio=1.0,
            confidence=0.85,
        )
        d = bp.to_dict()
        assert d['shoulder_hip_ratio'] == 1.5
        assert d['confidence'] == 0.85

    def test_from_dict(self):
        """Test creation from dict."""
        d = {
            'shoulder_hip_ratio': 1.5,
            'leg_torso_ratio': 1.3,
            'arm_span_height_ratio': 1.0,
            'confidence': 0.85,
        }
        bp = BodyProportions.from_dict(d)
        assert bp.shoulder_hip_ratio == 1.5
        assert bp.confidence == 0.85


class TestBodyProportionExtractor:
    """Test body proportion extraction."""

    @pytest.fixture
    def extractor(self):
        """Create extractor instance."""
        return BodyProportionExtractor()

    def test_extract_returns_none_for_blank_image(self, extractor):
        """Extraction returns None when no pose detected."""
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)
        result = extractor.extract(blank_image)
        assert result is None

    def test_extractor_has_required_methods(self, extractor):
        """Extractor has the expected interface."""
        assert hasattr(extractor, 'extract')
        assert callable(extractor.extract)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_body_proportions.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'body_proportions'"

**Step 3: Create body_proportions.py**

Port from trainer with minimal changes (already production-quality):

```python
"""
Body proportion extraction using MediaPipe Pose.

Extracts scale-invariant body proportions from images for use alongside
face recognition to improve performer identification accuracy.
"""
import logging
import math
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Model download URL
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
MODEL_FILENAME = "pose_landmarker_lite.task"


def _get_model_path() -> Path:
    """Get path to the pose landmarker model, downloading if needed."""
    # Store in models directory relative to data dir
    models_dir = Path(__file__).parent / "models" / "mediapipe"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / MODEL_FILENAME

    if not model_path.exists():
        logger.info(f"Downloading pose landmarker model to {model_path}...")
        try:
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
                with open(model_path, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(
                f"Failed to download pose landmarker model. Error: {e}"
            ) from e
        logger.info("Model download complete.")

    return model_path


@dataclass
class BodyProportions:
    """Body proportion ratios extracted from an image."""
    shoulder_hip_ratio: float
    leg_torso_ratio: float
    arm_span_height_ratio: float
    confidence: float

    def to_dict(self) -> dict:
        """Convert to dict for JSON storage."""
        return {
            'shoulder_hip_ratio': self.shoulder_hip_ratio,
            'leg_torso_ratio': self.leg_torso_ratio,
            'arm_span_height_ratio': self.arm_span_height_ratio,
            'confidence': self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'BodyProportions':
        """Create from dict (loaded from JSON)."""
        return cls(
            shoulder_hip_ratio=d['shoulder_hip_ratio'],
            leg_torso_ratio=d['leg_torso_ratio'],
            arm_span_height_ratio=d['arm_span_height_ratio'],
            confidence=d['confidence'],
        )


class BodyProportionExtractor:
    """
    Extracts body proportions from images using MediaPipe Pose.

    Uses lazy loading of the MediaPipe model to avoid startup cost
    if body proportions are not needed.
    """

    MIN_POSE_DETECTION_CONFIDENCE = 0.5
    MIN_VISIBILITY_THRESHOLD = 0.3

    # MediaPipe Pose landmark indices
    NOSE = 0
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_WRIST = 15
    RIGHT_WRIST = 16

    REQUIRED_LANDMARK_INDICES = [
        NOSE,
        LEFT_SHOULDER, RIGHT_SHOULDER,
        LEFT_HIP, RIGHT_HIP,
        LEFT_ANKLE, RIGHT_ANKLE,
        LEFT_WRIST, RIGHT_WRIST,
    ]

    def __init__(self):
        """Initialize extractor with lazy-loaded model."""
        self._landmarker = None

    @property
    def landmarker(self):
        """Lazy-load MediaPipe PoseLandmarker."""
        if self._landmarker is None:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            model_path = _get_model_path()

            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=self.MIN_POSE_DETECTION_CONFIDENCE,
                min_pose_presence_confidence=self.MIN_POSE_DETECTION_CONFIDENCE,
            )
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        return self._landmarker

    def extract(self, image: np.ndarray) -> Optional[BodyProportions]:
        """
        Extract body proportions from an image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            BodyProportions if pose detected, None otherwise
        """
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        results = self.landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        pose_landmarks = results.pose_landmarks[0]
        landmarks = self._extract_landmarks(pose_landmarks)

        if landmarks is None:
            return None

        ratios = self._compute_ratios(landmarks)
        confidence = self._compute_confidence(pose_landmarks)

        return BodyProportions(
            shoulder_hip_ratio=ratios['shoulder_hip_ratio'],
            leg_torso_ratio=ratios['leg_torso_ratio'],
            arm_span_height_ratio=ratios['arm_span_height_ratio'],
            confidence=confidence,
        )

    def _extract_landmarks(self, pose_landmarks) -> Optional[dict]:
        """Extract landmark positions as normalized (x, y) coordinates."""
        for idx in self.REQUIRED_LANDMARK_INDICES:
            if pose_landmarks[idx].visibility < self.MIN_VISIBILITY_THRESHOLD:
                return None

        return {
            'nose': (pose_landmarks[self.NOSE].x, pose_landmarks[self.NOSE].y),
            'left_shoulder': (pose_landmarks[self.LEFT_SHOULDER].x, pose_landmarks[self.LEFT_SHOULDER].y),
            'right_shoulder': (pose_landmarks[self.RIGHT_SHOULDER].x, pose_landmarks[self.RIGHT_SHOULDER].y),
            'left_hip': (pose_landmarks[self.LEFT_HIP].x, pose_landmarks[self.LEFT_HIP].y),
            'right_hip': (pose_landmarks[self.RIGHT_HIP].x, pose_landmarks[self.RIGHT_HIP].y),
            'left_ankle': (pose_landmarks[self.LEFT_ANKLE].x, pose_landmarks[self.LEFT_ANKLE].y),
            'right_ankle': (pose_landmarks[self.RIGHT_ANKLE].x, pose_landmarks[self.RIGHT_ANKLE].y),
            'left_wrist': (pose_landmarks[self.LEFT_WRIST].x, pose_landmarks[self.LEFT_WRIST].y),
            'right_wrist': (pose_landmarks[self.RIGHT_WRIST].x, pose_landmarks[self.RIGHT_WRIST].y),
        }

    def _compute_ratios(self, landmarks: dict) -> dict:
        """Compute body proportion ratios from landmarks."""
        shoulder_width = self._distance(
            landmarks['left_shoulder'],
            landmarks['right_shoulder']
        )
        hip_width = self._distance(
            landmarks['left_hip'],
            landmarks['right_hip']
        )
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0

        shoulder_mid = self._midpoint(landmarks['left_shoulder'], landmarks['right_shoulder'])
        hip_mid = self._midpoint(landmarks['left_hip'], landmarks['right_hip'])
        torso_length = self._distance(shoulder_mid, hip_mid)

        ankle_mid = self._midpoint(landmarks['left_ankle'], landmarks['right_ankle'])
        leg_length = self._distance(hip_mid, ankle_mid)
        leg_torso_ratio = leg_length / torso_length if torso_length > 0 else 0

        arm_span = self._distance(
            landmarks['left_wrist'],
            landmarks['right_wrist']
        )
        height = self._distance(landmarks['nose'], ankle_mid)
        arm_span_height_ratio = arm_span / height if height > 0 else 0

        return {
            'shoulder_hip_ratio': shoulder_hip_ratio,
            'leg_torso_ratio': leg_torso_ratio,
            'arm_span_height_ratio': arm_span_height_ratio,
        }

    def _compute_confidence(self, pose_landmarks) -> float:
        """Compute overall confidence as average visibility of key landmarks."""
        visibilities = [pose_landmarks[idx].visibility for idx in self.REQUIRED_LANDMARK_INDICES]
        return sum(visibilities) / len(visibilities)

    @staticmethod
    def _distance(p1: tuple, p2: tuple) -> float:
        """Compute Euclidean distance between two points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    @staticmethod
    def _midpoint(p1: tuple, p2: tuple) -> tuple:
        """Compute midpoint between two points."""
        return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_body_proportions.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/body_proportions.py api/tests/test_body_proportions.py
git commit -m "feat: add body proportion extractor using MediaPipe Pose"
```

---

## Task 3: Port TattooDetector from Trainer

**Files:**
- Create: `/home/carrot/code/stash-sense/api/tattoo_detector.py`
- Test: `/home/carrot/code/stash-sense/api/tests/test_tattoo_detector.py`

**Step 1: Write the failing test**

```python
"""Tests for tattoo detection."""
import pytest
import numpy as np

from tattoo_detector import TattooDetector, TattooResult, TattooDetection


class TestTattooResult:
    """Test TattooResult dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        result = TattooResult(
            detections=[
                TattooDetection(
                    bbox={'x': 0.1, 'y': 0.2, 'w': 0.1, 'h': 0.1},
                    confidence=0.8,
                    location_hint='left arm',
                )
            ],
            has_tattoos=True,
            confidence=0.8,
        )
        d = result.to_dict()
        assert d['has_tattoos'] is True
        assert d['confidence'] == 0.8
        assert len(d['detections']) == 1

    def test_from_dict(self):
        """Test creation from dict."""
        d = {
            'detections': [
                {'bbox': {'x': 0.1, 'y': 0.2, 'w': 0.1, 'h': 0.1}, 'confidence': 0.8, 'location_hint': 'left arm'}
            ],
            'has_tattoos': True,
            'confidence': 0.8,
        }
        result = TattooResult.from_dict(d)
        assert result.has_tattoos is True
        assert result.confidence == 0.8

    def test_empty_result(self):
        """Test empty detection result."""
        result = TattooResult(detections=[], has_tattoos=False, confidence=0.0)
        assert result.has_tattoos is False
        assert result.locations == set()

    def test_locations_property(self):
        """Test locations aggregation."""
        result = TattooResult(
            detections=[
                TattooDetection(bbox={}, confidence=0.8, location_hint='left arm'),
                TattooDetection(bbox={}, confidence=0.7, location_hint='torso'),
                TattooDetection(bbox={}, confidence=0.6, location_hint='left arm'),  # Duplicate
            ],
            has_tattoos=True,
            confidence=0.8,
        )
        assert result.locations == {'left arm', 'torso'}


class TestTattooDetector:
    """Test tattoo detection."""

    def test_detector_has_required_methods(self):
        """Detector has the expected interface."""
        # Don't instantiate (loads model), just check class
        assert hasattr(TattooDetector, 'detect')
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_tattoo_detector.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'tattoo_detector'"

**Step 3: Create tattoo_detector.py**

Port from trainer:

```python
"""
Tattoo detection using YOLOv5 model.

Downloads the pre-trained model on first use and
uses it to detect tattoos in images.
"""
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# Google Drive file ID for TattooTrace YOLOv5s model
GDRIVE_FILE_ID = "14VpsSrTkOxp0MTzN7uaVJp8uAZ167l-z"
MODEL_FILENAME = "tattoo_yolov5s.pt"


def _get_model_path() -> Path:
    """
    Get path to the tattoo detection model, downloading if needed.
    """
    models_dir = Path(__file__).parent / "models" / "tattoo"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / MODEL_FILENAME

    if not model_path.exists():
        logger.info(f"Downloading tattoo detection model to {model_path}...")
        try:
            import gdown

            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, str(model_path), quiet=False)

            if not model_path.exists():
                raise RuntimeError("Download completed but model file not found")

            logger.info("Model download complete.")
        except Exception as e:
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(
                f"Failed to download tattoo detection model. Error: {e}"
            ) from e

    return model_path


@dataclass
class TattooDetection:
    """A single detected tattoo."""
    bbox: dict  # {x, y, w, h} normalized 0-1
    confidence: float
    location_hint: Optional[str]  # "left arm", "torso", etc.


@dataclass
class TattooResult:
    """Result of tattoo detection on an image."""
    detections: list[TattooDetection]
    has_tattoos: bool
    confidence: float  # Overall confidence (max of detections, or 0)

    @property
    def locations(self) -> set[str]:
        """Get unique location hints from all detections."""
        return {d.location_hint for d in self.detections if d.location_hint}

    def to_dict(self) -> dict:
        """Convert to dict for JSON storage."""
        return {
            "detections": [
                {
                    "bbox": d.bbox,
                    "confidence": d.confidence,
                    "location_hint": d.location_hint,
                }
                for d in self.detections
            ],
            "has_tattoos": self.has_tattoos,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TattooResult":
        """Create from dict (loaded from JSON)."""
        detections = [
            TattooDetection(
                bbox=det["bbox"],
                confidence=det["confidence"],
                location_hint=det.get("location_hint"),
            )
            for det in d.get("detections", [])
        ]
        return cls(
            detections=detections,
            has_tattoos=d.get("has_tattoos", len(detections) > 0),
            confidence=d.get("confidence", 0.0),
        )


class TattooDetector:
    """
    Detects tattoos in images using YOLOv5.

    Uses lazy loading of the model to avoid startup cost if tattoo
    detection is not needed.
    """

    MIN_CONFIDENCE = 0.25

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize detector.

        Args:
            model_path: Path to model file. If None, uses default path
                       and downloads model if needed.
        """
        self._model = None
        self._model_path = model_path

    @property
    def model(self):
        """Lazy-load YOLOv5 model."""
        if self._model is None:
            import pathlib
            import torch

            # Fix for models saved on Windows
            pathlib.WindowsPath = pathlib.PosixPath

            if self._model_path:
                model_path = Path(self._model_path)
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Model file not found at {self._model_path}"
                    )
            else:
                model_path = _get_model_path()

            logger.info(f"Loading tattoo detection model from {model_path}...")
            self._model = torch.hub.load(
                "ultralytics/yolov5",
                "custom",
                path=str(model_path),
                trust_repo=True,
            )
            self._model.conf = self.MIN_CONFIDENCE
            logger.info("Tattoo detection model loaded.")

        return self._model

    def detect(self, image: np.ndarray) -> TattooResult:
        """
        Detect tattoos in an image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            TattooResult with detected tattoos
        """
        results = self.model(image)

        detections = []
        height, width = image.shape[:2]

        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = xyxy
            conf = float(conf)

            if conf < self.MIN_CONFIDENCE:
                continue

            bbox = {
                "x": float(x1 / width),
                "y": float(y1 / height),
                "w": float((x2 - x1) / width),
                "h": float((y2 - y1) / height),
            }

            location_hint = self._estimate_location(bbox)

            detections.append(
                TattooDetection(
                    bbox=bbox,
                    confidence=conf,
                    location_hint=location_hint,
                )
            )

        detections.sort(key=lambda d: d.confidence, reverse=True)
        overall_confidence = max((d.confidence for d in detections), default=0.0)

        return TattooResult(
            detections=detections,
            has_tattoos=len(detections) > 0,
            confidence=overall_confidence,
        )

    def _estimate_location(self, bbox: dict) -> Optional[str]:
        """Estimate body location based on bbox position."""
        cx = bbox["x"] + bbox["w"] / 2
        cy = bbox["y"] + bbox["h"] / 2

        if cx < 0.4:
            h_pos = "left"
        elif cx > 0.6:
            h_pos = "right"
        else:
            h_pos = None

        if cy < 0.4:
            v_pos = "upper body"
        elif cy > 0.7:
            v_pos = "lower body"
        else:
            v_pos = "torso"

        if h_pos and v_pos == "torso":
            return f"{h_pos} side"
        elif h_pos:
            return f"{h_pos} {v_pos}"
        else:
            return v_pos
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_tattoo_detector.py -v`
Expected: All tests PASS

**Step 5: Add gdown dependency**

Add to `requirements.txt`:
```
# Model downloads
gdown>=4.7.0
```

**Step 6: Commit**

```bash
git add api/tattoo_detector.py api/tests/test_tattoo_detector.py requirements.txt
git commit -m "feat: add tattoo detector using YOLOv5"
```

---

## Task 4: Add Database Reader Methods for Bulk Loading

**Files:**
- Modify: `/home/carrot/code/stash-sense/api/database_reader.py`
- Test: `/home/carrot/code/stash-sense/api/tests/test_multi_signal_data.py`

**Step 1: Write the failing test**

```python
"""Tests for multi-signal data loading."""
import pytest
import sqlite3
import tempfile
import os

from database_reader import PerformerDatabaseReader


@pytest.fixture
def test_db():
    """Create a temporary test database with body proportions and tattoo data."""
    fd, path = tempfile.mkstemp(suffix='.db')
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE performers (
            id INTEGER PRIMARY KEY,
            canonical_name TEXT,
            disambiguation TEXT,
            gender TEXT,
            country TEXT,
            ethnicity TEXT,
            birth_date TEXT,
            death_date TEXT,
            height_cm INTEGER,
            eye_color TEXT,
            hair_color TEXT,
            career_start_year INTEGER,
            career_end_year INTEGER,
            scene_count INTEGER,
            stashdb_updated_at TEXT,
            face_count INTEGER DEFAULT 0,
            image_url TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            updated_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE stashbox_ids (
            performer_id INTEGER,
            endpoint TEXT,
            stashbox_performer_id TEXT
        );

        CREATE TABLE body_proportions (
            id INTEGER PRIMARY KEY,
            performer_id INTEGER,
            shoulder_hip_ratio REAL,
            leg_torso_ratio REAL,
            arm_span_height_ratio REAL,
            confidence REAL
        );

        CREATE TABLE tattoos (
            id INTEGER PRIMARY KEY,
            performer_id INTEGER,
            location TEXT,
            description TEXT
        );

        CREATE TABLE tattoo_detections (
            id INTEGER PRIMARY KEY,
            performer_id INTEGER,
            has_tattoos BOOLEAN,
            confidence REAL,
            detections_json TEXT
        );

        -- Insert test data
        INSERT INTO performers (id, canonical_name, face_count) VALUES (1, 'Test Performer 1', 5);
        INSERT INTO performers (id, canonical_name, face_count) VALUES (2, 'Test Performer 2', 3);

        INSERT INTO stashbox_ids VALUES (1, 'stashdb', 'uuid-1');
        INSERT INTO stashbox_ids VALUES (2, 'stashdb', 'uuid-2');

        INSERT INTO body_proportions (performer_id, shoulder_hip_ratio, leg_torso_ratio, arm_span_height_ratio, confidence)
        VALUES (1, 1.45, 1.32, 1.01, 0.89);

        INSERT INTO tattoos (performer_id, location, description) VALUES (1, 'left arm', 'sleeve');
        INSERT INTO tattoos (performer_id, location, description) VALUES (1, 'back', 'large piece');

        INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
        VALUES (1, 1, 0.85, '[{"location_hint": "left arm"}, {"location_hint": "torso"}]');
        INSERT INTO tattoo_detections (performer_id, has_tattoos, confidence, detections_json)
        VALUES (2, 0, 0.0, '[]');
    """)
    conn.close()

    yield path
    os.unlink(path)


class TestBulkDataLoading:
    """Test bulk data loading for multi-signal matching."""

    def test_get_all_body_proportions(self, test_db):
        """Load all body proportions keyed by universal_id."""
        db = PerformerDatabaseReader(test_db)
        data = db.get_all_body_proportions()

        assert 'stashdb.org:uuid-1' in data
        assert data['stashdb.org:uuid-1']['shoulder_hip_ratio'] == 1.45
        assert data['stashdb.org:uuid-1']['confidence'] == 0.89

    def test_get_all_tattoo_info(self, test_db):
        """Load all tattoo info keyed by universal_id."""
        db = PerformerDatabaseReader(test_db)
        data = db.get_all_tattoo_info()

        assert 'stashdb.org:uuid-1' in data
        assert data['stashdb.org:uuid-1']['has_tattoos'] is True
        assert 'left arm' in data['stashdb.org:uuid-1']['locations']

        assert 'stashdb.org:uuid-2' in data
        assert data['stashdb.org:uuid-2']['has_tattoos'] is False
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_multi_signal_data.py -v`
Expected: FAIL with "AttributeError: 'PerformerDatabaseReader' object has no attribute 'get_all_body_proportions'"

**Step 3: Add methods to database_reader.py**

Add after the `get_piercings` method (around line 223):

```python
    # ==================== Multi-Signal Data ====================

    def get_all_body_proportions(self) -> dict[str, dict]:
        """
        Load all body proportions, keyed by universal_id.

        Returns:
            Dict mapping universal_id to body proportion data:
            {
                "stashdb.org:uuid": {
                    "shoulder_hip_ratio": 1.45,
                    "leg_torso_ratio": 1.32,
                    "arm_span_height_ratio": 1.01,
                    "confidence": 0.89
                }
            }
        """
        with self._connection() as conn:
            rows = conn.execute("""
                SELECT
                    'stashdb.org:' || s.stashbox_performer_id as universal_id,
                    bp.shoulder_hip_ratio,
                    bp.leg_torso_ratio,
                    bp.arm_span_height_ratio,
                    bp.confidence
                FROM body_proportions bp
                JOIN stashbox_ids s ON s.performer_id = bp.performer_id
                WHERE s.endpoint = 'stashdb'
                AND bp.confidence = (
                    SELECT MAX(confidence) FROM body_proportions
                    WHERE performer_id = bp.performer_id
                )
            """).fetchall()

            return {
                row[0]: {
                    'shoulder_hip_ratio': row[1],
                    'leg_torso_ratio': row[2],
                    'arm_span_height_ratio': row[3],
                    'confidence': row[4],
                }
                for row in rows
            }

    def get_all_tattoo_info(self) -> dict[str, dict]:
        """
        Load tattoo presence info for all performers.

        Combines data from tattoos table (text descriptions) and
        tattoo_detections table (detected locations).

        Returns:
            Dict mapping universal_id to tattoo info:
            {
                "stashdb.org:uuid": {
                    "has_tattoos": True,
                    "locations": ["left arm", "back"],
                    "count": 2
                }
            }
        """
        import json

        with self._connection() as conn:
            # Get all performers with stashdb IDs
            performers = conn.execute("""
                SELECT p.id, 'stashdb.org:' || s.stashbox_performer_id as universal_id
                FROM performers p
                JOIN stashbox_ids s ON s.performer_id = p.id
                WHERE s.endpoint = 'stashdb'
            """).fetchall()

            result = {}
            for performer_id, universal_id in performers:
                locations = set()

                # Get locations from tattoos table (text descriptions)
                tattoo_rows = conn.execute(
                    "SELECT location FROM tattoos WHERE performer_id = ?",
                    (performer_id,)
                ).fetchall()
                for row in tattoo_rows:
                    if row[0]:
                        locations.add(row[0].lower())

                # Get locations from tattoo_detections table
                detection_rows = conn.execute(
                    "SELECT detections_json FROM tattoo_detections WHERE performer_id = ? AND has_tattoos = 1",
                    (performer_id,)
                ).fetchall()
                for row in detection_rows:
                    if row[0]:
                        try:
                            detections = json.loads(row[0])
                            for det in detections:
                                if det.get('location_hint'):
                                    locations.add(det['location_hint'].lower())
                        except (json.JSONDecodeError, TypeError):
                            pass

                # Check if performer has tattoos from either source
                has_tattoos_text = len(tattoo_rows) > 0
                has_tattoos_detected = conn.execute(
                    "SELECT 1 FROM tattoo_detections WHERE performer_id = ? AND has_tattoos = 1 LIMIT 1",
                    (performer_id,)
                ).fetchone() is not None

                result[universal_id] = {
                    'has_tattoos': has_tattoos_text or has_tattoos_detected,
                    'locations': list(locations),
                    'count': len(locations),
                }

            return result
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_multi_signal_data.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/database_reader.py api/tests/test_multi_signal_data.py
git commit -m "feat: add bulk data loading for body proportions and tattoo info"
```

---

## Task 5: Create Signal Scoring Functions

**Files:**
- Create: `/home/carrot/code/stash-sense/api/signal_scoring.py`
- Test: `/home/carrot/code/stash-sense/api/tests/test_signal_scoring.py`

**Step 1: Write the failing test**

```python
"""Tests for signal scoring functions."""
import pytest

from body_proportions import BodyProportions
from tattoo_detector import TattooResult, TattooDetection
from signal_scoring import body_ratio_penalty, tattoo_adjustment


class TestBodyRatioPenalty:
    """Test body ratio penalty calculation."""

    def test_no_penalty_when_no_query_data(self):
        """Returns 1.0 when query has no body data."""
        result = body_ratio_penalty(None, {'shoulder_hip_ratio': 1.5})
        assert result == 1.0

    def test_no_penalty_when_no_candidate_data(self):
        """Returns 1.0 when candidate has no body data."""
        query = BodyProportions(1.5, 1.3, 1.0, 0.9)
        result = body_ratio_penalty(query, None)
        assert result == 1.0

    def test_no_penalty_for_compatible_ratios(self):
        """Returns 1.0 when ratios are compatible."""
        query = BodyProportions(1.5, 1.3, 1.0, 0.9)
        candidate = {'shoulder_hip_ratio': 1.52, 'leg_torso_ratio': 1.28}
        result = body_ratio_penalty(query, candidate)
        assert result == 1.0

    def test_moderate_penalty_for_moderate_mismatch(self):
        """Returns ~0.6 for moderate ratio mismatch."""
        query = BodyProportions(1.5, 1.3, 1.0, 0.9)
        candidate = {'shoulder_hip_ratio': 1.7, 'leg_torso_ratio': 1.3}  # 0.2 diff
        result = body_ratio_penalty(query, candidate)
        assert 0.5 < result < 0.8

    def test_severe_penalty_for_large_mismatch(self):
        """Returns ~0.3 for severe ratio mismatch."""
        query = BodyProportions(1.2, 1.3, 1.0, 0.9)
        candidate = {'shoulder_hip_ratio': 1.8, 'leg_torso_ratio': 1.3}  # 0.6 diff
        result = body_ratio_penalty(query, candidate)
        assert result < 0.5


class TestTattooAdjustment:
    """Test tattoo adjustment calculation."""

    def test_neutral_when_no_query_tattoos(self):
        """Returns 1.0 when no tattoos in query and candidate has none."""
        query = TattooResult(detections=[], has_tattoos=False, confidence=0.0)
        result = tattoo_adjustment(query, has_tattoos=False, locations=[])
        assert result == 1.0

    def test_penalty_when_query_has_tattoos_candidate_has_none(self):
        """Returns penalty when query shows tattoos but candidate has none."""
        query = TattooResult(
            detections=[TattooDetection({}, 0.8, 'left arm')],
            has_tattoos=True,
            confidence=0.8,
        )
        result = tattoo_adjustment(query, has_tattoos=False, locations=[])
        assert result < 0.8

    def test_slight_penalty_when_candidate_has_tattoos_not_visible(self):
        """Returns slight penalty when candidate has tattoos but query shows none."""
        query = TattooResult(detections=[], has_tattoos=False, confidence=0.0)
        result = tattoo_adjustment(query, has_tattoos=True, locations=['left arm'])
        assert 0.9 < result < 1.0

    def test_boost_when_locations_match(self):
        """Returns boost when tattoo locations match."""
        query = TattooResult(
            detections=[TattooDetection({}, 0.8, 'left arm')],
            has_tattoos=True,
            confidence=0.8,
        )
        result = tattoo_adjustment(query, has_tattoos=True, locations=['left arm', 'back'])
        assert result > 1.0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_signal_scoring.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'signal_scoring'"

**Step 3: Create signal_scoring.py**

```python
"""
Signal scoring functions for multi-signal performer identification.

These functions compute adjustment multipliers based on body proportion
and tattoo signals to re-rank face recognition candidates.
"""
from typing import Optional

from body_proportions import BodyProportions
from tattoo_detector import TattooResult


def body_ratio_penalty(
    query_ratios: Optional[BodyProportions],
    candidate_ratios: Optional[dict],
) -> float:
    """
    Compare query body ratios against a candidate performer.

    Args:
        query_ratios: Body proportions extracted from query image
        candidate_ratios: Body proportions from database for candidate

    Returns:
        Penalty multiplier (0.3 to 1.0):
        - 1.0 = no penalty (ratios compatible or no data)
        - 0.6 = moderate mismatch
        - 0.3 = severe mismatch (likely different person)
    """
    if query_ratios is None or candidate_ratios is None:
        return 1.0

    # Compare shoulder-hip ratio (most discriminating metric)
    query_sh = query_ratios.shoulder_hip_ratio
    candidate_sh = candidate_ratios.get('shoulder_hip_ratio')

    if candidate_sh is None:
        return 1.0

    sh_diff = abs(query_sh - candidate_sh)

    # Thresholds based on typical human variation
    # Shoulder-hip ratio typically ranges from 1.1 to 1.9
    if sh_diff > 0.35:
        return 0.3  # Severe mismatch - very unlikely same person
    elif sh_diff > 0.2:
        return 0.6  # Moderate mismatch
    elif sh_diff > 0.12:
        return 0.85  # Slight mismatch

    return 1.0  # Compatible


def tattoo_adjustment(
    query_result: Optional[TattooResult],
    has_tattoos: bool,
    locations: list[str],
) -> float:
    """
    Adjust score based on tattoo visibility consistency.

    Args:
        query_result: Tattoo detection result from query image
        has_tattoos: Whether candidate performer has tattoos in database
        locations: Known tattoo locations for candidate

    Returns:
        Adjustment multiplier (0.7 to 1.15):
        - 1.15 = tattoo locations match (boost)
        - 1.0 = neutral (no info or inconclusive)
        - 0.95 = candidate has tattoos but not visible in query
        - 0.7 = query shows tattoos but candidate has none
    """
    if query_result is None:
        return 1.0

    query_has_tattoos = query_result.has_tattoos
    query_locations = query_result.locations

    # Case 1: Query shows tattoos, candidate has none in DB
    if query_has_tattoos and not has_tattoos:
        return 0.7  # Significant penalty

    # Case 2: Query shows no tattoos, but candidate has tattoos
    # (tattoos might just not be visible in this shot)
    if not query_has_tattoos and has_tattoos:
        return 0.95  # Very slight penalty

    # Case 3: Both have tattoos - check location overlap
    if query_has_tattoos and locations:
        candidate_locs = {loc.lower() for loc in locations}
        query_locs = {loc.lower() for loc in query_locations}

        # Check for any overlap in locations
        overlap = query_locs & candidate_locs
        if overlap:
            return 1.15  # Boost - locations match!

        # Tattoos visible but different locations - slight penalty
        if query_locs and candidate_locs:
            return 0.9

    return 1.0  # Neutral
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_signal_scoring.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/signal_scoring.py api/tests/test_signal_scoring.py
git commit -m "feat: add signal scoring functions for body and tattoo signals"
```

---

## Task 6: Create MultiSignalMatcher

**Files:**
- Create: `/home/carrot/code/stash-sense/api/multi_signal_matcher.py`
- Test: `/home/carrot/code/stash-sense/api/tests/test_multi_signal_matcher.py`

**Step 1: Write the failing test**

```python
"""Tests for multi-signal matcher."""
import pytest
from unittest.mock import Mock, MagicMock
from dataclasses import dataclass

from multi_signal_matcher import MultiSignalMatcher, MultiSignalMatch


@dataclass
class MockPerformerMatch:
    """Mock performer match for testing."""
    universal_id: str
    name: str
    combined_score: float


class TestMultiSignalMatcher:
    """Test multi-signal matching."""

    @pytest.fixture
    def mock_face_recognizer(self):
        """Create mock face recognizer."""
        recognizer = Mock()
        return recognizer

    @pytest.fixture
    def mock_db_reader(self):
        """Create mock database reader."""
        reader = Mock()
        reader.get_all_body_proportions.return_value = {
            'stashdb.org:uuid-1': {'shoulder_hip_ratio': 1.45, 'leg_torso_ratio': 1.3},
            'stashdb.org:uuid-2': {'shoulder_hip_ratio': 1.7, 'leg_torso_ratio': 1.4},
        }
        reader.get_all_tattoo_info.return_value = {
            'stashdb.org:uuid-1': {'has_tattoos': True, 'locations': ['left arm']},
            'stashdb.org:uuid-2': {'has_tattoos': False, 'locations': []},
        }
        return reader

    @pytest.fixture
    def mock_body_extractor(self):
        """Create mock body extractor."""
        return Mock()

    @pytest.fixture
    def mock_tattoo_detector(self):
        """Create mock tattoo detector."""
        return Mock()

    def test_matcher_loads_data_on_init(
        self, mock_face_recognizer, mock_db_reader,
        mock_body_extractor, mock_tattoo_detector
    ):
        """Matcher preloads body and tattoo data on init."""
        matcher = MultiSignalMatcher(
            face_recognizer=mock_face_recognizer,
            db_reader=mock_db_reader,
            body_extractor=mock_body_extractor,
            tattoo_detector=mock_tattoo_detector,
        )

        assert len(matcher.body_data) == 2
        assert len(matcher.tattoo_data) == 2
        mock_db_reader.get_all_body_proportions.assert_called_once()
        mock_db_reader.get_all_tattoo_info.assert_called_once()

    def test_matcher_has_identify_method(
        self, mock_face_recognizer, mock_db_reader,
        mock_body_extractor, mock_tattoo_detector
    ):
        """Matcher has identify method."""
        matcher = MultiSignalMatcher(
            face_recognizer=mock_face_recognizer,
            db_reader=mock_db_reader,
            body_extractor=mock_body_extractor,
            tattoo_detector=mock_tattoo_detector,
        )
        assert hasattr(matcher, 'identify')
        assert callable(matcher.identify)
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_multi_signal_matcher.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'multi_signal_matcher'"

**Step 3: Create multi_signal_matcher.py**

```python
"""
Multi-signal performer identification.

Combines face recognition with body proportion filtering and
tattoo presence signals to improve identification accuracy.
"""
import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from body_proportions import BodyProportionExtractor, BodyProportions
from database_reader import PerformerDatabaseReader
from recognizer import FaceRecognizer, RecognitionResult, PerformerMatch
from signal_scoring import body_ratio_penalty, tattoo_adjustment
from tattoo_detector import TattooDetector, TattooResult

logger = logging.getLogger(__name__)


@dataclass
class MultiSignalMatch:
    """Result of multi-signal identification for one face."""
    face: object  # DetectedFace from recognizer
    matches: list[PerformerMatch]
    body_ratios: Optional[BodyProportions] = None
    tattoo_result: Optional[TattooResult] = None
    signals_used: list[str] = field(default_factory=list)


class MultiSignalMatcher:
    """
    Combines face recognition with body and tattoo signals.

    Wraps existing FaceRecognizer and adds multi-signal re-ranking.
    """

    def __init__(
        self,
        face_recognizer: FaceRecognizer,
        db_reader: PerformerDatabaseReader,
        body_extractor: Optional[BodyProportionExtractor] = None,
        tattoo_detector: Optional[TattooDetector] = None,
    ):
        """
        Initialize multi-signal matcher.

        Args:
            face_recognizer: Existing face recognizer
            db_reader: Database reader for loading performer data
            body_extractor: Body proportion extractor (optional)
            tattoo_detector: Tattoo detector (optional)
        """
        self.face_recognizer = face_recognizer
        self.body_extractor = body_extractor
        self.tattoo_detector = tattoo_detector

        # Preload multi-signal data
        logger.info("Loading body proportion data...")
        self.body_data = db_reader.get_all_body_proportions()
        logger.info(f"Loaded {len(self.body_data)} body proportion records")

        logger.info("Loading tattoo info...")
        self.tattoo_data = db_reader.get_all_tattoo_info()
        logger.info(f"Loaded {len(self.tattoo_data)} tattoo info records")

    def identify(
        self,
        image: np.ndarray,
        top_k: int = 5,
        use_body: bool = True,
        use_tattoo: bool = True,
        face_candidates: int = 20,
    ) -> list[MultiSignalMatch]:
        """
        Full multi-signal identification pipeline.

        Args:
            image: RGB image as numpy array
            top_k: Number of final matches to return per face
            use_body: Whether to use body proportion filtering
            use_tattoo: Whether to use tattoo presence signal
            face_candidates: Number of face candidates to consider for re-ranking

        Returns:
            List of MultiSignalMatch objects, one per detected face
        """
        signals_used = ['face']

        # Step 1: Face recognition (get more candidates for re-ranking)
        face_results = self.face_recognizer.recognize_image(
            image,
            top_k=face_candidates,
        )

        # Step 2: Body proportions (optional)
        body_ratios = None
        if use_body and self.body_extractor is not None:
            try:
                body_ratios = self.body_extractor.extract(image)
                if body_ratios is not None:
                    signals_used.append('body')
                    logger.debug(f"Body ratios: sh={body_ratios.shoulder_hip_ratio:.2f}")
            except Exception as e:
                logger.warning(f"Body extraction failed: {e}")

        # Step 3: Tattoo detection (optional)
        tattoo_result = None
        if use_tattoo and self.tattoo_detector is not None:
            try:
                tattoo_result = self.tattoo_detector.detect(image)
                if tattoo_result.has_tattoos:
                    signals_used.append('tattoo')
                    logger.debug(f"Tattoos detected: {tattoo_result.locations}")
            except Exception as e:
                logger.warning(f"Tattoo detection failed: {e}")

        # Step 4: Re-rank each face's candidates
        final_results = []
        for face_result in face_results:
            reranked = self._rerank_candidates(
                face_result.matches,
                body_ratios,
                tattoo_result,
                top_k,
            )
            final_results.append(MultiSignalMatch(
                face=face_result.face,
                matches=reranked,
                body_ratios=body_ratios,
                tattoo_result=tattoo_result,
                signals_used=signals_used.copy(),
            ))

        return final_results

    def _rerank_candidates(
        self,
        candidates: list[PerformerMatch],
        body_ratios: Optional[BodyProportions],
        tattoo_result: Optional[TattooResult],
        top_k: int,
    ) -> list[PerformerMatch]:
        """
        Apply body and tattoo signals to re-rank candidates.

        Args:
            candidates: Face recognition candidates
            body_ratios: Body proportions from query image
            tattoo_result: Tattoo detection from query image
            top_k: Number of results to return

        Returns:
            Re-ranked list of candidates
        """
        scored = []

        for candidate in candidates:
            # Start with face score (convert distance to similarity, higher = better)
            # combined_score is a distance (lower = better), so invert it
            base_score = 1.0 / (1.0 + candidate.combined_score)

            # Apply body penalty
            body_mult = 1.0
            if body_ratios is not None:
                candidate_body = self.body_data.get(candidate.universal_id)
                body_mult = body_ratio_penalty(body_ratios, candidate_body)

            # Apply tattoo adjustment
            tattoo_mult = 1.0
            if tattoo_result is not None:
                candidate_tattoo = self.tattoo_data.get(candidate.universal_id, {})
                tattoo_mult = tattoo_adjustment(
                    tattoo_result,
                    candidate_tattoo.get('has_tattoos', False),
                    candidate_tattoo.get('locations', []),
                )

            # Combined score (higher = better match)
            final_score = base_score * body_mult * tattoo_mult
            scored.append((candidate, final_score, body_mult, tattoo_mult))

        # Sort by final score (higher is better)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top_k candidates
        return [item[0] for item in scored[:top_k]]
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_multi_signal_matcher.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/multi_signal_matcher.py api/tests/test_multi_signal_matcher.py
git commit -m "feat: add MultiSignalMatcher for combined identification"
```

---

## Task 7: Integrate MultiSignalMatcher into API

**Files:**
- Modify: `/home/carrot/code/stash-sense/api/main.py`
- Modify: `/home/carrot/code/stash-sense/api/config.py`

**Step 1: Update config.py to add multi-signal settings**

Add after the `DatabaseConfig` class:

```python
@dataclass
class MultiSignalConfig:
    """Configuration for multi-signal identification."""
    enable_body: bool = True
    enable_tattoo: bool = True
    face_candidates: int = 20  # More candidates for re-ranking

    @classmethod
    def from_env(cls) -> "MultiSignalConfig":
        return cls(
            enable_body=os.environ.get("ENABLE_BODY_SIGNAL", "true").lower() == "true",
            enable_tattoo=os.environ.get("ENABLE_TATTOO_SIGNAL", "true").lower() == "true",
            face_candidates=int(os.environ.get("FACE_CANDIDATES", "20")),
        )
```

**Step 2: Update main.py imports**

Add to imports at top of file:

```python
from body_proportions import BodyProportionExtractor
from tattoo_detector import TattooDetector
from multi_signal_matcher import MultiSignalMatcher, MultiSignalMatch
from config import MultiSignalConfig
```

**Step 3: Add globals and initialization**

Add after existing globals:

```python
# Multi-signal components
multi_signal_matcher: Optional[MultiSignalMatcher] = None
body_extractor: Optional[BodyProportionExtractor] = None
tattoo_detector: Optional[TattooDetector] = None
multi_signal_config: Optional[MultiSignalConfig] = None
```

**Step 4: Update startup initialization**

In the `init_recognizer` function, add after recognizer initialization:

```python
    global multi_signal_matcher, body_extractor, tattoo_detector, multi_signal_config

    # Initialize multi-signal config
    multi_signal_config = MultiSignalConfig.from_env()

    # Initialize body extractor if enabled
    if multi_signal_config.enable_body:
        print("Initializing body proportion extractor...")
        body_extractor = BodyProportionExtractor()

    # Initialize tattoo detector if enabled
    if multi_signal_config.enable_tattoo:
        print("Initializing tattoo detector...")
        tattoo_detector = TattooDetector()

    # Initialize multi-signal matcher
    if db_reader and (body_extractor or tattoo_detector):
        print("Initializing multi-signal matcher...")
        multi_signal_matcher = MultiSignalMatcher(
            face_recognizer=recognizer,
            db_reader=db_reader,
            body_extractor=body_extractor,
            tattoo_detector=tattoo_detector,
        )
        print(f"Multi-signal ready: {len(multi_signal_matcher.body_data)} body, "
              f"{len(multi_signal_matcher.tattoo_data)} tattoo records")
```

**Step 5: Update IdentifyRequest model**

Add fields to the existing `IdentifyRequest` class:

```python
    use_multi_signal: bool = True
    use_body: bool = True
    use_tattoo: bool = True
```

**Step 6: Update identify endpoint**

Modify the `identify_performers` endpoint to use multi-signal when available:

```python
    # Use multi-signal matching if available and requested
    if request.use_multi_signal and multi_signal_matcher is not None:
        multi_results = multi_signal_matcher.identify(
            image,
            top_k=request.top_k,
            use_body=request.use_body,
            use_tattoo=request.use_tattoo,
        )
        # Convert to response format
        results = []
        for mr in multi_results:
            results.append({
                "face": {
                    "bbox": mr.face.bbox,
                    "confidence": mr.face.confidence,
                },
                "matches": [
                    {
                        "universal_id": m.universal_id,
                        "stashdb_id": m.stashdb_id,
                        "name": m.name,
                        "country": m.country,
                        "image_url": m.image_url,
                        "score": m.combined_score,
                    }
                    for m in mr.matches
                ],
                "signals_used": mr.signals_used,
                "body_detected": mr.body_ratios is not None,
                "tattoos_detected": mr.tattoo_result.has_tattoos if mr.tattoo_result else False,
            })
        return {"results": results, "multi_signal": True}
    else:
        # Fall back to face-only
        # ... existing code ...
```

**Step 7: Commit**

```bash
git add api/main.py api/config.py
git commit -m "feat: integrate multi-signal matching into API"
```

---

## Task 8: Add Integration Test

**Files:**
- Create: `/home/carrot/code/stash-sense/api/tests/test_multi_signal_integration.py`

**Step 1: Write integration test**

```python
"""Integration tests for multi-signal identification."""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestMultiSignalIntegration:
    """Test full multi-signal pipeline."""

    def test_identify_with_all_signals(self):
        """Test identification using face + body + tattoo signals."""
        # This is a smoke test - full integration requires running services
        from body_proportions import BodyProportionExtractor, BodyProportions
        from tattoo_detector import TattooDetector, TattooResult
        from signal_scoring import body_ratio_penalty, tattoo_adjustment

        # Test scoring functions work together
        body = BodyProportions(1.5, 1.3, 1.0, 0.9)
        candidate_body = {'shoulder_hip_ratio': 1.52}
        body_mult = body_ratio_penalty(body, candidate_body)
        assert body_mult == 1.0  # Compatible

        tattoo = TattooResult([], False, 0.0)
        tattoo_mult = tattoo_adjustment(tattoo, False, [])
        assert tattoo_mult == 1.0  # Neutral

        # Combined score
        base_score = 0.8
        final = base_score * body_mult * tattoo_mult
        assert final == 0.8

    def test_graceful_degradation(self):
        """Test that missing signals don't break identification."""
        from signal_scoring import body_ratio_penalty, tattoo_adjustment

        # No body data
        assert body_ratio_penalty(None, {'shoulder_hip_ratio': 1.5}) == 1.0
        assert body_ratio_penalty(None, None) == 1.0

        # No tattoo data
        assert tattoo_adjustment(None, True, ['arm']) == 1.0
```

**Step 2: Run test**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_multi_signal_integration.py -v`
Expected: All tests PASS

**Step 3: Commit**

```bash
git add api/tests/test_multi_signal_integration.py
git commit -m "test: add multi-signal integration tests"
```

---

## Task 9: Update Documentation

**Files:**
- Modify: `/home/carrot/code/stash-sense/docs/plans/SESSION-CONTEXT.md`

**Step 1: Update session context**

Add to the session context file:

```markdown
## Multi-Signal Identification (2026-02-04)

Added body proportion filtering and tattoo presence signals to improve performer identification.

### New Files
- `api/body_proportions.py` - MediaPipe-based body ratio extraction
- `api/tattoo_detector.py` - YOLO-based tattoo detection
- `api/signal_scoring.py` - Scoring functions for body and tattoo signals
- `api/multi_signal_matcher.py` - Combined multi-signal identification

### API Changes
- `/identify` endpoint now accepts `use_multi_signal`, `use_body`, `use_tattoo` flags
- Response includes `signals_used`, `body_detected`, `tattoos_detected` fields

### Configuration
- `ENABLE_BODY_SIGNAL=true/false` - Enable body proportion filtering
- `ENABLE_TATTOO_SIGNAL=true/false` - Enable tattoo presence signal
- `FACE_CANDIDATES=20` - Number of face candidates to consider for re-ranking
```

**Step 2: Commit**

```bash
git add docs/plans/SESSION-CONTEXT.md
git commit -m "docs: update session context with multi-signal feature"
```

---

## Summary

This plan implements multi-signal performer identification in 9 tasks:

| Task | Component | Files |
|------|-----------|-------|
| 1 | Add MediaPipe dependency | requirements.txt |
| 2 | Body proportion extractor | body_proportions.py |
| 3 | Tattoo detector | tattoo_detector.py |
| 4 | Database bulk loading | database_reader.py |
| 5 | Signal scoring functions | signal_scoring.py |
| 6 | Multi-signal matcher | multi_signal_matcher.py |
| 7 | API integration | main.py, config.py |
| 8 | Integration tests | test_multi_signal_integration.py |
| 9 | Documentation | SESSION-CONTEXT.md |

---

## Testing the Feature

After implementation, test with:

```bash
# Run all tests
cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v

# Start the API
python -m uvicorn main:app --reload

# Test endpoint with multi-signal
curl -X POST http://localhost:8000/identify \
  -H "Content-Type: application/json" \
  -d '{"image_url": "...", "use_multi_signal": true}'
```

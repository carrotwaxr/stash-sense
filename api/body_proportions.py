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
    # Store in models directory relative to this file
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
            # Clean up partial download if it exists
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(
                f"Failed to download pose landmarker model from {MODEL_URL}. "
                f"Please check your network connection and try again. Error: {e}"
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

    # Detection confidence thresholds
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

    # Required landmarks for body proportion extraction
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

        # Convert numpy array to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # Process image with MediaPipe PoseLandmarker
        results = self.landmarker.detect(mp_image)

        if not results.pose_landmarks or len(results.pose_landmarks) == 0:
            return None

        # Use first detected pose
        pose_landmarks = results.pose_landmarks[0]

        # Extract landmark positions
        landmarks = self._extract_landmarks(pose_landmarks)

        if landmarks is None:
            return None

        # Compute ratios
        ratios = self._compute_ratios(landmarks)

        # Compute confidence as average landmark visibility
        confidence = self._compute_confidence(pose_landmarks)

        return BodyProportions(
            shoulder_hip_ratio=ratios['shoulder_hip_ratio'],
            leg_torso_ratio=ratios['leg_torso_ratio'],
            arm_span_height_ratio=ratios['arm_span_height_ratio'],
            confidence=confidence,
        )

    def _extract_landmarks(self, pose_landmarks) -> Optional[dict]:
        """
        Extract landmark positions as normalized (x, y) coordinates.

        Args:
            pose_landmarks: List of MediaPipe NormalizedLandmark objects

        Returns:
            Dict mapping landmark names to (x, y) tuples, or None if key landmarks missing
        """
        # Check that all required landmarks are visible enough
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
        """
        Compute body proportion ratios from landmarks.

        All ratios are scale-invariant (based on relative positions).

        Args:
            landmarks: Dict mapping landmark names to (x, y) tuples

        Returns:
            Dict with computed ratios
        """
        # Shoulder width
        shoulder_width = self._distance(
            landmarks['left_shoulder'],
            landmarks['right_shoulder']
        )

        # Hip width
        hip_width = self._distance(
            landmarks['left_hip'],
            landmarks['right_hip']
        )

        # Shoulder to hip ratio
        shoulder_hip_ratio = shoulder_width / hip_width if hip_width > 0 else 0

        # Torso length (midpoint of shoulders to midpoint of hips)
        shoulder_mid = self._midpoint(landmarks['left_shoulder'], landmarks['right_shoulder'])
        hip_mid = self._midpoint(landmarks['left_hip'], landmarks['right_hip'])
        torso_length = self._distance(shoulder_mid, hip_mid)

        # Leg length (midpoint of hips to midpoint of ankles)
        ankle_mid = self._midpoint(landmarks['left_ankle'], landmarks['right_ankle'])
        leg_length = self._distance(hip_mid, ankle_mid)

        # Leg to torso ratio
        leg_torso_ratio = leg_length / torso_length if torso_length > 0 else 0

        # Arm span (wrist to wrist)
        arm_span = self._distance(
            landmarks['left_wrist'],
            landmarks['right_wrist']
        )

        # Height approximation (nose to ankle midpoint)
        height = self._distance(landmarks['nose'], ankle_mid)

        # Arm span to height ratio
        arm_span_height_ratio = arm_span / height if height > 0 else 0

        return {
            'shoulder_hip_ratio': shoulder_hip_ratio,
            'leg_torso_ratio': leg_torso_ratio,
            'arm_span_height_ratio': arm_span_height_ratio,
        }

    def _compute_confidence(self, pose_landmarks) -> float:
        """
        Compute overall confidence as average visibility of key landmarks.

        Args:
            pose_landmarks: List of MediaPipe NormalizedLandmark objects

        Returns:
            Average visibility score (0-1)
        """
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

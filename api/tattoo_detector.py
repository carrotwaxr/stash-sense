"""
Tattoo detection using TattooTrace YOLOv5s model.

Downloads the pre-trained model from Google Drive on first use and
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
DEFAULT_MODEL_DIR = Path(__file__).parent / "models" / "tattoo"
MODEL_FILENAME = "tattoo_yolov5s.pt"
FINETUNED_MODEL_FILENAME = "tattoo_yolov5s_latest.pt"  # Symlink to latest fine-tuned model


def _get_model_path(model_dir: Optional[Path] = None) -> Path:
    """
    Get path to the tattoo detection model, downloading if needed.

    Prefers fine-tuned model (tattoo_yolov5s_latest.pt) if available,
    otherwise falls back to base model (tattoo_yolov5s.pt).

    Args:
        model_dir: Directory to store the model file

    Returns:
        Path to the model file
    """
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR

    model_dir_path = Path(model_dir)

    # Prefer fine-tuned model if it exists
    finetuned_path = model_dir_path / FINETUNED_MODEL_FILENAME
    if finetuned_path.exists():
        logger.info(f"Using fine-tuned model: {finetuned_path}")
        return finetuned_path

    # Fall back to base model
    model_path = model_dir_path / MODEL_FILENAME

    if not model_path.exists():
        logger.info(f"Downloading tattoo detection model to {model_path}...")
        try:
            import gdown

            # Create directory if it doesn't exist
            model_path.parent.mkdir(parents=True, exist_ok=True)

            # Download from Google Drive
            url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
            gdown.download(url, str(model_path), quiet=False)

            if not model_path.exists():
                raise RuntimeError("Download completed but model file not found")

            logger.info("Model download complete.")
        except Exception as e:
            # Clean up partial download if it exists
            if model_path.exists():
                model_path.unlink()
            raise RuntimeError(
                f"Failed to download tattoo detection model. "
                f"Please check your network connection and try again. Error: {e}"
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
        """Return set of unique location hints from all detections."""
        return {d.location_hint for d in self.detections if d.location_hint is not None}

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

            # Fix for models saved on Windows - patch WindowsPath to PosixPath
            # This is needed because TattooTrace model was saved on Windows
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
        # Run inference
        results = self.model(image)

        # Parse detections
        detections = []
        height, width = image.shape[:2]

        # Results format: xyxy, conf, cls
        for *xyxy, conf, cls in results.xyxy[0].cpu().numpy():
            x1, y1, x2, y2 = xyxy
            conf = float(conf)

            if conf < self.MIN_CONFIDENCE:
                continue

            # Convert to normalized bbox {x, y, w, h}
            bbox = {
                "x": float(x1 / width),
                "y": float(y1 / height),
                "w": float((x2 - x1) / width),
                "h": float((y2 - y1) / height),
            }

            # Estimate location hint based on bbox position
            location_hint = self._estimate_location(bbox)

            detections.append(
                TattooDetection(
                    bbox=bbox,
                    confidence=conf,
                    location_hint=location_hint,
                )
            )

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

        # Calculate overall confidence
        overall_confidence = max((d.confidence for d in detections), default=0.0)

        return TattooResult(
            detections=detections,
            has_tattoos=len(detections) > 0,
            confidence=overall_confidence,
        )

    def _estimate_location(self, bbox: dict) -> Optional[str]:
        """
        Estimate body location based on bbox position.

        Uses simple heuristics based on normalized coordinates:
        - Horizontal position: left (0-0.4), center (0.4-0.6), right (0.6-1.0)
        - Vertical position: upper (0-0.4), middle (0.4-0.7), lower (0.7-1.0)

        Args:
            bbox: Normalized bounding box {x, y, w, h}

        Returns:
            Location hint string or None
        """
        # Center point of bbox
        cx = bbox["x"] + bbox["w"] / 2
        cy = bbox["y"] + bbox["h"] / 2

        # Determine horizontal position
        if cx < 0.4:
            h_pos = "left"
        elif cx > 0.6:
            h_pos = "right"
        else:
            h_pos = None

        # Determine vertical position
        if cy < 0.4:
            v_pos = "upper body"
        elif cy > 0.7:
            v_pos = "lower body"
        else:
            v_pos = "torso"

        # Combine hints
        if h_pos and v_pos == "torso":
            return f"{h_pos} side"
        elif h_pos:
            return f"{h_pos} {v_pos}"
        else:
            return v_pos

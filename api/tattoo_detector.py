"""
Tattoo detection using YOLOv5s model via ONNX Runtime.

Loads the pre-trained TattooTrace YOLOv5s ONNX model and uses it to
detect tattoos in images. All preprocessing (letterbox, normalization)
and post-processing (NMS, coordinate mapping) is done in numpy/cv2.
"""
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Model filename (ONNX format, converted from PyTorch via convert_models_to_onnx.py)
ONNX_MODEL_FILENAME = "tattoo_yolov5s.onnx"

# Model search paths: DATA_DIR/models first, then ./models (relative to this file)
DATA_DIR = os.environ.get("DATA_DIR", "./data")
LOCAL_MODELS_DIR = Path(__file__).parent / "models"


def _find_model_path() -> Path:
    """
    Find the ONNX tattoo detection model.

    Search order:
    1. {DATA_DIR}/models/tattoo_yolov5s.onnx (Docker / production)
    2. ./models/tattoo_yolov5s.onnx (local development, relative to this file)

    Returns:
        Path to the ONNX model file.

    Raises:
        FileNotFoundError: If the model cannot be found in any search path.
    """
    # Check DATA_DIR/models first (production / Docker)
    data_models_path = Path(DATA_DIR) / "models" / ONNX_MODEL_FILENAME
    if data_models_path.exists():
        return data_models_path

    # Fall back to local models directory
    local_path = LOCAL_MODELS_DIR / ONNX_MODEL_FILENAME
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        f"Tattoo detection ONNX model not found. "
        f"Searched: {data_models_path}, {local_path}. "
        f"Run convert_models_to_onnx.py to generate the model."
    )


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
    Detects tattoos in images using YOLOv5 via ONNX Runtime.

    Uses lazy loading of the ONNX session to avoid startup cost if tattoo
    detection is not needed. All preprocessing and post-processing is done
    in numpy/cv2 -- no PyTorch dependency.
    """

    MIN_CONFIDENCE = 0.25
    INPUT_SIZE = 640  # YOLOv5 input resolution
    IOU_THRESHOLD = 0.45  # NMS IoU threshold (matches YOLOv5 default)

    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize detector.

        Args:
            model_path: Path to ONNX model file. If None, searches default
                       locations (DATA_DIR/models, then ./models).
        """
        self._session = None
        self._model_path = model_path
        self._input_name = None
        self._output_name = None

    @property
    def session(self) -> ort.InferenceSession:
        """Lazy-load ONNX Runtime session."""
        if self._session is None:
            if self._model_path:
                model_path = Path(self._model_path)
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Model file not found at {self._model_path}"
                    )
            else:
                model_path = _find_model_path()

            # Use GPU if available, fall back to CPU
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                ort_providers = ["CPUExecutionProvider"]

            logger.info(f"Loading tattoo detection ONNX model from {model_path}...")
            self._session = ort.InferenceSession(
                str(model_path), providers=ort_providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info("Tattoo detection model loaded.")

        return self._session

    # Keep backward-compatible `model` alias so existing attribute checks
    # (e.g. `hasattr(TattooDetector, 'model')`) still pass.
    model = session

    def _letterbox(
        self, image: np.ndarray, new_shape: int = 640
    ) -> tuple[np.ndarray, float, tuple[int, int]]:
        """
        Resize image with letterboxing (preserve aspect ratio, pad with gray).

        This replicates YOLOv5's letterbox preprocessing:
        - Scale the image so its longest side fits within new_shape
        - Pad the shorter side with gray (114, 114, 114) to reach new_shape

        Args:
            image: Input image as numpy array (H, W, 3), any color space.
            new_shape: Target square size (default 640).

        Returns:
            Tuple of:
            - letterboxed: Padded image of shape (new_shape, new_shape, 3)
            - ratio: Scale factor applied to the image
            - pad: (pad_w, pad_h) padding added to each side (half-pad)
        """
        h, w = image.shape[:2]

        # Compute scale factor (fit longest side)
        ratio = new_shape / max(h, w)
        new_unpad_w = int(round(w * ratio))
        new_unpad_h = int(round(h * ratio))

        # Compute padding needed
        dw = new_shape - new_unpad_w
        dh = new_shape - new_unpad_h

        # Divide padding evenly on both sides
        pad_left = dw // 2
        pad_top = dh // 2

        # Resize image (only if needed)
        if (new_unpad_w, new_unpad_h) != (w, h):
            resized = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        # Pad with gray (114 is YOLOv5's default pad value)
        letterboxed = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
        letterboxed[pad_top:pad_top + new_unpad_h, pad_left:pad_left + new_unpad_w] = resized

        return letterboxed, ratio, (pad_left, pad_top)

    def _nms(self, detections: np.ndarray, iou_threshold: float = 0.45) -> np.ndarray:
        """
        Non-Maximum Suppression in pure numpy.

        Args:
            detections: Array of shape (N, 6) with columns
                       [x1, y1, x2, y2, confidence, class_id].
                       Must already be filtered by confidence threshold.
            iou_threshold: IoU threshold for suppression.

        Returns:
            Filtered detections array after NMS, shape (M, 6) where M <= N.
        """
        if len(detections) == 0:
            return detections

        x1 = detections[:, 0]
        y1 = detections[:, 1]
        x2 = detections[:, 2]
        y2 = detections[:, 3]
        scores = detections[:, 4]

        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]

        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)

            if len(order) == 1:
                break

            # Compute IoU of the picked box with the rest
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])

            inter_w = np.maximum(0.0, xx2 - xx1)
            inter_h = np.maximum(0.0, yy2 - yy1)
            intersection = inter_w * inter_h

            union = areas[i] + areas[rest] - intersection
            iou = intersection / np.maximum(union, 1e-6)

            # Keep boxes with IoU below threshold
            remaining = np.where(iou <= iou_threshold)[0]
            order = rest[remaining]

        return detections[keep]

    def detect(self, image: np.ndarray) -> TattooResult:
        """
        Detect tattoos in an image.

        Args:
            image: RGB image as numpy array (H, W, 3)

        Returns:
            TattooResult with detected tattoos
        """
        orig_h, orig_w = image.shape[:2]

        # --- Preprocessing (replicate YOLOv5 letterbox pipeline) ---
        letterboxed, ratio, (pad_left, pad_top) = self._letterbox(image, self.INPUT_SIZE)

        # HWC -> CHW, uint8 -> float32, normalize to [0, 1]
        blob = letterboxed.transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = blob[np.newaxis, ...]  # Add batch dimension: (1, 3, 640, 640)

        # --- ONNX Inference ---
        raw_output = self.session.run(
            [self._output_name], {self._input_name: blob}
        )[0]  # Shape: (1, 25200, 6)

        predictions = raw_output[0]  # (25200, 6)

        # --- Post-processing ---
        # Columns: [x_center, y_center, w, h, obj_conf, class_conf]
        obj_conf = predictions[:, 4]
        class_conf = predictions[:, 5]
        scores = obj_conf * class_conf

        # Filter by confidence threshold
        mask = scores >= self.MIN_CONFIDENCE
        filtered = predictions[mask]
        filtered_scores = scores[mask]

        if len(filtered) == 0:
            return TattooResult(detections=[], has_tattoos=False, confidence=0.0)

        # Convert xywh (center) to xyxy (corner) in 640x640 pixel space
        cx = filtered[:, 0]
        cy = filtered[:, 1]
        w = filtered[:, 2]
        h = filtered[:, 3]

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        # Build detection array for NMS: [x1, y1, x2, y2, score, class_id=0]
        nms_input = np.column_stack([
            x1, y1, x2, y2, filtered_scores, np.zeros(len(filtered_scores))
        ])

        # Apply NMS
        nms_output = self._nms(nms_input, self.IOU_THRESHOLD)

        # --- Map coordinates back to original image space ---
        detections = []
        for det in nms_output:
            det_x1, det_y1, det_x2, det_y2, conf, _ = det

            # Remove letterbox padding
            det_x1 -= pad_left
            det_y1 -= pad_top
            det_x2 -= pad_left
            det_y2 -= pad_top

            # Undo letterbox scaling to get original pixel coordinates
            det_x1 /= ratio
            det_y1 /= ratio
            det_x2 /= ratio
            det_y2 /= ratio

            # Clip to image bounds
            det_x1 = max(0.0, min(float(det_x1), orig_w))
            det_y1 = max(0.0, min(float(det_y1), orig_h))
            det_x2 = max(0.0, min(float(det_x2), orig_w))
            det_y2 = max(0.0, min(float(det_y2), orig_h))

            # Convert to normalized bbox {x, y, w, h}
            bbox = {
                "x": det_x1 / orig_w,
                "y": det_y1 / orig_h,
                "w": (det_x2 - det_x1) / orig_w,
                "h": (det_y2 - det_y1) / orig_h,
            }

            location_hint = self._estimate_location(bbox)

            detections.append(
                TattooDetection(
                    bbox=bbox,
                    confidence=float(conf),
                    location_hint=location_hint,
                )
            )

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)

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

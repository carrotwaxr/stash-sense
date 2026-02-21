"""
Tattoo embedding matcher using EfficientNet-B0 (ONNX) + Voyager kNN search.

Takes YOLO tattoo detection crops, generates EfficientNet-B0 embeddings
via ONNX Runtime, and queries the pre-built tattoo_embeddings.voy index
to find performers with visually similar tattoos.

Replaces the old binary has/doesn't-have tattoo signal with visual
similarity matching against 230k+ tattoo embeddings.
"""

import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

TATTOO_EMBEDDING_DIM = 1280

# Model search paths: DATA_DIR/models first, then ./models (relative to this file)
ONNX_MODEL_FILENAME = "tattoo_efficientnet_b0.onnx"
DATA_DIR = os.environ.get("DATA_DIR", "./data")
LOCAL_MODELS_DIR = Path(__file__).parent / "models"


def _find_embedder_model_path() -> Path:
    """Find the EfficientNet-B0 ONNX model for tattoo embedding.

    Search order:
    1. {DATA_DIR}/models/tattoo_efficientnet_b0.onnx (Docker / production)
    2. ./models/tattoo_efficientnet_b0.onnx (local development, relative to this file)

    Returns:
        Path to the ONNX model file.

    Raises:
        FileNotFoundError: If the model cannot be found in any search path.
    """
    data_models_path = Path(DATA_DIR) / "models" / ONNX_MODEL_FILENAME
    if data_models_path.exists():
        return data_models_path

    local_path = LOCAL_MODELS_DIR / ONNX_MODEL_FILENAME
    if local_path.exists():
        return local_path

    raise FileNotFoundError(
        f"Tattoo embedder ONNX model not found. "
        f"Searched: {data_models_path}, {local_path}. "
        f"Run convert_models_to_onnx.py --efficientnet to generate the model."
    )


class TattooMatcher:
    """Match tattoo crops against the pre-built tattoo embedding index."""

    def __init__(
        self,
        tattoo_index,
        tattoo_mapping: list,
        embedder_model_path: Optional[str] = None,
    ):
        """
        Initialize the tattoo matcher.

        Args:
            tattoo_index: Voyager Index (1280-dim cosine) loaded from tattoo_embeddings.voy
            tattoo_mapping: List mapping voyager_index -> universal_id
            embedder_model_path: Optional path to EfficientNet-B0 ONNX model.
                If None, searches DATA_DIR/models then ./models.
        """
        self.tattoo_index = tattoo_index
        self.tattoo_mapping = tattoo_mapping
        self._generator = None
        self._embedder_model_path = embedder_model_path

    @property
    def generator(self):
        """Lazy-load EfficientNet-B0 ONNX embedding generator."""
        if self._generator is None:
            self._generator = _TattooEmbeddingGenerator(
                model_path=self._embedder_model_path,
            )
        return self._generator

    @staticmethod
    def crop_tattoo(
        image: np.ndarray,
        bbox: dict,
        padding: float = 0.1,
    ) -> np.ndarray:
        """Crop a tattoo region from an image using normalized bbox coords.

        Args:
            image: Source image as numpy array (H, W, 3)
            bbox: Normalized bounding box {x, y, w, h} with values in [0, 1]
            padding: Fractional padding around the crop (0.1 = 10% on each side)

        Returns:
            Cropped image as numpy array
        """
        h, w = image.shape[:2]

        bw = bbox["w"]
        bh = bbox["h"]
        if bw <= 0 or bh <= 0:
            return None

        pad_w = bw * padding
        pad_h = bh * padding

        x1 = max(0, int((bbox["x"] - pad_w) * w))
        y1 = max(0, int((bbox["y"] - pad_h) * h))
        x2 = min(w, int((bbox["x"] + bw + pad_w) * w))
        y2 = min(h, int((bbox["y"] + bh + pad_h) * h))

        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop

    def match(
        self,
        image: np.ndarray,
        detections: list,
        k: int = 10,
    ) -> dict[str, float]:
        """Match tattoo crops against the index.

        Args:
            image: Full RGB image as numpy array (H, W, 3)
            detections: List of TattooDetection objects from TattooDetector
            k: Number of nearest neighbors per crop

        Returns:
            Dict mapping universal_id -> best similarity score (0-1, higher is better)
        """
        if not detections or self.tattoo_index is None:
            return {}

        scores_by_performer: dict[str, list[float]] = defaultdict(list)
        mapping_size = len(self.tattoo_mapping)

        for detection in detections:
            crop = self.crop_tattoo(image, detection.bbox)
            if crop is None:
                continue

            try:
                embedding = self.generator.get_embedding(crop)
            except Exception as e:
                logger.warning(f"Failed to generate tattoo embedding: {e}")
                continue

            # Query Voyager index
            try:
                neighbors, distances = self.tattoo_index.query(embedding, k=k)
            except Exception as e:
                logger.warning(f"Tattoo index query failed: {e}")
                continue

            for idx, dist in zip(neighbors, distances):
                if idx < 0 or idx >= mapping_size:
                    continue

                entry = self.tattoo_mapping[idx]
                if entry is None:
                    continue

                universal_id = entry["universal_id"]

                # Convert cosine distance to similarity (0-1)
                similarity = max(0.0, 1.0 - dist)
                scores_by_performer[universal_id].append(similarity)

        # Take best score per performer
        return {uid: max(scores) for uid, scores in scores_by_performer.items()}


class _TattooEmbeddingGenerator:
    """Generate EfficientNet-B0 embeddings from tattoo image crops using ONNX Runtime.

    Preprocessing replicates the torchvision EfficientNet_B0_Weights.DEFAULT transforms:
    1. Resize shortest side to 256 (bilinear interpolation)
    2. Center crop to 224x224
    3. Convert to float32 [0, 1]
    4. Normalize with ImageNet mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    5. HWC -> CHW layout, add batch dimension
    """

    # ImageNet normalization constants
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path: Optional[str] = None):
        """Initialize the embedding generator.

        Args:
            model_path: Path to EfficientNet-B0 ONNX model. If None, searches
                DATA_DIR/models then ./models for tattoo_efficientnet_b0.onnx.
        """
        self._model_path = model_path
        self._session = None
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
                        f"Tattoo embedder model not found at {self._model_path}"
                    )
            else:
                model_path = _find_embedder_model_path()

            # Use GPU if available, fall back to CPU
            providers = ort.get_available_providers()
            if "CUDAExecutionProvider" in providers:
                ort_providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                ort_providers = ["CPUExecutionProvider"]

            logger.info(f"Loading tattoo embedder ONNX model from {model_path}...")
            self._session = ort.InferenceSession(
                str(model_path), providers=ort_providers,
            )
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name
            logger.info("Tattoo embedder model loaded.")

        return self._session

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for EfficientNet-B0 inference.

        Replicates torchvision EfficientNet_B0_Weights.DEFAULT transforms:
        Resize(256) -> CenterCrop(224) -> ToTensor -> Normalize(ImageNet)

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            Preprocessed float32 array of shape (1, 3, 224, 224)
        """
        h, w = image.shape[:2]

        # Step 1: Resize shortest side to 256, preserve aspect ratio (bilinear)
        if h < w:
            new_h = 256
            new_w = int(round(w * 256 / h))
        else:
            new_w = 256
            new_h = int(round(h * 256 / w))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Step 2: Center crop to 224x224
        crop_h, crop_w = 224, 224
        start_y = (new_h - crop_h) // 2
        start_x = (new_w - crop_w) // 2
        cropped = resized[start_y : start_y + crop_h, start_x : start_x + crop_w]

        # Step 3: Convert to float32 [0, 1]
        img_float = cropped.astype(np.float32) / 255.0

        # Step 4: Normalize with ImageNet mean and std
        img_float = (img_float - self.IMAGENET_MEAN) / self.IMAGENET_STD

        # Step 5: HWC -> CHW, add batch dimension
        img_chw = np.transpose(img_float, (2, 0, 1))
        return np.expand_dims(img_chw, axis=0)

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate L2-normalized embedding from an image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            L2-normalized float32 array of shape (1280,)
        """
        input_tensor = self._preprocess(image)

        embedding = self.session.run(
            [self._output_name], {self._input_name: input_tensor}
        )[0].squeeze(0)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

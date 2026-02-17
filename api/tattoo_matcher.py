"""
Tattoo embedding matcher using EfficientNet-B0 + Voyager kNN search.

Takes YOLO tattoo detection crops, generates EfficientNet-B0 embeddings,
and queries the pre-built tattoo_embeddings.voy index to find performers
with visually similar tattoos.

Replaces the old binary has/doesn't-have tattoo signal with visual
similarity matching against 230k+ tattoo embeddings.
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

TATTOO_EMBEDDING_DIM = 1280


class TattooMatcher:
    """Match tattoo crops against the pre-built tattoo embedding index."""

    def __init__(self, tattoo_index, tattoo_mapping: list):
        """
        Initialize the tattoo matcher.

        Args:
            tattoo_index: Voyager Index (1280-dim cosine) loaded from tattoo_embeddings.voy
            tattoo_mapping: List mapping voyager_index -> universal_id
        """
        self.tattoo_index = tattoo_index
        self.tattoo_mapping = tattoo_mapping
        self._generator = None

    @property
    def generator(self):
        """Lazy-load EfficientNet-B0 embedding generator."""
        if self._generator is None:
            self._generator = _TattooEmbeddingGenerator()
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
    """Generate EfficientNet-B0 embeddings from tattoo image crops."""

    def __init__(self, device: str = "cpu"):
        self._device = device
        self._model = None
        self._transform = None

    @property
    def model(self):
        """Lazy-load EfficientNet-B0 model."""
        if self._model is None:
            import torch
            from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

            weights = EfficientNet_B0_Weights.DEFAULT
            self._model = efficientnet_b0(weights=weights)
            self._model.classifier = torch.nn.Identity()
            self._model = self._model.to(self._device)
            self._model.eval()
            self._transform = weights.transforms()
            logger.info("Loaded EfficientNet-B0 for tattoo matching")
        return self._model

    @property
    def transform(self):
        """Get the preprocessing transform (loads model if needed)."""
        if self._transform is None:
            _ = self.model
        return self._transform

    def get_embedding(self, image: np.ndarray) -> np.ndarray:
        """Generate L2-normalized embedding from an image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8

        Returns:
            L2-normalized float32 array of shape (1280,)
        """
        import torch
        from PIL import Image

        pil_img = Image.fromarray(image)
        tensor = self.transform(pil_img).unsqueeze(0).to(self._device)

        with torch.no_grad():
            embedding = self.model(tensor).squeeze(0).cpu().numpy()

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.astype(np.float32)

"""
Face processing pipeline for enrichment.

Downloads images, detects faces, generates embeddings, applies quality filters.

See: docs/plans/2026-01-29-face-enrichment-integration.md
"""
import io
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
from PIL import Image

from embeddings import FaceEmbedding, FaceEmbeddingGenerator
from quality_filters import QualityFilters, QualityFilter, FilterResult

logger = logging.getLogger(__name__)


@dataclass
class ProcessedFace:
    """A processed face ready for validation and storage."""
    embedding: FaceEmbedding
    quality_score: float
    bbox: dict


class FaceProcessor:
    """
    Processes images to extract face embeddings.

    Pipeline:
    1. Load image from bytes
    2. Detect faces using RetinaFace
    3. Apply quality filters (size, confidence, angle)
    4. Generate embeddings for valid faces
    5. Return ProcessedFace objects
    """

    def __init__(
        self,
        generator: FaceEmbeddingGenerator,
        quality_filters: QualityFilters,
    ):
        """
        Initialize processor.

        Args:
            generator: Face embedding generator
            quality_filters: Quality filter configuration
        """
        self.generator = generator
        self.quality_filter = QualityFilter(quality_filters)
        self.filters = quality_filters

    def process_image(
        self,
        image_bytes: bytes,
        trust_level: str = "medium",
    ) -> list[ProcessedFace]:
        """
        Process an image and extract face embeddings.

        Args:
            image_bytes: Raw image data
            trust_level: Source trust level for filtering behavior

        Returns:
            List of ProcessedFace objects (may be empty)
        """
        # Load image
        try:
            image = self._load_image(image_bytes)
        except Exception as e:
            logger.debug(f"Failed to load image: {e}")
            return []

        height, width = image.shape[:2]

        # Detect faces
        try:
            detected_faces = self.generator.detect_faces(image)
        except Exception as e:
            logger.debug(f"Face detection failed: {e}")
            return []

        if not detected_faces:
            return []

        # For high-trust sources, reject multi-face images
        if trust_level == "high" and len(detected_faces) > 1:
            if self.filters.prefer_single_face:
                logger.debug(f"Rejected multi-face image ({len(detected_faces)} faces) for high-trust source")
                return []

        # Apply quality filters
        valid_faces = []
        for face in detected_faces:
            result = self.quality_filter.check_face(
                face=face,
                image_width=width,
                image_height=height,
                total_faces_in_image=len(detected_faces),
                trust_level=trust_level,
            )
            if result.passed:
                valid_faces.append(face)

        if not valid_faces:
            return []

        # For non-high-trust sources with multiple faces, take largest
        if len(valid_faces) > 1:
            valid_faces.sort(
                key=lambda f: f.bbox.get("w", 0) * f.bbox.get("h", 0),
                reverse=True,
            )
            valid_faces = [valid_faces[0]]

        # Generate embeddings
        processed = []
        for face in valid_faces:
            try:
                embedding = self.generator.get_embedding(face.image)
            except Exception as e:
                logger.debug(f"Failed to generate embedding: {e}")
                continue

            # Calculate quality score based on face size and confidence
            face_area = face.bbox.get("w", 0) * face.bbox.get("h", 0)
            quality_score = min(1.0, (face_area / 10000) * face.confidence)

            processed.append(ProcessedFace(
                embedding=embedding,
                quality_score=quality_score,
                bbox=face.bbox,
            ))

        return processed

    def _load_image(self, image_bytes: bytes) -> np.ndarray:
        """Load image from bytes to numpy array."""
        img = Image.open(io.BytesIO(image_bytes))
        img = img.convert("RGB")
        return np.array(img)

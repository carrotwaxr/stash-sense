"""Tests for face processing pipeline."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch


class TestFaceProcessor:
    """Test image processing pipeline."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock embedding generator."""
        from embeddings import FaceEmbedding

        generator = MagicMock()

        # Mock detect_faces to return one valid face
        mock_face = MagicMock(spec=["bbox", "confidence", "image"])
        mock_face.bbox = {"x": 100, "y": 100, "w": 150, "h": 150}
        mock_face.confidence = 0.95
        mock_face.image = np.random.rand(160, 160, 3).astype(np.uint8)
        generator.detect_faces.return_value = [mock_face]

        # Mock get_embedding to return valid embedding
        generator.get_embedding.return_value = FaceEmbedding(
            facenet=np.random.rand(512).astype(np.float32),
            arcface=np.random.rand(512).astype(np.float32),
        )

        return generator

    @pytest.fixture
    def processor(self, mock_generator):
        """Create processor with mock generator."""
        from face_processor import FaceProcessor
        from quality_filters import QualityFilters

        return FaceProcessor(
            generator=mock_generator,
            quality_filters=QualityFilters(),
        )

    def test_process_image_returns_faces(self, processor):
        """Processing valid image returns detected faces."""
        from face_processor import ProcessedFace

        # Create fake image bytes (100x100 RGB)
        image_bytes = self._create_test_image(500, 500)

        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 1
        assert isinstance(faces[0], ProcessedFace)
        assert faces[0].embedding is not None

    def test_process_image_filters_small_faces(self, processor, mock_generator):
        """Small faces are filtered out."""
        # Mock a small face
        mock_face = MagicMock(spec=["bbox", "confidence", "image"])
        mock_face.bbox = {"x": 10, "y": 10, "w": 50, "h": 50}  # Too small
        mock_face.confidence = 0.95
        mock_generator.detect_faces.return_value = [mock_face]

        image_bytes = self._create_test_image(500, 500)
        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 0

    def test_process_image_filters_low_confidence(self, processor, mock_generator):
        """Low confidence faces are filtered out."""
        mock_face = MagicMock(spec=["bbox", "confidence", "image"])
        mock_face.bbox = {"x": 100, "y": 100, "w": 150, "h": 150}
        mock_face.confidence = 0.5  # Too low
        mock_generator.detect_faces.return_value = [mock_face]

        image_bytes = self._create_test_image(500, 500)
        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 0

    def test_process_image_filters_multi_face_for_high_trust(self, processor, mock_generator):
        """Multi-face images rejected for high trust sources."""
        mock_face1 = MagicMock(spec=["bbox", "confidence", "image"])
        mock_face1.bbox = {"x": 100, "y": 100, "w": 150, "h": 150}
        mock_face1.confidence = 0.95
        mock_face1.image = np.random.rand(160, 160, 3).astype(np.uint8)

        mock_face2 = MagicMock(spec=["bbox", "confidence", "image"])
        mock_face2.bbox = {"x": 300, "y": 100, "w": 150, "h": 150}
        mock_face2.confidence = 0.95
        mock_face2.image = np.random.rand(160, 160, 3).astype(np.uint8)

        mock_generator.detect_faces.return_value = [mock_face1, mock_face2]

        image_bytes = self._create_test_image(600, 400)
        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 0  # Rejected due to multi-face

    def test_process_image_accepts_multi_face_for_medium_trust(self, processor, mock_generator):
        """Multi-face images take largest face for medium trust sources."""
        mock_face1 = MagicMock(spec=["bbox", "confidence", "image"])
        mock_face1.bbox = {"x": 100, "y": 100, "w": 100, "h": 100}  # Smaller
        mock_face1.confidence = 0.95
        mock_face1.image = np.random.rand(160, 160, 3).astype(np.uint8)

        mock_face2 = MagicMock(spec=["bbox", "confidence", "image"])
        mock_face2.bbox = {"x": 300, "y": 100, "w": 150, "h": 150}  # Larger
        mock_face2.confidence = 0.95
        mock_face2.image = np.random.rand(160, 160, 3).astype(np.uint8)

        mock_generator.detect_faces.return_value = [mock_face1, mock_face2]

        image_bytes = self._create_test_image(600, 400)
        faces = processor.process_image(image_bytes, trust_level="medium")

        assert len(faces) == 1  # Takes largest

    def test_process_image_handles_no_faces(self, processor, mock_generator):
        """Returns empty list when no faces detected."""
        mock_generator.detect_faces.return_value = []

        image_bytes = self._create_test_image(500, 500)
        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 0

    def test_process_image_handles_invalid_image(self, processor):
        """Returns empty list for invalid image data."""
        faces = processor.process_image(b"not an image", trust_level="high")

        assert len(faces) == 0

    def _create_test_image(self, width: int, height: int) -> bytes:
        """Create a test PNG image."""
        from PIL import Image
        import io

        img = Image.new("RGB", (width, height), color="white")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        return buffer.getvalue()

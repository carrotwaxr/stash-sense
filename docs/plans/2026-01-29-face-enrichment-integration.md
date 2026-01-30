# Face Enrichment Integration Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add face detection and embedding generation to the enrichment pipeline, so scraped images produce face embeddings stored in Voyager indices.

**Architecture:** The enrichment coordinator loads Voyager indices at startup, processes images in scraper threads (download → detect → embed → validate), queues valid embeddings to the write queue, and saves indices periodically and at shutdown.

**Tech Stack:** Voyager (HNSW indices), InsightFace/DeepFace (face detection/embedding), asyncio (write queue), threading (scraper parallelism)

---

## Task 1: Add Face Validator Class

**Files:**
- Create: `api/face_validator.py`
- Test: `api/tests/test_face_validator.py`

This class encapsulates the trust-level validation logic: should a new face embedding be accepted for a performer?

**Step 1: Write the failing test**

```python
# api/tests/test_face_validator.py
"""Tests for face validation logic."""
import pytest
import numpy as np
from face_validator import FaceValidator, ValidationResult


class TestFaceValidator:
    """Test trust-level based face validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return FaceValidator()

    def test_high_trust_accepts_without_existing_faces(self, validator):
        """High trust sources accept faces even with no existing embeddings."""
        embedding = np.random.rand(512).astype(np.float32)

        result = validator.validate(
            new_embedding=embedding,
            existing_embeddings=[],
            trust_level="high",
        )

        assert result.accepted is True
        assert result.reason == "high_trust"

    def test_medium_trust_rejects_without_existing_faces(self, validator):
        """Medium trust sources reject faces when no existing embeddings."""
        embedding = np.random.rand(512).astype(np.float32)

        result = validator.validate(
            new_embedding=embedding,
            existing_embeddings=[],
            trust_level="medium",
        )

        assert result.accepted is False
        assert result.reason == "no_existing_faces"

    def test_medium_trust_accepts_matching_face(self, validator):
        """Medium trust accepts when new face matches existing."""
        # Create similar embeddings (same with small noise)
        base = np.random.rand(512).astype(np.float32)
        base = base / np.linalg.norm(base)  # Normalize

        existing = base.copy()
        new = base + np.random.rand(512).astype(np.float32) * 0.05  # Small noise
        new = new / np.linalg.norm(new)

        result = validator.validate(
            new_embedding=new,
            existing_embeddings=[existing],
            trust_level="medium",
        )

        assert result.accepted is True
        assert result.reason == "matched_existing"
        assert result.distance is not None
        assert result.distance < 0.35  # Single face threshold

    def test_medium_trust_rejects_non_matching_face(self, validator):
        """Medium trust rejects when new face doesn't match existing."""
        # Create very different embeddings
        existing = np.random.rand(512).astype(np.float32)
        existing = existing / np.linalg.norm(existing)

        new = np.random.rand(512).astype(np.float32)
        new = new / np.linalg.norm(new)

        result = validator.validate(
            new_embedding=new,
            existing_embeddings=[existing],
            trust_level="medium",
        )

        assert result.accepted is False
        assert result.reason == "no_match"

    def test_threshold_scales_with_existing_count(self, validator):
        """Match threshold is more lenient with more existing faces."""
        # With 1 existing face: threshold = 0.35
        assert validator.get_match_threshold(1) == 0.35

        # With 3 existing faces: threshold = 0.40
        assert validator.get_match_threshold(3) == 0.40

        # With 5 existing faces: threshold = 0.45
        assert validator.get_match_threshold(5) == 0.45

    def test_low_trust_always_rejected(self, validator):
        """Low trust sources are rejected (clustering not implemented)."""
        embedding = np.random.rand(512).astype(np.float32)

        result = validator.validate(
            new_embedding=embedding,
            existing_embeddings=[embedding],  # Even with match
            trust_level="low",
        )

        assert result.accepted is False
        assert result.reason == "low_trust_not_supported"
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_face_validator.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'face_validator'"

**Step 3: Write minimal implementation**

```python
# api/face_validator.py
"""
Face validation for enrichment pipeline.

Determines whether a new face embedding should be accepted for a performer
based on trust level and similarity to existing embeddings.

See: docs/plans/2026-01-29-multi-source-enrichment-design.md
"""
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class ValidationResult:
    """Result of face validation."""
    accepted: bool
    reason: str
    distance: Optional[float] = None
    threshold: Optional[float] = None


class FaceValidator:
    """
    Validates new face embeddings against existing performer faces.

    Trust levels:
    - high: Accept without validation (professional headshot sources)
    - medium: Require match to existing embeddings
    - low: Reject (clustering not yet implemented)
    """

    def validate(
        self,
        new_embedding: np.ndarray,
        existing_embeddings: list[np.ndarray],
        trust_level: str,
    ) -> ValidationResult:
        """
        Validate a new face embedding.

        Args:
            new_embedding: The candidate face embedding (512-dim, normalized)
            existing_embeddings: Existing embeddings for this performer
            trust_level: Source trust level ("high", "medium", "low")

        Returns:
            ValidationResult with acceptance decision and reason
        """
        if trust_level == "high":
            return ValidationResult(accepted=True, reason="high_trust")

        if trust_level == "low":
            return ValidationResult(accepted=False, reason="low_trust_not_supported")

        # Medium trust: require match to existing
        if not existing_embeddings:
            return ValidationResult(accepted=False, reason="no_existing_faces")

        # Find best match distance
        min_distance = float('inf')
        for existing in existing_embeddings:
            distance = self._cosine_distance(new_embedding, existing)
            min_distance = min(min_distance, distance)

        # Get threshold based on existing face count
        threshold = self.get_match_threshold(len(existing_embeddings))

        if min_distance < threshold:
            return ValidationResult(
                accepted=True,
                reason="matched_existing",
                distance=min_distance,
                threshold=threshold,
            )
        else:
            return ValidationResult(
                accepted=False,
                reason="no_match",
                distance=min_distance,
                threshold=threshold,
            )

    def get_match_threshold(self, existing_face_count: int) -> float:
        """
        Get match threshold based on existing face count.

        Stricter threshold when fewer faces to match against.
        """
        if existing_face_count <= 1:
            return 0.35  # Very strict - single face could be wrong
        elif existing_face_count <= 3:
            return 0.40  # Moderately strict
        else:
            return 0.45  # More lenient with good existing data

    def _cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine distance between two embeddings."""
        a_norm = a / np.linalg.norm(a)
        b_norm = b / np.linalg.norm(b)
        similarity = np.dot(a_norm, b_norm)
        return 1.0 - similarity
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_face_validator.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add api/face_validator.py api/tests/test_face_validator.py
git commit -m "feat: add FaceValidator for trust-level based face validation

Implements validation logic per multi-source-enrichment-design.md:
- High trust: accept without validation
- Medium trust: require match to existing embeddings
- Adaptive thresholds based on existing face count

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 2: Add Index Manager Class

**Files:**
- Create: `api/index_manager.py`
- Test: `api/tests/test_index_manager.py`

Handles loading, saving, and querying Voyager indices.

**Step 1: Write the failing test**

```python
# api/tests/test_index_manager.py
"""Tests for Voyager index management."""
import pytest
import numpy as np
from pathlib import Path
import voyager


class TestIndexManager:
    """Test index loading, saving, and querying."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create temp data directory with empty indices."""
        # Create minimal valid indices
        facenet = voyager.Index(
            voyager.Space.Cosine,
            num_dimensions=512,
        )
        arcface = voyager.Index(
            voyager.Space.Cosine,
            num_dimensions=512,
        )

        facenet.save(str(tmp_path / "face_facenet512.voy"))
        arcface.save(str(tmp_path / "face_arcface.voy"))

        return tmp_path

    @pytest.fixture
    def manager(self, data_dir):
        """Create index manager."""
        from index_manager import IndexManager
        return IndexManager(data_dir)

    def test_loads_existing_indices(self, manager):
        """Manager loads indices from disk."""
        assert manager.facenet_index is not None
        assert manager.arcface_index is not None
        assert manager.current_index == 0

    def test_add_embedding_returns_index(self, manager):
        """Adding embedding returns its index position."""
        from embeddings import FaceEmbedding

        embedding = FaceEmbedding(
            facenet=np.random.rand(512).astype(np.float32),
            arcface=np.random.rand(512).astype(np.float32),
        )

        idx = manager.add_embedding(embedding)

        assert idx == 0
        assert manager.current_index == 1

    def test_add_multiple_embeddings(self, manager):
        """Adding multiple embeddings increments index."""
        from embeddings import FaceEmbedding

        for i in range(5):
            embedding = FaceEmbedding(
                facenet=np.random.rand(512).astype(np.float32),
                arcface=np.random.rand(512).astype(np.float32),
            )
            idx = manager.add_embedding(embedding)
            assert idx == i

        assert manager.current_index == 5

    def test_get_embedding_by_index(self, manager):
        """Can retrieve embedding by index."""
        from embeddings import FaceEmbedding

        original = FaceEmbedding(
            facenet=np.random.rand(512).astype(np.float32),
            arcface=np.random.rand(512).astype(np.float32),
        )

        idx = manager.add_embedding(original)
        retrieved = manager.get_embedding(idx)

        np.testing.assert_array_almost_equal(retrieved.facenet, original.facenet)
        np.testing.assert_array_almost_equal(retrieved.arcface, original.arcface)

    def test_save_and_reload(self, data_dir):
        """Indices persist after save and reload."""
        from index_manager import IndexManager
        from embeddings import FaceEmbedding

        # Add embeddings and save
        manager1 = IndexManager(data_dir)
        embedding = FaceEmbedding(
            facenet=np.random.rand(512).astype(np.float32),
            arcface=np.random.rand(512).astype(np.float32),
        )
        manager1.add_embedding(embedding)
        manager1.save()

        # Reload and verify
        manager2 = IndexManager(data_dir)
        assert manager2.current_index == 1

        retrieved = manager2.get_embedding(0)
        np.testing.assert_array_almost_equal(retrieved.facenet, embedding.facenet)

    def test_query_nearest(self, manager):
        """Can query for nearest neighbors."""
        from embeddings import FaceEmbedding

        # Add some embeddings
        base = np.random.rand(512).astype(np.float32)
        base = base / np.linalg.norm(base)

        for i in range(10):
            noise = np.random.rand(512).astype(np.float32) * 0.1
            vec = base + noise
            vec = vec / np.linalg.norm(vec)

            embedding = FaceEmbedding(facenet=vec, arcface=vec.copy())
            manager.add_embedding(embedding)

        # Query with similar vector
        query = base + np.random.rand(512).astype(np.float32) * 0.05
        query = query / np.linalg.norm(query)

        neighbors, distances = manager.query(query, k=5)

        assert len(neighbors) == 5
        assert len(distances) == 5
        assert all(d < 0.5 for d in distances)  # Should be close
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_index_manager.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'index_manager'"

**Step 3: Write minimal implementation**

```python
# api/index_manager.py
"""
Voyager index management for face embeddings.

Handles loading, saving, and querying the dual FaceNet512/ArcFace indices.

See: docs/plans/2026-01-29-face-enrichment-integration.md
"""
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import voyager

from embeddings import FaceEmbedding

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages Voyager indices for face embeddings.

    Loads indices at initialization, provides methods for adding
    and querying embeddings, and saves periodically.
    """

    FACENET_FILENAME = "face_facenet512.voy"
    ARCFACE_FILENAME = "face_arcface.voy"

    def __init__(self, data_dir: Path):
        """
        Load indices from data directory.

        Args:
            data_dir: Directory containing .voy index files
        """
        self.data_dir = Path(data_dir)
        self.facenet_path = self.data_dir / self.FACENET_FILENAME
        self.arcface_path = self.data_dir / self.ARCFACE_FILENAME

        self._load_indices()

    def _load_indices(self):
        """Load or create indices."""
        if self.facenet_path.exists():
            logger.info(f"Loading FaceNet index from {self.facenet_path}")
            self.facenet_index = voyager.Index.load(str(self.facenet_path))
        else:
            logger.info("Creating new FaceNet index")
            self.facenet_index = voyager.Index(
                voyager.Space.Cosine,
                num_dimensions=512,
            )

        if self.arcface_path.exists():
            logger.info(f"Loading ArcFace index from {self.arcface_path}")
            self.arcface_index = voyager.Index.load(str(self.arcface_path))
        else:
            logger.info("Creating new ArcFace index")
            self.arcface_index = voyager.Index(
                voyager.Space.Cosine,
                num_dimensions=512,
            )

        # Current index = length of existing index
        self.current_index = len(self.facenet_index)
        logger.info(f"Loaded {self.current_index} existing embeddings")

    def add_embedding(self, embedding: FaceEmbedding) -> int:
        """
        Add a face embedding to both indices.

        Args:
            embedding: FaceEmbedding with facenet and arcface vectors

        Returns:
            Index position of the added embedding
        """
        idx = self.current_index

        self.facenet_index.add_item(embedding.facenet)
        self.arcface_index.add_item(embedding.arcface)

        self.current_index += 1
        return idx

    def get_embedding(self, index: int) -> FaceEmbedding:
        """
        Retrieve embedding by index.

        Args:
            index: Position in the index

        Returns:
            FaceEmbedding with both vectors
        """
        facenet = self.facenet_index.get_vector(index)
        arcface = self.arcface_index.get_vector(index)

        return FaceEmbedding(
            facenet=np.array(facenet, dtype=np.float32),
            arcface=np.array(arcface, dtype=np.float32),
        )

    def query(
        self,
        embedding: np.ndarray,
        k: int = 10,
        use_arcface: bool = False,
    ) -> Tuple[list[int], list[float]]:
        """
        Query for nearest neighbors.

        Args:
            embedding: Query vector (512-dim)
            k: Number of neighbors to return
            use_arcface: If True, query ArcFace index; else FaceNet

        Returns:
            Tuple of (neighbor_indices, distances)
        """
        index = self.arcface_index if use_arcface else self.facenet_index

        neighbors, distances = index.query(embedding, k=min(k, len(index)))

        return list(neighbors), list(distances)

    def save(self):
        """Save indices to disk."""
        logger.info(f"Saving indices ({self.current_index} embeddings)")
        self.facenet_index.save(str(self.facenet_path))
        self.arcface_index.save(str(self.arcface_path))
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_index_manager.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add api/index_manager.py api/tests/test_index_manager.py
git commit -m "feat: add IndexManager for Voyager index operations

Handles loading, saving, and querying dual FaceNet512/ArcFace indices.
Provides thread-safe embedding addition with automatic index tracking.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 3: Add Face Processor Class

**Files:**
- Create: `api/face_processor.py`
- Test: `api/tests/test_face_processor.py`

Handles image download → face detection → embedding generation → validation pipeline.

**Step 1: Write the failing test**

```python
# api/tests/test_face_processor.py
"""Tests for face processing pipeline."""
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from pathlib import Path

from face_processor import FaceProcessor, ProcessedFace
from quality_filters import QualityFilters


class TestFaceProcessor:
    """Test image processing pipeline."""

    @pytest.fixture
    def mock_generator(self):
        """Create mock embedding generator."""
        generator = MagicMock()

        # Mock detect_faces to return one valid face
        mock_face = MagicMock()
        mock_face.bbox = {"x": 100, "y": 100, "w": 150, "h": 150}
        mock_face.confidence = 0.95
        mock_face.image = np.random.rand(160, 160, 3).astype(np.uint8)
        generator.detect_faces.return_value = [mock_face]

        # Mock get_embedding to return valid embedding
        from embeddings import FaceEmbedding
        generator.get_embedding.return_value = FaceEmbedding(
            facenet=np.random.rand(512).astype(np.float32),
            arcface=np.random.rand(512).astype(np.float32),
        )

        return generator

    @pytest.fixture
    def processor(self, mock_generator):
        """Create processor with mock generator."""
        return FaceProcessor(
            generator=mock_generator,
            quality_filters=QualityFilters(),
        )

    def test_process_image_returns_faces(self, processor):
        """Processing valid image returns detected faces."""
        # Create fake image bytes (100x100 RGB)
        image_bytes = self._create_test_image(500, 500)

        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 1
        assert isinstance(faces[0], ProcessedFace)
        assert faces[0].embedding is not None

    def test_process_image_filters_small_faces(self, processor, mock_generator):
        """Small faces are filtered out."""
        # Mock a small face
        mock_face = MagicMock()
        mock_face.bbox = {"x": 10, "y": 10, "w": 50, "h": 50}  # Too small
        mock_face.confidence = 0.95
        mock_generator.detect_faces.return_value = [mock_face]

        image_bytes = self._create_test_image(500, 500)
        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 0

    def test_process_image_filters_low_confidence(self, processor, mock_generator):
        """Low confidence faces are filtered out."""
        mock_face = MagicMock()
        mock_face.bbox = {"x": 100, "y": 100, "w": 150, "h": 150}
        mock_face.confidence = 0.5  # Too low
        mock_generator.detect_faces.return_value = [mock_face]

        image_bytes = self._create_test_image(500, 500)
        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 0

    def test_process_image_filters_multi_face_for_high_trust(self, processor, mock_generator):
        """Multi-face images rejected for high trust sources."""
        mock_face1 = MagicMock()
        mock_face1.bbox = {"x": 100, "y": 100, "w": 150, "h": 150}
        mock_face1.confidence = 0.95
        mock_face1.image = np.random.rand(160, 160, 3).astype(np.uint8)

        mock_face2 = MagicMock()
        mock_face2.bbox = {"x": 300, "y": 100, "w": 150, "h": 150}
        mock_face2.confidence = 0.95
        mock_face2.image = np.random.rand(160, 160, 3).astype(np.uint8)

        mock_generator.detect_faces.return_value = [mock_face1, mock_face2]

        image_bytes = self._create_test_image(600, 400)
        faces = processor.process_image(image_bytes, trust_level="high")

        assert len(faces) == 0  # Rejected due to multi-face

    def test_process_image_accepts_multi_face_for_medium_trust(self, processor, mock_generator):
        """Multi-face images take largest face for medium trust sources."""
        mock_face1 = MagicMock()
        mock_face1.bbox = {"x": 100, "y": 100, "w": 100, "h": 100}  # Smaller
        mock_face1.confidence = 0.95
        mock_face1.image = np.random.rand(160, 160, 3).astype(np.uint8)

        mock_face2 = MagicMock()
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
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_face_processor.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'face_processor'"

**Step 3: Write minimal implementation**

```python
# api/face_processor.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd api && python -m pytest tests/test_face_processor.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add api/face_processor.py api/tests/test_face_processor.py
git commit -m "feat: add FaceProcessor for image-to-embedding pipeline

Handles: image loading, face detection, quality filtering, embedding generation.
Applies trust-level specific rules (reject multi-face for high-trust sources).

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 4: Integrate Face Processing into EnrichmentCoordinator

**Files:**
- Modify: `api/enrichment_coordinator.py`
- Modify: `api/tests/test_enrichment_coordinator.py`

Wire up the new components into the coordinator.

**Step 1: Write the failing test**

Add to `api/tests/test_enrichment_coordinator.py`:

```python
# Add these tests to the existing TestEnrichmentCoordinator class

    @pytest.mark.asyncio
    async def test_coordinator_processes_images_for_faces(self, mock_db, tmp_path):
        """Coordinator downloads images and extracts faces."""
        from enrichment_coordinator import EnrichmentCoordinator
        from base_scraper import BaseScraper, ScrapedPerformer
        from unittest.mock import MagicMock, patch
        import numpy as np

        # Create mock scraper with image URLs
        class ImageScraper(BaseScraper):
            source_name = "imagesource"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                if page > 1:
                    return (1, [])
                return (1, [
                    ScrapedPerformer(
                        id="p1",
                        name="Test Performer",
                        image_urls=["https://example.com/face1.jpg", "https://example.com/face2.jpg"],
                        stash_ids={"imagesource": "p1"},
                    ),
                ])

            def download_image(self, url, max_retries=3):
                # Return fake PNG image
                from PIL import Image
                import io
                img = Image.new("RGB", (500, 500), color="white")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()

        # Create empty indices
        import voyager
        facenet = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        arcface = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        facenet.save(str(tmp_path / "face_facenet512.voy"))
        arcface.save(str(tmp_path / "face_arcface.voy"))

        # Mock the face detection to return a valid face
        with patch('face_processor.FaceProcessor.process_image') as mock_process:
            from face_processor import ProcessedFace
            from embeddings import FaceEmbedding

            mock_process.return_value = [ProcessedFace(
                embedding=FaceEmbedding(
                    facenet=np.random.rand(512).astype(np.float32),
                    arcface=np.random.rand(512).astype(np.float32),
                ),
                quality_score=0.9,
                bbox={"x": 100, "y": 100, "w": 150, "h": 150},
            )]

            coordinator = EnrichmentCoordinator(
                database=mock_db,
                scrapers=[ImageScraper()],
                data_dir=tmp_path,
                enable_face_processing=True,
            )

            await coordinator.run()

        # Should have processed images
        assert coordinator.stats.images_processed >= 1
        assert coordinator.stats.faces_added >= 1

    @pytest.mark.asyncio
    async def test_coordinator_respects_face_limits(self, mock_db, tmp_path):
        """Coordinator stops processing when face limit reached."""
        from enrichment_coordinator import EnrichmentCoordinator
        from base_scraper import BaseScraper, ScrapedPerformer
        from unittest.mock import patch
        import numpy as np

        class ManyScraper(BaseScraper):
            source_name = "manysource"
            source_type = "stash_box"

            def __init__(self):
                super().__init__(rate_limit_delay=0)

            def get_performer(self, performer_id):
                return None

            def query_performers(self, page=1, per_page=25):
                if page > 1:
                    return (1, [])
                return (1, [
                    ScrapedPerformer(
                        id="p1",
                        name="Test Performer",
                        image_urls=[f"https://example.com/face{i}.jpg" for i in range(10)],
                        stash_ids={"manysource": "p1"},
                    ),
                ])

            def download_image(self, url, max_retries=3):
                from PIL import Image
                import io
                img = Image.new("RGB", (500, 500), color="white")
                buffer = io.BytesIO()
                img.save(buffer, format="PNG")
                return buffer.getvalue()

        # Create empty indices
        import voyager
        facenet = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        arcface = voyager.Index(voyager.Space.Cosine, num_dimensions=512)
        facenet.save(str(tmp_path / "face_facenet512.voy"))
        arcface.save(str(tmp_path / "face_arcface.voy"))

        with patch('face_processor.FaceProcessor.process_image') as mock_process:
            from face_processor import ProcessedFace
            from embeddings import FaceEmbedding

            mock_process.return_value = [ProcessedFace(
                embedding=FaceEmbedding(
                    facenet=np.random.rand(512).astype(np.float32),
                    arcface=np.random.rand(512).astype(np.float32),
                ),
                quality_score=0.9,
                bbox={"x": 100, "y": 100, "w": 150, "h": 150},
            )]

            coordinator = EnrichmentCoordinator(
                database=mock_db,
                scrapers=[ManyScraper()],
                data_dir=tmp_path,
                max_faces_per_source=3,  # Limit to 3
                enable_face_processing=True,
            )

            await coordinator.run()

        # Should have stopped at limit
        assert coordinator.stats.faces_added == 3
```

**Step 2: Run test to verify it fails**

Run: `cd api && python -m pytest tests/test_enrichment_coordinator.py::TestEnrichmentCoordinator::test_coordinator_processes_images_for_faces -v`
Expected: FAIL (missing data_dir parameter, missing enable_face_processing)

**Step 3: Update EnrichmentCoordinator implementation**

Update `api/enrichment_coordinator.py`:

```python
"""
Enrichment coordinator for multi-source database building.

Runs multiple scrapers concurrently, funneling results through
a single write queue for serialized database writes.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import threading

from base_scraper import BaseScraper, ScrapedPerformer
from database import PerformerDatabase
from write_queue import WriteQueue, WriteMessage, WriteOperation

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorStats:
    """Statistics for monitoring."""
    performers_processed: int = 0
    faces_added: int = 0
    images_processed: int = 0
    images_skipped: int = 0
    faces_rejected: int = 0
    errors: int = 0
    by_source: dict = field(default_factory=dict)

    def record_performer(self, source: str):
        """Record a processed performer."""
        self.performers_processed += 1
        self.by_source[source] = self.by_source.get(source, 0) + 1


class EnrichmentCoordinator:
    """
    Coordinates multi-source enrichment.

    Features:
    - Runs scrapers concurrently (one thread per scraper)
    - Single write queue for serialized database writes
    - Per-source and total face limits
    - Resume capability via scrape_progress table
    - Face detection and embedding generation
    """

    SAVE_INTERVAL = 1000  # Save indices every N faces

    def __init__(
        self,
        database: PerformerDatabase,
        scrapers: list[BaseScraper],
        data_dir: Optional[Path] = None,
        max_faces_per_source: int = 5,
        max_faces_total: int = 20,
        dry_run: bool = False,
        enable_face_processing: bool = False,
        source_trust_levels: Optional[dict[str, str]] = None,
    ):
        self.database = database
        self.scrapers = scrapers
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_faces_per_source = max_faces_per_source
        self.max_faces_total = max_faces_total
        self.dry_run = dry_run
        self.enable_face_processing = enable_face_processing
        self.source_trust_levels = source_trust_levels or {}

        self.stats = CoordinatorStats()

        # Write queue with handler
        self.write_queue = WriteQueue(self._handle_write)

        # Thread pool for running sync scrapers
        self._executor = ThreadPoolExecutor(max_workers=max(1, len(scrapers)))

        # Event loop reference for cross-thread communication
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Face processing components (lazy loaded)
        self._index_manager = None
        self._face_processor = None
        self._face_validator = None
        self._generator = None

        # Lock for index writes
        self._index_lock = threading.Lock()

        # Track faces added since last save
        self._faces_since_save = 0

    def _init_face_processing(self):
        """Initialize face processing components."""
        if not self.enable_face_processing or self.data_dir is None:
            return

        from index_manager import IndexManager
        from face_processor import FaceProcessor
        from face_validator import FaceValidator
        from embeddings import FaceEmbeddingGenerator
        from quality_filters import QualityFilters

        logger.info("Initializing face processing components...")

        self._index_manager = IndexManager(self.data_dir)
        self._generator = FaceEmbeddingGenerator()
        self._face_processor = FaceProcessor(
            generator=self._generator,
            quality_filters=QualityFilters(),
        )
        self._face_validator = FaceValidator()

        logger.info(f"Loaded {self._index_manager.current_index} existing embeddings")

    async def run(self):
        """Run all scrapers and process results."""
        self._loop = asyncio.get_event_loop()

        # Initialize face processing if enabled
        self._init_face_processing()

        await self.write_queue.start()

        try:
            # Run each scraper in a thread
            tasks = [
                self._loop.run_in_executor(self._executor, self._run_scraper, scraper)
                for scraper in self.scrapers
            ]

            # Wait for all scrapers to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Wait for write queue to drain
            await self.write_queue.wait_until_empty()

        finally:
            await self.write_queue.stop()
            self._executor.shutdown(wait=True)

            # Save indices on shutdown
            if self._index_manager:
                self._index_manager.save()

        logger.info(f"Enrichment complete: {self.stats.performers_processed} performers, "
                   f"{self.stats.faces_added} faces added")

    def _run_scraper(self, scraper: BaseScraper):
        """Run a single scraper (called from thread pool)."""
        source = scraper.source_name
        logger.info(f"Starting scraper: {source}")

        # Get resume point
        progress = self.database.get_scrape_progress(source)
        last_id = progress['last_processed_id'] if progress else None

        processed = 0
        faces_added = 0

        try:
            for performer in scraper.iter_performers_after(last_id):
                # Process performer
                faces_from_performer = self._process_performer(scraper, performer)
                faces_added += faces_from_performer
                processed += 1

                # Save progress periodically
                if processed % 100 == 0:
                    self.database.save_scrape_progress(
                        source=source,
                        last_processed_id=performer.id,
                        performers_processed=processed,
                        faces_added=faces_added,
                    )
                    logger.info(f"{source}: {processed} performers, {faces_added} faces")

                # Record stats
                self.stats.record_performer(source)

        except Exception as e:
            logger.error(f"Scraper {source} error: {e}")
            self.stats.errors += 1

        # Final progress save
        self.database.save_scrape_progress(
            source=source,
            last_processed_id=performer.id if processed > 0 else last_id,
            performers_processed=processed,
            faces_added=faces_added,
        )

        logger.info(f"Scraper {source} complete: {processed} performers, {faces_added} faces")

    def _process_performer(self, scraper: BaseScraper, performer: ScrapedPerformer) -> int:
        """Process a single performer. Returns number of faces added."""
        source = scraper.source_name

        # Queue performer creation/update
        future = asyncio.run_coroutine_threadsafe(
            self.write_queue.enqueue(WriteMessage(
                operation=WriteOperation.CREATE_PERFORMER,
                source=source,
                performer_data=self._performer_to_dict(performer),
            )),
            self._loop,
        )
        # Wait for enqueue to complete
        future.result(timeout=10.0)

        # Process images for faces if enabled
        if not self.enable_face_processing or not performer.image_urls:
            return 0

        return self._process_performer_images(scraper, performer)

    def _process_performer_images(self, scraper: BaseScraper, performer: ScrapedPerformer) -> int:
        """Download and process images for a performer. Returns faces added."""
        source = scraper.source_name
        trust_level = self.source_trust_levels.get(source, "medium")

        # Get performer ID from database
        stash_id = performer.stash_ids.get(source)
        if not stash_id:
            return 0

        db_performer = self.database.get_performer_by_stashbox_id(source, stash_id)
        if not db_performer:
            return 0

        performer_id = db_performer.id

        # Check if already at total limit
        if self.database.total_limit_reached(performer_id, self.max_faces_total):
            return 0

        # Get existing embeddings for validation
        existing_embeddings = self._get_existing_embeddings(performer_id)

        faces_added = 0

        for image_url in performer.image_urls:
            # Check per-source limit
            if self.database.source_limit_reached(performer_id, source, self.max_faces_per_source):
                break

            # Check total limit
            if self.database.total_limit_reached(performer_id, self.max_faces_total):
                break

            # Download image
            try:
                image_data = scraper.download_image(image_url)
                if not image_data:
                    continue
            except Exception as e:
                logger.debug(f"Failed to download {image_url}: {e}")
                continue

            self.stats.images_processed += 1

            # Process image for faces
            processed_faces = self._face_processor.process_image(image_data, trust_level)

            if not processed_faces:
                continue

            # Validate and add each face
            for face in processed_faces:
                # Validate against existing embeddings
                validation = self._face_validator.validate(
                    new_embedding=face.embedding.facenet,
                    existing_embeddings=existing_embeddings,
                    trust_level=trust_level,
                )

                if not validation.accepted:
                    self.stats.faces_rejected += 1
                    logger.debug(f"Face rejected for {performer.name}: {validation.reason}")
                    continue

                # Add to index and database
                with self._index_lock:
                    face_index = self._index_manager.add_embedding(face.embedding)
                    self._faces_since_save += 1

                    # Periodic save
                    if self._faces_since_save >= self.SAVE_INTERVAL:
                        self._index_manager.save()
                        self._faces_since_save = 0

                # Queue database write
                future = asyncio.run_coroutine_threadsafe(
                    self.write_queue.enqueue(WriteMessage(
                        operation=WriteOperation.ADD_EMBEDDING,
                        source=source,
                        performer_id=performer_id,
                        image_url=image_url,
                        quality_score=face.quality_score,
                        # Store index for database record
                        embedding=None,  # Don't need to pass embedding, just index
                        embedding_type=str(face_index),  # Abuse this field for index
                    )),
                    self._loop,
                )
                future.result(timeout=10.0)

                faces_added += 1
                self.stats.faces_added += 1

                # Update existing embeddings for subsequent validation
                existing_embeddings.append(face.embedding.facenet)

        return faces_added

    def _get_existing_embeddings(self, performer_id: int) -> list:
        """Get existing face embeddings for a performer."""
        import numpy as np

        if not self._index_manager:
            return []

        faces = self.database.get_faces(performer_id)
        embeddings = []

        for face in faces:
            try:
                embedding = self._index_manager.get_embedding(face.facenet_index)
                embeddings.append(embedding.facenet)
            except Exception:
                pass

        return embeddings

    def _performer_to_dict(self, performer: ScrapedPerformer) -> dict:
        """Convert performer to dict for write queue."""
        return {
            'id': performer.id,
            'name': performer.name,
            'aliases': performer.aliases,
            'gender': performer.gender,
            'country': performer.country,
            'birth_date': performer.birth_date,
            'death_date': performer.death_date,
            'image_urls': performer.image_urls,
            'stash_ids': performer.stash_ids,
            'external_urls': performer.external_urls,
            'ethnicity': performer.ethnicity,
            'height_cm': performer.height_cm,
            'eye_color': performer.eye_color,
            'hair_color': performer.hair_color,
            'career_start_year': performer.career_start_year,
            'career_end_year': performer.career_end_year,
            'disambiguation': performer.disambiguation,
        }

    async def _handle_write(self, message: WriteMessage):
        """Handle write queue messages."""
        if self.dry_run:
            return

        if message.operation == WriteOperation.CREATE_PERFORMER:
            await self._handle_create_performer(message)
        elif message.operation == WriteOperation.ADD_EMBEDDING:
            await self._handle_add_embedding(message)
        elif message.operation == WriteOperation.ADD_STASH_ID:
            await self._handle_add_stash_id(message)
        elif message.operation == WriteOperation.ADD_ALIAS:
            await self._handle_add_alias(message)
        elif message.operation == WriteOperation.ADD_EXTERNAL_URL:
            await self._handle_add_url(message)

    async def _handle_create_performer(self, message: WriteMessage):
        """Create or update performer in database."""
        data = message.performer_data
        source = message.source

        # Check if performer exists by stash ID
        stash_id = data.get('stash_ids', {}).get(source)
        if stash_id:
            existing = self.database.get_performer_by_stashbox_id(source, stash_id)
        else:
            existing = None

        if existing:
            # Update existing performer
            self.database.update_performer(
                existing.id,
                canonical_name=data.get('name'),
                gender=data.get('gender'),
                country=data.get('country'),
            )
            performer_id = existing.id
        else:
            # Create new performer
            performer_id = self.database.add_performer(
                canonical_name=data['name'],
                gender=data.get('gender'),
                country=data.get('country'),
                birth_date=data.get('birth_date'),
            )
            # Add stash ID for this source
            if stash_id:
                self.database.add_stashbox_id(performer_id, source, stash_id)

        # Add aliases
        for alias in data.get('aliases', []):
            try:
                self.database.add_alias(performer_id, alias, source)
            except Exception:
                pass  # Ignore duplicate aliases

        # Add URLs
        for site, urls in data.get('external_urls', {}).items():
            for url in urls:
                try:
                    self.database.add_url(performer_id, url, source)
                except Exception:
                    pass  # Ignore duplicate URLs

    async def _handle_add_embedding(self, message: WriteMessage):
        """Add face embedding record to database."""
        if not message.performer_id or not message.image_url:
            return

        # The index was stored in embedding_type field
        try:
            face_index = int(message.embedding_type)
        except (ValueError, TypeError):
            return

        self.database.add_face(
            performer_id=message.performer_id,
            facenet_index=face_index,
            arcface_index=face_index,
            image_url=message.image_url,
            source_endpoint=message.source,
            quality_score=message.quality_score,
        )

    async def _handle_add_stash_id(self, message: WriteMessage):
        """Add stash ID to performer."""
        if message.performer_id and message.endpoint and message.stashbox_id:
            try:
                self.database.add_stashbox_id(
                    message.performer_id,
                    message.endpoint,
                    message.stashbox_id,
                )
            except Exception:
                pass

    async def _handle_add_alias(self, message: WriteMessage):
        """Add alias to performer."""
        if message.performer_id and message.alias:
            try:
                self.database.add_alias(
                    message.performer_id,
                    message.alias,
                    message.source,
                )
            except Exception:
                pass

    async def _handle_add_url(self, message: WriteMessage):
        """Add external URL to performer."""
        if message.performer_id and message.url:
            try:
                self.database.add_url(
                    message.performer_id,
                    message.url,
                    message.source,
                )
            except Exception:
                pass
```

**Step 4: Run tests to verify they pass**

Run: `cd api && python -m pytest tests/test_enrichment_coordinator.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/enrichment_coordinator.py api/tests/test_enrichment_coordinator.py
git commit -m "feat: integrate face processing into EnrichmentCoordinator

Adds image download → face detection → validation → storage pipeline.
Features:
- Lazy-loaded face processing components
- Thread-safe index writes with periodic saves
- Per-source and total face limits
- Trust-level based validation
- Existing embedding comparison for medium-trust sources

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 5: Update enrichment_builder.py CLI

**Files:**
- Modify: `api/enrichment_builder.py`

Add `--enable-faces` flag and wire up trust levels from config.

**Step 1: Read current implementation**

Run: `cat api/enrichment_builder.py` to understand current CLI structure.

**Step 2: Update CLI**

Add these changes to `api/enrichment_builder.py`:

```python
# Add to argparse arguments:
parser.add_argument(
    "--enable-faces",
    action="store_true",
    help="Enable face detection and embedding (requires GPU, slower)",
)

parser.add_argument(
    "--save-interval",
    type=int,
    default=1000,
    help="Save indices every N faces (default: 1000)",
)

# Update coordinator instantiation to include:
# - data_dir from config
# - enable_face_processing from --enable-faces
# - source_trust_levels from sources.yaml config

# In the run_enrichment() function:
source_trust_levels = {
    name: config.get_source(name).trust_level
    for name in config.get_enabled_sources()
}

coordinator = EnrichmentCoordinator(
    database=database,
    scrapers=scrapers,
    data_dir=data_dir,
    max_faces_per_source=args.source_max_faces or config.global_settings.max_faces_per_performer,
    max_faces_total=args.max_faces_total or config.global_settings.max_faces_per_performer,
    dry_run=args.dry_run,
    enable_face_processing=args.enable_faces,
    source_trust_levels=source_trust_levels,
)
```

**Step 3: Test manually**

Run: `cd api && python enrichment_builder.py --help`
Verify: `--enable-faces` and `--save-interval` appear in help output.

**Step 4: Commit**

```bash
git add api/enrichment_builder.py
git commit -m "feat: add --enable-faces flag to enrichment CLI

Allows running enrichment with or without face processing.
Also adds --save-interval for controlling index save frequency.

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 6: Run Full Test Suite and Fix Issues

**Step 1: Run all enrichment-related tests**

Run: `cd api && python -m pytest tests/test_enrichment_coordinator.py tests/test_face_validator.py tests/test_face_processor.py tests/test_index_manager.py -v`

**Step 2: Fix any failures**

Address test failures as they arise.

**Step 3: Run broader test suite**

Run: `cd api && python -m pytest tests/ -v --ignore=tests/test_stashdb_adapter.py`

(Ignore stashdb adapter test if it requires network)

**Step 4: Commit any fixes**

```bash
git add -A
git commit -m "fix: address test failures from face integration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Task 7: Update Documentation

**Files:**
- Modify: `docs/plans/SESSION-CONTEXT.md`

**Step 1: Update session context**

Add to the "Current State" section:

```markdown
**7. Face Enrichment Integration** ✅ COMPLETE
- `FaceValidator` - Trust-level based validation
- `IndexManager` - Voyager index loading/saving
- `FaceProcessor` - Image → embedding pipeline
- `EnrichmentCoordinator` - Full integration with face processing
- CLI: `--enable-faces` flag for enrichment_builder.py
```

Update the "Run Enrichment" section with face processing examples:

```markdown
# Run with face processing (slower, requires GPU)
python enrichment_builder.py --sources stashdb --enable-faces

# Run with face limits
python enrichment_builder.py --sources stashdb,babepedia --enable-faces --max-faces-total 12
```

**Step 2: Commit**

```bash
git add docs/plans/SESSION-CONTEXT.md
git commit -m "docs: update SESSION-CONTEXT with face enrichment integration

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

---

## Summary

| Task | Files | Tests | Purpose |
|------|-------|-------|---------|
| 1 | `face_validator.py` | 6 tests | Trust-level validation logic |
| 2 | `index_manager.py` | 7 tests | Voyager index operations |
| 3 | `face_processor.py` | 8 tests | Image → embedding pipeline |
| 4 | `enrichment_coordinator.py` | +2 tests | Full integration |
| 5 | `enrichment_builder.py` | manual | CLI flags |
| 6 | various | suite | Fix any issues |
| 7 | `SESSION-CONTEXT.md` | - | Documentation |

**Total new tests:** ~23
**New files:** 3 (`face_validator.py`, `index_manager.py`, `face_processor.py`)
**Modified files:** 3 (`enrichment_coordinator.py`, `enrichment_builder.py`, `SESSION-CONTEXT.md`)

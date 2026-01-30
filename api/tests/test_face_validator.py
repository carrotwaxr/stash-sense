# api/tests/test_face_validator.py
"""Tests for face validation logic."""
import pytest
import numpy as np


class TestFaceValidator:
    """Test trust-level based face validation."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        from face_validator import FaceValidator
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
        # Create opposite embeddings (guaranteed to be very different)
        existing = np.random.rand(512).astype(np.float32)
        existing = existing / np.linalg.norm(existing)

        # Negate to ensure maximum distance (cosine distance of 2.0)
        new = -existing

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

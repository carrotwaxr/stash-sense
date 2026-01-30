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

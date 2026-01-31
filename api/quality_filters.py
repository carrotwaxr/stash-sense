"""
Quality filters for face detection during enrichment.

Filters out low-quality faces that would degrade recognition accuracy.

See: docs/plans/2026-01-29-multi-source-enrichment-design.md
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityFilters:
    """Quality filter configuration."""
    min_face_size: int = 80
    min_image_size: int = 300  # Lowered from 400 to support IAFD (340x400)
    min_detection_confidence: float = 0.8
    max_face_angle: float = 45.0
    prefer_single_face: bool = True


@dataclass
class FilterResult:
    """Result of quality filter check."""
    passed: bool
    rejection_reason: Optional[str] = None
    details: Optional[dict] = None


class QualityFilter:
    """
    Applies quality filters to detected faces.

    Usage:
        filters = QualityFilters(min_face_size=80)
        qf = QualityFilter(filters)

        result = qf.check_face(detected_face, image_width, image_height)
        if result.passed:
            # Process face
        else:
            print(f"Rejected: {result.rejection_reason}")
    """

    def __init__(self, filters: QualityFilters):
        self.filters = filters

    def check_face(
        self,
        face,
        image_width: int,
        image_height: int,
        total_faces_in_image: int = 1,
        trust_level: str = "medium",
    ) -> FilterResult:
        """
        Check if a detected face passes quality filters.

        Args:
            face: DetectedFace-like object with bbox, confidence, and optional yaw
            image_width: Width of source image
            image_height: Height of source image
            total_faces_in_image: Number of faces detected in the image
            trust_level: Source trust level ("high", "medium", "low")

        Returns:
            FilterResult with passed status and rejection reason if failed
        """
        # Check image resolution
        min_dimension = min(image_width, image_height)
        if min_dimension < self.filters.min_image_size:
            return FilterResult(
                passed=False,
                rejection_reason="image_too_small",
                details={"min_dimension": min_dimension, "required": self.filters.min_image_size},
            )

        # Check face size
        face_width = face.bbox.get("w", 0)
        face_height = face.bbox.get("h", 0)
        face_size = min(face_width, face_height)

        if face_size < self.filters.min_face_size:
            return FilterResult(
                passed=False,
                rejection_reason="face_too_small",
                details={"face_size": face_size, "required": self.filters.min_face_size},
            )

        # Check detection confidence
        if face.confidence < self.filters.min_detection_confidence:
            return FilterResult(
                passed=False,
                rejection_reason="low_confidence",
                details={"confidence": face.confidence, "required": self.filters.min_detection_confidence},
            )

        # Check face angle (if available)
        yaw = getattr(face, "yaw", None)
        if yaw is not None and abs(yaw) > self.filters.max_face_angle:
            return FilterResult(
                passed=False,
                rejection_reason="extreme_angle",
                details={"yaw": yaw, "max_allowed": self.filters.max_face_angle},
            )

        # Check single face preference (only for high-trust sources)
        if (
            self.filters.prefer_single_face
            and trust_level == "high"
            and total_faces_in_image > 1
        ):
            return FilterResult(
                passed=False,
                rejection_reason="multi_face",
                details={"faces_in_image": total_faces_in_image},
            )

        return FilterResult(passed=True)

    def check_image(self, image_width: int, image_height: int) -> FilterResult:
        """Quick check if image resolution is acceptable."""
        min_dimension = min(image_width, image_height)
        if min_dimension < self.filters.min_image_size:
            return FilterResult(
                passed=False,
                rejection_reason="image_too_small",
                details={"min_dimension": min_dimension, "required": self.filters.min_image_size},
            )
        return FilterResult(passed=True)

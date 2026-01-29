"""Tests for face quality filtering."""
import pytest
from unittest.mock import MagicMock


class TestQualityFilters:
    """Test quality filter logic."""

    def test_face_too_small_rejected(self):
        """Faces below minimum size are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_face_size=80)
        qf = QualityFilter(filters)

        # 50x50 face should be rejected
        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 50, "h": 50}
        face.confidence = 0.95

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is False
        assert "face_too_small" in result.rejection_reason

    def test_face_large_enough_accepted(self):
        """Faces at or above minimum size pass."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_face_size=80)
        qf = QualityFilter(filters)

        face = MagicMock(spec=[])
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 120}
        face.confidence = 0.95

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is True

    def test_image_too_small_rejected(self):
        """Images below minimum resolution are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_image_size=400)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95

        # 300x200 image should be rejected
        result = qf.check_face(face, image_width=300, image_height=200)
        assert result.passed is False
        assert "image_too_small" in result.rejection_reason

    def test_low_confidence_rejected(self):
        """Faces with low detection confidence are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_detection_confidence=0.8)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.6

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is False
        assert "low_confidence" in result.rejection_reason

    def test_extreme_angle_rejected(self):
        """Faces at extreme angles are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(max_face_angle=45.0)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95
        face.yaw = 60.0  # Looking sideways

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is False
        assert "extreme_angle" in result.rejection_reason

    def test_multi_face_rejected_for_high_trust(self):
        """Multiple faces in image rejected for high-trust sources."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(prefer_single_face=True)
        qf = QualityFilter(filters)

        face = MagicMock(spec=[])
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95

        result = qf.check_face(
            face,
            image_width=800,
            image_height=600,
            total_faces_in_image=3,
            trust_level="high"
        )
        assert result.passed is False
        assert "multi_face" in result.rejection_reason

    def test_multi_face_allowed_for_low_trust(self):
        """Multiple faces in image allowed for low-trust (will be clustered)."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(prefer_single_face=True)
        qf = QualityFilter(filters)

        face = MagicMock(spec=[])
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95

        result = qf.check_face(
            face,
            image_width=800,
            image_height=600,
            total_faces_in_image=3,
            trust_level="low"
        )
        # Low trust doesn't apply single-face filter
        assert result.passed is True

    def test_no_angle_info_passes(self):
        """Faces without angle info pass angle check."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(max_face_angle=45.0)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95
        face.yaw = None  # No angle info

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is True

    def test_check_image_resolution(self):
        """Image resolution can be checked independently."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_image_size=400)
        qf = QualityFilter(filters)

        # Too small
        result = qf.check_image(300, 200)
        assert result.passed is False

        # Large enough
        result = qf.check_image(800, 600)
        assert result.passed is True

        # Portrait orientation - min dimension matters
        result = qf.check_image(350, 600)
        assert result.passed is False

        result = qf.check_image(400, 600)
        assert result.passed is True

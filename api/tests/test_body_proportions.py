"""Tests for body proportion extraction using MediaPipe Pose."""

import numpy as np
import pytest


class TestBodyProportions:
    """Tests for BodyProportions dataclass."""

    def test_to_dict(self):
        """Test that BodyProportions converts to dict correctly."""
        from body_proportions import BodyProportions

        props = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.85,
        )

        result = props.to_dict()

        assert result == {
            'shoulder_hip_ratio': 1.2,
            'leg_torso_ratio': 1.5,
            'arm_span_height_ratio': 1.0,
            'confidence': 0.85,
        }

    def test_from_dict(self):
        """Test that BodyProportions can be created from dict."""
        from body_proportions import BodyProportions

        data = {
            'shoulder_hip_ratio': 1.3,
            'leg_torso_ratio': 1.4,
            'arm_span_height_ratio': 0.95,
            'confidence': 0.9,
        }

        props = BodyProportions.from_dict(data)

        assert props.shoulder_hip_ratio == 1.3
        assert props.leg_torso_ratio == 1.4
        assert props.arm_span_height_ratio == 0.95
        assert props.confidence == 0.9


class TestBodyProportionExtractor:
    """Tests for BodyProportionExtractor class."""

    def test_extract_returns_none_for_blank_image(self):
        """Test that extract returns None for a blank image with no pose."""
        from body_proportions import BodyProportionExtractor

        extractor = BodyProportionExtractor()

        # Create a blank black image (no pose detectable)
        blank_image = np.zeros((480, 640, 3), dtype=np.uint8)

        result = extractor.extract(blank_image)

        assert result is None

    def test_extractor_has_required_methods(self):
        """Test that BodyProportionExtractor has the required interface."""
        from body_proportions import BodyProportionExtractor

        extractor = BodyProportionExtractor()

        # Check required methods exist
        assert hasattr(extractor, 'extract')
        assert callable(extractor.extract)

        # Check internal methods exist
        assert hasattr(extractor, '_extract_landmarks')
        assert hasattr(extractor, '_compute_ratios')
        assert hasattr(extractor, '_compute_confidence')
        assert hasattr(extractor, '_distance')
        assert hasattr(extractor, '_midpoint')

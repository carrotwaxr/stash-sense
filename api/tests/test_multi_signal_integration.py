"""Integration tests for multi-signal identification."""
import pytest
import numpy as np
from unittest.mock import Mock, patch


class TestMultiSignalIntegration:
    """Test full multi-signal pipeline."""

    def test_identify_with_all_signals(self):
        """Test identification using face + body + tattoo signals."""
        # This is a smoke test - full integration requires running services
        from body_proportions import BodyProportionExtractor, BodyProportions
        from tattoo_detector import TattooDetector, TattooResult
        from signal_scoring import body_ratio_penalty, tattoo_adjustment

        # Test scoring functions work together
        body = BodyProportions(1.5, 1.3, 1.0, 0.9)
        candidate_body = BodyProportions(1.52, 1.3, 1.0, 0.9)
        body_mult = body_ratio_penalty(body, candidate_body)
        assert body_mult == 1.0  # Compatible

        tattoo = TattooResult([], False, 0.0)
        tattoo_mult = tattoo_adjustment(tattoo, False, set())
        assert tattoo_mult == 1.0  # Neutral

        # Combined score
        base_score = 0.8
        final = base_score * body_mult * tattoo_mult
        assert final == 0.8

    def test_graceful_degradation(self):
        """Test that missing signals don't break identification."""
        from signal_scoring import body_ratio_penalty, tattoo_adjustment
        from body_proportions import BodyProportions

        # No body data
        candidate = BodyProportions(1.5, 1.3, 1.0, 0.9)
        assert body_ratio_penalty(None, candidate) == 1.0
        assert body_ratio_penalty(None, None) == 1.0

        # No tattoo data
        assert tattoo_adjustment(None, True, {'arm'}) == 1.0

    def test_body_extractor_dataclass_serialization(self):
        """Test BodyProportions round-trip through dict."""
        from body_proportions import BodyProportions

        original = BodyProportions(1.5, 1.3, 1.0, 0.9)
        d = original.to_dict()
        restored = BodyProportions.from_dict(d)

        assert restored.shoulder_hip_ratio == original.shoulder_hip_ratio
        assert restored.leg_torso_ratio == original.leg_torso_ratio

    def test_tattoo_result_dataclass_serialization(self):
        """Test TattooResult round-trip through dict."""
        from tattoo_detector import TattooResult, TattooDetection

        original = TattooResult(
            detections=[TattooDetection({'x': 0.1, 'y': 0.2, 'w': 0.1, 'h': 0.1}, 0.8, 'left arm')],
            has_tattoos=True,
            confidence=0.8
        )
        d = original.to_dict()
        restored = TattooResult.from_dict(d)

        assert restored.has_tattoos == original.has_tattoos
        assert restored.confidence == original.confidence
        assert len(restored.detections) == 1

"""Tests for signal scoring functions."""

import pytest

from body_proportions import BodyProportions
from tattoo_detector import TattooResult, TattooDetection


class TestBodyRatioPenalty:
    """Tests for body_ratio_penalty function."""

    def test_none_query_ratios_returns_no_penalty(self):
        """Test that None query_ratios returns 1.0 (no penalty)."""
        from signal_scoring import body_ratio_penalty

        candidate = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )

        result = body_ratio_penalty(None, candidate)

        assert result == 1.0

    def test_none_candidate_ratios_returns_no_penalty(self):
        """Test that None candidate_ratios returns 1.0 (no penalty)."""
        from signal_scoring import body_ratio_penalty

        query = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )

        result = body_ratio_penalty(query, None)

        assert result == 1.0

    def test_both_none_returns_no_penalty(self):
        """Test that both None returns 1.0 (no penalty)."""
        from signal_scoring import body_ratio_penalty

        result = body_ratio_penalty(None, None)

        assert result == 1.0

    def test_compatible_ratios_returns_no_penalty(self):
        """Test that compatible ratios (diff <= 0.12) return 1.0."""
        from signal_scoring import body_ratio_penalty

        query = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )
        candidate = BodyProportions(
            shoulder_hip_ratio=1.25,  # diff = 0.05, <= 0.12
            leg_torso_ratio=1.6,
            arm_span_height_ratio=1.1,
            confidence=0.85,
        )

        result = body_ratio_penalty(query, candidate)

        assert result == 1.0

    def test_slight_mismatch_returns_085(self):
        """Test that slight mismatch (diff > 0.12, <= 0.2) returns 0.85."""
        from signal_scoring import body_ratio_penalty

        query = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )
        candidate = BodyProportions(
            shoulder_hip_ratio=1.35,  # diff = 0.15, > 0.12 and <= 0.2
            leg_torso_ratio=1.6,
            arm_span_height_ratio=1.1,
            confidence=0.85,
        )

        result = body_ratio_penalty(query, candidate)

        assert result == 0.85

    def test_moderate_mismatch_returns_06(self):
        """Test that moderate mismatch (diff > 0.2, <= 0.35) returns 0.6."""
        from signal_scoring import body_ratio_penalty

        query = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )
        candidate = BodyProportions(
            shoulder_hip_ratio=1.45,  # diff = 0.25, > 0.2 and <= 0.35
            leg_torso_ratio=1.6,
            arm_span_height_ratio=1.1,
            confidence=0.85,
        )

        result = body_ratio_penalty(query, candidate)

        assert result == 0.6

    def test_severe_mismatch_returns_03(self):
        """Test that severe mismatch (diff > 0.35) returns 0.3."""
        from signal_scoring import body_ratio_penalty

        query = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )
        candidate = BodyProportions(
            shoulder_hip_ratio=1.6,  # diff = 0.4, > 0.35
            leg_torso_ratio=1.6,
            arm_span_height_ratio=1.1,
            confidence=0.85,
        )

        result = body_ratio_penalty(query, candidate)

        assert result == 0.3

    def test_boundary_at_012_is_compatible(self):
        """Test that difference at boundary (just under 0.12) is compatible."""
        from signal_scoring import body_ratio_penalty

        query = BodyProportions(
            shoulder_hip_ratio=1.0,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )
        candidate = BodyProportions(
            shoulder_hip_ratio=1.11,  # diff = 0.11, clearly <= 0.12
            leg_torso_ratio=1.6,
            arm_span_height_ratio=1.1,
            confidence=0.85,
        )

        result = body_ratio_penalty(query, candidate)

        assert result == 1.0


class TestTattooAdjustment:
    """Tests for tattoo_adjustment function."""

    def test_none_query_result_returns_neutral(self):
        """Test that None query_result returns 1.0 (neutral)."""
        from signal_scoring import tattoo_adjustment

        result = tattoo_adjustment(None, has_tattoos=True, locations={"left arm"})

        assert result == 1.0

    def test_query_has_tattoos_candidate_has_none_returns_penalty(self):
        """Test that query with tattoos but candidate without returns 0.7."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[
                TattooDetection(
                    bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                    confidence=0.9,
                    location_hint="left arm",
                )
            ],
            has_tattoos=True,
            confidence=0.9,
        )

        result = tattoo_adjustment(query_result, has_tattoos=False, locations=set())

        assert result == 0.7

    def test_query_no_tattoos_candidate_has_tattoos_returns_slight_penalty(self):
        """Test that query without tattoos but candidate with returns 0.95."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[],
            has_tattoos=False,
            confidence=0.0,
        )

        result = tattoo_adjustment(query_result, has_tattoos=True, locations={"left arm"})

        assert result == 0.95

    def test_both_have_tattoos_same_location_returns_boost(self):
        """Test that both having tattoos at same location returns 1.15."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[
                TattooDetection(
                    bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                    confidence=0.9,
                    location_hint="left arm",
                )
            ],
            has_tattoos=True,
            confidence=0.9,
        )

        result = tattoo_adjustment(
            query_result, has_tattoos=True, locations={"left arm", "torso"}
        )

        assert result == 1.15

    def test_both_have_tattoos_different_locations_returns_penalty(self):
        """Test that both having tattoos at different locations returns 0.9."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[
                TattooDetection(
                    bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                    confidence=0.9,
                    location_hint="left arm",
                )
            ],
            has_tattoos=True,
            confidence=0.9,
        )

        result = tattoo_adjustment(
            query_result, has_tattoos=True, locations={"right leg", "torso"}
        )

        assert result == 0.9

    def test_neither_has_tattoos_returns_neutral(self):
        """Test that neither having tattoos returns 1.0 (neutral)."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[],
            has_tattoos=False,
            confidence=0.0,
        )

        result = tattoo_adjustment(query_result, has_tattoos=False, locations=set())

        assert result == 1.0

    def test_multiple_query_locations_one_matches_returns_boost(self):
        """Test that multiple query locations with one matching returns boost."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[
                TattooDetection(
                    bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                    confidence=0.9,
                    location_hint="left arm",
                ),
                TattooDetection(
                    bbox={"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
                    confidence=0.8,
                    location_hint="torso",
                ),
            ],
            has_tattoos=True,
            confidence=0.9,
        )

        result = tattoo_adjustment(
            query_result, has_tattoos=True, locations={"torso", "right leg"}
        )

        assert result == 1.15

    def test_empty_candidate_locations_with_tattoos_returns_penalty(self):
        """Test that candidate with tattoos but empty locations returns penalty."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[
                TattooDetection(
                    bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                    confidence=0.9,
                    location_hint="left arm",
                )
            ],
            has_tattoos=True,
            confidence=0.9,
        )

        # Candidate has tattoos but locations set is empty (no location hints)
        result = tattoo_adjustment(query_result, has_tattoos=True, locations=set())

        assert result == 0.9

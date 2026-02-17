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
    """Tests for tattoo_adjustment function (v2 embedding-based)."""

    def test_none_query_result_returns_neutral(self):
        """Test that None query_result returns 1.0 (neutral)."""
        from signal_scoring import tattoo_adjustment

        result = tattoo_adjustment(
            None, candidate_id="stashdb.org:uuid1",
            tattoo_scores={"stashdb.org:uuid1": 0.9},
            has_tattoo_embeddings=True,
        )

        assert result == 1.0

    def test_query_has_tattoos_candidate_has_no_embeddings_returns_penalty(self):
        """Test that query with tattoos but candidate without embeddings returns 0.7."""
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
            query_result, candidate_id="stashdb.org:uuid1",
            tattoo_scores={}, has_tattoo_embeddings=False,
        )

        assert result == 0.7

    def test_query_no_tattoos_candidate_has_embeddings_returns_slight_penalty(self):
        """Test that query without tattoos but candidate with embeddings returns 0.95."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[],
            has_tattoos=False,
            confidence=0.0,
        )

        result = tattoo_adjustment(
            query_result, candidate_id="stashdb.org:uuid1",
            tattoo_scores={}, has_tattoo_embeddings=True,
        )

        assert result == 0.95

    def test_high_similarity_returns_strong_boost(self):
        """Test that high tattoo similarity (>0.7) returns 1.3-1.5 boost."""
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
            query_result, candidate_id="stashdb.org:uuid1",
            tattoo_scores={"stashdb.org:uuid1": 0.85},
            has_tattoo_embeddings=True,
        )

        assert 1.3 <= result <= 1.5

    def test_moderate_similarity_returns_modest_boost(self):
        """Test that moderate tattoo similarity (>0.5, <=0.7) returns 1.1."""
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
            query_result, candidate_id="stashdb.org:uuid1",
            tattoo_scores={"stashdb.org:uuid1": 0.6},
            has_tattoo_embeddings=True,
        )

        assert result == 1.1

    def test_low_similarity_returns_neutral(self):
        """Test that low similarity (<=0.5) with embeddings returns 1.0."""
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
            query_result, candidate_id="stashdb.org:uuid1",
            tattoo_scores={"stashdb.org:uuid1": 0.3},
            has_tattoo_embeddings=True,
        )

        assert result == 1.0

    def test_neither_has_tattoos_returns_neutral(self):
        """Test that neither having tattoos returns 1.0 (neutral)."""
        from signal_scoring import tattoo_adjustment

        query_result = TattooResult(
            detections=[],
            has_tattoos=False,
            confidence=0.0,
        )

        result = tattoo_adjustment(
            query_result, candidate_id="stashdb.org:uuid1",
            tattoo_scores={}, has_tattoo_embeddings=False,
        )

        assert result == 1.0

    def test_no_scores_dict_returns_penalty_if_no_embeddings(self):
        """Test that missing scores dict with no embeddings returns penalty."""
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
            query_result, candidate_id="stashdb.org:uuid1",
            tattoo_scores=None, has_tattoo_embeddings=False,
        )

        assert result == 0.7

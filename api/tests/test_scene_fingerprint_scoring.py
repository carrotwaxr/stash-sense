"""Tests for scene fingerprint match quality scoring."""

import pytest
from scene_fingerprint_scoring import score_match, is_high_confidence


class TestScoreMatch:
    """Test quality score computation for a candidate match."""

    def test_single_md5_match(self):
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "abc123", "duration": 1834.5, "submissions": 3}
            ],
            total_local_fingerprints=3,
            local_duration=1835.0,
        )
        assert result["match_count"] == 1
        assert result["match_percentage"] == pytest.approx(33.3, abs=0.1)
        assert result["has_exact_hash"] is True

    def test_multiple_fingerprints_all_match(self):
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "abc", "duration": 1800.0, "submissions": 5},
                {"algorithm": "OSHASH", "hash": "def", "duration": 1800.0, "submissions": 3},
                {"algorithm": "PHASH", "hash": "ghi", "duration": 1800.0, "submissions": 1},
            ],
            total_local_fingerprints=3,
            local_duration=1800.0,
        )
        assert result["match_count"] == 3
        assert result["match_percentage"] == pytest.approx(100.0)
        assert result["has_exact_hash"] is True

    def test_phash_only_match(self):
        result = score_match(
            matching_fingerprints=[
                {"algorithm": "PHASH", "hash": "abc", "duration": 1800.0, "submissions": 1}
            ],
            total_local_fingerprints=2,
            local_duration=1800.0,
        )
        assert result["has_exact_hash"] is False

    def test_duration_agreement(self):
        close = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1834.0, "submissions": 1}
            ],
            total_local_fingerprints=1,
            local_duration=1835.0,
        )
        assert close["duration_agreement"] is True

        far = score_match(
            matching_fingerprints=[
                {"algorithm": "MD5", "hash": "a", "duration": 1700.0, "submissions": 1}
            ],
            total_local_fingerprints=1,
            local_duration=1835.0,
        )
        assert far["duration_agreement"] is False

    def test_empty_fingerprints(self):
        result = score_match(
            matching_fingerprints=[],
            total_local_fingerprints=3,
            local_duration=1800.0,
        )
        assert result["match_count"] == 0
        assert result["match_percentage"] == 0.0


class TestIsHighConfidence:
    def test_meets_both_thresholds(self):
        assert is_high_confidence(match_count=2, match_percentage=66.7) is True

    def test_below_count_threshold(self):
        assert is_high_confidence(match_count=1, match_percentage=100.0) is False

    def test_below_percentage_threshold(self):
        assert is_high_confidence(match_count=2, match_percentage=50.0) is False

    def test_custom_thresholds(self):
        assert is_high_confidence(match_count=3, match_percentage=80.0, min_count=3, min_percentage=80) is True

    def test_custom_thresholds_fail(self):
        assert is_high_confidence(match_count=2, match_percentage=80.0, min_count=3, min_percentage=80) is False

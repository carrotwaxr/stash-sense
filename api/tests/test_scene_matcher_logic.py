"""Tests for scene_matcher.py pure functions - cosine distance and cluster merging."""

import sys
from unittest.mock import Mock

# Mock recognizer before importing scene_matcher
sys.modules['recognizer'] = Mock()

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from scene_matcher import _cosine_distance, merge_clusters_by_match  # noqa: E402


class TestCosineDistance:
    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        assert pytest.approx(_cosine_distance(v, v), abs=1e-6) == 0.0

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        assert pytest.approx(_cosine_distance(a, b), abs=1e-6) == 1.0

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        assert pytest.approx(_cosine_distance(a, b), abs=1e-6) == 2.0

    def test_zero_vector_returns_one(self):
        a = np.array([0.0, 0.0, 0.0])
        b = np.array([1.0, 2.0, 3.0])
        assert _cosine_distance(a, b) == 1.0

    def test_both_zero_vectors(self):
        a = np.array([0.0, 0.0])
        b = np.array([0.0, 0.0])
        assert _cosine_distance(a, b) == 1.0

    def test_similar_vectors_small_distance(self):
        a = np.array([1.0, 1.0, 1.0])
        b = np.array([1.0, 1.0, 1.1])
        dist = _cosine_distance(a, b)
        assert dist < 0.01  # Very close vectors


def _make_match(stashdb_id, combined_score):
    """Create a mock match object."""
    match = Mock()
    match.stashdb_id = stashdb_id
    match.combined_score = combined_score
    return match


def _make_result(matches):
    """Create a mock RecognitionResult with given matches."""
    result = Mock()
    result.matches = matches
    return result


class TestMergeClusters:
    def test_single_cluster_unchanged(self):
        match = _make_match("perf-1", 0.2)
        result = _make_result([match])
        clusters = [[(0, result)]]

        merged = merge_clusters_by_match(clusters)

        assert len(merged) == 1
        assert len(merged[0]) == 1

    def test_two_clusters_same_match_merged(self):
        match1 = _make_match("perf-1", 0.2)
        result1 = _make_result([match1])
        match2 = _make_match("perf-1", 0.3)
        result2 = _make_result([match2])

        clusters = [[(0, result1)], [(1, result2)]]

        merged = merge_clusters_by_match(clusters)

        # Should merge into one cluster with 2 entries
        assert len(merged) == 1
        assert len(merged[0]) == 2

    def test_two_clusters_different_matches_unchanged(self):
        match1 = _make_match("perf-1", 0.2)
        result1 = _make_result([match1])
        match2 = _make_match("perf-2", 0.3)
        result2 = _make_result([match2])

        clusters = [[(0, result1)], [(1, result2)]]

        merged = merge_clusters_by_match(clusters)

        assert len(merged) == 2

    def test_clusters_with_no_matches_preserved(self):
        result_no_match = _make_result([])
        match = _make_match("perf-1", 0.2)
        result_with_match = _make_result([match])

        clusters = [[(0, result_no_match)], [(1, result_with_match)]]

        merged = merge_clusters_by_match(clusters)

        # Both should be preserved (no-match cluster is kept separately)
        assert len(merged) == 2

    def test_empty_list_returns_empty(self):
        assert merge_clusters_by_match([]) == []

    def test_three_clusters_two_same_one_different(self):
        match_a1 = _make_match("perf-1", 0.2)
        result_a1 = _make_result([match_a1])
        match_a2 = _make_match("perf-1", 0.25)
        result_a2 = _make_result([match_a2])
        match_b = _make_match("perf-2", 0.3)
        result_b = _make_result([match_b])

        clusters = [[(0, result_a1)], [(1, result_a2)], [(2, result_b)]]

        merged = merge_clusters_by_match(clusters)

        # perf-1 clusters merge, perf-2 stays separate
        assert len(merged) == 2
        sizes = sorted(len(c) for c in merged)
        assert sizes == [1, 2]

    def test_merge_picks_best_score_across_cluster(self):
        # Both results in a cluster have different best matches, but
        # the cluster's best is whichever has lowest combined_score
        match1 = _make_match("perf-1", 0.4)
        match2 = _make_match("perf-2", 0.1)  # Better score
        result1 = _make_result([match1])
        result2 = _make_result([match2])

        clusters = [[(0, result1), (1, result2)]]

        merged = merge_clusters_by_match(clusters)

        assert len(merged) == 1

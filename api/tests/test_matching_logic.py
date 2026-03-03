"""Tests for matching.py pure numpy functions - model health detection and result fusion."""

import sys
from unittest.mock import Mock

# Mock voyager before importing matching
sys.modules['voyager'] = Mock()

import numpy as np
import pytest

from matching import (
    check_model_health,
    fuse_results,
    ModelHealth,
    ModelQueryResult,
    MatchingConfig,
)


class TestCheckModelHealth:
    def test_healthy_normal_distances(self):
        # min_dist >= 0.20, reasonable variance
        distances = np.array([0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60])
        health, reason = check_model_health(distances)
        assert health == ModelHealth.HEALTHY

    def test_degenerate_low_min_dist_tight_clustering(self):
        # min_dist < 0.10, very tight clustering (low variance)
        distances = np.array([0.05, 0.051, 0.052, 0.053, 0.054, 0.055])
        health, reason = check_model_health(distances)
        assert health == ModelHealth.DEGENERATE

    def test_edge_case_moderate_min_very_tight_clustering(self):
        # min_dist 0.15-0.20, very tight clustering (range < 0.03, variance < 0.001)
        distances = np.array([0.16, 0.161, 0.162, 0.163, 0.164, 0.165])
        health, reason = check_model_health(distances)
        assert health == ModelHealth.DEGENERATE

    def test_empty_distances_no_results(self):
        distances = np.array([])
        health, reason = check_model_health(distances)
        assert health == ModelHealth.NO_RESULTS

    def test_large_spread_low_min_still_healthy(self):
        # min_dist < 0.10, but high variance and wide range prevents degenerate flag
        distances = np.array([0.08, 0.20, 0.40, 0.60, 0.80, 1.00, 1.20, 1.40])
        health, reason = check_model_health(distances)
        assert health == ModelHealth.HEALTHY

    def test_barely_healthy_at_threshold(self):
        # min_dist exactly 0.20 -> healthy
        distances = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
        health, reason = check_model_health(distances)
        assert health == ModelHealth.HEALTHY

    def test_custom_config_thresholds(self):
        config = MatchingConfig(max_suspicious_min_distance=0.20)
        distances = np.array([0.15, 0.151, 0.152, 0.153, 0.154])
        health, reason = check_model_health(distances, config)
        assert health == ModelHealth.DEGENERATE


class TestFuseResults:
    def _make_result(self, health, neighbors, distances):
        return ModelQueryResult(
            neighbors=np.array(neighbors, dtype=np.int64),
            distances=np.array(distances, dtype=np.float32),
            health=health,
            health_reason="test",
            min_distance=float(min(distances)) if distances else 0.0,
            max_distance=float(max(distances)) if distances else 0.0,
            distance_variance=float(np.var(distances)) if distances else 0.0,
        )

    def _make_faces_mapping(self, n):
        return [f"stashdb.org:uuid-{i}" for i in range(n)]

    def _make_performers(self, n):
        return {f"stashdb.org:uuid-{i}": {"name": f"Performer {i}"} for i in range(n)}

    def test_both_healthy_weighted_combination(self):
        fn = self._make_result(ModelHealth.HEALTHY, [0, 1, 2], [0.2, 0.3, 0.4])
        af = self._make_result(ModelHealth.HEALTHY, [0, 1, 2], [0.25, 0.35, 0.45])
        faces = self._make_faces_mapping(3)
        performers = self._make_performers(3)

        result = fuse_results(fn, af, faces, performers)

        assert result.fusion_strategy == "weighted"
        assert len(result.matches) > 0
        # Candidate 0 has both: combined = 0.2*0.5 + 0.25*0.5 = 0.225
        match_0 = next(m for m in result.matches if m.face_index == 0)
        assert pytest.approx(match_0.combined_distance, abs=0.01) == 0.225

    def test_only_facenet_healthy(self):
        fn = self._make_result(ModelHealth.HEALTHY, [0, 1], [0.3, 0.4])
        af = self._make_result(ModelHealth.DEGENERATE, [0, 1], [0.05, 0.06])
        faces = self._make_faces_mapping(2)
        performers = self._make_performers(2)

        result = fuse_results(fn, af, faces, performers)

        assert result.fusion_strategy == "facenet_only"

    def test_only_arcface_healthy(self):
        fn = self._make_result(ModelHealth.DEGENERATE, [0, 1], [0.05, 0.06])
        af = self._make_result(ModelHealth.HEALTHY, [0, 1], [0.3, 0.4])
        faces = self._make_faces_mapping(2)
        performers = self._make_performers(2)

        result = fuse_results(fn, af, faces, performers)

        assert result.fusion_strategy == "arcface_only"

    def test_neither_healthy_fallback(self):
        fn = self._make_result(ModelHealth.DEGENERATE, [0], [0.05])
        af = self._make_result(ModelHealth.DEGENERATE, [0], [0.05])
        faces = self._make_faces_mapping(1)
        performers = self._make_performers(1)

        result = fuse_results(fn, af, faces, performers)

        assert result.fusion_strategy == "facenet_fallback"

    def test_filtering_by_max_distance(self):
        config = MatchingConfig(max_distance=0.3)
        fn = self._make_result(ModelHealth.HEALTHY, [0, 1], [0.2, 0.9])
        af = self._make_result(ModelHealth.HEALTHY, [0, 1], [0.2, 0.9])
        faces = self._make_faces_mapping(2)
        performers = self._make_performers(2)

        result = fuse_results(fn, af, faces, performers, config)

        # Only candidate 0 should pass the 0.3 max_distance filter
        assert all(m.combined_distance <= 0.3 for m in result.matches)

    def test_candidate_in_both_models_combined(self):
        fn = self._make_result(ModelHealth.HEALTHY, [0], [0.3])
        af = self._make_result(ModelHealth.HEALTHY, [0], [0.2])
        faces = self._make_faces_mapping(1)
        performers = self._make_performers(1)

        result = fuse_results(fn, af, faces, performers)

        assert result.candidates_in_both == 1
        match = result.matches[0]
        assert match.in_facenet is True
        assert match.in_arcface is True
        assert match.facenet_distance == pytest.approx(0.3)
        assert match.arcface_distance == pytest.approx(0.2)

    def test_null_face_mapping_entries_skipped(self):
        fn = self._make_result(ModelHealth.HEALTHY, [0, 1], [0.3, 0.4])
        af = self._make_result(ModelHealth.HEALTHY, [0, 1], [0.3, 0.4])
        faces = ["stashdb.org:uuid-0", None]  # index 1 is null
        performers = {"stashdb.org:uuid-0": {"name": "Performer 0"}}

        result = fuse_results(fn, af, faces, performers)

        # Only candidate 0 should be returned (index 1 is None)
        assert len(result.matches) == 1
        assert result.matches[0].universal_id == "stashdb.org:uuid-0"

    def test_out_of_bounds_index_skipped(self):
        fn = self._make_result(ModelHealth.HEALTHY, [0, 99], [0.3, 0.4])
        af = self._make_result(ModelHealth.HEALTHY, [0], [0.3])
        faces = self._make_faces_mapping(2)  # Only 2 entries
        performers = self._make_performers(2)

        result = fuse_results(fn, af, faces, performers)

        # Index 99 is out of bounds and should be skipped
        face_indices = [m.face_index for m in result.matches]
        assert 99 not in face_indices

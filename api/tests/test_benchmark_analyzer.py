"""Tests for benchmark analyzer module."""

import pytest
from benchmark.models import (
    ExpectedPerformer,
    TestScene,
    SceneResult,
)
from benchmark.analyzer import Analyzer, MIN_DB_COVERAGE


def make_performer(
    stashdb_id: str = "perf-1",
    name: str = "Test Performer",
    faces_in_db: int = 10,
    has_body_data: bool = False,
    has_tattoo_data: bool = False,
) -> ExpectedPerformer:
    """Helper to create an ExpectedPerformer."""
    return ExpectedPerformer(
        stashdb_id=stashdb_id,
        name=name,
        faces_in_db=faces_in_db,
        has_body_data=has_body_data,
        has_tattoo_data=has_tattoo_data,
    )


def make_scene(
    scene_id: str = "scene-1",
    resolution: str = "1080p",
    db_coverage_tier: str = "well-covered",
    expected_performers: list[ExpectedPerformer] | None = None,
) -> TestScene:
    """Helper to create a TestScene."""
    return TestScene(
        scene_id=scene_id,
        stashdb_id=f"stashdb-{scene_id}",
        title=f"Test Scene {scene_id}",
        resolution=resolution,
        width=1920,
        height=1080,
        duration_sec=600.0,
        expected_performers=expected_performers or [],
        db_coverage_tier=db_coverage_tier,
    )


def make_result(
    scene_id: str = "scene-1",
    true_positives: int = 1,
    false_negatives: int = 0,
    false_positives: int = 0,
) -> SceneResult:
    """Helper to create a SceneResult."""
    return SceneResult(
        scene_id=scene_id,
        params={},
        true_positives=true_positives,
        false_negatives=false_negatives,
        false_positives=false_positives,
        expected_in_top_1=true_positives,
        expected_in_top_3=true_positives,
        correct_match_scores=[0.9] * true_positives,
        incorrect_match_scores=[0.5] * false_positives,
        score_gap=0.1,
        faces_detected=10,
        faces_after_filter=8,
        persons_clustered=true_positives + false_positives,
        elapsed_sec=5.0,
    )


class TestComputeAggregateMetrics:
    """Tests for compute_aggregate_metrics method."""

    def test_compute_aggregate_metrics(self):
        """Test computing aggregate metrics from scene results."""
        results = [
            make_result(scene_id="s1", true_positives=8, false_negatives=2, false_positives=1),
            make_result(scene_id="s2", true_positives=5, false_negatives=5, false_positives=2),
        ]

        analyzer = Analyzer()
        metrics = analyzer.compute_aggregate_metrics(results)

        assert metrics.total_scenes == 2
        assert metrics.total_expected == 20  # (8+2) + (5+5)
        assert metrics.total_true_positives == 13  # 8 + 5
        assert metrics.total_false_positives == 3  # 1 + 2
        assert metrics.total_false_negatives == 7  # 2 + 5
        assert metrics.accuracy == 0.65  # 13 / 20
        assert metrics.precision == 0.8125  # 13 / 16
        assert metrics.recall == 0.65  # 13 / 20

    def test_compute_aggregate_empty(self):
        """Test computing aggregate metrics from empty list."""
        analyzer = Analyzer()
        metrics = analyzer.compute_aggregate_metrics([])

        assert metrics.total_scenes == 0
        assert metrics.total_expected == 0
        assert metrics.total_true_positives == 0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0


class TestAccuracyByResolution:
    """Tests for compute_accuracy_by_resolution method."""

    def test_accuracy_by_resolution(self):
        """Test computing accuracy grouped by resolution tier."""
        scenes = [
            make_scene(scene_id="s1", resolution="1080p"),
            make_scene(scene_id="s2", resolution="1080p"),
            make_scene(scene_id="s3", resolution="720p"),
            make_scene(scene_id="s4", resolution="4k"),
        ]

        results = [
            make_result(scene_id="s1", true_positives=2, false_negatives=0),  # 1080p, 100%
            make_result(scene_id="s2", true_positives=1, false_negatives=1),  # 1080p, 50%
            make_result(scene_id="s3", true_positives=3, false_negatives=1),  # 720p, 75%
            make_result(scene_id="s4", true_positives=4, false_negatives=0),  # 4k, 100%
        ]

        analyzer = Analyzer()
        accuracy_by_res = analyzer.compute_accuracy_by_resolution(scenes, results)

        # 1080p: (2+1) / (2+0+1+1) = 3/4 = 0.75
        assert accuracy_by_res["1080p"] == 0.75
        # 720p: 3 / (3+1) = 0.75
        assert accuracy_by_res["720p"] == 0.75
        # 4k: 4 / (4+0) = 1.0
        assert accuracy_by_res["4k"] == 1.0

    def test_accuracy_by_resolution_empty(self):
        """Test accuracy by resolution with no data."""
        analyzer = Analyzer()
        accuracy_by_res = analyzer.compute_accuracy_by_resolution([], [])

        assert accuracy_by_res == {}

    def test_accuracy_by_resolution_handles_division_by_zero(self):
        """Test accuracy by resolution handles zero expected performers."""
        scenes = [make_scene(scene_id="s1", resolution="480p")]
        results = [make_result(scene_id="s1", true_positives=0, false_negatives=0)]

        analyzer = Analyzer()
        accuracy_by_res = analyzer.compute_accuracy_by_resolution(scenes, results)

        # With no expected performers, accuracy should be 0.0
        assert accuracy_by_res["480p"] == 0.0


class TestAccuracyByCoverage:
    """Tests for compute_accuracy_by_coverage method."""

    def test_accuracy_by_coverage(self):
        """Test computing accuracy grouped by database coverage tier."""
        scenes = [
            make_scene(scene_id="s1", db_coverage_tier="well-covered"),
            make_scene(scene_id="s2", db_coverage_tier="well-covered"),
            make_scene(scene_id="s3", db_coverage_tier="sparse"),
        ]

        results = [
            make_result(scene_id="s1", true_positives=4, false_negatives=0),  # well-covered, 100%
            make_result(scene_id="s2", true_positives=3, false_negatives=1),  # well-covered, 75%
            make_result(scene_id="s3", true_positives=1, false_negatives=3),  # sparse, 25%
        ]

        analyzer = Analyzer()
        accuracy_by_cov = analyzer.compute_accuracy_by_coverage(scenes, results)

        # well-covered: (4+3) / (4+0+3+1) = 7/8 = 0.875
        assert accuracy_by_cov["well-covered"] == 0.875
        # sparse: 1 / (1+3) = 0.25
        assert accuracy_by_cov["sparse"] == 0.25

    def test_accuracy_by_coverage_empty(self):
        """Test accuracy by coverage with no data."""
        analyzer = Analyzer()
        accuracy_by_cov = analyzer.compute_accuracy_by_coverage([], [])

        assert accuracy_by_cov == {}


class TestAccuracyByFaceCount:
    """Tests for compute_accuracy_by_face_count method."""

    def test_accuracy_by_face_count_placeholder(self):
        """Test that accuracy by face count returns placeholder values."""
        analyzer = Analyzer()
        accuracy = analyzer.compute_accuracy_by_face_count([], [])

        # Placeholder implementation returns 0.0 for all buckets
        assert accuracy == {"1-2": 0.0, "3-5": 0.0, "6+": 0.0}


class TestClassifyFailureReason:
    """Tests for classify_failure_reason method."""

    def test_classify_failure_not_detected(self):
        """Test classification when performer was not detected."""
        performer = make_performer(stashdb_id="perf-1", faces_in_db=10)
        analyzer = Analyzer()

        reason = analyzer.classify_failure_reason(
            performer,
            was_detected=False,
            top_matches=[],
        )

        assert reason == "not_detected"

    def test_classify_failure_insufficient_db(self):
        """Test classification when performer has insufficient DB coverage."""
        performer = make_performer(stashdb_id="perf-1", faces_in_db=2)  # Below MIN_DB_COVERAGE
        analyzer = Analyzer()

        reason = analyzer.classify_failure_reason(
            performer,
            was_detected=True,
            top_matches=[
                {"stashdb_id": "other-1", "confidence": 0.9},
                {"stashdb_id": "perf-1", "confidence": 0.7},  # Performer is in matches but ranked lower
            ],
        )

        assert reason == "insufficient_db_coverage"

    def test_classify_failure_similar_performer_won(self):
        """Test classification when similar performer beat expected performer."""
        performer = make_performer(stashdb_id="perf-1", faces_in_db=10)
        analyzer = Analyzer()

        reason = analyzer.classify_failure_reason(
            performer,
            was_detected=True,
            top_matches=[
                {"stashdb_id": "other-1", "confidence": 0.9},
                {"stashdb_id": "perf-1", "confidence": 0.7},  # Expected performer ranked 2nd
            ],
        )

        assert reason == "similar_performer_won"

    def test_classify_failure_low_confidence(self):
        """Test classification defaults to low_confidence."""
        performer = make_performer(stashdb_id="perf-1", faces_in_db=10)
        analyzer = Analyzer()

        reason = analyzer.classify_failure_reason(
            performer,
            was_detected=True,
            top_matches=[
                {"stashdb_id": "other-1", "confidence": 0.9},
                {"stashdb_id": "other-2", "confidence": 0.85},
            ],
        )

        assert reason == "low_confidence"


class TestFindFailurePatterns:
    """Tests for find_failure_patterns method."""

    def test_find_failure_patterns_counts_false_negatives(self):
        """Test that failure patterns count false negatives as not_detected."""
        scenes = [make_scene(scene_id="s1"), make_scene(scene_id="s2")]
        results = [
            make_result(scene_id="s1", true_positives=2, false_negatives=1),
            make_result(scene_id="s2", true_positives=0, false_negatives=3),
        ]

        analyzer = Analyzer()
        patterns = analyzer.find_failure_patterns(scenes, results)

        # Simplified implementation counts false_negatives into "not_detected"
        assert patterns["not_detected"] == 4  # 1 + 3
        assert patterns["insufficient_db_coverage"] == 0
        assert patterns["similar_performer_won"] == 0
        assert patterns["low_confidence"] == 0

    def test_find_failure_patterns_no_failures(self):
        """Test failure patterns when there are no failures."""
        scenes = [make_scene(scene_id="s1")]
        results = [make_result(scene_id="s1", true_positives=5, false_negatives=0)]

        analyzer = Analyzer()
        patterns = analyzer.find_failure_patterns(scenes, results)

        assert patterns["not_detected"] == 0
        assert patterns["insufficient_db_coverage"] == 0
        assert patterns["similar_performer_won"] == 0
        assert patterns["low_confidence"] == 0


class TestCompareParameters:
    """Tests for compare_parameters method."""

    def test_compare_parameters(self):
        """Test comparing two sets of benchmark results."""
        results_a = [
            make_result(scene_id="s1", true_positives=6, false_negatives=4, false_positives=2),
            make_result(scene_id="s2", true_positives=4, false_negatives=6, false_positives=1),
        ]

        results_b = [
            make_result(scene_id="s1", true_positives=8, false_negatives=2, false_positives=1),
            make_result(scene_id="s2", true_positives=7, false_negatives=3, false_positives=2),
        ]

        analyzer = Analyzer()
        comparison = analyzer.compare_parameters(results_a, results_b)

        # Results A: TP=10, FN=10, FP=3 => accuracy=0.5, precision=10/13, recall=0.5
        assert comparison["accuracy_a"] == 0.5
        assert comparison["precision_a"] == pytest.approx(10 / 13, rel=1e-6)
        assert comparison["recall_a"] == 0.5

        # Results B: TP=15, FN=5, FP=3 => accuracy=0.75, precision=15/18, recall=0.75
        assert comparison["accuracy_b"] == 0.75
        assert comparison["precision_b"] == pytest.approx(15 / 18, rel=1e-6)
        assert comparison["recall_b"] == 0.75

        # Improvement = 0.75 - 0.5 = 0.25
        assert comparison["improvement"] == 0.25

    def test_compare_parameters_empty(self):
        """Test comparing empty result sets."""
        analyzer = Analyzer()
        comparison = analyzer.compare_parameters([], [])

        assert comparison["accuracy_a"] == 0.0
        assert comparison["accuracy_b"] == 0.0
        assert comparison["improvement"] == 0.0


class TestBuildPerformerResults:
    """Tests for build_performer_results method."""

    def test_build_performer_results_found(self):
        """Test building results when performer is found."""
        performer = make_performer(stashdb_id="perf-1", name="Test Performer", faces_in_db=10)
        scene = make_scene(scene_id="s1", expected_performers=[performer])

        identification_results = [
            {
                "stashdb_id": "perf-1",
                "name": "Test Performer",
                "confidence": 0.92,
                "distance": 0.35,
            },
        ]

        analyzer = Analyzer()
        results = analyzer.build_performer_results(scene, identification_results)

        assert len(results) == 1
        result = results[0]
        assert result.stashdb_id == "perf-1"
        assert result.was_found is True
        assert result.rank_if_found == 1
        assert result.confidence_if_found == 0.92
        assert result.distance_if_found == 0.35
        assert result.who_beat_them == []

    def test_build_performer_results_not_found(self):
        """Test building results when performer is not found."""
        performer = make_performer(stashdb_id="perf-1", name="Missed Performer", faces_in_db=5)
        scene = make_scene(scene_id="s1", expected_performers=[performer])

        identification_results = [
            {"stashdb_id": "other-1", "name": "Other 1", "confidence": 0.85, "distance": 0.4},
            {"stashdb_id": "other-2", "name": "Other 2", "confidence": 0.75, "distance": 0.45},
            {"stashdb_id": "other-3", "name": "Other 3", "confidence": 0.65, "distance": 0.5},
        ]

        analyzer = Analyzer()
        results = analyzer.build_performer_results(scene, identification_results)

        assert len(results) == 1
        result = results[0]
        assert result.stashdb_id == "perf-1"
        assert result.was_found is False
        assert result.rank_if_found is None
        assert result.confidence_if_found is None
        assert result.distance_if_found is None
        # Top 3 who beat them
        assert len(result.who_beat_them) == 3
        assert result.who_beat_them[0] == ("other-1", 0.85)
        assert result.best_match_for_missed == "other-1"

    def test_build_performer_results_multiple_performers(self):
        """Test building results with multiple expected performers."""
        performer1 = make_performer(stashdb_id="perf-1", name="Found Performer", faces_in_db=10)
        performer2 = make_performer(stashdb_id="perf-2", name="Missed Performer", faces_in_db=5)
        scene = make_scene(scene_id="s1", expected_performers=[performer1, performer2])

        identification_results = [
            {"stashdb_id": "perf-1", "name": "Found Performer", "confidence": 0.9, "distance": 0.3},
            {"stashdb_id": "other-1", "name": "Other", "confidence": 0.7, "distance": 0.5},
        ]

        analyzer = Analyzer()
        results = analyzer.build_performer_results(scene, identification_results)

        assert len(results) == 2
        # First performer was found
        assert results[0].stashdb_id == "perf-1"
        assert results[0].was_found is True
        assert results[0].rank_if_found == 1
        # Second performer was not found
        assert results[1].stashdb_id == "perf-2"
        assert results[1].was_found is False
        assert results[1].best_match_for_missed == "perf-1"


class TestMinDbCoverageConstant:
    """Tests for MIN_DB_COVERAGE constant."""

    def test_min_db_coverage_value(self):
        """Test that MIN_DB_COVERAGE has expected value."""
        assert MIN_DB_COVERAGE == 3

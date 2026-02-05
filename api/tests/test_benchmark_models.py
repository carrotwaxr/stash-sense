"""Tests for benchmark framework data models."""

import json
import pytest
from benchmark.models import (
    ExpectedPerformer,
    TestScene,
    BenchmarkParams,
    SceneResult,
    PerformerResult,
    AggregateMetrics,
    BenchmarkState,
)


class TestExpectedPerformer:
    """Tests for ExpectedPerformer dataclass."""

    def test_create_expected_performer(self):
        """Test creating an ExpectedPerformer with all fields."""
        performer = ExpectedPerformer(
            stashdb_id="abc-123-def",
            name="Test Performer",
            faces_in_db=15,
            has_body_data=True,
            has_tattoo_data=False,
        )

        assert performer.stashdb_id == "abc-123-def"
        assert performer.name == "Test Performer"
        assert performer.faces_in_db == 15
        assert performer.has_body_data is True
        assert performer.has_tattoo_data is False


class TestTestScene:
    """Tests for TestScene dataclass."""

    def test_create_test_scene(self):
        """Test creating a TestScene with expected performers."""
        performer = ExpectedPerformer(
            stashdb_id="perf-1",
            name="Performer One",
            faces_in_db=10,
            has_body_data=True,
            has_tattoo_data=True,
        )

        scene = TestScene(
            scene_id="scene-123",
            stashdb_id="stashdb-scene-456",
            title="Test Scene Title",
            resolution="1080p",
            width=1920,
            height=1080,
            duration_sec=1800.5,
            expected_performers=[performer],
            db_coverage_tier="well-covered",
        )

        assert scene.scene_id == "scene-123"
        assert scene.stashdb_id == "stashdb-scene-456"
        assert scene.title == "Test Scene Title"
        assert scene.resolution == "1080p"
        assert scene.width == 1920
        assert scene.height == 1080
        assert scene.duration_sec == 1800.5
        assert len(scene.expected_performers) == 1
        assert scene.expected_performers[0].name == "Performer One"
        assert scene.db_coverage_tier == "well-covered"

    def test_is_well_covered_true(self):
        """Test is_well_covered returns True for well-covered tier."""
        scene = TestScene(
            scene_id="scene-1",
            stashdb_id="stashdb-1",
            title="Test",
            resolution="720p",
            width=1280,
            height=720,
            duration_sec=600.0,
            expected_performers=[],
            db_coverage_tier="well-covered",
        )
        assert scene.is_well_covered() is True

    def test_is_well_covered_false(self):
        """Test is_well_covered returns False for sparse tier."""
        scene = TestScene(
            scene_id="scene-2",
            stashdb_id="stashdb-2",
            title="Sparse Scene",
            resolution="4k",
            width=3840,
            height=2160,
            duration_sec=900.0,
            expected_performers=[],
            db_coverage_tier="sparse",
        )
        assert scene.is_well_covered() is False


class TestBenchmarkParams:
    """Tests for BenchmarkParams dataclass."""

    def test_default_values(self):
        """Test BenchmarkParams has correct default values."""
        params = BenchmarkParams()

        assert params.matching_mode == "frequency"
        assert params.max_distance == 0.7
        assert params.min_face_size == 40
        assert params.use_multi_signal is True
        assert params.num_frames == 40
        assert params.start_offset_pct == 0.05
        assert params.end_offset_pct == 0.95
        assert params.min_face_confidence == 0.5
        assert params.top_k == 5
        assert params.cluster_threshold == 0.6

    def test_custom_values(self):
        """Test BenchmarkParams with custom values."""
        params = BenchmarkParams(
            matching_mode="weighted",
            max_distance=0.5,
            min_face_size=60,
            use_multi_signal=False,
            num_frames=100,
            start_offset_pct=0.1,
            end_offset_pct=0.9,
            min_face_confidence=0.7,
            top_k=10,
            cluster_threshold=0.5,
        )

        assert params.matching_mode == "weighted"
        assert params.max_distance == 0.5
        assert params.min_face_size == 60
        assert params.use_multi_signal is False
        assert params.num_frames == 100

    def test_to_dict(self):
        """Test to_dict returns dictionary with all fields."""
        params = BenchmarkParams()
        result = params.to_dict()

        assert isinstance(result, dict)
        assert result["matching_mode"] == "frequency"
        assert result["max_distance"] == 0.7
        assert result["min_face_size"] == 40
        assert result["use_multi_signal"] is True
        assert result["num_frames"] == 40
        assert result["start_offset_pct"] == 0.05
        assert result["end_offset_pct"] == 0.95
        assert result["min_face_confidence"] == 0.5
        assert result["top_k"] == 5
        assert result["cluster_threshold"] == 0.6


class TestSceneResult:
    """Tests for SceneResult dataclass."""

    def test_create_scene_result(self):
        """Test creating a SceneResult."""
        result = SceneResult(
            scene_id="scene-1",
            params={"max_distance": 0.7},
            true_positives=3,
            false_negatives=1,
            false_positives=2,
            expected_in_top_1=2,
            expected_in_top_3=3,
            correct_match_scores=[0.85, 0.90, 0.78],
            incorrect_match_scores=[0.65, 0.55],
            score_gap=0.15,
            faces_detected=50,
            faces_after_filter=45,
            persons_clustered=4,
            elapsed_sec=12.5,
        )

        assert result.scene_id == "scene-1"
        assert result.true_positives == 3
        assert result.false_negatives == 1
        assert result.false_positives == 2
        assert result.expected_in_top_1 == 2
        assert result.expected_in_top_3 == 3
        assert len(result.correct_match_scores) == 3
        assert len(result.incorrect_match_scores) == 2
        assert result.score_gap == 0.15
        assert result.elapsed_sec == 12.5

    def test_accuracy_property(self):
        """Test accuracy property calculates correctly."""
        result = SceneResult(
            scene_id="scene-1",
            params={},
            true_positives=8,
            false_negatives=2,
            false_positives=1,
            expected_in_top_1=6,
            expected_in_top_3=8,
            correct_match_scores=[],
            incorrect_match_scores=[],
            score_gap=0.0,
            faces_detected=0,
            faces_after_filter=0,
            persons_clustered=0,
            elapsed_sec=0.0,
        )

        # accuracy = TP / (TP + FN) = 8 / (8 + 2) = 0.8
        assert result.accuracy == 0.8

    def test_accuracy_division_by_zero(self):
        """Test accuracy handles division by zero."""
        result = SceneResult(
            scene_id="scene-empty",
            params={},
            true_positives=0,
            false_negatives=0,
            false_positives=5,
            expected_in_top_1=0,
            expected_in_top_3=0,
            correct_match_scores=[],
            incorrect_match_scores=[],
            score_gap=0.0,
            faces_detected=0,
            faces_after_filter=0,
            persons_clustered=0,
            elapsed_sec=0.0,
        )

        # When TP + FN = 0, accuracy should be 0.0 (not a crash)
        assert result.accuracy == 0.0


class TestPerformerResult:
    """Tests for PerformerResult dataclass."""

    def test_performer_found(self):
        """Test PerformerResult when performer was found."""
        result = PerformerResult(
            stashdb_id="perf-123",
            name="Found Performer",
            faces_in_db=10,
            has_body_data=True,
            has_tattoo_data=False,
            was_found=True,
            rank_if_found=1,
            confidence_if_found=0.92,
            distance_if_found=0.35,
            who_beat_them=[],
            best_match_for_missed=None,
        )

        assert result.was_found is True
        assert result.rank_if_found == 1
        assert result.confidence_if_found == 0.92
        assert result.distance_if_found == 0.35
        assert result.who_beat_them == []
        assert result.best_match_for_missed is None

    def test_performer_not_found(self):
        """Test PerformerResult when performer was missed."""
        result = PerformerResult(
            stashdb_id="perf-456",
            name="Missed Performer",
            faces_in_db=3,
            has_body_data=False,
            has_tattoo_data=True,
            was_found=False,
            rank_if_found=None,
            confidence_if_found=None,
            distance_if_found=None,
            who_beat_them=[("other-perf-1", 0.85), ("other-perf-2", 0.75)],
            best_match_for_missed="other-perf-1",
        )

        assert result.was_found is False
        assert result.rank_if_found is None
        assert result.confidence_if_found is None
        assert result.distance_if_found is None
        assert len(result.who_beat_them) == 2
        assert result.who_beat_them[0] == ("other-perf-1", 0.85)
        assert result.best_match_for_missed == "other-perf-1"


class TestAggregateMetrics:
    """Tests for AggregateMetrics dataclass."""

    def test_create_aggregate_metrics(self):
        """Test creating AggregateMetrics."""
        metrics = AggregateMetrics(
            total_scenes=50,
            total_expected=100,
            total_true_positives=85,
            total_false_positives=10,
            total_false_negatives=15,
            accuracy=0.85,
            precision=0.895,
            recall=0.85,
            accuracy_by_resolution={"1080p": 0.9, "720p": 0.8},
            accuracy_by_coverage={"well-covered": 0.92, "sparse": 0.75},
            accuracy_by_face_count={"high": 0.95, "low": 0.70},
        )

        assert metrics.total_scenes == 50
        assert metrics.total_expected == 100
        assert metrics.total_true_positives == 85
        assert metrics.accuracy == 0.85
        assert metrics.precision == 0.895
        assert metrics.recall == 0.85
        assert metrics.accuracy_by_resolution["1080p"] == 0.9
        assert metrics.accuracy_by_coverage["well-covered"] == 0.92

    def test_default_dicts(self):
        """Test that dict fields default to empty dicts."""
        metrics = AggregateMetrics(
            total_scenes=0,
            total_expected=0,
            total_true_positives=0,
            total_false_positives=0,
            total_false_negatives=0,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
        )

        assert metrics.accuracy_by_resolution == {}
        assert metrics.accuracy_by_coverage == {}
        assert metrics.accuracy_by_face_count == {}

    def test_from_results(self):
        """Test from_results classmethod computes metrics correctly."""
        results = [
            SceneResult(
                scene_id="s1",
                params={},
                true_positives=8,
                false_negatives=2,
                false_positives=1,
                expected_in_top_1=6,
                expected_in_top_3=8,
                correct_match_scores=[],
                incorrect_match_scores=[],
                score_gap=0.0,
                faces_detected=0,
                faces_after_filter=0,
                persons_clustered=0,
                elapsed_sec=0.0,
            ),
            SceneResult(
                scene_id="s2",
                params={},
                true_positives=5,
                false_negatives=5,
                false_positives=2,
                expected_in_top_1=3,
                expected_in_top_3=5,
                correct_match_scores=[],
                incorrect_match_scores=[],
                score_gap=0.0,
                faces_detected=0,
                faces_after_filter=0,
                persons_clustered=0,
                elapsed_sec=0.0,
            ),
        ]

        metrics = AggregateMetrics.from_results(results)

        assert metrics.total_scenes == 2
        # total_expected = (8+2) + (5+5) = 20
        assert metrics.total_expected == 20
        # total_true_positives = 8 + 5 = 13
        assert metrics.total_true_positives == 13
        # total_false_positives = 1 + 2 = 3
        assert metrics.total_false_positives == 3
        # total_false_negatives = 2 + 5 = 7
        assert metrics.total_false_negatives == 7
        # accuracy = TP / (TP + FN) = 13 / 20 = 0.65
        assert metrics.accuracy == 0.65
        # precision = TP / (TP + FP) = 13 / 16 = 0.8125
        assert metrics.precision == 0.8125
        # recall = TP / (TP + FN) = 13 / 20 = 0.65
        assert metrics.recall == 0.65

    def test_from_results_empty(self):
        """Test from_results handles empty list."""
        metrics = AggregateMetrics.from_results([])

        assert metrics.total_scenes == 0
        assert metrics.total_expected == 0
        assert metrics.total_true_positives == 0
        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    def test_from_results_zero_denominators(self):
        """Test from_results handles zero denominators gracefully."""
        results = [
            SceneResult(
                scene_id="s1",
                params={},
                true_positives=0,
                false_negatives=0,
                false_positives=0,
                expected_in_top_1=0,
                expected_in_top_3=0,
                correct_match_scores=[],
                incorrect_match_scores=[],
                score_gap=0.0,
                faces_detected=0,
                faces_after_filter=0,
                persons_clustered=0,
                elapsed_sec=0.0,
            ),
        ]

        metrics = AggregateMetrics.from_results(results)

        assert metrics.accuracy == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0


class TestBenchmarkState:
    """Tests for BenchmarkState dataclass."""

    def test_create_benchmark_state(self):
        """Test creating a BenchmarkState."""
        scene = TestScene(
            scene_id="scene-1",
            stashdb_id="stashdb-1",
            title="Test Scene",
            resolution="1080p",
            width=1920,
            height=1080,
            duration_sec=600.0,
            expected_performers=[],
            db_coverage_tier="well-covered",
        )

        result = SceneResult(
            scene_id="scene-1",
            params={"max_distance": 0.7},
            true_positives=5,
            false_negatives=1,
            false_positives=0,
            expected_in_top_1=4,
            expected_in_top_3=5,
            correct_match_scores=[0.9],
            incorrect_match_scores=[],
            score_gap=0.1,
            faces_detected=10,
            faces_after_filter=10,
            persons_clustered=2,
            elapsed_sec=5.0,
        )

        state = BenchmarkState(
            round_num=2,
            scenes=[scene],
            results_by_round={1: [result]},
            current_best_params={"max_distance": 0.7},
            current_best_accuracy=0.833,
            parameters_eliminated=["min_face_size"],
        )

        assert state.round_num == 2
        assert len(state.scenes) == 1
        assert len(state.results_by_round[1]) == 1
        assert state.current_best_accuracy == 0.833
        assert "min_face_size" in state.parameters_eliminated

    def test_to_json(self):
        """Test to_json serializes state correctly."""
        performer = ExpectedPerformer(
            stashdb_id="perf-1",
            name="Test Performer",
            faces_in_db=5,
            has_body_data=True,
            has_tattoo_data=False,
        )

        scene = TestScene(
            scene_id="scene-1",
            stashdb_id="stashdb-1",
            title="Test Scene",
            resolution="720p",
            width=1280,
            height=720,
            duration_sec=300.0,
            expected_performers=[performer],
            db_coverage_tier="sparse",
        )

        result = SceneResult(
            scene_id="scene-1",
            params={"max_distance": 0.6},
            true_positives=1,
            false_negatives=0,
            false_positives=0,
            expected_in_top_1=1,
            expected_in_top_3=1,
            correct_match_scores=[0.95],
            incorrect_match_scores=[],
            score_gap=0.2,
            faces_detected=5,
            faces_after_filter=5,
            persons_clustered=1,
            elapsed_sec=2.5,
        )

        state = BenchmarkState(
            round_num=1,
            scenes=[scene],
            results_by_round={1: [result]},
            current_best_params={"max_distance": 0.6},
            current_best_accuracy=1.0,
            parameters_eliminated=[],
        )

        json_str = state.to_json()

        # Verify it's valid JSON
        data = json.loads(json_str)

        assert data["round_num"] == 1
        assert len(data["scenes"]) == 1
        assert data["scenes"][0]["scene_id"] == "scene-1"
        assert data["scenes"][0]["expected_performers"][0]["name"] == "Test Performer"
        assert "1" in data["results_by_round"]
        assert data["current_best_accuracy"] == 1.0

    def test_from_json(self):
        """Test from_json deserializes state correctly."""
        json_str = json.dumps({
            "round_num": 3,
            "scenes": [
                {
                    "scene_id": "s-100",
                    "stashdb_id": "stashdb-100",
                    "title": "Deserialized Scene",
                    "resolution": "4k",
                    "width": 3840,
                    "height": 2160,
                    "duration_sec": 1200.0,
                    "expected_performers": [
                        {
                            "stashdb_id": "p-200",
                            "name": "Deserialized Performer",
                            "faces_in_db": 20,
                            "has_body_data": False,
                            "has_tattoo_data": True,
                        }
                    ],
                    "db_coverage_tier": "well-covered",
                }
            ],
            "results_by_round": {
                "2": [
                    {
                        "scene_id": "s-100",
                        "params": {"max_distance": 0.65},
                        "true_positives": 1,
                        "false_negatives": 0,
                        "false_positives": 0,
                        "expected_in_top_1": 1,
                        "expected_in_top_3": 1,
                        "correct_match_scores": [0.88],
                        "incorrect_match_scores": [],
                        "score_gap": 0.15,
                        "faces_detected": 8,
                        "faces_after_filter": 8,
                        "persons_clustered": 1,
                        "elapsed_sec": 3.0,
                    }
                ]
            },
            "current_best_params": {"max_distance": 0.65},
            "current_best_accuracy": 1.0,
            "parameters_eliminated": ["min_face_size", "num_frames"],
        })

        state = BenchmarkState.from_json(json_str)

        assert state.round_num == 3
        assert len(state.scenes) == 1
        assert state.scenes[0].scene_id == "s-100"
        assert state.scenes[0].resolution == "4k"
        assert state.scenes[0].expected_performers[0].name == "Deserialized Performer"
        assert state.scenes[0].expected_performers[0].has_tattoo_data is True
        assert state.scenes[0].is_well_covered() is True

        assert 2 in state.results_by_round
        assert len(state.results_by_round[2]) == 1
        assert state.results_by_round[2][0].true_positives == 1

        assert state.current_best_params == {"max_distance": 0.65}
        assert state.current_best_accuracy == 1.0
        assert len(state.parameters_eliminated) == 2

    def test_json_roundtrip(self):
        """Test serialization and deserialization roundtrip."""
        performer = ExpectedPerformer(
            stashdb_id="perf-roundtrip",
            name="Roundtrip Performer",
            faces_in_db=12,
            has_body_data=True,
            has_tattoo_data=True,
        )

        scene = TestScene(
            scene_id="scene-roundtrip",
            stashdb_id="stashdb-roundtrip",
            title="Roundtrip Scene",
            resolution="1080p",
            width=1920,
            height=1080,
            duration_sec=900.0,
            expected_performers=[performer],
            db_coverage_tier="well-covered",
        )

        result = SceneResult(
            scene_id="scene-roundtrip",
            params={"max_distance": 0.7, "num_frames": 40},
            true_positives=1,
            false_negatives=0,
            false_positives=0,
            expected_in_top_1=1,
            expected_in_top_3=1,
            correct_match_scores=[0.92],
            incorrect_match_scores=[],
            score_gap=0.25,
            faces_detected=15,
            faces_after_filter=14,
            persons_clustered=1,
            elapsed_sec=4.5,
        )

        original = BenchmarkState(
            round_num=5,
            scenes=[scene],
            results_by_round={4: [result]},
            current_best_params={"max_distance": 0.7},
            current_best_accuracy=0.95,
            parameters_eliminated=["cluster_threshold"],
        )

        # Roundtrip
        json_str = original.to_json()
        restored = BenchmarkState.from_json(json_str)

        assert restored.round_num == original.round_num
        assert len(restored.scenes) == len(original.scenes)
        assert restored.scenes[0].scene_id == original.scenes[0].scene_id
        assert restored.scenes[0].expected_performers[0].stashdb_id == \
            original.scenes[0].expected_performers[0].stashdb_id
        assert 4 in restored.results_by_round
        assert restored.results_by_round[4][0].true_positives == 1
        assert restored.current_best_params == original.current_best_params
        assert restored.current_best_accuracy == original.current_best_accuracy
        assert restored.parameters_eliminated == original.parameters_eliminated

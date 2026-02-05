"""Tests for the benchmark Reporter class."""

import csv
import json
import os
import tempfile
from pathlib import Path

import pytest

from benchmark.models import (
    AggregateMetrics,
    BenchmarkParams,
    BenchmarkState,
    ExpectedPerformer,
    PerformerResult,
    SceneResult,
    TestScene,
)
from benchmark.reporter import Reporter


@pytest.fixture
def sample_metrics():
    """Create sample aggregate metrics for testing."""
    return AggregateMetrics(
        total_scenes=10,
        total_expected=25,
        total_true_positives=20,
        total_false_positives=3,
        total_false_negatives=5,
        accuracy=0.8,
        precision=0.87,
        recall=0.8,
        accuracy_by_resolution={"480p": 0.75, "720p": 0.85, "1080p": 0.9},
        accuracy_by_coverage={"well-covered": 0.92, "sparse": 0.65},
        accuracy_by_face_count={"low": 0.7, "high": 0.9},
    )


@pytest.fixture
def sample_scene_results():
    """Create sample scene results for testing."""
    return [
        SceneResult(
            scene_id="scene-001",
            params={"max_distance": 0.7},
            true_positives=2,
            false_negatives=1,
            false_positives=0,
            expected_in_top_1=2,
            expected_in_top_3=2,
            correct_match_scores=[0.85, 0.90],
            incorrect_match_scores=[],
            score_gap=0.15,
            faces_detected=10,
            faces_after_filter=8,
            persons_clustered=3,
            elapsed_sec=2.5,
        ),
        SceneResult(
            scene_id="scene-002",
            params={"max_distance": 0.7},
            true_positives=1,
            false_negatives=0,
            false_positives=1,
            expected_in_top_1=1,
            expected_in_top_3=1,
            correct_match_scores=[0.95],
            incorrect_match_scores=[0.72],
            score_gap=0.23,
            faces_detected=5,
            faces_after_filter=4,
            persons_clustered=2,
            elapsed_sec=1.8,
        ),
    ]


@pytest.fixture
def sample_performer_results():
    """Create sample performer results for testing."""
    return [
        PerformerResult(
            stashdb_id="perf-001",
            name="Test Performer 1",
            faces_in_db=15,
            has_body_data=True,
            has_tattoo_data=False,
            was_found=True,
            rank_if_found=1,
            confidence_if_found=0.956,
            distance_if_found=0.312,
            who_beat_them=[],
            best_match_for_missed=None,
        ),
        PerformerResult(
            stashdb_id="perf-002",
            name="Test Performer 2",
            faces_in_db=3,
            has_body_data=False,
            has_tattoo_data=True,
            was_found=False,
            rank_if_found=None,
            confidence_if_found=None,
            distance_if_found=None,
            who_beat_them=[("perf-003", 0.85)],
            best_match_for_missed="perf-003",
        ),
    ]


@pytest.fixture
def sample_benchmark_state(sample_scene_results):
    """Create sample benchmark state for testing."""
    performers = [
        ExpectedPerformer(
            stashdb_id="perf-001",
            name="Test Performer",
            faces_in_db=10,
            has_body_data=True,
            has_tattoo_data=False,
        )
    ]
    scenes = [
        TestScene(
            scene_id="scene-001",
            stashdb_id="stash-001",
            title="Test Scene",
            resolution="1080p",
            width=1920,
            height=1080,
            duration_sec=600.0,
            expected_performers=performers,
            db_coverage_tier="well-covered",
        )
    ]
    return BenchmarkState(
        round_num=3,
        scenes=scenes,
        results_by_round={1: sample_scene_results, 2: sample_scene_results},
        current_best_params={"max_distance": 0.65, "min_face_size": 40},
        current_best_accuracy=0.85,
        parameters_eliminated=["max_distance=0.8", "min_face_size=30"],
    )


class TestGenerateSummary:
    """Tests for generate_summary method."""

    def test_generate_summary(self, sample_metrics):
        """Test that summary contains expected strings."""
        reporter = Reporter()
        summary = reporter.generate_summary(sample_metrics)

        # Check header
        assert "=== Benchmark Summary ===" in summary

        # Check scene count
        assert "Scenes tested: 10" in summary

        # Check expected performers
        assert "Total expected performers: 25" in summary

        # Check overall accuracy
        assert "Overall accuracy: 80.0% (20/25)" in summary

        # Check precision and recall
        assert "Precision: 87.0%" in summary
        assert "Recall: 80.0%" in summary

        # Check resolution breakdown
        assert "By Resolution:" in summary
        assert "480p: 75.0%" in summary
        assert "720p: 85.0%" in summary
        assert "1080p: 90.0%" in summary

        # Check coverage breakdown
        assert "By DB Coverage:" in summary
        assert "well-covered: 92.0%" in summary
        assert "sparse: 65.0%" in summary


class TestFormatProgress:
    """Tests for format_progress method."""

    def test_format_progress_without_eta(self):
        """Test progress bar without ETA."""
        reporter = Reporter()
        progress = reporter.format_progress(current=5, total=10, accuracy=0.8)

        # Check format
        assert "[##########..........]" in progress
        assert "5/10 scenes" in progress
        assert "80.0% accuracy" in progress
        assert "ETA" not in progress

    def test_format_progress_with_eta(self):
        """Test progress bar with ETA."""
        reporter = Reporter()
        progress = reporter.format_progress(
            current=3, total=10, accuracy=0.75, eta_sec=180.0
        )

        # Check format
        assert "[######..............]" in progress
        assert "3/10 scenes" in progress
        assert "75.0% accuracy" in progress
        assert "ETA 3m" in progress

    def test_format_progress_full(self):
        """Test progress bar at 100%."""
        reporter = Reporter()
        progress = reporter.format_progress(current=10, total=10, accuracy=0.95)

        assert "[####################]" in progress
        assert "10/10 scenes" in progress
        assert "95.0% accuracy" in progress

    def test_format_progress_empty(self):
        """Test progress bar at 0%."""
        reporter = Reporter()
        progress = reporter.format_progress(current=0, total=10, accuracy=0.0)

        assert "[....................]" in progress
        assert "0/10 scenes" in progress
        assert "0.0% accuracy" in progress


class TestExportSceneResultsCsv:
    """Tests for export_scene_results_csv method."""

    def test_export_scene_results_csv(self, sample_scene_results):
        """Test writing scene results to CSV and reading back."""
        reporter = Reporter()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            reporter.export_scene_results_csv(sample_scene_results, tmp_path)

            # Read back and verify
            with open(tmp_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2

            # Check first row
            row1 = rows[0]
            assert row1["scene_id"] == "scene-001"
            assert row1["true_positives"] == "2"
            assert row1["false_negatives"] == "1"
            assert row1["false_positives"] == "0"
            assert row1["expected_in_top_1"] == "2"
            assert row1["expected_in_top_3"] == "2"
            assert row1["score_gap"] == "0.15"
            assert row1["faces_detected"] == "10"
            assert row1["faces_after_filter"] == "8"
            assert row1["persons_clustered"] == "3"
            assert row1["elapsed_sec"] == "2.5"
            # Accuracy is 2/(2+1) = 0.667
            assert "0.667" in row1["accuracy"]

        finally:
            os.unlink(tmp_path)


class TestExportPerformerResultsCsv:
    """Tests for export_performer_results_csv method."""

    def test_export_performer_results_csv(self, sample_performer_results):
        """Test writing performer results to CSV and reading back."""
        reporter = Reporter()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            reporter.export_performer_results_csv(sample_performer_results, tmp_path)

            # Read back and verify
            with open(tmp_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2

            # Check found performer
            row1 = rows[0]
            assert row1["stashdb_id"] == "perf-001"
            assert row1["name"] == "Test Performer 1"
            assert row1["faces_in_db"] == "15"
            assert row1["has_body_data"] == "True"
            assert row1["has_tattoo_data"] == "False"
            assert row1["was_found"] == "True"
            assert row1["rank_if_found"] == "1"
            assert row1["confidence_if_found"] == "0.956"
            assert row1["distance_if_found"] == "0.312"
            assert row1["best_match_for_missed"] == ""

            # Check missed performer (has None values)
            row2 = rows[1]
            assert row2["stashdb_id"] == "perf-002"
            assert row2["was_found"] == "False"
            assert row2["rank_if_found"] == ""
            assert row2["confidence_if_found"] == ""
            assert row2["distance_if_found"] == ""
            assert row2["best_match_for_missed"] == "perf-003"

        finally:
            os.unlink(tmp_path)


class TestExportParameterComparisonCsv:
    """Tests for export_parameter_comparison_csv method."""

    def test_export_parameter_comparison_csv(self):
        """Test writing parameter comparisons to CSV."""
        reporter = Reporter()
        comparisons = [
            {
                "param_name": "max_distance",
                "value_a": "0.6",
                "value_b": "0.7",
                "accuracy_a": 0.85,
                "accuracy_b": 0.80,
                "improvement": 0.05,
            },
            {
                "param_name": "min_face_size",
                "value_a": "30",
                "value_b": "40",
                "accuracy_a": 0.78,
                "accuracy_b": 0.82,
                "improvement": -0.04,
            },
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            reporter.export_parameter_comparison_csv(comparisons, tmp_path)

            with open(tmp_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            assert len(rows) == 2
            assert rows[0]["param_name"] == "max_distance"
            assert rows[0]["value_a"] == "0.6"
            assert rows[0]["value_b"] == "0.7"
            assert rows[1]["improvement"] == "-0.04"

        finally:
            os.unlink(tmp_path)


class TestGenerateRecommendation:
    """Tests for generate_recommendation method."""

    def test_generate_recommendation_json(self):
        """Test recommendation dict structure."""
        reporter = Reporter()
        best_params = BenchmarkParams(
            matching_mode="frequency",
            max_distance=0.65,
            min_face_size=40,
        )

        result = reporter.generate_recommendation(
            best_params=best_params,
            accuracy=0.92,
            baseline_accuracy=0.85,
            total_scenes=50,
            total_performers=120,
            notes=["Multi-signal matching improved sparse coverage"],
        )

        assert "recommended_config" in result
        assert result["recommended_config"]["max_distance"] == 0.65
        assert result["recommended_config"]["min_face_size"] == 40

        assert result["accuracy"] == 0.92
        assert result["baseline_accuracy"] == 0.85
        assert result["improvement"] == pytest.approx(0.07)

        assert "validated_on" in result
        assert result["validated_on"]["scenes"] == 50
        assert result["validated_on"]["performers"] == 120

        assert "notes" in result
        assert len(result["notes"]) == 1


class TestGenerateFinalReport:
    """Tests for generate_final_report method."""

    def test_generate_final_report_markdown(self, sample_metrics, sample_benchmark_state):
        """Test final report markdown structure."""
        reporter = Reporter()
        report = reporter.generate_final_report(sample_benchmark_state, sample_metrics)

        # Check header
        assert "# Benchmark Final Report" in report

        # Check summary section
        assert "## Summary" in report
        assert "Rounds completed: 3" in report or "round" in report.lower()
        assert "Scenes tested: 10" in report
        assert "Total performers: 25" in report
        assert "80.0%" in report  # accuracy

        # Check best parameters section
        assert "## Best Parameters" in report
        assert "```json" in report
        assert "max_distance" in report
        assert "0.65" in report

        # Check eliminated parameters
        assert "## Parameters Eliminated" in report
        assert "max_distance=0.8" in report
        assert "min_face_size=30" in report


class TestCheckpointSaveLoad:
    """Tests for checkpoint save/load methods."""

    def test_save_checkpoint(self, sample_benchmark_state):
        """Test saving checkpoint to file."""
        reporter = Reporter()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name

        try:
            reporter.save_checkpoint(sample_benchmark_state, tmp_path)

            with open(tmp_path, "r") as f:
                data = json.load(f)

            assert data["round_num"] == 3
            assert len(data["scenes"]) == 1
            assert data["current_best_accuracy"] == 0.85

        finally:
            os.unlink(tmp_path)

    def test_load_checkpoint(self, sample_benchmark_state):
        """Test loading checkpoint from file."""
        reporter = Reporter()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            tmp_path = tmp.name
            tmp.write(sample_benchmark_state.to_json())

        try:
            loaded_state = reporter.load_checkpoint(tmp_path)

            assert loaded_state is not None
            assert loaded_state.round_num == 3
            assert len(loaded_state.scenes) == 1
            assert loaded_state.current_best_accuracy == 0.85
            assert len(loaded_state.parameters_eliminated) == 2

        finally:
            os.unlink(tmp_path)

    def test_load_checkpoint_file_not_found(self):
        """Test loading checkpoint returns None for missing file."""
        reporter = Reporter()

        result = reporter.load_checkpoint("/nonexistent/path/checkpoint.json")

        assert result is None

"""Tests for benchmark runner."""

import pytest
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import os
import tempfile
import json

from api.benchmark.models import (
    ExpectedPerformer,
    TestScene,
    BenchmarkParams,
    SceneResult,
    BenchmarkState,
    AggregateMetrics,
)
from api.benchmark.runner import BenchmarkRunner


# Fixtures for common test data
@pytest.fixture
def sample_performers():
    """Create sample expected performers."""
    return [
        ExpectedPerformer(
            stashdb_id="perf-1-uuid",
            name="Performer One",
            faces_in_db=10,
            has_body_data=True,
            has_tattoo_data=False,
        ),
        ExpectedPerformer(
            stashdb_id="perf-2-uuid",
            name="Performer Two",
            faces_in_db=15,
            has_body_data=True,
            has_tattoo_data=True,
        ),
    ]


@pytest.fixture
def sample_scenes(sample_performers):
    """Create sample test scenes."""
    return [
        TestScene(
            scene_id=f"scene-{i}",
            stashdb_id=f"stashdb-scene-{i}",
            title=f"Test Scene {i}",
            resolution="1080p",
            width=1920,
            height=1080,
            duration_sec=1800.0,
            expected_performers=sample_performers,
            db_coverage_tier="well-covered",
        )
        for i in range(5)
    ]


@pytest.fixture
def sample_results():
    """Create sample scene results."""
    return [
        SceneResult(
            scene_id=f"scene-{i}",
            params={"max_distance": 0.7},
            true_positives=2,
            false_negatives=0,
            false_positives=0,
            expected_in_top_1=2,
            expected_in_top_3=2,
            correct_match_scores=[0.3, 0.4],
            incorrect_match_scores=[],
            score_gap=0.0,
            faces_detected=50,
            faces_after_filter=45,
            persons_clustered=2,
            elapsed_sec=1.0,
        )
        for i in range(5)
    ]


@pytest.fixture
def mock_scene_selector():
    """Create a mock scene selector."""
    selector = Mock()
    selector.select_scenes = AsyncMock()
    selector.sample_stratified = Mock()
    return selector


@pytest.fixture
def mock_executor():
    """Create a mock test executor."""
    executor = Mock()
    executor.run_batch = AsyncMock()
    return executor


@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer."""
    analyzer = Mock()
    analyzer.compute_aggregate_metrics = Mock()
    analyzer.compute_accuracy_by_resolution = Mock(return_value={})
    analyzer.compute_accuracy_by_coverage = Mock(return_value={})
    return analyzer


@pytest.fixture
def mock_reporter():
    """Create a mock reporter."""
    reporter = Mock()
    reporter.save_checkpoint = Mock()
    reporter.load_checkpoint = Mock()
    reporter.export_scene_results_csv = Mock()
    reporter.generate_summary = Mock(return_value="Summary")
    reporter.generate_final_report = Mock(return_value="# Final Report")
    reporter.generate_recommendation = Mock(return_value={"recommended_config": {}})
    return reporter


@pytest.fixture
def runner(mock_scene_selector, mock_executor, mock_analyzer, mock_reporter, tmp_path):
    """Create a BenchmarkRunner with mocked dependencies."""
    return BenchmarkRunner(
        scene_selector=mock_scene_selector,
        executor=mock_executor,
        analyzer=mock_analyzer,
        reporter=mock_reporter,
        output_dir=str(tmp_path),
    )


class TestBenchmarkRunnerInit:
    """Tests for BenchmarkRunner initialization."""

    def test_init_sets_dependencies(
        self, mock_scene_selector, mock_executor, mock_analyzer, mock_reporter, tmp_path
    ):
        """Test that constructor sets all dependencies correctly."""
        runner = BenchmarkRunner(
            scene_selector=mock_scene_selector,
            executor=mock_executor,
            analyzer=mock_analyzer,
            reporter=mock_reporter,
            output_dir=str(tmp_path),
        )

        assert runner.scene_selector == mock_scene_selector
        assert runner.executor == mock_executor
        assert runner.analyzer == mock_analyzer
        assert runner.reporter == mock_reporter
        assert runner.output_dir == str(tmp_path)

    def test_init_sets_default_best_params(
        self, mock_scene_selector, mock_executor, mock_analyzer, mock_reporter, tmp_path
    ):
        """Test that constructor sets default best parameters."""
        runner = BenchmarkRunner(
            scene_selector=mock_scene_selector,
            executor=mock_executor,
            analyzer=mock_analyzer,
            reporter=mock_reporter,
            output_dir=str(tmp_path),
        )

        assert runner._best_distance == 0.7
        assert runner._best_face_size == 40
        assert runner._best_multi_signal is True

    def test_constants_defined(self, runner):
        """Test that class constants are defined correctly."""
        assert BenchmarkRunner.MAX_ROUNDS == 6
        assert BenchmarkRunner.IMPROVEMENT_THRESHOLD == 0.01
        assert BenchmarkRunner.SAMPLE_FRACTION == 0.3


class TestRunBaseline:
    """Tests for _run_baseline method."""

    @pytest.mark.asyncio
    async def test_run_baseline_calls_executor_run_batch(
        self, runner, mock_executor, sample_scenes, sample_results
    ):
        """Test that _run_baseline calls executor.run_batch with correct args."""
        mock_executor.run_batch.return_value = sample_results
        params = BenchmarkParams()

        results = await runner._run_baseline(sample_scenes, params)

        mock_executor.run_batch.assert_called_once_with(sample_scenes, params)
        assert results == sample_results

    @pytest.mark.asyncio
    async def test_run_baseline_returns_results(
        self, runner, mock_executor, sample_scenes, sample_results
    ):
        """Test that _run_baseline returns results from executor."""
        mock_executor.run_batch.return_value = sample_results
        params = BenchmarkParams(max_distance=0.6)

        results = await runner._run_baseline(sample_scenes, params)

        assert len(results) == 5
        assert results[0].scene_id == "scene-0"


class TestBuildParameterGrid:
    """Tests for _build_parameter_grid method."""

    def test_build_parameter_grid_round_1_has_baseline_and_distance_variations(
        self, runner
    ):
        """Test that round 1 parameter grid has baseline + distance variations."""
        grid = runner._build_parameter_grid(round_num=1)

        # Should have baseline (0.7) + variations (0.5, 0.6, 0.8)
        assert len(grid) >= 4

        distances = [p.max_distance for p in grid]
        assert 0.7 in distances  # baseline
        assert 0.5 in distances  # variation
        assert 0.6 in distances  # variation
        assert 0.8 in distances  # variation

    def test_build_parameter_grid_round_2_has_finer_tuning(self, runner):
        """Test that round 2 parameter grid has finer distance tuning and face size."""
        # Set best distance from round 1
        runner._best_distance = 0.6

        grid = runner._build_parameter_grid(round_num=2)

        # Should include finer distance tuning around 0.6 + face size variations
        distances = [p.max_distance for p in grid]
        face_sizes = [p.min_face_size for p in grid]

        # Should have variations around best distance
        assert any(0.55 <= d <= 0.65 for d in distances)

        # Should include face size variations
        assert 60 in face_sizes or 80 in face_sizes

    def test_build_parameter_grid_round_3_has_multi_signal_comparison(self, runner):
        """Test that round 3 parameter grid compares multi-signal True vs False."""
        grid = runner._build_parameter_grid(round_num=3)

        multi_signal_values = [p.use_multi_signal for p in grid]

        assert True in multi_signal_values
        assert False in multi_signal_values

    def test_build_parameter_grid_round_4_combines_best_params(self, runner):
        """Test that round 4+ uses combination of best params."""
        runner._best_distance = 0.55
        runner._best_face_size = 60
        runner._best_multi_signal = True

        grid = runner._build_parameter_grid(round_num=4)

        # Should have at least one config with best params
        assert len(grid) >= 1


class TestShouldContinue:
    """Tests for _should_continue method."""

    def test_should_continue_improvement_above_threshold_returns_true(self, runner):
        """Test that improvement above threshold returns True."""
        # 5% improvement (above 1% threshold)
        result = runner._should_continue(
            current_accuracy=0.85,
            previous_accuracy=0.80,
            round_num=2,
        )

        assert result is True

    def test_should_continue_improvement_below_threshold_returns_false(self, runner):
        """Test that improvement below threshold returns False."""
        # 0.5% improvement (below 1% threshold)
        result = runner._should_continue(
            current_accuracy=0.805,
            previous_accuracy=0.80,
            round_num=2,
        )

        assert result is False

    def test_should_continue_max_rounds_returns_false(self, runner):
        """Test that reaching MAX_ROUNDS returns False."""
        result = runner._should_continue(
            current_accuracy=0.90,
            previous_accuracy=0.80,
            round_num=BenchmarkRunner.MAX_ROUNDS,
        )

        assert result is False

    def test_should_continue_exactly_at_threshold(self, runner):
        """Test behavior when improvement is exactly at threshold."""
        # Exactly 1% improvement
        result = runner._should_continue(
            current_accuracy=0.81,
            previous_accuracy=0.80,
            round_num=2,
        )

        # Improvement >= threshold should return True
        assert result is True

    def test_should_continue_negative_improvement(self, runner):
        """Test that negative improvement (regression) returns False."""
        result = runner._should_continue(
            current_accuracy=0.75,
            previous_accuracy=0.80,
            round_num=2,
        )

        assert result is False


class TestUpdateBestParams:
    """Tests for _update_best_params method."""

    def test_update_best_params_updates_distance(self, runner):
        """Test that _update_best_params updates _best_distance."""
        params = BenchmarkParams(max_distance=0.55)

        runner._update_best_params(params)

        assert runner._best_distance == 0.55

    def test_update_best_params_updates_face_size(self, runner):
        """Test that _update_best_params updates _best_face_size."""
        params = BenchmarkParams(min_face_size=60)

        runner._update_best_params(params)

        assert runner._best_face_size == 60

    def test_update_best_params_updates_multi_signal(self, runner):
        """Test that _update_best_params updates _best_multi_signal."""
        params = BenchmarkParams(use_multi_signal=False)

        runner._update_best_params(params)

        assert runner._best_multi_signal is False

    def test_update_best_params_updates_all_fields(self, runner):
        """Test that _update_best_params updates all tracked parameters."""
        params = BenchmarkParams(
            max_distance=0.5,
            min_face_size=80,
            use_multi_signal=False,
        )

        runner._update_best_params(params)

        assert runner._best_distance == 0.5
        assert runner._best_face_size == 80
        assert runner._best_multi_signal is False


class TestRunRound:
    """Tests for _run_round method."""

    @pytest.mark.asyncio
    async def test_run_round_first_round_runs_baseline_on_all_scenes(
        self, runner, mock_executor, sample_scenes, sample_results
    ):
        """Test that round 1 runs baseline on all scenes."""
        mock_executor.run_batch.return_value = sample_results
        state = BenchmarkState(
            round_num=0,
            scenes=sample_scenes,
            results_by_round={},
            current_best_params={},
            current_best_accuracy=0.0,
            parameters_eliminated=[],
        )

        results = await runner._run_round(
            round_num=1, scenes=sample_scenes, state=state
        )

        # Should call run_batch with all scenes
        assert mock_executor.run_batch.called
        call_args = mock_executor.run_batch.call_args
        assert len(call_args[0][0]) == len(sample_scenes)

    @pytest.mark.asyncio
    async def test_run_round_subsequent_rounds_sample_scenes(
        self,
        runner,
        mock_scene_selector,
        mock_executor,
        mock_analyzer,
        sample_scenes,
        sample_results,
    ):
        """Test that rounds > 1 sample scenes using SAMPLE_FRACTION."""
        # Setup mocks
        sampled_scenes = sample_scenes[:2]  # 30% of 5 scenes
        mock_scene_selector.sample_stratified.return_value = sampled_scenes
        mock_executor.run_batch.return_value = sample_results[:2]
        mock_analyzer.compute_aggregate_metrics.return_value = AggregateMetrics(
            total_scenes=2,
            total_expected=4,
            total_true_positives=4,
            total_false_positives=0,
            total_false_negatives=0,
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
        )

        state = BenchmarkState(
            round_num=1,
            scenes=sample_scenes,
            results_by_round={1: sample_results},
            current_best_params={"max_distance": 0.7},
            current_best_accuracy=0.8,
            parameters_eliminated=[],
        )

        results = await runner._run_round(
            round_num=2, scenes=sample_scenes, state=state
        )

        # Should call sample_stratified for sampling
        assert mock_scene_selector.sample_stratified.called


class TestGenerateFinalOutputs:
    """Tests for _generate_final_outputs method."""

    @pytest.mark.asyncio
    async def test_generate_final_outputs_creates_output_dir(
        self, runner, mock_reporter, mock_analyzer, sample_scenes, sample_results
    ):
        """Test that _generate_final_outputs creates output directory."""
        mock_analyzer.compute_aggregate_metrics.return_value = AggregateMetrics(
            total_scenes=5,
            total_expected=10,
            total_true_positives=10,
            total_false_positives=0,
            total_false_negatives=0,
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
        )

        state = BenchmarkState(
            round_num=2,
            scenes=sample_scenes,
            results_by_round={1: sample_results, 2: sample_results},
            current_best_params={"max_distance": 0.7},
            current_best_accuracy=1.0,
            parameters_eliminated=[],
        )

        await runner._generate_final_outputs(state)

        assert os.path.exists(runner.output_dir)

    @pytest.mark.asyncio
    async def test_generate_final_outputs_exports_csv(
        self, runner, mock_reporter, mock_analyzer, sample_scenes, sample_results
    ):
        """Test that _generate_final_outputs exports scene results CSV."""
        mock_analyzer.compute_aggregate_metrics.return_value = AggregateMetrics(
            total_scenes=5,
            total_expected=10,
            total_true_positives=10,
            total_false_positives=0,
            total_false_negatives=0,
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
        )

        state = BenchmarkState(
            round_num=2,
            scenes=sample_scenes,
            results_by_round={1: sample_results, 2: sample_results},
            current_best_params={"max_distance": 0.7},
            current_best_accuracy=1.0,
            parameters_eliminated=[],
        )

        await runner._generate_final_outputs(state)

        # Should call export_scene_results_csv
        assert mock_reporter.export_scene_results_csv.called


class TestRun:
    """Tests for main run method."""

    @pytest.mark.asyncio
    async def test_run_selects_scenes_when_not_resuming(
        self, runner, mock_scene_selector, mock_executor, mock_analyzer, sample_scenes, sample_results
    ):
        """Test that run selects scenes when resume=False."""
        mock_scene_selector.select_scenes.return_value = sample_scenes
        mock_executor.run_batch.return_value = sample_results
        mock_analyzer.compute_aggregate_metrics.return_value = AggregateMetrics(
            total_scenes=5,
            total_expected=10,
            total_true_positives=9,
            total_false_positives=0,
            total_false_negatives=1,
            accuracy=0.9,
            precision=1.0,
            recall=0.9,
        )

        # Mock _should_continue to stop after first round
        with patch.object(runner, "_should_continue", return_value=False):
            state = await runner.run(min_scenes=100, max_rounds=4, resume=False)

        mock_scene_selector.select_scenes.assert_called_once_with(min_count=100)

    @pytest.mark.asyncio
    async def test_run_loads_checkpoint_when_resuming(
        self, runner, mock_reporter, mock_executor, mock_analyzer, sample_scenes, sample_results
    ):
        """Test that run loads checkpoint when resume=True."""
        # Create a checkpoint state
        checkpoint_state = BenchmarkState(
            round_num=1,
            scenes=sample_scenes,
            results_by_round={1: sample_results},
            current_best_params={"max_distance": 0.7},
            current_best_accuracy=0.9,
            parameters_eliminated=[],
        )
        mock_reporter.load_checkpoint.return_value = checkpoint_state
        mock_executor.run_batch.return_value = sample_results
        mock_analyzer.compute_aggregate_metrics.return_value = AggregateMetrics(
            total_scenes=5,
            total_expected=10,
            total_true_positives=9,
            total_false_positives=0,
            total_false_negatives=1,
            accuracy=0.9,
            precision=1.0,
            recall=0.9,
        )

        # Mock _should_continue to stop after one round
        with patch.object(runner, "_should_continue", return_value=False):
            state = await runner.run(min_scenes=100, max_rounds=4, resume=True)

        mock_reporter.load_checkpoint.assert_called()

    @pytest.mark.asyncio
    async def test_run_saves_checkpoint_after_each_round(
        self, runner, mock_scene_selector, mock_executor, mock_analyzer, mock_reporter, sample_scenes, sample_results
    ):
        """Test that run saves checkpoint after each round."""
        mock_scene_selector.select_scenes.return_value = sample_scenes
        mock_executor.run_batch.return_value = sample_results
        mock_analyzer.compute_aggregate_metrics.return_value = AggregateMetrics(
            total_scenes=5,
            total_expected=10,
            total_true_positives=9,
            total_false_positives=0,
            total_false_negatives=1,
            accuracy=0.9,
            precision=1.0,
            recall=0.9,
        )

        # Allow 2 rounds
        call_count = 0

        def should_continue_side_effect(*args):
            nonlocal call_count
            call_count += 1
            return call_count < 2

        with patch.object(runner, "_should_continue", side_effect=should_continue_side_effect):
            state = await runner.run(min_scenes=100, max_rounds=4, resume=False)

        # Should save checkpoint at least once
        assert mock_reporter.save_checkpoint.called

    @pytest.mark.asyncio
    async def test_run_returns_benchmark_state(
        self, runner, mock_scene_selector, mock_executor, mock_analyzer, sample_scenes, sample_results
    ):
        """Test that run returns a BenchmarkState."""
        mock_scene_selector.select_scenes.return_value = sample_scenes
        mock_executor.run_batch.return_value = sample_results
        mock_analyzer.compute_aggregate_metrics.return_value = AggregateMetrics(
            total_scenes=5,
            total_expected=10,
            total_true_positives=9,
            total_false_positives=0,
            total_false_negatives=1,
            accuracy=0.9,
            precision=1.0,
            recall=0.9,
        )

        with patch.object(runner, "_should_continue", return_value=False):
            state = await runner.run(min_scenes=100, max_rounds=4, resume=False)

        assert isinstance(state, BenchmarkState)
        assert state.round_num >= 1
        assert len(state.scenes) == len(sample_scenes)

"""Benchmark runner for orchestrating iterative benchmark runs.

This module provides the BenchmarkRunner class that orchestrates multi-round
benchmark execution with parameter grid search, stopping criteria, and
checkpoint/resume support.
"""

import os
import json
import random
from typing import TYPE_CHECKING

from api.benchmark.models import (
    BenchmarkParams,
    BenchmarkState,
    SceneResult,
    TestScene,
    AggregateMetrics,
)

if TYPE_CHECKING:
    from api.benchmark.scene_selector import SceneSelector
    from api.benchmark.test_executor import TestExecutor
    from api.benchmark.analyzer import Analyzer
    from api.benchmark.reporter import Reporter


class BenchmarkRunner:
    """Orchestrates iterative benchmark runs with parameter tuning.

    The runner executes multi-round benchmarks, starting with a baseline
    configuration and iteratively exploring parameter variations to find
    optimal settings. It supports checkpoint/resume functionality and
    generates comprehensive final reports.

    Usage:
        runner = BenchmarkRunner(scene_selector, executor, analyzer, reporter, output_dir)
        state = await runner.run(min_scenes=100, max_rounds=4)
    """

    # Class constants
    MAX_ROUNDS = 6
    IMPROVEMENT_THRESHOLD = 0.01  # 1% minimum improvement to continue
    SAMPLE_FRACTION = 0.3  # 30% of scenes for parameter variations

    def __init__(
        self,
        scene_selector: "SceneSelector",
        executor: "TestExecutor",
        analyzer: "Analyzer",
        reporter: "Reporter",
        output_dir: str,
    ):
        """Initialize the benchmark runner.

        Args:
            scene_selector: SceneSelector for selecting and sampling test scenes.
            executor: TestExecutor for running identification on scenes.
            analyzer: Analyzer for computing metrics and analyzing results.
            reporter: Reporter for generating reports and saving checkpoints.
            output_dir: Directory path for output files.
        """
        self.scene_selector = scene_selector
        self.executor = executor
        self.analyzer = analyzer
        self.reporter = reporter
        self.output_dir = output_dir

        # Track best parameters found
        self._best_distance: float = 0.7
        self._best_face_size: int = 40
        self._best_multi_signal: bool = True

    async def run(
        self,
        min_scenes: int = 100,
        max_rounds: int = 4,
        resume: bool = False,
    ) -> BenchmarkState:
        """Run the benchmark with iterative parameter tuning.

        Args:
            min_scenes: Minimum number of scenes to select for testing.
            max_rounds: Maximum number of rounds to run.
            resume: If True, load from checkpoint and continue.

        Returns:
            BenchmarkState containing all results and best parameters.
        """
        # Initialize or load state
        if resume:
            checkpoint_path = os.path.join(self.output_dir, "checkpoint.json")
            loaded_state = self.reporter.load_checkpoint(checkpoint_path)
            if loaded_state is not None:
                scenes = loaded_state.scenes
                start_round = loaded_state.round_num + 1
                state = loaded_state
            else:
                # No checkpoint found, start fresh
                scenes = await self.scene_selector.select_scenes(min_count=min_scenes)
                start_round = 1
                state = BenchmarkState(
                    round_num=0,
                    scenes=scenes,
                    results_by_round={},
                    current_best_params=BenchmarkParams().to_dict(),
                    current_best_accuracy=0.0,
                    parameters_eliminated=[],
                )
        else:
            scenes = await self.scene_selector.select_scenes(min_count=min_scenes)
            start_round = 1
            state = BenchmarkState(
                round_num=0,
                scenes=scenes,
                results_by_round={},
                current_best_params=BenchmarkParams().to_dict(),
                current_best_accuracy=0.0,
                parameters_eliminated=[],
            )

        previous_accuracy = state.current_best_accuracy

        # Run rounds
        for round_num in range(start_round, max_rounds + 1):
            print(f"\n=== Round {round_num} ===")

            # Run this round
            results = await self._run_round(round_num, scenes, state)

            # Compute metrics for this round
            metrics = self.analyzer.compute_aggregate_metrics(results)
            current_accuracy = metrics.accuracy

            # Update state
            state.round_num = round_num
            state.results_by_round[round_num] = results
            state.current_best_accuracy = current_accuracy

            print(f"Round {round_num} accuracy: {current_accuracy:.2%}")

            # Save checkpoint
            os.makedirs(self.output_dir, exist_ok=True)
            checkpoint_path = os.path.join(self.output_dir, "checkpoint.json")
            self.reporter.save_checkpoint(state, checkpoint_path)

            # Check stopping criteria
            if not self._should_continue(current_accuracy, previous_accuracy, round_num):
                print(f"Stopping: improvement threshold not met or max rounds reached")
                break

            previous_accuracy = current_accuracy

        # Generate final outputs
        await self._generate_final_outputs(state)

        return state

    async def _run_round(
        self,
        round_num: int,
        scenes: list[TestScene],
        state: BenchmarkState,
    ) -> list[SceneResult]:
        """Run a single benchmark round.

        Args:
            round_num: The round number (1-indexed).
            scenes: All test scenes.
            state: Current benchmark state.

        Returns:
            List of SceneResult from the best configuration in this round.
        """
        if round_num == 1:
            # Round 1: run baseline on all scenes with default params
            params = BenchmarkParams(
                max_distance=self._best_distance,
                min_face_size=self._best_face_size,
                use_multi_signal=self._best_multi_signal,
            )
            results = await self._run_baseline(scenes, params)
            self._update_best_params(params)
            state.current_best_params = params.to_dict()
            return results
        else:
            # Subsequent rounds: sample scenes, try parameter grid, find best
            sample_count = max(1, int(len(scenes) * self.SAMPLE_FRACTION))
            sampled_scenes = self.scene_selector.sample_stratified(
                scenes, sample_count
            )

            # Build parameter grid
            param_grid = self._build_parameter_grid(round_num)

            best_results = None
            best_accuracy = 0.0
            best_params = None

            for params in param_grid:
                results = await self._run_baseline(sampled_scenes, params)
                metrics = self.analyzer.compute_aggregate_metrics(results)

                if metrics.accuracy > best_accuracy:
                    best_accuracy = metrics.accuracy
                    best_results = results
                    best_params = params

            # Update best parameters if we found better ones
            if best_params is not None:
                self._update_best_params(best_params)
                state.current_best_params = best_params.to_dict()

            return best_results if best_results is not None else []

    async def _run_baseline(
        self,
        scenes: list[TestScene],
        params: BenchmarkParams,
    ) -> list[SceneResult]:
        """Run baseline identification on scenes.

        Args:
            scenes: Test scenes to run identification on.
            params: Benchmark parameters to use.

        Returns:
            List of SceneResult from the executor.
        """
        return await self.executor.run_batch(scenes, params)

    def _build_parameter_grid(self, round_num: int) -> list[BenchmarkParams]:
        """Build parameter grid for a given round.

        Args:
            round_num: The round number (1-indexed).

        Returns:
            List of BenchmarkParams configurations to try.
        """
        grid = []

        if round_num == 1:
            # Round 1: baseline + distance variations [0.5, 0.6, 0.8]
            for distance in [0.5, 0.6, 0.7, 0.8]:
                grid.append(BenchmarkParams(
                    max_distance=distance,
                    min_face_size=self._best_face_size,
                    use_multi_signal=self._best_multi_signal,
                ))
        elif round_num == 2:
            # Round 2: finer distance tuning around best + face size variations
            fine_distances = [
                self._best_distance - 0.05,
                self._best_distance,
                self._best_distance + 0.05,
            ]
            face_sizes = [40, 60, 80]

            for distance in fine_distances:
                for face_size in face_sizes:
                    grid.append(BenchmarkParams(
                        max_distance=distance,
                        min_face_size=face_size,
                        use_multi_signal=self._best_multi_signal,
                    ))
        elif round_num == 3:
            # Round 3: multi-signal comparison (True vs False)
            for use_multi_signal in [True, False]:
                grid.append(BenchmarkParams(
                    max_distance=self._best_distance,
                    min_face_size=self._best_face_size,
                    use_multi_signal=use_multi_signal,
                ))
        else:
            # Round 4+: combination of best params with slight variations
            grid.append(BenchmarkParams(
                max_distance=self._best_distance,
                min_face_size=self._best_face_size,
                use_multi_signal=self._best_multi_signal,
            ))

            # Add small variations
            variations = [
                (self._best_distance - 0.02, self._best_face_size),
                (self._best_distance + 0.02, self._best_face_size),
                (self._best_distance, self._best_face_size - 10),
                (self._best_distance, self._best_face_size + 10),
            ]
            for distance, face_size in variations:
                if distance > 0 and face_size > 0:
                    grid.append(BenchmarkParams(
                        max_distance=distance,
                        min_face_size=face_size,
                        use_multi_signal=self._best_multi_signal,
                    ))

        return grid

    def _update_best_params(self, params: BenchmarkParams) -> None:
        """Update the best parameters found.

        Args:
            params: The best performing parameters to store.
        """
        self._best_distance = params.max_distance
        self._best_face_size = params.min_face_size
        self._best_multi_signal = params.use_multi_signal

    def _should_continue(
        self,
        current_accuracy: float,
        previous_accuracy: float,
        round_num: int,
    ) -> bool:
        """Determine if benchmark should continue to next round.

        Args:
            current_accuracy: Accuracy from the current round.
            previous_accuracy: Accuracy from the previous round.
            round_num: The current round number.

        Returns:
            True if benchmark should continue, False otherwise.
        """
        # Stop if we've reached max rounds
        if round_num >= self.MAX_ROUNDS:
            return False

        # Continue if improvement is at or above threshold
        improvement = current_accuracy - previous_accuracy
        return improvement >= self.IMPROVEMENT_THRESHOLD

    async def _generate_final_outputs(self, state: BenchmarkState) -> None:
        """Generate final outputs after benchmark completion.

        Creates output directory, exports CSVs, computes final metrics,
        generates summary and final report, and saves recommendation JSON.

        Args:
            state: Final benchmark state.
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # Get all results from the last round
        last_round = max(state.results_by_round.keys()) if state.results_by_round else 1
        final_results = state.results_by_round.get(last_round, [])

        # Export scene results CSV
        csv_path = os.path.join(self.output_dir, "scene_results.csv")
        self.reporter.export_scene_results_csv(final_results, csv_path)

        # Compute final metrics with breakdowns
        metrics = self.analyzer.compute_aggregate_metrics(final_results)
        metrics.accuracy_by_resolution = self.analyzer.compute_accuracy_by_resolution(
            state.scenes, final_results
        )
        metrics.accuracy_by_coverage = self.analyzer.compute_accuracy_by_coverage(
            state.scenes, final_results
        )

        # Generate and print summary
        summary = self.reporter.generate_summary(metrics)
        print(f"\n{summary}")

        # Save summary to file
        summary_path = os.path.join(self.output_dir, "summary.txt")
        with open(summary_path, "w") as f:
            f.write(summary)

        # Generate and save final report
        report = self.reporter.generate_final_report(state, metrics)
        report_path = os.path.join(self.output_dir, "final_report.md")
        with open(report_path, "w") as f:
            f.write(report)

        # Generate and save recommendation JSON
        best_params = BenchmarkParams(**state.current_best_params)

        # Get baseline accuracy (round 1)
        baseline_results = state.results_by_round.get(1, [])
        baseline_metrics = self.analyzer.compute_aggregate_metrics(baseline_results)
        baseline_accuracy = baseline_metrics.accuracy

        recommendation = self.reporter.generate_recommendation(
            best_params=best_params,
            accuracy=metrics.accuracy,
            baseline_accuracy=baseline_accuracy,
            total_scenes=metrics.total_scenes,
            total_performers=metrics.total_expected,
            notes=[
                f"Optimized over {state.round_num} rounds",
                f"Improvement from baseline: {(metrics.accuracy - baseline_accuracy) * 100:.1f}%",
            ],
        )

        recommendation_path = os.path.join(self.output_dir, "recommendation.json")
        with open(recommendation_path, "w") as f:
            json.dump(recommendation, f, indent=2)

        print(f"\nOutputs saved to: {self.output_dir}")

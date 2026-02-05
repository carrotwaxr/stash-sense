"""Reporter module for generating benchmark reports and exports.

This module provides the Reporter class for generating summaries, progress bars,
CSV exports, and markdown reports from benchmark results.
"""

import csv
import json
from typing import Optional

from benchmark.models import (
    AggregateMetrics,
    BenchmarkParams,
    BenchmarkState,
    PerformerResult,
    SceneResult,
)


class Reporter:
    """Generates reports and exports from benchmark results.

    Provides methods for creating text summaries, progress bars, CSV exports,
    and markdown reports to track and analyze benchmark progress.
    """

    def generate_summary(self, metrics: AggregateMetrics) -> str:
        """Generate a multi-line string summary of aggregate metrics.

        Args:
            metrics: The aggregate metrics to summarize.

        Returns:
            Multi-line string with formatted summary including overall stats
            and breakdowns by resolution and coverage.
        """
        lines = [
            "=== Benchmark Summary ===",
            f"Scenes tested: {metrics.total_scenes}",
            f"Total expected performers: {metrics.total_expected}",
            f"Overall accuracy: {metrics.accuracy * 100:.1f}% "
            f"({metrics.total_true_positives}/{metrics.total_expected})",
            f"Precision: {metrics.precision * 100:.1f}%",
            f"Recall: {metrics.recall * 100:.1f}%",
            "",
            "By Resolution:",
        ]

        for resolution, acc in sorted(metrics.accuracy_by_resolution.items()):
            lines.append(f"  {resolution}: {acc * 100:.1f}%")

        lines.append("")
        lines.append("By DB Coverage:")
        for coverage, acc in sorted(metrics.accuracy_by_coverage.items()):
            lines.append(f"  {coverage}: {acc * 100:.1f}%")

        return "\n".join(lines)

    def format_progress(
        self,
        current: int,
        total: int,
        accuracy: float,
        eta_sec: Optional[float] = None,
    ) -> str:
        """Format a progress line with progress bar.

        Args:
            current: Current number of completed scenes.
            total: Total number of scenes.
            accuracy: Current accuracy as a float (0.0 to 1.0).
            eta_sec: Optional estimated time remaining in seconds.

        Returns:
            Progress line string with bar, counts, and optional ETA.
        """
        bar_width = 20
        if total > 0:
            filled = int((current / total) * bar_width)
        else:
            filled = 0
        empty = bar_width - filled

        bar = "[" + "#" * filled + "." * empty + "]"

        progress_str = f"{bar} {current}/{total} scenes | {accuracy * 100:.1f}% accuracy"

        if eta_sec is not None:
            minutes = int(eta_sec / 60)
            progress_str += f" | ETA {minutes}m"

        return progress_str

    def export_scene_results_csv(
        self, results: list[SceneResult], output_path: str
    ) -> None:
        """Export scene results to a CSV file.

        Args:
            results: List of SceneResult objects to export.
            output_path: Path to the output CSV file.
        """
        fieldnames = [
            "scene_id",
            "true_positives",
            "false_negatives",
            "false_positives",
            "expected_in_top_1",
            "expected_in_top_3",
            "score_gap",
            "faces_detected",
            "faces_after_filter",
            "persons_clustered",
            "elapsed_sec",
            "accuracy",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "scene_id": result.scene_id,
                    "true_positives": result.true_positives,
                    "false_negatives": result.false_negatives,
                    "false_positives": result.false_positives,
                    "expected_in_top_1": result.expected_in_top_1,
                    "expected_in_top_3": result.expected_in_top_3,
                    "score_gap": result.score_gap,
                    "faces_detected": result.faces_detected,
                    "faces_after_filter": result.faces_after_filter,
                    "persons_clustered": result.persons_clustered,
                    "elapsed_sec": result.elapsed_sec,
                    "accuracy": f"{result.accuracy:.3f}",
                }
                writer.writerow(row)

    def export_performer_results_csv(
        self, results: list[PerformerResult], output_path: str
    ) -> None:
        """Export performer results to a CSV file.

        Args:
            results: List of PerformerResult objects to export.
            output_path: Path to the output CSV file.
        """
        fieldnames = [
            "stashdb_id",
            "name",
            "faces_in_db",
            "has_body_data",
            "has_tattoo_data",
            "was_found",
            "rank_if_found",
            "confidence_if_found",
            "distance_if_found",
            "best_match_for_missed",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "stashdb_id": result.stashdb_id,
                    "name": result.name,
                    "faces_in_db": result.faces_in_db,
                    "has_body_data": result.has_body_data,
                    "has_tattoo_data": result.has_tattoo_data,
                    "was_found": result.was_found,
                    "rank_if_found": (
                        result.rank_if_found if result.rank_if_found is not None else ""
                    ),
                    "confidence_if_found": (
                        f"{result.confidence_if_found:.3f}"
                        if result.confidence_if_found is not None
                        else ""
                    ),
                    "distance_if_found": (
                        f"{result.distance_if_found:.3f}"
                        if result.distance_if_found is not None
                        else ""
                    ),
                    "best_match_for_missed": (
                        result.best_match_for_missed
                        if result.best_match_for_missed is not None
                        else ""
                    ),
                }
                writer.writerow(row)

    def export_parameter_comparison_csv(
        self, comparisons: list[dict], output_path: str
    ) -> None:
        """Export parameter comparisons to a CSV file.

        Args:
            comparisons: List of comparison dictionaries with keys:
                param_name, value_a, value_b, accuracy_a, accuracy_b, improvement
            output_path: Path to the output CSV file.
        """
        fieldnames = [
            "param_name",
            "value_a",
            "value_b",
            "accuracy_a",
            "accuracy_b",
            "improvement",
        ]

        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for comparison in comparisons:
                writer.writerow(comparison)

    def generate_recommendation(
        self,
        best_params: BenchmarkParams,
        accuracy: float,
        baseline_accuracy: float,
        total_scenes: int,
        total_performers: int,
        notes: list[str],
    ) -> dict:
        """Generate a recommendation dictionary with best parameters.

        Args:
            best_params: The best performing parameters.
            accuracy: The accuracy achieved with best params.
            baseline_accuracy: The baseline accuracy for comparison.
            total_scenes: Number of scenes tested.
            total_performers: Number of performers tested.
            notes: List of notes about the recommendation.

        Returns:
            Dictionary with recommended_config, accuracy, baseline_accuracy,
            improvement, validated_on, and notes.
        """
        return {
            "recommended_config": best_params.to_dict(),
            "accuracy": accuracy,
            "baseline_accuracy": baseline_accuracy,
            "improvement": accuracy - baseline_accuracy,
            "validated_on": {
                "scenes": total_scenes,
                "performers": total_performers,
            },
            "notes": notes,
        }

    def generate_final_report(
        self, state: BenchmarkState, metrics: AggregateMetrics
    ) -> str:
        """Generate a final markdown report.

        Args:
            state: The benchmark state containing round info and parameters.
            metrics: The aggregate metrics to include.

        Returns:
            Markdown formatted final report string.
        """
        lines = [
            "# Benchmark Final Report",
            "",
            "## Summary",
            "",
            f"- Rounds completed: {state.round_num}",
            f"- Scenes tested: {metrics.total_scenes}",
            f"- Total performers: {metrics.total_expected}",
            f"- Accuracy: {metrics.accuracy * 100:.1f}%",
            f"- Precision: {metrics.precision * 100:.1f}%",
            f"- Recall: {metrics.recall * 100:.1f}%",
            "",
            "## Best Parameters",
            "",
            "```json",
            json.dumps(state.current_best_params, indent=2),
            "```",
            "",
            "## Parameters Eliminated",
            "",
        ]

        for param in state.parameters_eliminated:
            lines.append(f"- {param}")

        return "\n".join(lines)

    def save_checkpoint(self, state: BenchmarkState, output_path: str) -> None:
        """Save benchmark state to a checkpoint file.

        Args:
            state: The benchmark state to save.
            output_path: Path to the output checkpoint file.
        """
        with open(output_path, "w") as f:
            f.write(state.to_json())

    def load_checkpoint(self, checkpoint_path: str) -> Optional[BenchmarkState]:
        """Load benchmark state from a checkpoint file.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Returns:
            BenchmarkState if file exists and is valid, None if file not found.
        """
        try:
            with open(checkpoint_path, "r") as f:
                json_str = f.read()
            return BenchmarkState.from_json(json_str)
        except FileNotFoundError:
            return None

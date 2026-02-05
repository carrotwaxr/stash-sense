"""Analyzer for benchmark metrics and failure pattern analysis.

Provides methods to compute aggregate metrics, breakdown by various dimensions,
and analyze failure patterns from benchmark results.
"""

from benchmark.models import (
    ExpectedPerformer,
    TestScene,
    SceneResult,
    AggregateMetrics,
    PerformerResult,
)

# Minimum number of faces needed in database for reliable matching
MIN_DB_COVERAGE = 3


class Analyzer:
    """Analyzes benchmark results to compute metrics and identify patterns.

    Provides methods for computing aggregate metrics, breaking down accuracy
    by various dimensions (resolution, coverage, face count), classifying
    failure reasons, and comparing different parameter configurations.
    """

    def compute_aggregate_metrics(self, results: list[SceneResult]) -> AggregateMetrics:
        """Compute aggregate metrics from a list of scene results.

        Args:
            results: List of SceneResult objects to aggregate.

        Returns:
            AggregateMetrics with computed summary statistics.
        """
        return AggregateMetrics.from_results(results)

    def compute_accuracy_by_resolution(
        self, scenes: list[TestScene], results: list[SceneResult]
    ) -> dict[str, float]:
        """Compute accuracy grouped by resolution tier.

        Args:
            scenes: List of TestScene objects with resolution information.
            results: List of SceneResult objects with identification results.

        Returns:
            Dictionary mapping resolution tier to accuracy (TP / (TP + FN)).
        """
        if not scenes or not results:
            return {}

        # Map scene_id to result
        result_by_scene = {r.scene_id: r for r in results}

        # Group by resolution
        resolution_stats: dict[str, dict[str, int]] = {}

        for scene in scenes:
            result = result_by_scene.get(scene.scene_id)
            if result is None:
                continue

            resolution = scene.resolution
            if resolution not in resolution_stats:
                resolution_stats[resolution] = {"tp": 0, "fn": 0}

            resolution_stats[resolution]["tp"] += result.true_positives
            resolution_stats[resolution]["fn"] += result.false_negatives

        # Compute accuracy per resolution
        accuracy_by_resolution: dict[str, float] = {}
        for resolution, stats in resolution_stats.items():
            total = stats["tp"] + stats["fn"]
            if total == 0:
                accuracy_by_resolution[resolution] = 0.0
            else:
                accuracy_by_resolution[resolution] = stats["tp"] / total

        return accuracy_by_resolution

    def compute_accuracy_by_coverage(
        self, scenes: list[TestScene], results: list[SceneResult]
    ) -> dict[str, float]:
        """Compute accuracy grouped by database coverage tier.

        Args:
            scenes: List of TestScene objects with db_coverage_tier information.
            results: List of SceneResult objects with identification results.

        Returns:
            Dictionary mapping coverage tier to accuracy (TP / (TP + FN)).
        """
        if not scenes or not results:
            return {}

        # Map scene_id to result
        result_by_scene = {r.scene_id: r for r in results}

        # Group by db_coverage_tier
        coverage_stats: dict[str, dict[str, int]] = {}

        for scene in scenes:
            result = result_by_scene.get(scene.scene_id)
            if result is None:
                continue

            tier = scene.db_coverage_tier
            if tier not in coverage_stats:
                coverage_stats[tier] = {"tp": 0, "fn": 0}

            coverage_stats[tier]["tp"] += result.true_positives
            coverage_stats[tier]["fn"] += result.false_negatives

        # Compute accuracy per coverage tier
        accuracy_by_coverage: dict[str, float] = {}
        for tier, stats in coverage_stats.items():
            total = stats["tp"] + stats["fn"]
            if total == 0:
                accuracy_by_coverage[tier] = 0.0
            else:
                accuracy_by_coverage[tier] = stats["tp"] / total

        return accuracy_by_coverage

    def compute_accuracy_by_face_count(
        self, scenes: list[TestScene], results: list[SceneResult]
    ) -> dict[str, float]:
        """Compute accuracy grouped by expected face count bucket.

        Buckets: "1-2", "3-5", "6+"

        Args:
            scenes: List of TestScene objects with expected performer count.
            results: List of SceneResult objects with identification results.

        Returns:
            Dictionary mapping bucket to accuracy (placeholder implementation returns 0.0).
        """
        # Simplified placeholder implementation
        return {"1-2": 0.0, "3-5": 0.0, "6+": 0.0}

    def classify_failure_reason(
        self,
        performer: ExpectedPerformer,
        was_detected: bool,
        top_matches: list[dict],
    ) -> str:
        """Classify the reason for a performer identification failure.

        Args:
            performer: The expected performer that failed to match.
            was_detected: Whether the performer's face was detected at all.
            top_matches: List of top match candidates with 'stashdb_id' and 'confidence'.

        Returns:
            One of: "not_detected", "insufficient_db_coverage",
            "similar_performer_won", "low_confidence"
        """
        if not was_detected:
            return "not_detected"

        if performer.faces_in_db < MIN_DB_COVERAGE:
            return "insufficient_db_coverage"

        # Check if performer is in top matches but ranked lower
        for i, match in enumerate(top_matches):
            if match.get("stashdb_id") == performer.stashdb_id:
                if i > 0:  # Not the top match
                    return "similar_performer_won"
                break

        return "low_confidence"

    def find_failure_patterns(
        self, scenes: list[TestScene], results: list[SceneResult]
    ) -> dict[str, int]:
        """Find patterns in identification failures.

        Args:
            scenes: List of TestScene objects.
            results: List of SceneResult objects.

        Returns:
            Dictionary mapping failure reason to count.
            Simplified: counts false_negatives into "not_detected".
        """
        patterns = {
            "not_detected": 0,
            "insufficient_db_coverage": 0,
            "similar_performer_won": 0,
            "low_confidence": 0,
        }

        # Simplified implementation: count all false_negatives as not_detected
        for result in results:
            patterns["not_detected"] += result.false_negatives

        return patterns

    def compare_parameters(
        self, results_a: list[SceneResult], results_b: list[SceneResult]
    ) -> dict:
        """Compare metrics between two parameter configurations.

        Args:
            results_a: Results from first parameter configuration.
            results_b: Results from second parameter configuration.

        Returns:
            Dictionary with accuracy, precision, recall for both configurations
            and the improvement (accuracy_b - accuracy_a).
        """
        metrics_a = AggregateMetrics.from_results(results_a)
        metrics_b = AggregateMetrics.from_results(results_b)

        return {
            "accuracy_a": metrics_a.accuracy,
            "accuracy_b": metrics_b.accuracy,
            "improvement": metrics_b.accuracy - metrics_a.accuracy,
            "precision_a": metrics_a.precision,
            "precision_b": metrics_b.precision,
            "recall_a": metrics_a.recall,
            "recall_b": metrics_b.recall,
        }

    def build_performer_results(
        self, scene: TestScene, identification_results: list[dict]
    ) -> list[PerformerResult]:
        """Build PerformerResult objects for each expected performer.

        Args:
            scene: TestScene with expected performers.
            identification_results: List of identification matches with
                'stashdb_id', 'name', 'confidence', and 'distance'.

        Returns:
            List of PerformerResult objects for each expected performer.
        """
        results: list[PerformerResult] = []

        # Build lookup for identification results by stashdb_id
        result_lookup = {r["stashdb_id"]: r for r in identification_results}

        # Find rank of each identification result
        rank_lookup = {
            r["stashdb_id"]: i + 1
            for i, r in enumerate(identification_results)
        }

        for performer in scene.expected_performers:
            match = result_lookup.get(performer.stashdb_id)

            if match is not None:
                # Performer was found
                result = PerformerResult(
                    stashdb_id=performer.stashdb_id,
                    name=performer.name,
                    faces_in_db=performer.faces_in_db,
                    has_body_data=performer.has_body_data,
                    has_tattoo_data=performer.has_tattoo_data,
                    was_found=True,
                    rank_if_found=rank_lookup[performer.stashdb_id],
                    confidence_if_found=match["confidence"],
                    distance_if_found=match["distance"],
                    who_beat_them=[],
                    best_match_for_missed=None,
                )
            else:
                # Performer was not found - build who_beat_them list (top 3)
                top_3 = [
                    (r["stashdb_id"], r["confidence"])
                    for r in identification_results[:3]
                ]
                best_match = identification_results[0]["stashdb_id"] if identification_results else None

                result = PerformerResult(
                    stashdb_id=performer.stashdb_id,
                    name=performer.name,
                    faces_in_db=performer.faces_in_db,
                    has_body_data=performer.has_body_data,
                    has_tattoo_data=performer.has_tattoo_data,
                    was_found=False,
                    rank_if_found=None,
                    confidence_if_found=None,
                    distance_if_found=None,
                    who_beat_them=top_3,
                    best_match_for_missed=best_match,
                )

            results.append(result)

        return results

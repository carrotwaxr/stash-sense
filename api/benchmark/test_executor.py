"""Test executor for benchmark scene identification.

Runs identification on test scenes and compares results to ground truth
to compute accuracy metrics.
"""

import time
from typing import TYPE_CHECKING

from benchmark.models import (
    TestScene,
    BenchmarkParams,
    SceneResult,
)

if TYPE_CHECKING:
    from recognizer import FaceRecognizer
    from multi_signal_matcher import MultiSignalMatcher


class TestExecutor:
    """Executes identification tests and compares to ground truth.

    This class runs the identification system on test scenes with configurable
    parameters and computes metrics by comparing results to expected performers.

    Usage:
        executor = TestExecutor(recognizer, multi_signal_matcher)
        result = await executor.identify_scene(scene, params)
        results = await executor.run_batch(scenes, params)
    """

    def __init__(
        self,
        recognizer: "FaceRecognizer",
        multi_signal_matcher: "MultiSignalMatcher",
    ):
        """Initialize the test executor.

        Args:
            recognizer: FaceRecognizer instance for face-only identification
            multi_signal_matcher: MultiSignalMatcher instance for multi-signal identification
        """
        self.recognizer = recognizer
        self.multi_signal_matcher = multi_signal_matcher

    def _compare_to_ground_truth(
        self,
        scene: TestScene,
        identification_results: list[dict],
    ) -> tuple[int, int, int]:
        """Compare identification results to ground truth.

        Extracts expected performer IDs from the scene and found performer IDs
        from the identification results, then computes true positives, false
        negatives, and false positives.

        Args:
            scene: TestScene with expected_performers
            identification_results: List of dicts with "stashdb_id" key

        Returns:
            Tuple of (true_positives, false_negatives, false_positives)
            - TP = len(expected_ids & found_ids)
            - FN = len(expected_ids - found_ids)
            - FP = len(found_ids - expected_ids)
        """
        # Extract expected IDs from scene
        expected_ids = {p.stashdb_id for p in scene.expected_performers}

        # Extract found IDs from results
        found_ids = {r["stashdb_id"] for r in identification_results}

        # Calculate metrics
        true_positives = len(expected_ids & found_ids)
        false_negatives = len(expected_ids - found_ids)
        false_positives = len(found_ids - expected_ids)

        return true_positives, false_negatives, false_positives

    def _count_in_top_n(
        self,
        results: list[dict],
        expected_ids: set[str],
        n: int,
    ) -> int:
        """Count how many expected performers have rank <= n in results.

        Args:
            results: List of dicts with "stashdb_id" and "rank" keys
            expected_ids: Set of expected performer stashdb_ids
            n: Maximum rank to consider (inclusive)

        Returns:
            Count of expected performers with rank <= n
        """
        count = 0
        for result in results:
            if result["stashdb_id"] in expected_ids and result["rank"] <= n:
                count += 1
        return count

    def _compute_score_gap(
        self,
        correct_scores: list[float],
        incorrect_scores: list[float],
    ) -> float:
        """Compute the gap between average correct and incorrect scores.

        A larger positive gap indicates better separation between correct
        and incorrect matches.

        Args:
            correct_scores: Distance scores for correctly identified performers
            incorrect_scores: Distance scores for incorrectly identified performers

        Returns:
            avg(correct_scores) - avg(incorrect_scores), or 0.0 if either list is empty
        """
        if not correct_scores or not incorrect_scores:
            return 0.0

        avg_correct = sum(correct_scores) / len(correct_scores)
        avg_incorrect = sum(incorrect_scores) / len(incorrect_scores)

        return avg_correct - avg_incorrect

    async def _run_scene_identification(
        self,
        scene: TestScene,
        params: BenchmarkParams,
    ) -> dict:
        """Run identification on a scene with given parameters.

        Builds the request dict and calls either the multi_signal_matcher or
        recognizer depending on params.use_multi_signal.

        Args:
            scene: TestScene to identify
            params: BenchmarkParams with identification settings

        Returns:
            Dict with:
                - performers: list of dicts with stashdb_id, rank, distance
                - faces_detected: int
                - faces_after_filter: int
                - persons_clustered: int
        """
        # Build request dict
        request = {
            "scene_id": scene.scene_id,
            "num_frames": params.num_frames,
            "start_offset_pct": params.start_offset_pct,
            "end_offset_pct": params.end_offset_pct,
            "matching_mode": params.matching_mode,
            "max_distance": params.max_distance,
            "min_face_size": params.min_face_size,
            "top_k": params.top_k,
        }

        # Call appropriate identification method
        if params.use_multi_signal:
            raw_result = await self.multi_signal_matcher.identify_scene(**request)
        else:
            raw_result = await self.recognizer.identify_scene(**request)

        # Transform results
        performers = []
        for i, person in enumerate(raw_result.get("persons", [])):
            best_match = person.get("best_match", {})
            universal_id = best_match.get("stashdb_id", "")

            # Extract stashdb_id from universal_id (split on ":", take last part)
            if ":" in universal_id:
                stashdb_id = universal_id.split(":")[-1]
            else:
                stashdb_id = universal_id

            distance = best_match.get("distance", 1.0)

            performers.append({
                "stashdb_id": stashdb_id,
                "rank": i + 1,  # 1-indexed rank based on order
                "distance": distance,
            })

        return {
            "performers": performers,
            "faces_detected": raw_result.get("faces_detected", 0),
            "faces_after_filter": raw_result.get("faces_after_filter", 0),
            "persons_clustered": len(performers),
        }

    async def identify_scene(
        self,
        scene: TestScene,
        params: BenchmarkParams,
    ) -> SceneResult:
        """Run identification on a scene and return metrics.

        Times the execution, runs identification, compares to ground truth,
        and computes all metrics for the SceneResult.

        Args:
            scene: TestScene to identify
            params: BenchmarkParams with identification settings

        Returns:
            SceneResult with all metrics and diagnostic information
        """
        # Time the execution
        start_time = time.time()

        # Run identification
        id_result = await self._run_scene_identification(scene, params)

        elapsed_sec = time.time() - start_time

        # Get list of performers with results
        performers = id_result["performers"]

        # Compare to ground truth
        true_positives, false_negatives, false_positives = self._compare_to_ground_truth(
            scene, performers
        )

        # Get expected IDs for ranking metrics
        expected_ids = {p.stashdb_id for p in scene.expected_performers}

        # Count in top-N
        expected_in_top_1 = self._count_in_top_n(performers, expected_ids, n=1)
        expected_in_top_3 = self._count_in_top_n(performers, expected_ids, n=3)

        # Separate correct and incorrect scores
        correct_match_scores = []
        incorrect_match_scores = []

        for perf in performers:
            if perf["stashdb_id"] in expected_ids:
                correct_match_scores.append(perf["distance"])
            else:
                incorrect_match_scores.append(perf["distance"])

        # Compute score gap
        score_gap = self._compute_score_gap(correct_match_scores, incorrect_match_scores)

        return SceneResult(
            scene_id=scene.scene_id,
            params=params.to_dict(),
            true_positives=true_positives,
            false_negatives=false_negatives,
            false_positives=false_positives,
            expected_in_top_1=expected_in_top_1,
            expected_in_top_3=expected_in_top_3,
            correct_match_scores=correct_match_scores,
            incorrect_match_scores=incorrect_match_scores,
            score_gap=score_gap,
            faces_detected=id_result["faces_detected"],
            faces_after_filter=id_result["faces_after_filter"],
            persons_clustered=id_result["persons_clustered"],
            elapsed_sec=elapsed_sec,
        )

    async def run_batch(
        self,
        scenes: list[TestScene],
        params: BenchmarkParams,
    ) -> list[SceneResult]:
        """Run identification on multiple scenes.

        Processes each scene sequentially, catching exceptions to continue
        with remaining scenes if one fails.

        Args:
            scenes: List of TestScene objects to process
            params: BenchmarkParams with identification settings

        Returns:
            List of SceneResult objects for successfully processed scenes
        """
        results = []

        for scene in scenes:
            try:
                result = await self.identify_scene(scene, params)
                results.append(result)
            except Exception as e:
                print(f"Error processing scene {scene.scene_id}: {e}")
                continue

        return results

"""Test executor for benchmark scene identification.

Runs identification on test scenes and compares results to ground truth
to compute accuracy metrics.
"""

import os
import time
from typing import TYPE_CHECKING

import httpx

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
        executor = TestExecutor(recognizer, multi_signal_matcher, stash_url, stash_api_key)
        result = await executor.identify_scene(scene, params)
        results = await executor.run_batch(scenes, params)
    """

    def __init__(
        self,
        recognizer: "FaceRecognizer",
        multi_signal_matcher: "MultiSignalMatcher",
        stash_url: str = None,
        stash_api_key: str = None,
    ):
        """Initialize the test executor.

        Args:
            recognizer: FaceRecognizer instance for face-only identification
            multi_signal_matcher: MultiSignalMatcher instance for multi-signal identification
            stash_url: Stash server URL (defaults to STASH_URL env var)
            stash_api_key: Stash API key (defaults to STASH_API_KEY env var)
        """
        self.recognizer = recognizer
        self.multi_signal_matcher = multi_signal_matcher
        self.stash_url = (stash_url or os.environ.get("STASH_URL", "")).rstrip("/")
        self.stash_api_key = stash_api_key or os.environ.get("STASH_API_KEY", "")

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

    async def _get_scene_duration(self, scene_id: str) -> float:
        """Get scene duration from Stash.

        Args:
            scene_id: The Stash scene ID

        Returns:
            Duration in seconds
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            gql_query = {
                "query": f'''{{
                    findScene(id: "{scene_id}") {{
                        files {{
                            duration
                        }}
                    }}
                }}'''
            }
            headers = {"ApiKey": self.stash_api_key, "Content-Type": "application/json"}
            response = await client.post(
                f"{self.stash_url}/graphql",
                json=gql_query,
                headers=headers
            )
            response.raise_for_status()
            data = response.json()

            scene_data = data.get("data", {}).get("findScene", {})
            if not scene_data or not scene_data.get("files"):
                raise ValueError(f"Scene {scene_id} not found or has no files")

            file_info = scene_data["files"][0]
            duration = file_info.get("duration", 0)
            if not duration:
                raise ValueError(f"Scene {scene_id} has no duration")

            return duration

    async def _run_scene_identification(
        self,
        scene: TestScene,
        params: BenchmarkParams,
    ) -> dict:
        """Run identification on a scene with given parameters.

        Extracts frames from the scene, detects faces, runs recognition,
        and clusters results.

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
        from frame_extractor import FrameExtractionConfig, extract_frames_from_stash_scene
        from matching import MatchingConfig
        from recognizer import RecognitionResult

        # Get scene duration
        duration_sec = await self._get_scene_duration(scene.scene_id)

        # Configure frame extraction
        config = FrameExtractionConfig(
            num_frames=params.num_frames,
            start_offset_pct=params.start_offset_pct,
            end_offset_pct=params.end_offset_pct,
            min_face_size=params.min_face_size,
            min_face_confidence=0.8,
        )

        # Extract frames
        extraction_result = await extract_frames_from_stash_scene(
            stash_url=self.stash_url,
            scene_id=scene.scene_id,
            duration_sec=duration_sec,
            api_key=self.stash_api_key,
            config=config,
        )

        # Configure matching
        match_config = MatchingConfig(
            query_k=100,
            facenet_weight=0.6,
            arcface_weight=0.4,
            max_results=params.top_k * 2,
            max_distance=params.max_distance,
        )

        # Detect and recognize faces in each frame
        all_results: list[tuple[int, RecognitionResult]] = []
        total_faces = 0
        filtered_faces = 0

        for frame in extraction_result.frames:
            faces = self.recognizer.generator.detect_faces(
                frame.image,
                min_confidence=0.8,
            )

            for face in faces:
                total_faces += 1

                # Apply minimum face size filter
                if face.bbox["w"] < params.min_face_size or face.bbox["h"] < params.min_face_size:
                    continue

                filtered_faces += 1

                # Recognize this face
                matches, _ = self.recognizer.recognize_face_v2(face, match_config)
                result = RecognitionResult(face=face, matches=matches)
                all_results.append((frame.frame_index, result))

        # Apply matching based on mode
        from main import (
            cluster_faces_by_person,
            merge_clusters_by_match,
            aggregate_matches,
            frequency_based_matching,
            hybrid_matching,
        )

        performers = []
        num_clusters = 0

        if params.matching_mode == "hybrid":
            # Use hybrid matching with improved false positive filtering
            persons = hybrid_matching(
                all_results,
                self.recognizer,
                cluster_threshold=params.cluster_threshold,
                top_k=params.top_k,
                max_distance=params.max_distance,
                min_appearances=2,
                min_unique_frames=2,
                min_confidence=0.35,
            )
            # Get cluster count for stats
            clusters = cluster_faces_by_person(
                all_results, self.recognizer, params.cluster_threshold
            )
            num_clusters = len(clusters)

            for rank, person in enumerate(persons):
                stashdb_id = person.best_match.stashdb_id
                if ":" in stashdb_id:
                    stashdb_id = stashdb_id.split(":")[-1]
                performers.append({
                    "stashdb_id": stashdb_id,
                    "rank": rank + 1,
                    "distance": person.best_match.distance,
                })

        elif params.matching_mode == "frequency":
            # Use frequency-based matching with improved filtering
            persons = frequency_based_matching(
                all_results,
                top_k=params.top_k,
                min_appearances=2,
                min_unique_frames=2,
                max_distance=params.max_distance,
                min_confidence=0.35,
            )
            num_clusters = len(persons)

            for rank, person in enumerate(persons):
                stashdb_id = person.best_match.stashdb_id
                if ":" in stashdb_id:
                    stashdb_id = stashdb_id.split(":")[-1]
                performers.append({
                    "stashdb_id": stashdb_id,
                    "rank": rank + 1,
                    "distance": person.best_match.distance,
                })

        else:
            # Default cluster-based matching
            clusters = cluster_faces_by_person(
                all_results,
                self.recognizer,
                distance_threshold=params.cluster_threshold,
            )
            clusters = merge_clusters_by_match(clusters)
            num_clusters = len(clusters)

            # Build response with deduplication
            used_performers: set[str] = set()
            all_persons = []

            for person_id, cluster in enumerate(clusters):
                aggregated_matches = aggregate_matches(cluster, top_k=params.top_k)
                if aggregated_matches:
                    all_persons.append((len(cluster), aggregated_matches[0]))

            all_persons.sort(key=lambda x: x[0], reverse=True)

            for rank, (_, best_match) in enumerate(all_persons):
                if best_match.stashdb_id in used_performers:
                    continue

                stashdb_id = best_match.stashdb_id
                if ":" in stashdb_id:
                    stashdb_id = stashdb_id.split(":")[-1]

                used_performers.add(best_match.stashdb_id)
                performers.append({
                    "stashdb_id": stashdb_id,
                    "rank": rank + 1,
                    "distance": best_match.distance,
                })

        return {
            "performers": performers,
            "faces_detected": total_faces,
            "faces_after_filter": filtered_faces,
            "persons_clustered": num_clusters,
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
        max_retries: int = 2,
    ) -> list[SceneResult]:
        """Run identification on multiple scenes with retry logic.

        Processes each scene sequentially, with automatic retry on failure.
        Failed scenes are retried up to max_retries times with a small delay.

        Args:
            scenes: List of TestScene objects to process
            params: BenchmarkParams with identification settings
            max_retries: Maximum number of retries per scene (default 2)

        Returns:
            List of SceneResult objects for successfully processed scenes
        """
        import asyncio

        results = []
        failed_count = 0
        retry_count = 0

        for i, scene in enumerate(scenes):
            success = False
            last_error = None

            for attempt in range(max_retries + 1):
                try:
                    result = await self.identify_scene(scene, params)
                    results.append(result)
                    success = True
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        retry_count += 1
                        # Brief delay before retry to let resources recover
                        await asyncio.sleep(1.0)

            if not success:
                failed_count += 1
                print(f"[{i+1}/{len(scenes)}] Failed scene {scene.scene_id} after {max_retries + 1} attempts: {type(last_error).__name__}: {str(last_error)[:80]}")

        if failed_count > 0 or retry_count > 0:
            print(f"Batch complete: {len(results)}/{len(scenes)} succeeded, {failed_count} failed, {retry_count} retries")

        return results

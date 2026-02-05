"""Tests for benchmark test executor."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from dataclasses import dataclass

from benchmark.models import (
    ExpectedPerformer,
    TestScene,
    BenchmarkParams,
    SceneResult,
)
from benchmark.test_executor import TestExecutor


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
def sample_scene(sample_performers):
    """Create a sample test scene."""
    return TestScene(
        scene_id="scene-123",
        stashdb_id="stashdb-scene-456",
        title="Test Scene Title",
        resolution="1080p",
        width=1920,
        height=1080,
        duration_sec=1800.0,
        expected_performers=sample_performers,
        db_coverage_tier="well-covered",
    )


@pytest.fixture
def mock_recognizer():
    """Create a mock recognizer."""
    return Mock()


@pytest.fixture
def mock_multi_signal_matcher():
    """Create a mock multi-signal matcher."""
    return Mock()


@pytest.fixture
def executor(mock_recognizer, mock_multi_signal_matcher):
    """Create a TestExecutor with mocked dependencies."""
    return TestExecutor(
        recognizer=mock_recognizer,
        multi_signal_matcher=mock_multi_signal_matcher,
    )


class TestCompareToGroundTruth:
    """Tests for _compare_to_ground_truth method."""

    def test_compare_to_ground_truth_all_found(self, executor, sample_scene):
        """Test when all expected performers are found (TP=2, FN=0, FP=0)."""
        identification_results = [
            {"stashdb_id": "perf-1-uuid", "rank": 1, "distance": 0.3},
            {"stashdb_id": "perf-2-uuid", "rank": 2, "distance": 0.4},
        ]

        tp, fn, fp = executor._compare_to_ground_truth(sample_scene, identification_results)

        assert tp == 2
        assert fn == 0
        assert fp == 0

    def test_compare_to_ground_truth_one_missed(self, executor, sample_scene):
        """Test when one expected performer is missed (TP=1, FN=1, FP=0)."""
        identification_results = [
            {"stashdb_id": "perf-1-uuid", "rank": 1, "distance": 0.3},
        ]

        tp, fn, fp = executor._compare_to_ground_truth(sample_scene, identification_results)

        assert tp == 1
        assert fn == 1
        assert fp == 0

    def test_compare_to_ground_truth_false_positive(self, executor, sample_scene):
        """Test when there are false positives (TP=2, FN=0, FP=1)."""
        identification_results = [
            {"stashdb_id": "perf-1-uuid", "rank": 1, "distance": 0.3},
            {"stashdb_id": "perf-2-uuid", "rank": 2, "distance": 0.4},
            {"stashdb_id": "unexpected-uuid", "rank": 3, "distance": 0.5},
        ]

        tp, fn, fp = executor._compare_to_ground_truth(sample_scene, identification_results)

        assert tp == 2
        assert fn == 0
        assert fp == 1

    def test_compare_to_ground_truth_mixed(self, executor, sample_scene):
        """Test with mix of found, missed, and false positives."""
        identification_results = [
            {"stashdb_id": "perf-1-uuid", "rank": 1, "distance": 0.3},
            {"stashdb_id": "wrong-perf-uuid", "rank": 2, "distance": 0.45},
        ]

        tp, fn, fp = executor._compare_to_ground_truth(sample_scene, identification_results)

        assert tp == 1  # Only perf-1 found
        assert fn == 1  # perf-2 missed
        assert fp == 1  # wrong-perf is a false positive

    def test_compare_to_ground_truth_empty_results(self, executor, sample_scene):
        """Test when no performers are identified."""
        identification_results = []

        tp, fn, fp = executor._compare_to_ground_truth(sample_scene, identification_results)

        assert tp == 0
        assert fn == 2  # Both expected performers missed
        assert fp == 0


class TestCountInTopN:
    """Tests for _count_in_top_n method."""

    def test_count_in_top_n_all_in_top_1(self, executor):
        """Test counting when expected performers are all rank 1."""
        results = [
            {"stashdb_id": "perf-1", "rank": 1},
            {"stashdb_id": "perf-2", "rank": 1},
        ]
        expected_ids = {"perf-1", "perf-2"}

        count = executor._count_in_top_n(results, expected_ids, n=1)

        assert count == 2

    def test_count_in_top_n_mixed_ranks(self, executor):
        """Test counting with mixed ranks."""
        results = [
            {"stashdb_id": "perf-1", "rank": 1},
            {"stashdb_id": "perf-2", "rank": 2},
            {"stashdb_id": "perf-3", "rank": 3},
            {"stashdb_id": "perf-4", "rank": 5},
        ]
        expected_ids = {"perf-1", "perf-2", "perf-3", "perf-4"}

        # Top 1
        count_top1 = executor._count_in_top_n(results, expected_ids, n=1)
        assert count_top1 == 1  # Only perf-1

        # Top 3
        count_top3 = executor._count_in_top_n(results, expected_ids, n=3)
        assert count_top3 == 3  # perf-1, perf-2, perf-3

    def test_count_in_top_n_some_not_in_expected(self, executor):
        """Test that results not in expected_ids are ignored."""
        results = [
            {"stashdb_id": "perf-1", "rank": 1},
            {"stashdb_id": "unexpected", "rank": 2},
            {"stashdb_id": "perf-2", "rank": 3},
        ]
        expected_ids = {"perf-1", "perf-2"}

        count = executor._count_in_top_n(results, expected_ids, n=3)

        assert count == 2  # perf-1 and perf-2

    def test_count_in_top_n_empty_results(self, executor):
        """Test with empty results."""
        results = []
        expected_ids = {"perf-1", "perf-2"}

        count = executor._count_in_top_n(results, expected_ids, n=3)

        assert count == 0

    def test_count_in_top_n_empty_expected(self, executor):
        """Test with empty expected set."""
        results = [
            {"stashdb_id": "perf-1", "rank": 1},
        ]
        expected_ids = set()

        count = executor._count_in_top_n(results, expected_ids, n=3)

        assert count == 0


class TestComputeScoreGap:
    """Tests for _compute_score_gap method."""

    def test_compute_score_gap(self, executor):
        """Test score gap calculation."""
        correct_scores = [0.8, 0.9, 0.7]  # avg = 0.8
        incorrect_scores = [0.4, 0.5, 0.3]  # avg = 0.4

        gap = executor._compute_score_gap(correct_scores, incorrect_scores)

        assert gap == pytest.approx(0.4, abs=0.001)  # 0.8 - 0.4

    def test_compute_score_gap_no_incorrect(self, executor):
        """Test score gap with no incorrect scores returns 0.0."""
        correct_scores = [0.8, 0.9]
        incorrect_scores = []

        gap = executor._compute_score_gap(correct_scores, incorrect_scores)

        assert gap == 0.0

    def test_compute_score_gap_no_correct(self, executor):
        """Test score gap with no correct scores returns 0.0."""
        correct_scores = []
        incorrect_scores = [0.4, 0.5]

        gap = executor._compute_score_gap(correct_scores, incorrect_scores)

        assert gap == 0.0

    def test_compute_score_gap_both_empty(self, executor):
        """Test score gap with both lists empty returns 0.0."""
        gap = executor._compute_score_gap([], [])

        assert gap == 0.0

    def test_compute_score_gap_negative(self, executor):
        """Test score gap can be negative if incorrect scores are higher."""
        correct_scores = [0.3]  # avg = 0.3
        incorrect_scores = [0.6]  # avg = 0.6

        gap = executor._compute_score_gap(correct_scores, incorrect_scores)

        assert gap == pytest.approx(-0.3, abs=0.001)


class TestIdentifySceneAsync:
    """Tests for async identify_scene method."""

    @pytest.mark.asyncio
    async def test_identify_scene_returns_scene_result(
        self, mock_recognizer, mock_multi_signal_matcher, sample_scene
    ):
        """Test that identify_scene returns a SceneResult with correct fields."""
        # Create executor with mocked _run_scene_identification
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        # Mock the internal method that runs identification
        mock_identification_result = {
            "performers": [
                {"stashdb_id": "perf-1-uuid", "rank": 1, "distance": 0.3},
                {"stashdb_id": "perf-2-uuid", "rank": 2, "distance": 0.4},
            ],
            "faces_detected": 50,
            "faces_after_filter": 45,
            "persons_clustered": 4,
        }

        with patch.object(
            executor,
            "_run_scene_identification",
            new_callable=AsyncMock,
            return_value=mock_identification_result,
        ):
            params = BenchmarkParams()
            result = await executor.identify_scene(sample_scene, params)

        # Verify result type and fields
        assert isinstance(result, SceneResult)
        assert result.scene_id == "scene-123"
        assert result.true_positives == 2
        assert result.false_negatives == 0
        assert result.false_positives == 0
        assert result.expected_in_top_1 >= 0
        assert result.expected_in_top_3 >= 0
        assert result.faces_detected == 50
        assert result.faces_after_filter == 45
        assert result.persons_clustered == 4
        assert result.elapsed_sec >= 0.0
        assert isinstance(result.params, dict)

    @pytest.mark.asyncio
    async def test_identify_scene_with_false_positives(
        self, mock_recognizer, mock_multi_signal_matcher, sample_scene
    ):
        """Test identify_scene correctly counts false positives."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        mock_identification_result = {
            "performers": [
                {"stashdb_id": "perf-1-uuid", "rank": 1, "distance": 0.3},
                {"stashdb_id": "wrong-person", "rank": 2, "distance": 0.4},
            ],
            "faces_detected": 30,
            "faces_after_filter": 25,
            "persons_clustered": 2,
        }

        with patch.object(
            executor,
            "_run_scene_identification",
            new_callable=AsyncMock,
            return_value=mock_identification_result,
        ):
            params = BenchmarkParams()
            result = await executor.identify_scene(sample_scene, params)

        assert result.true_positives == 1
        assert result.false_negatives == 1  # perf-2-uuid missed
        assert result.false_positives == 1  # wrong-person

    @pytest.mark.asyncio
    async def test_identify_scene_separates_correct_incorrect_scores(
        self, mock_recognizer, mock_multi_signal_matcher, sample_scene
    ):
        """Test that correct and incorrect match scores are separated."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        mock_identification_result = {
            "performers": [
                {"stashdb_id": "perf-1-uuid", "rank": 1, "distance": 0.3},
                {"stashdb_id": "perf-2-uuid", "rank": 2, "distance": 0.4},
                {"stashdb_id": "wrong-person", "rank": 3, "distance": 0.5},
            ],
            "faces_detected": 30,
            "faces_after_filter": 25,
            "persons_clustered": 3,
        }

        with patch.object(
            executor,
            "_run_scene_identification",
            new_callable=AsyncMock,
            return_value=mock_identification_result,
        ):
            params = BenchmarkParams()
            result = await executor.identify_scene(sample_scene, params)

        # Correct scores should have distances from perf-1-uuid and perf-2-uuid
        assert len(result.correct_match_scores) == 2
        assert 0.3 in result.correct_match_scores
        assert 0.4 in result.correct_match_scores

        # Incorrect scores should have distance from wrong-person
        assert len(result.incorrect_match_scores) == 1
        assert 0.5 in result.incorrect_match_scores


class TestRunBatch:
    """Tests for run_batch method."""

    @pytest.mark.asyncio
    async def test_run_batch_processes_all_scenes(
        self, mock_recognizer, mock_multi_signal_matcher, sample_performers
    ):
        """Test that run_batch processes all scenes and returns results."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        scenes = [
            TestScene(
                scene_id=f"scene-{i}",
                stashdb_id=f"stashdb-{i}",
                title=f"Test Scene {i}",
                resolution="1080p",
                width=1920,
                height=1080,
                duration_sec=600.0,
                expected_performers=sample_performers,
                db_coverage_tier="well-covered",
            )
            for i in range(3)
        ]

        # Mock identify_scene to return SceneResult
        async def mock_identify(scene, params):
            return SceneResult(
                scene_id=scene.scene_id,
                params=params.to_dict(),
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

        with patch.object(executor, "identify_scene", side_effect=mock_identify):
            params = BenchmarkParams()
            results = await executor.run_batch(scenes, params)

        assert len(results) == 3
        assert results[0].scene_id == "scene-0"
        assert results[1].scene_id == "scene-1"
        assert results[2].scene_id == "scene-2"

    @pytest.mark.asyncio
    async def test_run_batch_continues_after_exception(
        self, mock_recognizer, mock_multi_signal_matcher, sample_performers
    ):
        """Test that run_batch continues processing after an exception."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        scenes = [
            TestScene(
                scene_id=f"scene-{i}",
                stashdb_id=f"stashdb-{i}",
                title=f"Test Scene {i}",
                resolution="1080p",
                width=1920,
                height=1080,
                duration_sec=600.0,
                expected_performers=sample_performers,
                db_coverage_tier="well-covered",
            )
            for i in range(3)
        ]

        # Mock identify_scene to raise exception on second scene
        call_count = 0

        async def mock_identify(scene, params):
            nonlocal call_count
            call_count += 1
            if scene.scene_id == "scene-1":
                raise Exception("Test error")
            return SceneResult(
                scene_id=scene.scene_id,
                params=params.to_dict(),
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

        with patch.object(executor, "identify_scene", side_effect=mock_identify):
            params = BenchmarkParams()
            results = await executor.run_batch(scenes, params)

        # Should have 2 results (scenes 0 and 2), scene 1 failed
        assert len(results) == 2
        assert results[0].scene_id == "scene-0"
        assert results[1].scene_id == "scene-2"

    @pytest.mark.asyncio
    async def test_run_batch_empty_scenes(
        self, mock_recognizer, mock_multi_signal_matcher
    ):
        """Test run_batch with empty scenes list."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        params = BenchmarkParams()
        results = await executor.run_batch([], params)

        assert results == []


class TestRunSceneIdentification:
    """Tests for _run_scene_identification method."""

    @pytest.mark.asyncio
    async def test_run_scene_identification_uses_recognizer_when_multi_signal_disabled(
        self, mock_recognizer, mock_multi_signal_matcher, sample_scene
    ):
        """Test that recognizer is used when use_multi_signal is False."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        # Mock the recognizer's identify_scene method
        mock_recognizer.identify_scene = AsyncMock(
            return_value={
                "persons": [
                    {
                        "best_match": {
                            "stashdb_id": "stashdb.org:perf-1-uuid",
                            "distance": 0.3,
                        },
                        "person_id": 0,
                    }
                ],
                "faces_detected": 40,
                "faces_after_filter": 35,
            }
        )

        params = BenchmarkParams(use_multi_signal=False)
        result = await executor._run_scene_identification(sample_scene, params)

        mock_recognizer.identify_scene.assert_called_once()
        mock_multi_signal_matcher.identify_scene.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_scene_identification_uses_multi_signal_when_enabled(
        self, mock_recognizer, mock_multi_signal_matcher, sample_scene
    ):
        """Test that multi_signal_matcher is used when use_multi_signal is True."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        # Mock the multi_signal_matcher's identify_scene method
        mock_multi_signal_matcher.identify_scene = AsyncMock(
            return_value={
                "persons": [
                    {
                        "best_match": {
                            "stashdb_id": "stashdb.org:perf-1-uuid",
                            "distance": 0.3,
                        },
                        "person_id": 0,
                    }
                ],
                "faces_detected": 40,
                "faces_after_filter": 35,
            }
        )

        params = BenchmarkParams(use_multi_signal=True)
        result = await executor._run_scene_identification(sample_scene, params)

        mock_multi_signal_matcher.identify_scene.assert_called_once()
        mock_recognizer.identify_scene.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_scene_identification_extracts_stashdb_id_from_universal_id(
        self, mock_recognizer, mock_multi_signal_matcher, sample_scene
    ):
        """Test that stashdb_id is extracted from universal_id correctly."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        # Mock response with universal_id format
        mock_multi_signal_matcher.identify_scene = AsyncMock(
            return_value={
                "persons": [
                    {
                        "best_match": {
                            "stashdb_id": "stashdb.org:abc-123-def",
                            "distance": 0.3,
                        },
                        "person_id": 0,
                    }
                ],
                "faces_detected": 40,
                "faces_after_filter": 35,
            }
        )

        params = BenchmarkParams(use_multi_signal=True)
        result = await executor._run_scene_identification(sample_scene, params)

        # The stashdb_id should be extracted (taking last part after ":")
        assert len(result["performers"]) == 1
        assert result["performers"][0]["stashdb_id"] == "abc-123-def"

    @pytest.mark.asyncio
    async def test_run_scene_identification_builds_correct_request(
        self, mock_recognizer, mock_multi_signal_matcher, sample_scene
    ):
        """Test that request dict is built with correct parameters."""
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        mock_multi_signal_matcher.identify_scene = AsyncMock(
            return_value={
                "persons": [],
                "faces_detected": 0,
                "faces_after_filter": 0,
            }
        )

        params = BenchmarkParams(
            matching_mode="frequency",
            max_distance=0.6,
            min_face_size=50,
            num_frames=30,
            start_offset_pct=0.1,
            end_offset_pct=0.9,
            top_k=10,
        )

        await executor._run_scene_identification(sample_scene, params)

        # Verify the call arguments
        call_kwargs = mock_multi_signal_matcher.identify_scene.call_args[1]
        assert call_kwargs["scene_id"] == "scene-123"
        assert call_kwargs["num_frames"] == 30
        assert call_kwargs["start_offset_pct"] == 0.1
        assert call_kwargs["end_offset_pct"] == 0.9
        assert call_kwargs["matching_mode"] == "frequency"
        assert call_kwargs["max_distance"] == 0.6
        assert call_kwargs["min_face_size"] == 50
        assert call_kwargs["top_k"] == 10

"""Integration tests for benchmark framework.

Tests that all benchmark components (SceneSelector, TestExecutor, Analyzer,
Reporter, BenchmarkRunner) work together correctly with mocked dependencies.
"""

import os
import pytest
from unittest.mock import Mock, AsyncMock, MagicMock, patch

from benchmark.models import (
    ExpectedPerformer,
    TestScene,
    BenchmarkParams,
    SceneResult,
    BenchmarkState,
    AggregateMetrics,
)
from benchmark.scene_selector import SceneSelector, STASHDB_ENDPOINT
from benchmark.test_executor import TestExecutor
from benchmark.analyzer import Analyzer
from benchmark.reporter import Reporter
from benchmark.runner import BenchmarkRunner


# =============================================================================
# Shared Fixtures
# =============================================================================


@pytest.fixture
def mock_stash_response():
    """Mock response from Stash GraphQL API with 2 valid test scenes."""
    return {
        "findScenes": {
            "count": 2,
            "scenes": [
                {
                    "id": "123",
                    "title": "Test Scene 1",
                    "files": [{"width": 1920, "height": 1080, "duration": 1800}],
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-abc"}],
                    "performers": [
                        {
                            "name": "Performer A",
                            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-a"}],
                        },
                        {
                            "name": "Performer B",
                            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-b"}],
                        },
                    ],
                },
                {
                    "id": "456",
                    "title": "Test Scene 2",
                    "files": [{"width": 3840, "height": 2160, "duration": 2400}],
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-def"}],
                    "performers": [
                        {
                            "name": "Performer C",
                            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-c"}],
                        },
                        {
                            "name": "Performer D",
                            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-d"}],
                        },
                    ],
                },
            ],
        }
    }


@pytest.fixture
def mock_recognition_result():
    """Mock result from recognition system identifying performers."""
    return {
        "persons": [
            {
                "best_match": {
                    "stashdb_id": "stashdb.org:perf-a",
                    "name": "Performer A",
                    "distance": 0.32,
                },
                "person_id": 0,
            },
            {
                "best_match": {
                    "stashdb_id": "stashdb.org:perf-b",
                    "name": "Performer B",
                    "distance": 0.38,
                },
                "person_id": 1,
            },
        ],
        "faces_detected": 15,
        "faces_after_filter": 12,
    }


@pytest.fixture
def sample_expected_performers():
    """Create sample expected performers for test scenes."""
    return [
        ExpectedPerformer(
            stashdb_id="perf-a",
            name="Performer A",
            faces_in_db=10,
            has_body_data=True,
            has_tattoo_data=False,
        ),
        ExpectedPerformer(
            stashdb_id="perf-b",
            name="Performer B",
            faces_in_db=8,
            has_body_data=True,
            has_tattoo_data=True,
        ),
    ]


@pytest.fixture
def sample_test_scene(sample_expected_performers):
    """Create a sample test scene with expected performers."""
    return TestScene(
        scene_id="123",
        stashdb_id="scene-abc",
        title="Test Scene 1",
        resolution="1080p",
        width=1920,
        height=1080,
        duration_sec=1800.0,
        expected_performers=sample_expected_performers,
        db_coverage_tier="well-covered",
    )


# =============================================================================
# TestSceneSelectorIntegration
# =============================================================================


class TestSceneSelectorIntegration:
    """Integration tests for SceneSelector with mocked Stash client."""

    @pytest.mark.asyncio
    async def test_select_scenes_from_stash(self, mock_stash_response):
        """Test SceneSelector returns proper TestScene objects from Stash GraphQL."""
        # Create mock stash client with _query method
        mock_stash_client = MagicMock()
        mock_stash_client._query = MagicMock(return_value=mock_stash_response)

        # Create mock database reader
        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=10)
        mock_db_reader.has_body_data = MagicMock(return_value=True)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        # Create SceneSelector and select scenes
        selector = SceneSelector(mock_stash_client, mock_db_reader)
        scenes = await selector.select_scenes(min_count=2)

        # Verify results
        assert len(scenes) == 2

        # Verify first scene
        scene1 = scenes[0]
        assert isinstance(scene1, TestScene)
        assert scene1.scene_id == "123"
        assert scene1.stashdb_id == "scene-abc"
        assert scene1.title == "Test Scene 1"
        assert scene1.resolution == "1080p"
        assert scene1.width == 1920
        assert scene1.height == 1080
        assert scene1.duration_sec == 1800
        assert len(scene1.expected_performers) == 2

        # Verify first scene's performers
        assert scene1.expected_performers[0].stashdb_id == "perf-a"
        assert scene1.expected_performers[0].name == "Performer A"
        assert scene1.expected_performers[0].faces_in_db == 10
        assert scene1.expected_performers[1].stashdb_id == "perf-b"
        assert scene1.expected_performers[1].name == "Performer B"

        # Verify second scene
        scene2 = scenes[1]
        assert scene2.scene_id == "456"
        assert scene2.stashdb_id == "scene-def"
        assert scene2.resolution == "4k"  # 3840x2160
        assert scene2.width == 3840
        assert scene2.height == 2160

        # Verify Stash client was called
        mock_stash_client._query.assert_called()

        # Verify database reader was called for each performer
        assert mock_db_reader.get_face_count_for_performer.call_count == 4
        assert mock_db_reader.has_body_data.call_count == 4
        assert mock_db_reader.has_tattoo_data.call_count == 4

    @pytest.mark.asyncio
    async def test_select_scenes_with_coverage_tiers(self, mock_stash_response):
        """Test SceneSelector properly assigns coverage tiers based on face counts."""
        mock_stash_client = MagicMock()
        mock_stash_client._query = MagicMock(return_value=mock_stash_response)

        mock_db_reader = MagicMock()
        # First two performers (scene 1): well-covered (>= 5 faces)
        # Second two performers (scene 2): sparse (< 5 faces for at least one)
        call_count = [0]

        def get_face_count(universal_id):
            call_count[0] += 1
            # First 2 calls return >= 5, next 2 return mixed (sparse)
            if call_count[0] <= 2:
                return 10
            elif call_count[0] == 3:
                return 8
            else:
                return 3  # Below threshold

        mock_db_reader.get_face_count_for_performer = MagicMock(side_effect=get_face_count)
        mock_db_reader.has_body_data = MagicMock(return_value=False)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        selector = SceneSelector(mock_stash_client, mock_db_reader)
        scenes = await selector.select_scenes(min_count=2)

        assert len(scenes) == 2
        assert scenes[0].db_coverage_tier == "well-covered"
        assert scenes[1].db_coverage_tier == "sparse"

    @pytest.mark.asyncio
    async def test_select_scenes_filters_invalid_scenes(self):
        """Test SceneSelector filters out scenes that don't meet criteria."""
        # Create response with mix of valid and invalid scenes
        # Page 1 has the scenes
        response_page1 = {
            "findScenes": {
                "count": 3,
                "scenes": [
                    # Valid scene
                    {
                        "id": "valid-1",
                        "title": "Valid Scene",
                        "files": [{"width": 1920, "height": 1080, "duration": 1800}],
                        "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-valid"}],
                        "performers": [
                            {"name": "P1", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p1"}]},
                            {"name": "P2", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p2"}]},
                        ],
                    },
                    # Invalid: low resolution
                    {
                        "id": "invalid-low-res",
                        "title": "Low Res Scene",
                        "files": [{"width": 320, "height": 240, "duration": 600}],
                        "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-lowres"}],
                        "performers": [
                            {"name": "P3", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p3"}]},
                            {"name": "P4", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p4"}]},
                        ],
                    },
                    # Invalid: only 1 performer
                    {
                        "id": "invalid-solo",
                        "title": "Solo Scene",
                        "files": [{"width": 1920, "height": 1080, "duration": 1200}],
                        "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-solo"}],
                        "performers": [
                            {"name": "P5", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p5"}]},
                        ],
                    },
                ],
            }
        }

        # Page 2 is empty to signal no more scenes
        response_page2 = {
            "findScenes": {
                "count": 3,
                "scenes": [],
            }
        }

        mock_stash_client = MagicMock()
        # Return page 1 first, then empty page 2
        mock_stash_client._query = MagicMock(side_effect=[response_page1, response_page2])

        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=5)
        mock_db_reader.has_body_data = MagicMock(return_value=False)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        selector = SceneSelector(mock_stash_client, mock_db_reader)
        scenes = await selector.select_scenes(min_count=3)

        # Only the valid scene should be selected (pagination stops when page 2 is empty)
        assert len(scenes) == 1
        assert scenes[0].scene_id == "valid-1"


# =============================================================================
# TestExecutorIntegration
# =============================================================================


class TestExecutorIntegration:
    """Integration tests for TestExecutor with mocked recognizer."""

    @pytest.mark.asyncio
    async def test_identify_scene_with_recognizer(
        self, mock_recognition_result, sample_test_scene
    ):
        """Test TestExecutor properly computes TP/FN/FP from recognition results."""
        # Create mock recognizer
        mock_recognizer = MagicMock()
        mock_recognizer.identify_scene = AsyncMock(return_value=mock_recognition_result)

        # Create mock multi_signal_matcher
        mock_multi_signal_matcher = MagicMock()
        mock_multi_signal_matcher.identify_scene = AsyncMock(return_value=mock_recognition_result)

        # Create TestExecutor
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        # Run identification with multi-signal enabled (default)
        params = BenchmarkParams(use_multi_signal=True)
        result = await executor.identify_scene(sample_test_scene, params)

        # Verify result type
        assert isinstance(result, SceneResult)
        assert result.scene_id == "123"

        # Both expected performers (perf-a, perf-b) were found
        assert result.true_positives == 2
        assert result.false_negatives == 0
        assert result.false_positives == 0

        # Verify diagnostic info
        assert result.faces_detected == 15
        assert result.faces_after_filter == 12
        assert result.persons_clustered == 2
        assert result.elapsed_sec >= 0.0

        # Verify multi_signal_matcher was used
        mock_multi_signal_matcher.identify_scene.assert_called_once()
        mock_recognizer.identify_scene.assert_not_called()

    @pytest.mark.asyncio
    async def test_identify_scene_with_false_positives(self, sample_test_scene):
        """Test TestExecutor correctly counts false positives."""
        # Recognition result includes an unexpected performer
        recognition_result = {
            "persons": [
                {
                    "best_match": {
                        "stashdb_id": "stashdb.org:perf-a",
                        "distance": 0.30,
                    },
                    "person_id": 0,
                },
                {
                    "best_match": {
                        "stashdb_id": "stashdb.org:unexpected-performer",
                        "distance": 0.45,
                    },
                    "person_id": 1,
                },
            ],
            "faces_detected": 20,
            "faces_after_filter": 18,
        }

        mock_multi_signal_matcher = MagicMock()
        mock_multi_signal_matcher.identify_scene = AsyncMock(return_value=recognition_result)

        executor = TestExecutor(
            recognizer=MagicMock(),
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        params = BenchmarkParams(use_multi_signal=True)
        result = await executor.identify_scene(sample_test_scene, params)

        # perf-a found, perf-b missed, unexpected is false positive
        assert result.true_positives == 1
        assert result.false_negatives == 1
        assert result.false_positives == 1

    @pytest.mark.asyncio
    async def test_identify_scene_computes_score_gap(self, sample_test_scene):
        """Test TestExecutor correctly computes score gap between correct/incorrect."""
        recognition_result = {
            "persons": [
                {
                    "best_match": {
                        "stashdb_id": "stashdb.org:perf-a",
                        "distance": 0.30,  # Correct match
                    },
                    "person_id": 0,
                },
                {
                    "best_match": {
                        "stashdb_id": "stashdb.org:perf-b",
                        "distance": 0.35,  # Correct match
                    },
                    "person_id": 1,
                },
                {
                    "best_match": {
                        "stashdb_id": "stashdb.org:wrong-person",
                        "distance": 0.60,  # Incorrect match
                    },
                    "person_id": 2,
                },
            ],
            "faces_detected": 25,
            "faces_after_filter": 22,
        }

        mock_multi_signal_matcher = MagicMock()
        mock_multi_signal_matcher.identify_scene = AsyncMock(return_value=recognition_result)

        executor = TestExecutor(
            recognizer=MagicMock(),
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        params = BenchmarkParams(use_multi_signal=True)
        result = await executor.identify_scene(sample_test_scene, params)

        # Verify correct/incorrect score separation
        assert len(result.correct_match_scores) == 2
        assert 0.30 in result.correct_match_scores
        assert 0.35 in result.correct_match_scores

        assert len(result.incorrect_match_scores) == 1
        assert 0.60 in result.incorrect_match_scores

        # Score gap should be avg(correct) - avg(incorrect) = 0.325 - 0.60 = -0.275
        assert result.score_gap == pytest.approx(-0.275, abs=0.001)

    @pytest.mark.asyncio
    async def test_run_batch_processes_multiple_scenes(self, sample_expected_performers):
        """Test TestExecutor.run_batch processes multiple scenes."""
        scenes = [
            TestScene(
                scene_id=f"scene-{i}",
                stashdb_id=f"stashdb-{i}",
                title=f"Test Scene {i}",
                resolution="1080p",
                width=1920,
                height=1080,
                duration_sec=1200.0,
                expected_performers=sample_expected_performers,
                db_coverage_tier="well-covered",
            )
            for i in range(3)
        ]

        def make_result(scene_id):
            return {
                "persons": [
                    {
                        "best_match": {"stashdb_id": "stashdb.org:perf-a", "distance": 0.3},
                        "person_id": 0,
                    },
                    {
                        "best_match": {"stashdb_id": "stashdb.org:perf-b", "distance": 0.35},
                        "person_id": 1,
                    },
                ],
                "faces_detected": 20,
                "faces_after_filter": 18,
            }

        mock_multi_signal_matcher = MagicMock()
        mock_multi_signal_matcher.identify_scene = AsyncMock(
            side_effect=[make_result(f"scene-{i}") for i in range(3)]
        )

        executor = TestExecutor(
            recognizer=MagicMock(),
            multi_signal_matcher=mock_multi_signal_matcher,
        )

        params = BenchmarkParams()
        results = await executor.run_batch(scenes, params)

        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.scene_id == f"scene-{i}"
            assert result.true_positives == 2
            assert result.false_negatives == 0


# =============================================================================
# TestFullBenchmarkIntegration
# =============================================================================


class TestFullBenchmarkIntegration:
    """Integration tests for full benchmark flow with all mocked dependencies."""

    @pytest.mark.asyncio
    async def test_full_benchmark_run(self, tmp_path, sample_expected_performers):
        """Test full benchmark run with all components wired together."""
        # Create test scenes
        test_scenes = [
            TestScene(
                scene_id=f"scene-{i}",
                stashdb_id=f"stashdb-{i}",
                title=f"Test Scene {i}",
                resolution=["480p", "720p", "1080p", "4k", "1080p"][i],
                width=[854, 1280, 1920, 3840, 1920][i],
                height=[480, 720, 1080, 2160, 1080][i],
                duration_sec=1800.0,
                expected_performers=sample_expected_performers,
                db_coverage_tier="well-covered" if i < 3 else "sparse",
            )
            for i in range(5)
        ]

        # Create mock stash client
        mock_stash_client = MagicMock()

        # Create mock database reader
        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=10)
        mock_db_reader.has_body_data = MagicMock(return_value=True)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        # Create SceneSelector with pre-selected scenes
        scene_selector = SceneSelector(mock_stash_client, mock_db_reader)
        scene_selector.select_scenes = AsyncMock(return_value=test_scenes)
        scene_selector.sample_stratified = MagicMock(return_value=test_scenes[:2])

        # Create mock recognizer
        mock_recognizer = MagicMock()

        # Create mock multi_signal_matcher
        def make_recognition_result():
            return {
                "persons": [
                    {
                        "best_match": {"stashdb_id": "stashdb.org:perf-a", "distance": 0.32},
                        "person_id": 0,
                    },
                    {
                        "best_match": {"stashdb_id": "stashdb.org:perf-b", "distance": 0.38},
                        "person_id": 1,
                    },
                ],
                "faces_detected": 15,
                "faces_after_filter": 12,
            }

        mock_multi_signal_matcher = MagicMock()
        mock_multi_signal_matcher.identify_scene = AsyncMock(side_effect=lambda **kwargs: make_recognition_result())

        # Create real components
        executor = TestExecutor(
            recognizer=mock_recognizer,
            multi_signal_matcher=mock_multi_signal_matcher,
        )
        analyzer = Analyzer()
        reporter = Reporter()

        # Create output directory
        output_dir = str(tmp_path / "benchmark_output")

        # Create BenchmarkRunner
        runner = BenchmarkRunner(
            scene_selector=scene_selector,
            executor=executor,
            analyzer=analyzer,
            reporter=reporter,
            output_dir=output_dir,
        )

        # Run benchmark
        state = await runner.run(min_scenes=5, max_rounds=2, resume=False)

        # Verify state
        assert isinstance(state, BenchmarkState)
        assert state.round_num >= 1
        assert len(state.scenes) == 5

        # Verify results were recorded
        assert len(state.results_by_round) >= 1
        assert 1 in state.results_by_round

        # Verify first round results
        round_1_results = state.results_by_round[1]
        assert len(round_1_results) == 5  # All scenes processed in round 1

        # Verify each result
        for result in round_1_results:
            assert isinstance(result, SceneResult)
            assert result.true_positives == 2
            assert result.false_negatives == 0

        # Verify output files exist
        assert os.path.exists(output_dir)
        assert os.path.exists(os.path.join(output_dir, "scene_results.csv"))
        assert os.path.exists(os.path.join(output_dir, "checkpoint.json"))
        assert os.path.exists(os.path.join(output_dir, "summary.txt"))
        assert os.path.exists(os.path.join(output_dir, "final_report.md"))
        assert os.path.exists(os.path.join(output_dir, "recommendation.json"))

    @pytest.mark.asyncio
    async def test_full_benchmark_with_checkpoint_resume(
        self, tmp_path, sample_expected_performers
    ):
        """Test benchmark can save and resume from checkpoint."""
        test_scenes = [
            TestScene(
                scene_id=f"scene-{i}",
                stashdb_id=f"stashdb-{i}",
                title=f"Test Scene {i}",
                resolution="1080p",
                width=1920,
                height=1080,
                duration_sec=1800.0,
                expected_performers=sample_expected_performers,
                db_coverage_tier="well-covered",
            )
            for i in range(5)
        ]

        # Create initial state with round 1 complete
        initial_state = BenchmarkState(
            round_num=1,
            scenes=test_scenes,
            results_by_round={
                1: [
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
                        faces_detected=20,
                        faces_after_filter=18,
                        persons_clustered=2,
                        elapsed_sec=1.0,
                    )
                    for i in range(5)
                ]
            },
            current_best_params={"max_distance": 0.7, "min_face_size": 40, "use_multi_signal": True},
            current_best_accuracy=1.0,
            parameters_eliminated=[],
        )

        output_dir = str(tmp_path / "benchmark_checkpoint")
        os.makedirs(output_dir, exist_ok=True)

        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, "checkpoint.json")
        with open(checkpoint_path, "w") as f:
            f.write(initial_state.to_json())

        # Create mocked components
        mock_stash_client = MagicMock()
        mock_db_reader = MagicMock()
        scene_selector = SceneSelector(mock_stash_client, mock_db_reader)
        scene_selector.select_scenes = AsyncMock(return_value=test_scenes)
        scene_selector.sample_stratified = MagicMock(return_value=test_scenes[:2])

        def make_recognition_result():
            return {
                "persons": [
                    {"best_match": {"stashdb_id": "stashdb.org:perf-a", "distance": 0.32}, "person_id": 0},
                    {"best_match": {"stashdb_id": "stashdb.org:perf-b", "distance": 0.38}, "person_id": 1},
                ],
                "faces_detected": 15,
                "faces_after_filter": 12,
            }

        mock_multi_signal_matcher = MagicMock()
        mock_multi_signal_matcher.identify_scene = AsyncMock(side_effect=lambda **kwargs: make_recognition_result())

        executor = TestExecutor(
            recognizer=MagicMock(),
            multi_signal_matcher=mock_multi_signal_matcher,
        )
        analyzer = Analyzer()
        reporter = Reporter()

        runner = BenchmarkRunner(
            scene_selector=scene_selector,
            executor=executor,
            analyzer=analyzer,
            reporter=reporter,
            output_dir=output_dir,
        )

        # Run with resume=True
        state = await runner.run(min_scenes=5, max_rounds=2, resume=True)

        # Should have resumed from round 1 and potentially run round 2
        assert state.round_num >= 1
        assert len(state.scenes) == 5
        assert len(state.results_by_round) >= 1

    @pytest.mark.asyncio
    async def test_analyzer_integration_with_scene_results(self, sample_expected_performers):
        """Test Analyzer computes correct aggregate metrics from SceneResults."""
        # Create scene results with varying accuracy
        results = [
            SceneResult(
                scene_id="scene-1",
                params={"max_distance": 0.7},
                true_positives=2,  # Both found
                false_negatives=0,
                false_positives=0,
                expected_in_top_1=2,
                expected_in_top_3=2,
                correct_match_scores=[0.3, 0.35],
                incorrect_match_scores=[],
                score_gap=0.0,
                faces_detected=20,
                faces_after_filter=18,
                persons_clustered=2,
                elapsed_sec=1.0,
            ),
            SceneResult(
                scene_id="scene-2",
                params={"max_distance": 0.7},
                true_positives=1,  # One found
                false_negatives=1,  # One missed
                false_positives=1,  # One false positive
                expected_in_top_1=1,
                expected_in_top_3=1,
                correct_match_scores=[0.32],
                incorrect_match_scores=[0.55],
                score_gap=-0.23,
                faces_detected=25,
                faces_after_filter=22,
                persons_clustered=2,
                elapsed_sec=1.2,
            ),
            SceneResult(
                scene_id="scene-3",
                params={"max_distance": 0.7},
                true_positives=0,  # None found
                false_negatives=2,  # Both missed
                false_positives=0,
                expected_in_top_1=0,
                expected_in_top_3=0,
                correct_match_scores=[],
                incorrect_match_scores=[],
                score_gap=0.0,
                faces_detected=10,
                faces_after_filter=8,
                persons_clustered=0,
                elapsed_sec=0.8,
            ),
        ]

        # Create test scenes for resolution/coverage breakdown
        scenes = [
            TestScene(
                scene_id="scene-1",
                stashdb_id="sd-1",
                title="1080p Scene",
                resolution="1080p",
                width=1920,
                height=1080,
                duration_sec=1800,
                expected_performers=sample_expected_performers,
                db_coverage_tier="well-covered",
            ),
            TestScene(
                scene_id="scene-2",
                stashdb_id="sd-2",
                title="720p Scene",
                resolution="720p",
                width=1280,
                height=720,
                duration_sec=1200,
                expected_performers=sample_expected_performers,
                db_coverage_tier="sparse",
            ),
            TestScene(
                scene_id="scene-3",
                stashdb_id="sd-3",
                title="4k Scene",
                resolution="4k",
                width=3840,
                height=2160,
                duration_sec=2400,
                expected_performers=sample_expected_performers,
                db_coverage_tier="sparse",
            ),
        ]

        analyzer = Analyzer()

        # Compute aggregate metrics
        metrics = analyzer.compute_aggregate_metrics(results)

        assert metrics.total_scenes == 3
        assert metrics.total_expected == 6  # 2 + 2 + 2 expected performers
        assert metrics.total_true_positives == 3  # 2 + 1 + 0
        assert metrics.total_false_negatives == 3  # 0 + 1 + 2
        assert metrics.total_false_positives == 1  # 0 + 1 + 0
        assert metrics.accuracy == pytest.approx(0.5, abs=0.01)  # 3/6 = 0.5

        # Compute accuracy by resolution
        accuracy_by_res = analyzer.compute_accuracy_by_resolution(scenes, results)
        assert "1080p" in accuracy_by_res
        assert "720p" in accuracy_by_res
        assert "4k" in accuracy_by_res
        assert accuracy_by_res["1080p"] == pytest.approx(1.0, abs=0.01)  # 2/2
        assert accuracy_by_res["720p"] == pytest.approx(0.5, abs=0.01)  # 1/2
        assert accuracy_by_res["4k"] == pytest.approx(0.0, abs=0.01)  # 0/2

        # Compute accuracy by coverage
        accuracy_by_cov = analyzer.compute_accuracy_by_coverage(scenes, results)
        assert "well-covered" in accuracy_by_cov
        assert "sparse" in accuracy_by_cov
        assert accuracy_by_cov["well-covered"] == pytest.approx(1.0, abs=0.01)  # 2/2
        assert accuracy_by_cov["sparse"] == pytest.approx(0.25, abs=0.01)  # 1/4

    @pytest.mark.asyncio
    async def test_reporter_integration_with_results(self, tmp_path, sample_expected_performers):
        """Test Reporter generates correct outputs from benchmark results."""
        results = [
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
                faces_detected=20,
                faces_after_filter=18,
                persons_clustered=2,
                elapsed_sec=1.0,
            )
            for i in range(5)
        ]

        scenes = [
            TestScene(
                scene_id=f"scene-{i}",
                stashdb_id=f"sd-{i}",
                title=f"Scene {i}",
                resolution="1080p",
                width=1920,
                height=1080,
                duration_sec=1800,
                expected_performers=sample_expected_performers,
                db_coverage_tier="well-covered",
            )
            for i in range(5)
        ]

        state = BenchmarkState(
            round_num=2,
            scenes=scenes,
            results_by_round={1: results, 2: results},
            current_best_params={"max_distance": 0.7, "min_face_size": 40},
            current_best_accuracy=1.0,
            parameters_eliminated=["max_distance=0.5"],
        )

        metrics = AggregateMetrics(
            total_scenes=5,
            total_expected=10,
            total_true_positives=10,
            total_false_positives=0,
            total_false_negatives=0,
            accuracy=1.0,
            precision=1.0,
            recall=1.0,
            accuracy_by_resolution={"1080p": 1.0},
            accuracy_by_coverage={"well-covered": 1.0},
        )

        reporter = Reporter()

        # Test CSV export
        csv_path = str(tmp_path / "results.csv")
        reporter.export_scene_results_csv(results, csv_path)
        assert os.path.exists(csv_path)

        # Verify CSV content
        with open(csv_path, "r") as f:
            content = f.read()
            assert "scene_id" in content
            assert "scene-0" in content
            assert "true_positives" in content

        # Test summary generation
        summary = reporter.generate_summary(metrics)
        assert "=== Benchmark Summary ===" in summary
        assert "Scenes tested: 5" in summary
        assert "accuracy: 100.0%" in summary.lower()

        # Test final report generation
        report = reporter.generate_final_report(state, metrics)
        assert "# Benchmark Final Report" in report
        assert "Rounds completed: 2" in report
        assert "Accuracy: 100.0%" in report

        # Test checkpoint save/load
        checkpoint_path = str(tmp_path / "checkpoint.json")
        reporter.save_checkpoint(state, checkpoint_path)
        assert os.path.exists(checkpoint_path)

        loaded_state = reporter.load_checkpoint(checkpoint_path)
        assert loaded_state is not None
        assert loaded_state.round_num == 2
        assert len(loaded_state.scenes) == 5
        assert len(loaded_state.results_by_round) == 2

"""Tests for benchmark scene selector."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import random

from api.benchmark.scene_selector import SceneSelector, STASHDB_ENDPOINT
from api.benchmark.models import ExpectedPerformer, TestScene


class TestResolutionTier:
    """Tests for resolution tier classification.

    Resolution tier is determined by max(width, height):
    - >= 2160 -> "4k"
    - >= 1080 -> "1080p"
    - >= 720 -> "720p"
    - else -> "480p"
    """

    def test_resolution_tier_480p(self):
        """Test 480p classification for low resolution (max < 720)."""
        selector = SceneSelector(MagicMock(), MagicMock())
        # max(640, 480) = 640 < 720 -> 480p
        assert selector._get_resolution_tier(640, 480) == "480p"
        # max(719, 400) = 719 < 720 -> 480p
        assert selector._get_resolution_tier(719, 400) == "480p"
        # max(500, 500) = 500 < 720 -> 480p
        assert selector._get_resolution_tier(500, 500) == "480p"

    def test_resolution_tier_720p(self):
        """Test 720p classification (720 <= max < 1080)."""
        selector = SceneSelector(MagicMock(), MagicMock())
        # max(720, 480) = 720 -> 720p (exactly at threshold)
        assert selector._get_resolution_tier(720, 480) == "720p"
        # max(960, 540) = 960 -> 720p
        assert selector._get_resolution_tier(960, 540) == "720p"
        # max(1079, 720) = 1079 < 1080 -> 720p
        assert selector._get_resolution_tier(1079, 720) == "720p"

    def test_resolution_tier_1080p(self):
        """Test 1080p classification (1080 <= max < 2160)."""
        selector = SceneSelector(MagicMock(), MagicMock())
        # max(1920, 1080) = 1920 -> 1080p
        assert selector._get_resolution_tier(1920, 1080) == "1080p"
        # max(1080, 1920) = 1920 -> 1080p (portrait)
        assert selector._get_resolution_tier(1080, 1920) == "1080p"
        # max(1280, 720) = 1280 >= 1080 -> 1080p
        assert selector._get_resolution_tier(1280, 720) == "1080p"
        # max(2159, 1080) = 2159 < 2160 -> 1080p
        assert selector._get_resolution_tier(2159, 1080) == "1080p"

    def test_resolution_tier_4k(self):
        """Test 4k classification (max >= 2160)."""
        selector = SceneSelector(MagicMock(), MagicMock())
        # max(3840, 2160) = 3840 -> 4k
        assert selector._get_resolution_tier(3840, 2160) == "4k"
        # max(2160, 3840) = 3840 -> 4k (portrait)
        assert selector._get_resolution_tier(2160, 3840) == "4k"
        # max(2160, 2160) = 2160 -> 4k (exactly at threshold)
        assert selector._get_resolution_tier(2160, 2160) == "4k"
        # max(4096, 2160) = 4096 -> 4k
        assert selector._get_resolution_tier(4096, 2160) == "4k"


class TestCoverageTier:
    """Tests for coverage tier classification."""

    def test_coverage_tier_well_covered(self):
        """Test well-covered when all performers have >= 5 faces."""
        selector = SceneSelector(MagicMock(), MagicMock())
        performers = [
            ExpectedPerformer(
                stashdb_id="p1",
                name="Performer 1",
                faces_in_db=5,
                has_body_data=True,
                has_tattoo_data=False,
            ),
            ExpectedPerformer(
                stashdb_id="p2",
                name="Performer 2",
                faces_in_db=10,
                has_body_data=False,
                has_tattoo_data=True,
            ),
        ]
        assert selector._get_coverage_tier(performers) == "well-covered"

    def test_coverage_tier_sparse(self):
        """Test sparse when any performer has < 5 faces."""
        selector = SceneSelector(MagicMock(), MagicMock())
        performers = [
            ExpectedPerformer(
                stashdb_id="p1",
                name="Performer 1",
                faces_in_db=5,
                has_body_data=True,
                has_tattoo_data=False,
            ),
            ExpectedPerformer(
                stashdb_id="p2",
                name="Performer 2",
                faces_in_db=4,  # Below threshold
                has_body_data=False,
                has_tattoo_data=True,
            ),
        ]
        assert selector._get_coverage_tier(performers) == "sparse"

    def test_coverage_tier_sparse_single_performer(self):
        """Test sparse with single performer below threshold."""
        selector = SceneSelector(MagicMock(), MagicMock())
        performers = [
            ExpectedPerformer(
                stashdb_id="p1",
                name="Performer 1",
                faces_in_db=0,
                has_body_data=False,
                has_tattoo_data=False,
            ),
        ]
        assert selector._get_coverage_tier(performers) == "sparse"

    def test_coverage_tier_well_covered_boundary(self):
        """Test exactly at threshold is well-covered."""
        selector = SceneSelector(MagicMock(), MagicMock())
        performers = [
            ExpectedPerformer(
                stashdb_id="p1",
                name="Performer 1",
                faces_in_db=5,  # Exactly at threshold
                has_body_data=True,
                has_tattoo_data=False,
            ),
        ]
        assert selector._get_coverage_tier(performers) == "well-covered"


class TestBuildSceneQuery:
    """Tests for GraphQL query building."""

    @pytest.mark.asyncio
    async def test_build_scene_query(self):
        """Test that query contains expected fields."""
        selector = SceneSelector(MagicMock(), MagicMock())
        query = selector._build_scene_query(page=1, per_page=25)

        # Check for expected fields in query
        assert "findScenes" in query
        assert "id" in query
        assert "title" in query
        assert "files" in query
        assert "width" in query
        assert "height" in query
        assert "duration" in query
        assert "stash_ids" in query
        assert "endpoint" in query
        assert "stash_id" in query
        assert "performers" in query
        assert "name" in query


class TestGetStashdbId:
    """Tests for extracting StashDB ID."""

    def test_get_stashdb_id_found(self):
        """Test extracting StashDB ID when present."""
        selector = SceneSelector(MagicMock(), MagicMock())
        stash_ids = [
            {"endpoint": "https://other.com", "stash_id": "other-id"},
            {"endpoint": STASHDB_ENDPOINT, "stash_id": "stashdb-uuid"},
        ]
        assert selector._get_stashdb_id(stash_ids) == "stashdb-uuid"

    def test_get_stashdb_id_not_found(self):
        """Test returning None when StashDB ID not present."""
        selector = SceneSelector(MagicMock(), MagicMock())
        stash_ids = [
            {"endpoint": "https://other.com", "stash_id": "other-id"},
        ]
        assert selector._get_stashdb_id(stash_ids) is None

    def test_get_stashdb_id_empty_list(self):
        """Test returning None for empty list."""
        selector = SceneSelector(MagicMock(), MagicMock())
        assert selector._get_stashdb_id([]) is None


class TestIsValidTestScene:
    """Tests for scene validation."""

    def test_filter_scene_valid(self):
        """Test valid scene passes all checks."""
        selector = SceneSelector(MagicMock(), MagicMock())
        scene_data = {
            "id": "scene-1",
            "title": "Test Scene",
            "files": [{"width": 1920, "height": 1080, "duration": 1800}],
            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-stashdb-id"}],
            "performers": [
                {
                    "name": "Performer 1",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-1-id"}],
                },
                {
                    "name": "Performer 2",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-2-id"}],
                },
            ],
        }
        assert selector._is_valid_test_scene(scene_data) is True

    def test_filter_scene_too_few_performers(self):
        """Test scene with fewer than 2 performers is invalid."""
        selector = SceneSelector(MagicMock(), MagicMock())
        scene_data = {
            "id": "scene-1",
            "title": "Solo Scene",
            "files": [{"width": 1920, "height": 1080, "duration": 1800}],
            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-stashdb-id"}],
            "performers": [
                {
                    "name": "Performer 1",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-1-id"}],
                },
            ],
        }
        assert selector._is_valid_test_scene(scene_data) is False

    def test_filter_scene_no_stashdb_id(self):
        """Test scene without StashDB ID is invalid."""
        selector = SceneSelector(MagicMock(), MagicMock())
        scene_data = {
            "id": "scene-1",
            "title": "Test Scene",
            "files": [{"width": 1920, "height": 1080, "duration": 1800}],
            "stash_ids": [{"endpoint": "https://other.com", "stash_id": "other-id"}],
            "performers": [
                {
                    "name": "Performer 1",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-1-id"}],
                },
                {
                    "name": "Performer 2",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-2-id"}],
                },
            ],
        }
        assert selector._is_valid_test_scene(scene_data) is False

    def test_filter_scene_low_resolution(self):
        """Test scene with low resolution is invalid."""
        selector = SceneSelector(MagicMock(), MagicMock())
        scene_data = {
            "id": "scene-1",
            "title": "Low Res Scene",
            "files": [{"width": 640, "height": 360, "duration": 1800}],
            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-stashdb-id"}],
            "performers": [
                {
                    "name": "Performer 1",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-1-id"}],
                },
                {
                    "name": "Performer 2",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-2-id"}],
                },
            ],
        }
        assert selector._is_valid_test_scene(scene_data) is False

    def test_filter_scene_no_files(self):
        """Test scene without file info is invalid."""
        selector = SceneSelector(MagicMock(), MagicMock())
        scene_data = {
            "id": "scene-1",
            "title": "No Files Scene",
            "files": [],
            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-stashdb-id"}],
            "performers": [
                {
                    "name": "Performer 1",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-1-id"}],
                },
                {
                    "name": "Performer 2",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-2-id"}],
                },
            ],
        }
        assert selector._is_valid_test_scene(scene_data) is False

    def test_filter_scene_performer_no_stashdb_id(self):
        """Test scene where a performer lacks StashDB ID is invalid."""
        selector = SceneSelector(MagicMock(), MagicMock())
        scene_data = {
            "id": "scene-1",
            "title": "Test Scene",
            "files": [{"width": 1920, "height": 1080, "duration": 1800}],
            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "scene-stashdb-id"}],
            "performers": [
                {
                    "name": "Performer 1",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-1-id"}],
                },
                {
                    "name": "Performer 2",
                    "stash_ids": [{"endpoint": "https://other.com", "stash_id": "other-id"}],
                },
            ],
        }
        assert selector._is_valid_test_scene(scene_data) is False


class TestGetPerformerDbCoverage:
    """Tests for performer database coverage lookup."""

    @pytest.mark.asyncio
    async def test_get_performer_db_coverage(self):
        """Test getting performer database coverage."""
        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=10)
        mock_db_reader.has_body_data = MagicMock(return_value=True)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        selector = SceneSelector(MagicMock(), mock_db_reader)
        coverage = await selector._get_performer_db_coverage("test-uuid")

        assert coverage["faces_in_db"] == 10
        assert coverage["has_body_data"] is True
        assert coverage["has_tattoo_data"] is False

        # Verify universal_id format
        mock_db_reader.get_face_count_for_performer.assert_called_once_with(
            "stashdb.org:test-uuid"
        )
        mock_db_reader.has_body_data.assert_called_once_with("stashdb.org:test-uuid")
        mock_db_reader.has_tattoo_data.assert_called_once_with("stashdb.org:test-uuid")

    @pytest.mark.asyncio
    async def test_get_performer_db_coverage_none_face_count(self):
        """Test handling None face count."""
        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=None)
        mock_db_reader.has_body_data = MagicMock(return_value=False)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        selector = SceneSelector(MagicMock(), mock_db_reader)
        coverage = await selector._get_performer_db_coverage("test-uuid")

        assert coverage["faces_in_db"] == 0


class TestConvertToTestScene:
    """Tests for converting scene data to TestScene."""

    @pytest.mark.asyncio
    async def test_convert_to_test_scene(self):
        """Test converting scene data to TestScene object."""
        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=8)
        mock_db_reader.has_body_data = MagicMock(return_value=True)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        selector = SceneSelector(MagicMock(), mock_db_reader)

        scene_data = {
            "id": "scene-123",
            "title": "Test Scene Title",
            "files": [{"width": 1920, "height": 1080, "duration": 1800.5}],
            "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "stashdb-scene-456"}],
            "performers": [
                {
                    "name": "Performer One",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-1-uuid"}],
                },
                {
                    "name": "Performer Two",
                    "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "perf-2-uuid"}],
                },
            ],
        }

        test_scene = await selector._convert_to_test_scene(scene_data)

        assert test_scene.scene_id == "scene-123"
        assert test_scene.stashdb_id == "stashdb-scene-456"
        assert test_scene.title == "Test Scene Title"
        assert test_scene.resolution == "1080p"
        assert test_scene.width == 1920
        assert test_scene.height == 1080
        assert test_scene.duration_sec == 1800.5
        assert len(test_scene.expected_performers) == 2
        assert test_scene.expected_performers[0].stashdb_id == "perf-1-uuid"
        assert test_scene.expected_performers[0].name == "Performer One"
        assert test_scene.expected_performers[0].faces_in_db == 8
        assert test_scene.db_coverage_tier == "well-covered"


class TestSelectScenes:
    """Tests for scene selection with pagination."""

    @pytest.mark.asyncio
    async def test_select_scenes_single_page(self):
        """Test selecting scenes from a single page."""
        mock_stash_client = MagicMock()
        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=5)
        mock_db_reader.has_body_data = MagicMock(return_value=False)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        # Mock response with 2 valid scenes
        scenes_response = {
            "findScenes": {
                "count": 2,
                "scenes": [
                    {
                        "id": "scene-1",
                        "title": "Scene 1",
                        "files": [{"width": 1920, "height": 1080, "duration": 1800}],
                        "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "s1-uuid"}],
                        "performers": [
                            {"name": "P1", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p1-uuid"}]},
                            {"name": "P2", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p2-uuid"}]},
                        ],
                    },
                    {
                        "id": "scene-2",
                        "title": "Scene 2",
                        "files": [{"width": 1280, "height": 720, "duration": 1200}],
                        "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "s2-uuid"}],
                        "performers": [
                            {"name": "P3", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p3-uuid"}]},
                            {"name": "P4", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": "p4-uuid"}]},
                        ],
                    },
                ],
            }
        }

        mock_stash_client._query = MagicMock(return_value=scenes_response)

        selector = SceneSelector(mock_stash_client, mock_db_reader)
        scenes = await selector.select_scenes(min_count=2)

        assert len(scenes) == 2
        assert scenes[0].scene_id == "scene-1"
        assert scenes[1].scene_id == "scene-2"

    @pytest.mark.asyncio
    async def test_select_scenes_stops_at_min_count(self):
        """Test that selection stops when min_count is reached."""
        mock_stash_client = MagicMock()
        mock_db_reader = MagicMock()
        mock_db_reader.get_face_count_for_performer = MagicMock(return_value=5)
        mock_db_reader.has_body_data = MagicMock(return_value=False)
        mock_db_reader.has_tattoo_data = MagicMock(return_value=False)

        # Mock response with more scenes than needed
        scenes_response = {
            "findScenes": {
                "count": 100,
                "scenes": [
                    {
                        "id": f"scene-{i}",
                        "title": f"Scene {i}",
                        "files": [{"width": 1920, "height": 1080, "duration": 1800}],
                        "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": f"s{i}-uuid"}],
                        "performers": [
                            {"name": f"P{i}a", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": f"p{i}a-uuid"}]},
                            {"name": f"P{i}b", "stash_ids": [{"endpoint": STASHDB_ENDPOINT, "stash_id": f"p{i}b-uuid"}]},
                        ],
                    }
                    for i in range(10)
                ],
            }
        }

        mock_stash_client._query = MagicMock(return_value=scenes_response)

        selector = SceneSelector(mock_stash_client, mock_db_reader)
        scenes = await selector.select_scenes(min_count=5)

        assert len(scenes) == 5


class TestStratifyScenes:
    """Tests for stratifying scenes by resolution."""

    def test_stratify_by_resolution(self):
        """Test stratifying scenes into resolution tiers."""
        selector = SceneSelector(MagicMock(), MagicMock())

        scenes = [
            TestScene(
                scene_id="s1", stashdb_id="sd1", title="480p Scene",
                resolution="480p", width=854, height=480, duration_sec=600,
                expected_performers=[], db_coverage_tier="sparse"
            ),
            TestScene(
                scene_id="s2", stashdb_id="sd2", title="720p Scene",
                resolution="720p", width=1280, height=720, duration_sec=600,
                expected_performers=[], db_coverage_tier="sparse"
            ),
            TestScene(
                scene_id="s3", stashdb_id="sd3", title="1080p Scene",
                resolution="1080p", width=1920, height=1080, duration_sec=600,
                expected_performers=[], db_coverage_tier="sparse"
            ),
            TestScene(
                scene_id="s4", stashdb_id="sd4", title="4k Scene",
                resolution="4k", width=3840, height=2160, duration_sec=600,
                expected_performers=[], db_coverage_tier="sparse"
            ),
            TestScene(
                scene_id="s5", stashdb_id="sd5", title="Another 1080p Scene",
                resolution="1080p", width=1920, height=1080, duration_sec=600,
                expected_performers=[], db_coverage_tier="sparse"
            ),
        ]

        stratified = selector.stratify_scenes(scenes)

        assert len(stratified["480p"]) == 1
        assert len(stratified["720p"]) == 1
        assert len(stratified["1080p"]) == 2
        assert len(stratified["4k"]) == 1

        assert stratified["480p"][0].scene_id == "s1"
        assert stratified["4k"][0].scene_id == "s4"

    def test_stratify_empty_scenes(self):
        """Test stratifying empty scene list."""
        selector = SceneSelector(MagicMock(), MagicMock())
        stratified = selector.stratify_scenes([])

        assert stratified["480p"] == []
        assert stratified["720p"] == []
        assert stratified["1080p"] == []
        assert stratified["4k"] == []


class TestSampleStratified:
    """Tests for stratified sampling."""

    def test_sample_stratified(self):
        """Test stratified sampling distributes evenly."""
        selector = SceneSelector(MagicMock(), MagicMock())

        # Create 12 scenes: 3 per tier
        scenes = []
        for i, res in enumerate(["480p", "720p", "1080p", "4k"]):
            for j in range(3):
                width, height = {
                    "480p": (854, 480),
                    "720p": (1280, 720),
                    "1080p": (1920, 1080),
                    "4k": (3840, 2160),
                }[res]
                scenes.append(TestScene(
                    scene_id=f"s{i}-{j}", stashdb_id=f"sd{i}-{j}", title=f"{res} Scene {j}",
                    resolution=res, width=width, height=height, duration_sec=600,
                    expected_performers=[], db_coverage_tier="sparse"
                ))

        # Sample 8 scenes (2 per tier)
        sampled = selector.sample_stratified(scenes, count=8, seed=42)

        assert len(sampled) == 8

        # Count by resolution
        counts = {"480p": 0, "720p": 0, "1080p": 0, "4k": 0}
        for s in sampled:
            counts[s.resolution] += 1

        # Should be evenly distributed
        assert counts["480p"] == 2
        assert counts["720p"] == 2
        assert counts["1080p"] == 2
        assert counts["4k"] == 2

    def test_sample_stratified_with_remainder(self):
        """Test stratified sampling distributes remainder to first tiers."""
        selector = SceneSelector(MagicMock(), MagicMock())

        # Create 12 scenes: 3 per tier
        scenes = []
        for i, res in enumerate(["480p", "720p", "1080p", "4k"]):
            for j in range(3):
                width, height = {
                    "480p": (854, 480),
                    "720p": (1280, 720),
                    "1080p": (1920, 1080),
                    "4k": (3840, 2160),
                }[res]
                scenes.append(TestScene(
                    scene_id=f"s{i}-{j}", stashdb_id=f"sd{i}-{j}", title=f"{res} Scene {j}",
                    resolution=res, width=width, height=height, duration_sec=600,
                    expected_performers=[], db_coverage_tier="sparse"
                ))

        # Sample 10 scenes: 2 per tier + 2 remainder
        sampled = selector.sample_stratified(scenes, count=10, seed=42)

        assert len(sampled) == 10

    def test_sample_stratified_reproducible_with_seed(self):
        """Test that same seed produces same results."""
        selector = SceneSelector(MagicMock(), MagicMock())

        scenes = []
        for i, res in enumerate(["480p", "720p", "1080p", "4k"]):
            for j in range(5):
                width, height = {
                    "480p": (854, 480),
                    "720p": (1280, 720),
                    "1080p": (1920, 1080),
                    "4k": (3840, 2160),
                }[res]
                scenes.append(TestScene(
                    scene_id=f"s{i}-{j}", stashdb_id=f"sd{i}-{j}", title=f"{res} Scene {j}",
                    resolution=res, width=width, height=height, duration_sec=600,
                    expected_performers=[], db_coverage_tier="sparse"
                ))

        sample1 = selector.sample_stratified(scenes, count=8, seed=123)
        sample2 = selector.sample_stratified(scenes, count=8, seed=123)

        # Same seed should produce same result
        assert [s.scene_id for s in sample1] == [s.scene_id for s in sample2]

    def test_sample_stratified_fewer_scenes_than_requested(self):
        """Test sampling when fewer scenes available than requested."""
        selector = SceneSelector(MagicMock(), MagicMock())

        scenes = [
            TestScene(
                scene_id="s1", stashdb_id="sd1", title="1080p Scene",
                resolution="1080p", width=1920, height=1080, duration_sec=600,
                expected_performers=[], db_coverage_tier="sparse"
            ),
            TestScene(
                scene_id="s2", stashdb_id="sd2", title="4k Scene",
                resolution="4k", width=3840, height=2160, duration_sec=600,
                expected_performers=[], db_coverage_tier="sparse"
            ),
        ]

        sampled = selector.sample_stratified(scenes, count=10, seed=42)

        # Should return all available scenes
        assert len(sampled) == 2

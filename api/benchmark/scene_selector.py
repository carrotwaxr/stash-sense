"""Scene selector for benchmark framework.

This module provides functionality to query Stash for test scenes and
perform stratified sampling by resolution tier.
"""

import random
from typing import Optional

from benchmark.models import ExpectedPerformer, TestScene


# Constants
STASHDB_ENDPOINT = "https://stashdb.org"
MIN_RESOLUTION_WIDTH = 854
MIN_RESOLUTION_HEIGHT = 480
MIN_PERFORMERS = 2
WELL_COVERED_THRESHOLD = 5


class SceneSelector:
    """Selects scenes from Stash for benchmarking.

    Queries Stash for scenes meeting benchmark criteria and performs
    stratified sampling by resolution tier.
    """

    def __init__(self, stash_client, database_reader):
        """Initialize the scene selector.

        Args:
            stash_client: Client for Stash GraphQL API
            database_reader: Reader for performer database (faces counts, etc.)
        """
        self.stash_client = stash_client
        self.database_reader = database_reader

    def _get_resolution_tier(self, width: int, height: int) -> str:
        """Classify resolution into tier based on max dimension.

        Args:
            width: Video width in pixels
            height: Video height in pixels

        Returns:
            Resolution tier: "480p", "720p", "1080p", or "4k"
        """
        max_dim = max(width, height)

        if max_dim >= 2160:
            return "4k"
        elif max_dim >= 1080:
            return "1080p"
        elif max_dim >= 720:
            return "720p"
        else:
            return "480p"

    def _get_coverage_tier(self, performers: list[ExpectedPerformer]) -> str:
        """Determine coverage tier based on performer face counts.

        Args:
            performers: List of expected performers with face counts

        Returns:
            "well-covered" if ALL performers have faces_in_db >= WELL_COVERED_THRESHOLD,
            "sparse" otherwise
        """
        for performer in performers:
            if performer.faces_in_db < WELL_COVERED_THRESHOLD:
                return "sparse"
        return "well-covered"

    def _build_scene_query(self, page: int = 1, per_page: int = 25) -> str:
        """Build GraphQL query for findScenes.

        Args:
            page: Page number (1-indexed)
            per_page: Number of scenes per page

        Returns:
            GraphQL query string
        """
        return f"""
        query {{
            findScenes(
                filter: {{
                    page: {page}
                    per_page: {per_page}
                    sort: "random"
                }}
            ) {{
                count
                scenes {{
                    id
                    title
                    files {{
                        width
                        height
                        duration
                    }}
                    stash_ids {{
                        endpoint
                        stash_id
                    }}
                    performers {{
                        name
                        stash_ids {{
                            endpoint
                            stash_id
                        }}
                    }}
                }}
            }}
        }}
        """

    def _get_stashdb_id(self, stash_ids: list[dict]) -> Optional[str]:
        """Extract stash_id from list where endpoint matches STASHDB_ENDPOINT.

        Args:
            stash_ids: List of stash_id dicts with endpoint and stash_id keys

        Returns:
            The stash_id if found, None otherwise
        """
        for sid in stash_ids:
            if sid.get("endpoint") == STASHDB_ENDPOINT:
                return sid.get("stash_id")
        return None

    def _is_valid_test_scene(self, scene_data: dict) -> bool:
        """Check if scene meets benchmark criteria.

        A valid test scene must:
        - Have file info
        - Have resolution >= MIN_RESOLUTION_WIDTH or MIN_RESOLUTION_HEIGHT
        - Have a StashDB ID
        - Have >= MIN_PERFORMERS performers
        - All performers must have StashDB IDs

        Args:
            scene_data: Scene data from GraphQL response

        Returns:
            True if scene is valid for benchmarking
        """
        # Must have file info
        files = scene_data.get("files", [])
        if not files:
            return False

        file_info = files[0]
        width = file_info.get("width", 0)
        height = file_info.get("height", 0)

        # Resolution check: either dimension must meet minimum
        if width < MIN_RESOLUTION_WIDTH and height < MIN_RESOLUTION_HEIGHT:
            return False

        # Must have StashDB ID
        stash_ids = scene_data.get("stash_ids", [])
        if not self._get_stashdb_id(stash_ids):
            return False

        # Must have minimum number of performers
        performers = scene_data.get("performers", [])
        if len(performers) < MIN_PERFORMERS:
            return False

        # All performers must have StashDB IDs
        for performer in performers:
            performer_stash_ids = performer.get("stash_ids", [])
            if not self._get_stashdb_id(performer_stash_ids):
                return False

        return True

    async def _get_performer_db_coverage(self, stashdb_id: str) -> dict:
        """Get database coverage for a performer.

        Args:
            stashdb_id: The performer's StashDB ID

        Returns:
            Dict with faces_in_db, has_body_data, has_tattoo_data
        """
        universal_id = f"stashdb.org:{stashdb_id}"

        face_count = self.database_reader.get_face_count_for_performer(universal_id)
        has_body = self.database_reader.has_body_data(universal_id)
        has_tattoo = self.database_reader.has_tattoo_data(universal_id)

        return {
            "faces_in_db": face_count if face_count is not None else 0,
            "has_body_data": has_body,
            "has_tattoo_data": has_tattoo,
        }

    async def _convert_to_test_scene(self, scene_data: dict) -> TestScene:
        """Convert scene data from GraphQL to TestScene object.

        Args:
            scene_data: Scene data from GraphQL response

        Returns:
            TestScene object with all fields populated
        """
        file_info = scene_data["files"][0]
        width = file_info["width"]
        height = file_info["height"]
        duration = file_info.get("duration", 0)

        stashdb_id = self._get_stashdb_id(scene_data["stash_ids"])
        resolution = self._get_resolution_tier(width, height)

        # Build expected performers list with DB coverage
        expected_performers = []
        for performer_data in scene_data["performers"]:
            performer_stashdb_id = self._get_stashdb_id(performer_data["stash_ids"])
            coverage = await self._get_performer_db_coverage(performer_stashdb_id)

            performer = ExpectedPerformer(
                stashdb_id=performer_stashdb_id,
                name=performer_data["name"],
                faces_in_db=coverage["faces_in_db"],
                has_body_data=coverage["has_body_data"],
                has_tattoo_data=coverage["has_tattoo_data"],
            )
            expected_performers.append(performer)

        coverage_tier = self._get_coverage_tier(expected_performers)

        return TestScene(
            scene_id=scene_data["id"],
            stashdb_id=stashdb_id,
            title=scene_data["title"],
            resolution=resolution,
            width=width,
            height=height,
            duration_sec=duration,
            expected_performers=expected_performers,
            db_coverage_tier=coverage_tier,
        )

    async def select_scenes(self, min_count: int = 100) -> list[TestScene]:
        """Select scenes from Stash for benchmarking.

        Paginates through Stash scenes, filtering for valid test scenes,
        until min_count is reached or no more scenes are available.

        Args:
            min_count: Minimum number of scenes to select

        Returns:
            List of TestScene objects
        """
        scenes = []
        page = 1
        per_page = 25

        while len(scenes) < min_count:
            query = self._build_scene_query(page=page, per_page=per_page)
            response = self.stash_client._query(query)

            find_scenes = response.get("findScenes", {})
            scene_data_list = find_scenes.get("scenes", [])

            if not scene_data_list:
                # No more scenes available
                break

            for scene_data in scene_data_list:
                if self._is_valid_test_scene(scene_data):
                    test_scene = await self._convert_to_test_scene(scene_data)
                    scenes.append(test_scene)

                    if len(scenes) >= min_count:
                        break

            page += 1

        return scenes

    def stratify_scenes(self, scenes: list[TestScene]) -> dict[str, list[TestScene]]:
        """Group scenes by resolution tier.

        Args:
            scenes: List of TestScene objects

        Returns:
            Dict mapping resolution tier to list of scenes
        """
        stratified = {
            "480p": [],
            "720p": [],
            "1080p": [],
            "4k": [],
        }

        for scene in scenes:
            stratified[scene.resolution].append(scene)

        return stratified

    def sample_stratified(
        self,
        scenes: list[TestScene],
        count: int,
        seed: Optional[int] = None
    ) -> list[TestScene]:
        """Sample scenes with equal representation from each resolution tier.

        Args:
            scenes: List of TestScene objects to sample from
            count: Target number of scenes to sample
            seed: Random seed for reproducibility

        Returns:
            List of sampled TestScene objects
        """
        if seed is not None:
            random.seed(seed)

        stratified = self.stratify_scenes(scenes)
        tiers = ["480p", "720p", "1080p", "4k"]

        # Calculate base count per tier and remainder
        num_tiers = len(tiers)
        base_per_tier = count // num_tiers
        remainder = count % num_tiers

        sampled = []

        for i, tier in enumerate(tiers):
            tier_scenes = stratified[tier]
            # Distribute remainder to first tiers
            tier_count = base_per_tier + (1 if i < remainder else 0)

            if tier_scenes:
                # Sample up to tier_count from this tier
                sample_size = min(tier_count, len(tier_scenes))
                tier_sample = random.sample(tier_scenes, sample_size)
                sampled.extend(tier_sample)

        return sampled

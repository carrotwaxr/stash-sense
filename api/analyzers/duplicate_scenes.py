"""
Duplicate Scenes Analyzer

Detects duplicate scenes using multi-signal analysis:
- Stash-box ID matching (authoritative, 100%)
- Face fingerprint similarity (up to 85%)
- Metadata heuristics (up to 60%)

See: docs/plans/2026-01-30-duplicate-scene-detection-design.md

Note: Face fingerprint similarity requires fingerprints to be generated first.
Use /recommendations/fingerprints/generate to generate fingerprints for your library.
The analyzer will still find duplicates via stash-box IDs and metadata without fingerprints.
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Optional

from .base import BaseAnalyzer, AnalysisResult
from duplicate_detection import (
    SceneMetadata,
    SceneFingerprint,
    FaceAppearance,
    calculate_duplicate_confidence,
)

if TYPE_CHECKING:
    from stash_client_unified import StashClientUnified
    from recommendations_db import RecommendationsDB


logger = logging.getLogger(__name__)


@dataclass
class DuplicateScenesResult(AnalysisResult):
    """Extended result with fingerprint coverage info."""
    total_scenes: int = 0
    scenes_with_fingerprints: int = 0
    fingerprint_coverage_pct: float = 0.0
    comparisons_made: int = 0
    duplicates_found: int = 0


class DuplicateScenesAnalyzer(BaseAnalyzer):
    """
    Detects duplicate scenes using multi-signal analysis.

    Configuration:
        min_confidence: Minimum confidence (0-100) to create recommendation
        batch_size: Scenes to process per batch
        max_comparisons: Safety limit on O(n²) comparisons
    """

    type = "duplicate_scenes"

    def __init__(
        self,
        stash: "StashClientUnified",
        rec_db: "RecommendationsDB",
        min_confidence: float = 50.0,
        batch_size: int = 100,
        max_comparisons: int = None,  # No limit
        max_scenes: int = None,  # No limit
    ):
        super().__init__(stash, rec_db)
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.max_comparisons = max_comparisons
        self.max_scenes = max_scenes

    async def run(self, incremental: bool = True) -> DuplicateScenesResult:
        """
        Run duplicate scene detection.

        Note: Incremental mode is not supported for this analyzer since
        we need to compare all scenes against each other. Always runs full scan.

        Phase 1: Load scenes and existing fingerprints
        Phase 2: Compare all pairs for duplicates
        Phase 3: Create recommendations

        Returns DuplicateScenesResult with fingerprint coverage info.
        """
        # Phase 1: Load scenes
        logger.info("Loading scenes from Stash...")
        all_scenes: list[dict] = []
        offset = 0

        while True:
            scenes, total = await self.stash.get_scenes_for_fingerprinting(
                limit=self.batch_size,
                offset=offset,
            )
            all_scenes.extend(scenes)

            if len(all_scenes) >= total or not scenes:
                break

            if self.max_scenes and len(all_scenes) >= self.max_scenes:
                logger.warning(f"Hit max scenes limit ({self.max_scenes})")
                all_scenes = all_scenes[: self.max_scenes]
                break

            offset += self.batch_size
            # Rate limiting handled by StashClientUnified

        logger.info(f"Loaded {len(all_scenes)} scenes")

        if len(all_scenes) < 2:
            return DuplicateScenesResult(
                items_processed=len(all_scenes),
                recommendations_created=0,
                total_scenes=len(all_scenes),
            )

        # Convert to metadata objects
        scene_metadata = [SceneMetadata.from_stash(s) for s in all_scenes]
        scene_ids = {s.scene_id for s in scene_metadata}

        # Load existing fingerprints
        fingerprints: dict[str, SceneFingerprint] = {}
        for fp_data in self.rec_db.get_all_scene_fingerprints(status="complete"):
            try:
                scene_id = str(fp_data["stash_scene_id"])
                faces_data = self.rec_db.get_fingerprint_faces(fp_data["id"])

                faces = {}
                for f in faces_data:
                    faces[f["performer_id"]] = FaceAppearance(
                        performer_id=f["performer_id"],
                        face_count=f["face_count"],
                        avg_confidence=f["avg_confidence"],
                        proportion=f["proportion"],
                    )

                fingerprints[scene_id] = SceneFingerprint(
                    stash_scene_id=fp_data["stash_scene_id"],
                    faces=faces,
                    total_faces_detected=fp_data["total_faces"],
                    frames_analyzed=fp_data["frames_analyzed"],
                )
            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping malformed fingerprint {fp_data.get('id')}: {e}")
                continue

        # Calculate fingerprint coverage for loaded scenes
        scenes_with_fp = sum(1 for s in scene_metadata if s.scene_id in fingerprints)
        coverage_pct = (scenes_with_fp / len(scene_metadata) * 100) if scene_metadata else 0

        logger.info(
            f"Loaded {len(fingerprints)} fingerprints, "
            f"{scenes_with_fp}/{len(scene_metadata)} scenes have fingerprints ({coverage_pct:.1f}%)"
        )

        if coverage_pct < 10:
            logger.warning(
                f"Low fingerprint coverage ({coverage_pct:.1f}%). "
                "Run /recommendations/fingerprints/generate to improve duplicate detection accuracy."
            )

        # Phase 2: Find duplicates
        duplicates = []
        comparisons = 0

        for i, scene_a in enumerate(scene_metadata):
            fp_a = fingerprints.get(scene_a.scene_id)

            for scene_b in scene_metadata[i + 1 :]:
                comparisons += 1

                if self.max_comparisons and comparisons > self.max_comparisons:
                    logger.warning(f"Hit max comparisons limit ({self.max_comparisons})")
                    break

                # Yield periodically
                if comparisons % 1000 == 0:
                    await asyncio.sleep(0)

                fp_b = fingerprints.get(scene_b.scene_id)

                match = calculate_duplicate_confidence(scene_a, scene_b, fp_a, fp_b)

                if match and match.confidence >= self.min_confidence:
                    duplicates.append(match)

            if self.max_comparisons and comparisons > self.max_comparisons:
                break

        logger.info(f"Found {len(duplicates)} potential duplicates from {comparisons} comparisons")

        # Phase 3: Create recommendations
        created = 0
        for match in duplicates:
            rec_id = self.create_recommendation(
                target_type="scene",
                target_id=str(match.scene_a_id),
                details={
                    "scene_b_id": match.scene_b_id,
                    "confidence": match.confidence,
                    "reasoning": match.reasoning,
                    "signal_breakdown": asdict(match.signal_breakdown),
                },
                confidence=match.confidence / 100.0,  # Normalize to 0-1
            )

            if rec_id:
                created += 1

        return DuplicateScenesResult(
            items_processed=len(all_scenes),
            recommendations_created=created,
            total_scenes=len(all_scenes),
            scenes_with_fingerprints=scenes_with_fp,
            fingerprint_coverage_pct=round(coverage_pct, 1),
            comparisons_made=comparisons,
            duplicates_found=len(duplicates),
        )

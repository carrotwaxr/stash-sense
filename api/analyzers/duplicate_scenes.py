"""
Duplicate Scenes Analyzer

Detects duplicate scenes using multi-signal analysis:
- Stash-box ID matching (authoritative, 100%)
- Face fingerprint similarity (up to 85%)
- Metadata heuristics (up to 60%)

See: docs/plans/2026-01-30-duplicate-scene-detection-design.md
"""

import asyncio
import logging
from dataclasses import asdict
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
        max_comparisons: int = 50000,
    ):
        super().__init__(stash, rec_db)
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.max_comparisons = max_comparisons

    async def run(self, incremental: bool = True) -> AnalysisResult:
        """
        Run duplicate scene detection.

        Phase 1: Load scenes and existing fingerprints
        Phase 2: Compare all pairs for duplicates
        Phase 3: Create recommendations
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

            offset += self.batch_size
            await asyncio.sleep(0.1)  # Rate limiting

        logger.info(f"Loaded {len(all_scenes)} scenes")

        if len(all_scenes) < 2:
            return AnalysisResult(items_processed=len(all_scenes), recommendations_created=0)

        # Convert to metadata objects
        scene_metadata = [SceneMetadata.from_stash(s) for s in all_scenes]

        # Load existing fingerprints
        fingerprints: dict[str, SceneFingerprint] = {}
        for fp_data in self.rec_db.get_all_scene_fingerprints(status="complete"):
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

        logger.info(f"Loaded {len(fingerprints)} existing fingerprints")

        # Phase 2: Find duplicates
        duplicates = []
        comparisons = 0

        for i, scene_a in enumerate(scene_metadata):
            fp_a = fingerprints.get(scene_a.scene_id)

            for scene_b in scene_metadata[i + 1 :]:
                comparisons += 1

                if comparisons > self.max_comparisons:
                    logger.warning(f"Hit max comparisons limit ({self.max_comparisons})")
                    break

                # Yield periodically
                if comparisons % 1000 == 0:
                    await asyncio.sleep(0)

                fp_b = fingerprints.get(scene_b.scene_id)

                match = calculate_duplicate_confidence(scene_a, scene_b, fp_a, fp_b)

                if match and match.confidence >= self.min_confidence:
                    duplicates.append(match)

            if comparisons > self.max_comparisons:
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

        return AnalysisResult(
            items_processed=len(all_scenes),
            recommendations_created=created,
        )

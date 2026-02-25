"""
Duplicate Scenes Analyzer

Detects duplicate scenes using a two-phase pipeline:
  Phase 1: Candidate generation via stash-box IDs, phash similarity, and metadata indices
  Phase 2: Sequential scoring of candidate pairs

See: docs/plans/2026-02-15-scalable-duplicate-scene-detection-design.md
"""

import asyncio
import logging
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

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
    Detects duplicate scenes using a two-phase candidate-then-score pipeline.

    Phase 1 (Candidate Generation):
        - Stash-box ID grouping: scenes sharing (endpoint, stash_id)
        - Phash Hamming distance: scenes with perceptual hash distance <= 10
        - Metadata intersection: scenes sharing (studio_id, performer_id)
        All candidates stored in duplicate_candidates table.

    Phase 2 (Sequential Scoring):
        - Iterate candidates with cursor-based pagination
        - Score each pair with calculate_duplicate_confidence()
        - Write recommendations immediately

    Configuration:
        min_confidence: Minimum confidence (0-100) to create recommendation
        batch_size: Scenes to fetch per Stash API page
    """

    type = "duplicate_scenes"

    def __init__(
        self,
        stash: "StashClientUnified",
        rec_db: "RecommendationsDB",
        min_confidence: float = 50.0,
        batch_size: int = 100,
        **kwargs,
    ):
        super().__init__(stash, rec_db, **kwargs)
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self._phash_distances: dict[tuple[int, int], int] = {}

    async def run(self, incremental: bool = True) -> DuplicateScenesResult:
        """
        Run duplicate scene detection via candidate-then-score pipeline.

        Note: Incremental mode is not yet supported — always runs a full scan.
        """
        run_id = self.run_id

        # Phase 1: Generate candidates
        logger.warning("Phase 1: Generating candidates...")
        candidates_count = await self._generate_candidates(run_id)
        logger.warning(f"Phase 1 complete: {candidates_count} candidates generated")

        if candidates_count == 0:
            return DuplicateScenesResult(
                items_processed=0,
                recommendations_created=0,
            )

        # Phase 2: Score candidates
        self.set_items_total(candidates_count)
        logger.warning(f"Phase 2: Scoring {candidates_count} candidates...")
        scored, created, total_scenes, scenes_with_fp, coverage_pct = await self._score_candidates(run_id)
        logger.warning(f"Phase 2 complete: {scored} scored, {created} recommendations created")

        return DuplicateScenesResult(
            items_processed=scored,
            recommendations_created=created,
            total_scenes=total_scenes,
            scenes_with_fingerprints=scenes_with_fp,
            fingerprint_coverage_pct=round(coverage_pct, 1),
            comparisons_made=scored,
            duplicates_found=created,
        )

    async def _generate_candidates(self, run_id: int) -> int:
        """
        Phase 1: Generate candidate pairs from three sources:
          A) Stash-box ID grouping
          B) Phash Hamming distance (replaces face fingerprint self-join)
          C) Metadata intersection (studio + performer)
        Returns total number of unique candidates.
        """
        # Clean up orphaned candidates from previous broken runs (NULL run_id)
        orphans = self.rec_db.clear_orphaned_candidates()
        if orphans:
            logger.warning(f"Cleaned up {orphans} orphaned candidates (NULL run_id)")

        # Clear any stale candidates from a previous failed run
        self.rec_db.clear_candidates(run_id)
        self._phash_distances = {}

        # Source A: Stash-box IDs + Source B: Phash fingerprints
        # (single pass over paginated scenes with fingerprints)
        stashbox_index: dict[tuple[str, str], list[int]] = {}
        phash_list: list[tuple[int, str]] = []  # (scene_id, phash_hex)

        offset = 0
        total_scenes = 0
        while True:
            scenes, total = await self.stash.get_scenes_with_fingerprints(
                limit=self.batch_size, offset=offset,
            )
            total_scenes = total

            for scene in scenes:
                sid = int(scene["id"])

                # Build stash-box index
                for stash_id_entry in scene.get("stash_ids", []):
                    key = (stash_id_entry["endpoint"], stash_id_entry["stash_id"])
                    stashbox_index.setdefault(key, []).append(sid)

                # Extract phash from file fingerprints
                for file_entry in scene.get("files", []):
                    for fp in file_entry.get("fingerprints", []):
                        if fp.get("type") == "phash" and fp.get("value"):
                            phash_list.append((sid, fp["value"]))
                            break  # one phash per scene
                    else:
                        continue
                    break  # found phash, stop checking files

            if len(scenes) == 0 or offset + len(scenes) >= total:
                break
            offset += self.batch_size
            await asyncio.sleep(0)

        logger.warning(f"Loaded {total_scenes} scenes ({len(phash_list)} with phash). Building candidates...")

        # Generate stash-box candidates
        stashbox_pairs = []
        for key, scene_ids in stashbox_index.items():
            if len(scene_ids) > 1:
                for i, a in enumerate(scene_ids):
                    for b in scene_ids[i + 1:]:
                        stashbox_pairs.append((a, b, "stashbox"))
        if stashbox_pairs:
            self.rec_db.insert_candidates_batch(stashbox_pairs, run_id)
            logger.warning(f"  Stash-box ID: {len(stashbox_pairs)} candidates")

        # Source B: Phash candidates (Hamming distance)
        if phash_list:
            stored = self.rec_db.store_scene_phashes(phash_list)
            phash_triples = self.rec_db.generate_phash_candidates(max_distance=10)
            if phash_triples:
                phash_pairs = []
                for a, b, dist in phash_triples:
                    pair_key = (min(a, b), max(a, b))
                    self._phash_distances[pair_key] = dist
                    phash_pairs.append((a, b, "phash"))
                self.rec_db.insert_candidates_batch(phash_pairs, run_id)
                logger.warning(f"  Phash: {len(phash_pairs)} candidates ({self.rec_db.count_candidates(run_id)} after dedup)")
            else:
                logger.warning(f"  Phash: 0 candidates (from {stored} scenes with phash)")

        # Source C: Metadata candidates (studio + performer index)
        # Requires a separate pass over get_scenes_for_fingerprinting for studio/performer data
        combined_index: dict[tuple[str, str], set[int]] = {}
        offset = 0
        while True:
            scenes, total = await self.stash.get_scenes_for_fingerprinting(
                limit=self.batch_size, offset=offset,
            )

            for scene in scenes:
                sid = int(scene["id"])
                studio = scene.get("studio")
                if studio and studio.get("id"):
                    studio_id = studio["id"]
                    for performer in scene.get("performers", []):
                        key = (studio_id, performer["id"])
                        combined_index.setdefault(key, set()).add(sid)

            if len(scenes) == 0 or offset + len(scenes) >= total:
                break
            offset += self.batch_size
            await asyncio.sleep(0)

        metadata_pairs = []
        for key, scene_ids in combined_index.items():
            if len(scene_ids) > 1:
                sorted_ids = sorted(scene_ids)
                for i, a in enumerate(sorted_ids):
                    for b in sorted_ids[i + 1:]:
                        metadata_pairs.append((a, b, "metadata"))
        if metadata_pairs:
            self.rec_db.insert_candidates_batch(metadata_pairs, run_id)
            logger.warning(f"  Metadata: {len(metadata_pairs)} candidates ({self.rec_db.count_candidates(run_id)} after dedup)")

        return self.rec_db.count_candidates(run_id)

    async def _score_candidates(self, run_id: int) -> tuple[int, int, int, int, float]:
        """
        Phase 2: Score candidate pairs sequentially.
        Returns (scored, created, total_scenes, scenes_with_fp, coverage_pct).
        """
        # Load scene metadata for candidate scenes only
        candidate_scene_ids = self.rec_db.get_candidate_scene_ids(run_id)
        scene_metadata = await self._load_scene_metadata(candidate_scene_ids)

        # Load fingerprints via JOIN query (filtered to candidate scenes)
        fp_data = self.rec_db.get_fingerprints_with_faces(scene_ids=candidate_scene_ids)
        fingerprints: dict[str, SceneFingerprint] = {}
        for scene_id_str, data in fp_data.items():
            faces = {}
            for pid, face in data["faces"].items():
                faces[pid] = FaceAppearance(
                    performer_id=pid,
                    face_count=face["face_count"],
                    avg_confidence=face["avg_confidence"],
                    proportion=face["proportion"],
                )
            fingerprints[scene_id_str] = SceneFingerprint(
                stash_scene_id=data["stash_scene_id"],
                faces=faces,
                total_faces_detected=data["total_faces"],
                frames_analyzed=data["frames_analyzed"],
            )

        total_scenes = len(scene_metadata)
        scenes_with_fp = sum(1 for sid in scene_metadata if sid in fingerprints)
        coverage_pct = (scenes_with_fp / total_scenes * 100) if total_scenes else 0

        if coverage_pct < 10 and total_scenes > 0:
            logger.warning(
                f"Low fingerprint coverage ({coverage_pct:.1f}%). "
                "Run /recommendations/fingerprints/generate to improve accuracy."
            )

        # Score candidates with cursor-based pagination
        scored = 0
        created = 0
        last_id = 0

        while True:
            if self.is_stop_requested():
                logger.warning(f"Stop requested after scoring {scored} candidates")
                break

            batch = self.rec_db.get_candidates_batch(run_id, after_id=last_id, limit=100)
            if not batch:
                break

            for candidate in batch:
                scene_a = scene_metadata.get(str(candidate["scene_a_id"]))
                scene_b = scene_metadata.get(str(candidate["scene_b_id"]))

                if not scene_a or not scene_b:
                    scored += 1
                    last_id = candidate["id"]
                    continue

                fp_a = fingerprints.get(str(candidate["scene_a_id"]))
                fp_b = fingerprints.get(str(candidate["scene_b_id"]))

                # Look up phash distance for this pair
                pair_key = (min(candidate["scene_a_id"], candidate["scene_b_id"]),
                            max(candidate["scene_a_id"], candidate["scene_b_id"]))
                phash_dist = self._phash_distances.get(pair_key)

                match = calculate_duplicate_confidence(scene_a, scene_b, fp_a, fp_b, phash_distance=phash_dist)

                if match and match.confidence >= self.min_confidence:
                    rec_id = self.create_recommendation(
                        target_type="scene",
                        target_id=str(match.scene_a_id),
                        details={
                            "scene_b_id": match.scene_b_id,
                            "confidence": match.confidence,
                            "reasoning": match.reasoning,
                            "signal_breakdown": asdict(match.signal_breakdown),
                        },
                        confidence=match.confidence / 100.0,
                    )
                    if rec_id:
                        created += 1

                scored += 1
                last_id = candidate["id"]

                # Periodic progress logging (every 500 scored)
                if scored % 500 == 0:
                    total = self._items_total or "?"
                    logger.warning(f"Scoring progress: {scored}/{total} scored, {created} recommendations")

            self.update_progress(scored, created)
            await asyncio.sleep(0)

        return scored, created, total_scenes, scenes_with_fp, coverage_pct

    async def _load_scene_metadata(
        self, scene_ids: set[int]
    ) -> dict[str, SceneMetadata]:
        """
        Load SceneMetadata from Stash for specific scene IDs.
        Fetches all scenes (paginated) and filters to needed IDs.
        """
        if not scene_ids:
            return {}

        result: dict[str, SceneMetadata] = {}
        offset = 0

        while True:
            scenes, total = await self.stash.get_scenes_for_fingerprinting(
                limit=self.batch_size, offset=offset,
            )

            for scene in scenes:
                if int(scene["id"]) in scene_ids:
                    metadata = SceneMetadata.from_stash(scene)
                    result[scene["id"]] = metadata

            # Stop early if we've found all needed scenes
            if len(result) >= len(scene_ids):
                break

            if len(scenes) == 0 or offset + len(scenes) >= total:
                break

            offset += self.batch_size
            await asyncio.sleep(0)

        return result

"""Analyzer: match local scenes to stash-box entries via fingerprints.

Extends BaseAnalyzer (not BaseUpstreamAnalyzer) because this finds
unlinked scenes rather than diffing already-linked entities.
"""

import logging
from typing import Optional

from .base import BaseAnalyzer, AnalysisResult
from scene_fingerprint_scoring import score_match, is_high_confidence
from stashbox_client import StashBoxClient

logger = logging.getLogger(__name__)

# Max scenes per stash-box batch query (matches Stash tagger batching)
BATCH_SIZE = 40


class SceneFingerprintMatchAnalyzer(BaseAnalyzer):
    type = "scene_fingerprint_match"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        connections = await self.stash.get_stashbox_connections()
        if not connections:
            return AnalysisResult(items_processed=0, recommendations_created=0)

        # Load user-configurable thresholds
        min_count = self._get_setting("scene_fp_min_count", 2)
        min_percentage = self._get_setting("scene_fp_min_percentage", 66)

        total_processed = 0
        total_created = 0

        for conn in connections:
            endpoint = conn["endpoint"]
            api_key = conn.get("api_key", "")
            endpoint_name = conn.get("name", endpoint)

            processed, created = await self._process_endpoint(
                endpoint, api_key, endpoint_name,
                incremental, min_count, min_percentage,
            )
            total_processed += processed
            total_created += created

        return AnalysisResult(
            items_processed=total_processed,
            recommendations_created=total_created,
        )

    async def _process_endpoint(
        self,
        endpoint: str,
        api_key: str,
        endpoint_name: str,
        incremental: bool,
        min_count: int,
        min_percentage: int,
    ) -> tuple[int, int]:
        """Process one stash-box endpoint. Returns (processed, created)."""
        watermark_key = f"scene_fp_match_{endpoint}"

        # Get watermark for incremental mode
        watermark_ts = None
        if incremental:
            wm = self.rec_db.get_watermark(watermark_key)
            if wm:
                watermark_ts = wm.get("last_stash_updated_at")

        # Fetch all local scenes with fingerprint data
        scenes_needing_match = []
        offset = 0
        latest_updated = watermark_ts

        while True:
            scenes, total = await self.stash.get_scenes_with_fingerprints(
                updated_after=watermark_ts, limit=100, offset=offset,
            )
            if not scenes:
                break

            for scene in scenes:
                # Track latest updated_at for watermark
                updated_at = scene.get("updated_at")
                if updated_at and (latest_updated is None or updated_at > latest_updated):
                    latest_updated = updated_at

                # Skip scenes already linked to this endpoint
                linked_endpoints = {
                    sid["endpoint"] for sid in (scene.get("stash_ids") or [])
                }
                if endpoint in linked_endpoints:
                    continue

                # Collect fingerprints from all files
                fingerprints = []
                duration = None
                for f in scene.get("files") or []:
                    if duration is None and f.get("duration"):
                        duration = f["duration"]
                    for fp in f.get("fingerprints") or []:
                        fingerprints.append({
                            "hash": fp["value"],
                            "algorithm": fp["type"].upper(),
                        })

                if fingerprints:
                    scenes_needing_match.append({
                        "scene": scene,
                        "fingerprints": fingerprints,
                        "duration": duration,
                    })

            offset += len(scenes)
            if offset >= total:
                break

        if not scenes_needing_match:
            if latest_updated:
                self.rec_db.set_watermark(watermark_key, last_stash_updated_at=latest_updated)
            return 0, 0

        # Batch query stash-box
        stashbox = StashBoxClient(endpoint, api_key)
        logger.warning(
            "[%s] Starting scan of %d scenes with fingerprints",
            endpoint_name, len(scenes_needing_match),
        )
        created = 0

        for batch_start in range(0, len(scenes_needing_match), BATCH_SIZE):
            batch = scenes_needing_match[batch_start:batch_start + BATCH_SIZE]
            fp_sets = [item["fingerprints"] for item in batch]

            results = await stashbox.find_scenes_by_fingerprints(fp_sets)

            for i, matches in enumerate(results):
                item = batch[i]
                scene = item["scene"]
                local_fps = item["fingerprints"]
                local_duration = item["duration"]
                is_ambiguous = len(matches) > 1

                for match in matches:
                    # Build composite target_id for pair-based dismissal
                    target_id = f"{scene['id']}|{endpoint}|{match['id']}"

                    if self.is_dismissed("scene", target_id):
                        continue

                    # Find which local fingerprints matched this stash-box scene
                    local_hashes = {fp["hash"] for fp in local_fps}
                    matching_fps = [
                        fp for fp in match.get("fingerprints", [])
                        if fp["hash"] in local_hashes
                    ]

                    score_result = score_match(
                        matching_fingerprints=matching_fps,
                        total_local_fingerprints=len(local_fps),
                        local_duration=local_duration or 0,
                    )

                    high_conf = (
                        not is_ambiguous
                        and is_high_confidence(
                            score_result["match_count"],
                            score_result["match_percentage"],
                            min_count=min_count,
                            min_percentage=min_percentage,
                        )
                    )

                    performers = [
                        p["performer"]["name"]
                        for p in (match.get("performers") or [])
                        if p.get("performer")
                    ]
                    studio = match.get("studio")
                    images = match.get("images") or []

                    details = {
                        "local_scene_id": scene["id"],
                        "local_scene_title": scene.get("title") or f"Scene {scene['id']}",
                        "endpoint": endpoint,
                        "endpoint_name": endpoint_name,
                        "stashbox_scene_id": match["id"],
                        "stashbox_scene_title": match.get("title"),
                        "stashbox_studio": studio.get("name") if studio else None,
                        "stashbox_performers": performers,
                        "stashbox_date": match.get("date"),
                        "stashbox_cover_url": images[0]["url"] if images else None,
                        "matching_fingerprints": matching_fps,
                        "total_local_fingerprints": len(local_fps),
                        "match_count": score_result["match_count"],
                        "match_percentage": score_result["match_percentage"],
                        "has_exact_hash": score_result["has_exact_hash"],
                        "duration_local": local_duration,
                        "duration_remote": match.get("duration"),
                        "duration_agreement": score_result["duration_agreement"],
                        "duration_diff": score_result["duration_diff"],
                        "total_submissions": score_result["total_submissions"],
                        "high_confidence": high_conf,
                    }

                    confidence = score_result["match_percentage"] / 100.0

                    rec_id = self.create_recommendation(
                        target_type="scene",
                        target_id=target_id,
                        details=details,
                        confidence=confidence,
                    )
                    if rec_id:
                        created += 1

            batch_end = min(batch_start + BATCH_SIZE, len(scenes_needing_match))
            logger.warning(
                "[%s] Processed %d/%d scenes, %d matches found",
                endpoint_name, batch_end, len(scenes_needing_match), created,
            )
            self.update_progress(batch_end, created)

        processed = len(scenes_needing_match)

        logger.warning(
            "[%s] Complete: %d matches found from %d scenes",
            endpoint_name, created, len(scenes_needing_match),
        )

        if latest_updated:
            self.rec_db.set_watermark(watermark_key, last_stash_updated_at=latest_updated)

        return processed, created

    def _get_setting(self, key: str, default):
        """Read a user setting with fallback."""
        val = self.rec_db.get_user_setting(key)
        if val is None:
            return default
        try:
            return int(val)
        except (ValueError, TypeError):
            return default

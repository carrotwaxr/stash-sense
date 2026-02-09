"""
Upstream Performer Changes Analyzer

Detects changes in stash-box linked performers by comparing upstream
data against local Stash data using 3-way diffing with stored snapshots.
"""

import logging

from .base import BaseAnalyzer, AnalysisResult
from stashbox_client import StashBoxClient
from upstream_field_mapper import (
    normalize_upstream_performer,
    diff_performer_fields,
    DEFAULT_PERFORMER_FIELDS,
)
from config import get_stashbox_shortname

logger = logging.getLogger(__name__)


def _build_local_performer_data(performer: dict) -> dict:
    """Extract comparable field values from a local Stash performer."""
    from upstream_field_mapper import parse_measurements, parse_career_length

    measurements = parse_measurements(performer.get("measurements"))
    career = parse_career_length(performer.get("career_length"))

    return {
        "name": performer.get("name"),
        "disambiguation": performer.get("disambiguation") or "",
        "aliases": performer.get("alias_list") or [],
        "gender": performer.get("gender"),
        "birthdate": performer.get("birthdate"),
        "death_date": performer.get("death_date"),
        "ethnicity": performer.get("ethnicity"),
        "country": performer.get("country"),
        "eye_color": performer.get("eye_color"),
        "hair_color": performer.get("hair_color"),
        "height": performer.get("height_cm"),
        "cup_size": measurements["cup_size"],
        "band_size": measurements["band_size"],
        "waist_size": measurements["waist_size"],
        "hip_size": measurements["hip_size"],
        "breast_type": performer.get("fake_tits"),
        "tattoos": performer.get("tattoos") or "",
        "piercings": performer.get("piercings") or "",
        "career_start_year": career["career_start_year"],
        "career_end_year": career["career_end_year"],
        "urls": performer.get("urls") or [],
        "details": performer.get("details") or "",
        "favorite": performer.get("favorite", False),
    }


class UpstreamPerformerAnalyzer(BaseAnalyzer):
    """
    Detects upstream changes in stash-box linked performers.

    Compares performer data from configured stash-box endpoints against
    local Stash data, using stored snapshots for 3-way diffing.
    """

    type = "upstream_performer_changes"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        connections = await self.stash.get_stashbox_connections()

        settings = self.rec_db.get_settings(self.type)
        endpoint_config = {}
        if settings and settings.config:
            endpoint_config = settings.config.get("endpoints", {})

        total_processed = 0
        total_created = 0
        errors = []

        logger.info(f"Starting upstream performer analysis (incremental={incremental}), {len(connections)} endpoints")

        for conn in connections:
            endpoint = conn["endpoint"]
            api_key = conn.get("api_key", "")

            ep_cfg = endpoint_config.get(endpoint, {})
            if not ep_cfg.get("enabled", True):
                continue

            try:
                created, processed = await self._process_endpoint(
                    endpoint, api_key, incremental
                )
                total_created += created
                total_processed += processed
            except Exception as e:
                error_msg = f"Error processing {endpoint}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return AnalysisResult(
            items_processed=total_processed,
            recommendations_created=total_created,
            errors=errors if errors else None,
        )

    async def _process_endpoint(
        self, endpoint: str, api_key: str, incremental: bool
    ) -> tuple[int, int]:
        """Process a single stash-box endpoint. Returns (created, processed)."""
        local_performers = await self.stash.get_performers_for_endpoint(endpoint)

        if not local_performers:
            return 0, 0

        # Build lookup: stash_box_id -> local performer
        local_lookup = {}
        for p in local_performers:
            for sid in p.get("stash_ids", []):
                if sid["endpoint"] == endpoint:
                    local_lookup[sid["stash_id"]] = p

        logger.warning(f"Endpoint {endpoint}: {len(local_lookup)} linked performers to check")
        self.set_items_total(len(local_lookup))

        watermark = None
        if incremental:
            wm = self.rec_db.get_watermark(f"{self.type}:{endpoint}")
            if wm and wm.get("last_cursor"):
                watermark = wm["last_cursor"]

        enabled_fields = self.rec_db.get_enabled_fields(endpoint, "performer")
        if enabled_fields is None:
            enabled_fields = DEFAULT_PERFORMER_FIELDS

        sbc = StashBoxClient(endpoint, api_key)
        latest_updated_at = None
        created = 0
        processed = 0
        skipped = 0

        for i, (stash_box_id, local_performer) in enumerate(local_lookup.items()):
            local_id = local_performer["id"]

            if self.rec_db.is_permanently_dismissed(self.type, "performer", local_id):
                skipped += 1
                continue

            try:
                up = await sbc.get_performer(stash_box_id)
            except Exception as e:
                logger.warning(f"Failed to fetch performer {stash_box_id}: {e}")
                continue

            if not up:
                continue

            updated_at = up.get("updated")

            if latest_updated_at is None or (updated_at and updated_at > latest_updated_at):
                latest_updated_at = updated_at

            # In incremental mode, skip performers not updated since last run
            if watermark and updated_at and updated_at <= watermark:
                skipped += 1
                continue

            if up.get("deleted") or up.get("merged_into_id"):
                continue

            normalized = normalize_upstream_performer(up)

            snapshot_row = self.rec_db.get_upstream_snapshot(
                entity_type="performer",
                endpoint=endpoint,
                stash_box_id=stash_box_id,
            )
            snapshot = snapshot_row["upstream_data"] if snapshot_row else None

            local_data = _build_local_performer_data(local_performer)

            changes = diff_performer_fields(
                local_data, normalized, snapshot, enabled_fields
            )

            processed += 1

            if (i + 1) % 50 == 0:
                logger.warning(f"Progress: {i + 1}/{len(local_lookup)} checked, {processed} compared, {created} new recs")
                self.update_progress(processed, created)

            if not changes:
                # No differences: safe to update snapshot baseline
                self.rec_db.upsert_upstream_snapshot(
                    entity_type="performer",
                    local_entity_id=local_id,
                    endpoint=endpoint,
                    stash_box_id=stash_box_id,
                    upstream_data=normalized,
                    upstream_updated_at=updated_at or "",
                )
                # Auto-resolve any stale pending recommendation for this performer
                stale = self.rec_db.get_recommendation_by_target(
                    self.type, "performer", local_id, status="pending"
                )
                if stale:
                    self.rec_db.resolve_recommendation(
                        stale.id, action="auto_resolved",
                        details={"reason": "no_differences"},
                    )
                continue

            endpoint_name = get_stashbox_shortname(endpoint)
            details = {
                "endpoint": endpoint,
                "endpoint_name": endpoint_name,
                "stash_box_id": stash_box_id,
                "performer_id": local_id,
                "performer_name": local_performer.get("name", ""),
                "performer_image_path": local_performer.get("image_path"),
                "upstream_updated_at": updated_at,
                "changes": changes,
            }

            # Check for existing pending recommendation for this performer
            existing_pending = self.rec_db.get_recommendation_by_target(
                self.type, "performer", local_id, status="pending"
            )

            if existing_pending:
                self.rec_db.update_recommendation_details(existing_pending.id, details)
            else:
                # Check for dismissed recommendation that can be reopened
                existing_dismissed = self.rec_db.get_recommendation_by_target(
                    self.type, "performer", local_id, status="dismissed"
                )

                if existing_dismissed:
                    self.rec_db.undismiss(self.type, "performer", local_id)
                    self.rec_db.reopen_recommendation(existing_dismissed.id, details)
                    created += 1
                else:
                    rec_id = self.create_recommendation(
                        target_type="performer",
                        target_id=local_id,
                        details=details,
                        confidence=1.0,
                    )
                    if rec_id:
                        created += 1

        self.update_progress(processed, created)

        logger.warning(
            f"Endpoint {endpoint} complete: {processed} compared, "
            f"{created} recs created, {skipped} skipped"
        )

        if latest_updated_at:
            self.rec_db.set_watermark(
                f"{self.type}:{endpoint}",
                last_cursor=latest_updated_at,
            )

        return created, processed

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
        "cup_size": None,
        "band_size": None,
        "waist_size": None,
        "hip_size": None,
        "breast_type": performer.get("fake_tits"),
        "tattoos": performer.get("tattoos") or "",
        "piercings": performer.get("piercings") or "",
        "career_start_year": None,
        "career_end_year": None,
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

        local_lookup = {}
        for p in local_performers:
            for sid in p.get("stash_ids", []):
                if sid["endpoint"] == endpoint:
                    local_lookup[sid["stash_id"]] = p

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
        page = 1

        while True:
            upstream_performers, total_count = await sbc.query_performers(
                page=page, per_page=25
            )

            if not upstream_performers:
                break

            for up in upstream_performers:
                updated_at = up.get("updated")

                if latest_updated_at is None or (updated_at and updated_at > latest_updated_at):
                    latest_updated_at = updated_at

                if watermark and updated_at and updated_at <= watermark:
                    upstream_performers = []
                    break

                stash_box_id = up.get("id")
                if not stash_box_id or stash_box_id not in local_lookup:
                    continue

                if up.get("deleted"):
                    continue

                if up.get("merged_into_id"):
                    continue

                local_performer = local_lookup[stash_box_id]
                local_id = local_performer["id"]

                if self.rec_db.is_permanently_dismissed(self.type, "performer", local_id):
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

                self.rec_db.upsert_upstream_snapshot(
                    entity_type="performer",
                    local_entity_id=local_id,
                    endpoint=endpoint,
                    stash_box_id=stash_box_id,
                    upstream_data=normalized,
                    upstream_updated_at=updated_at or "",
                )

                processed += 1

                if not changes:
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
                existing = None
                for r in self.rec_db.get_recommendations(type=self.type, status="pending"):
                    if r.target_id == local_id:
                        existing = r
                        break

                if existing:
                    self.rec_db.update_recommendation_details(existing.id, details)
                else:
                    if self.rec_db.is_dismissed(self.type, "performer", local_id):
                        self.rec_db.undismiss(self.type, "performer", local_id)

                    rec_id = self.create_recommendation(
                        target_type="performer",
                        target_id=local_id,
                        details=details,
                        confidence=1.0,
                    )
                    if rec_id:
                        created += 1

            if not upstream_performers:
                break
            page += 1

        if latest_updated_at:
            self.rec_db.set_watermark(
                f"{self.type}:{endpoint}",
                last_cursor=latest_updated_at,
            )

        return created, processed

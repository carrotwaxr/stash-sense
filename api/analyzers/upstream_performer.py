"""
Upstream Performer Changes Analyzer

Detects changes in stash-box linked performers by comparing upstream
data against local Stash data using 3-way diffing with stored snapshots.
"""

import logging
from typing import Optional

from .base_upstream import BaseUpstreamAnalyzer
from stashbox_client import StashBoxClient
from upstream_field_mapper import (
    normalize_upstream_performer,
    diff_performer_fields,
    DEFAULT_PERFORMER_FIELDS,
    FIELD_LABELS,
)

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


class UpstreamPerformerAnalyzer(BaseUpstreamAnalyzer):
    """
    Detects upstream changes in stash-box linked performers.

    Compares performer data from configured stash-box endpoints against
    local Stash data, using stored snapshots for 3-way diffing.
    """

    type = "upstream_performer_changes"

    @property
    def entity_type(self) -> str:
        return "performer"

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        return await self.stash.get_performers_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client: StashBoxClient, stashbox_id: str) -> Optional[dict]:
        return await stashbox_client.get_performer(stashbox_id)

    def _build_local_data(self, entity: dict) -> dict:
        return _build_local_performer_data(entity)

    def _normalize_upstream(self, raw_data: dict) -> dict:
        return normalize_upstream_performer(raw_data)

    def _get_default_fields(self) -> set[str]:
        return DEFAULT_PERFORMER_FIELDS

    def _get_field_labels(self) -> dict[str, str]:
        return FIELD_LABELS

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        return diff_performer_fields(local_data, upstream_data, snapshot, enabled_fields)

    def _build_recommendation_details(
        self,
        endpoint: str,
        endpoint_name: str,
        stash_box_id: str,
        local_entity: dict,
        updated_at: Optional[str],
        changes: list[dict],
    ) -> dict:
        details = super()._build_recommendation_details(
            endpoint=endpoint,
            endpoint_name=endpoint_name,
            stash_box_id=stash_box_id,
            local_entity=local_entity,
            updated_at=updated_at,
            changes=changes,
        )
        # Add performer-specific fields
        details["performer_image_path"] = local_entity.get("image_path")
        return details

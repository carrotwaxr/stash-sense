"""
Upstream Tag Changes Analyzer

Detects changes in stash-box linked tags by comparing upstream
data against local Stash data using 3-way diffing with stored snapshots.
"""

import logging
from typing import Optional

from .base_upstream import BaseUpstreamAnalyzer
from stashbox_client import StashBoxClient
from upstream_field_mapper import (
    normalize_upstream_tag,
    diff_tag_fields,
    DEFAULT_TAG_FIELDS,
    TAG_FIELD_LABELS,
)

logger = logging.getLogger(__name__)


def _build_local_tag_data(tag: dict) -> dict:
    """Extract comparable field values from a local Stash tag."""
    return {
        "name": tag.get("name"),
        "description": tag.get("description") or "",
        "aliases": tag.get("aliases") or [],
    }


class UpstreamTagAnalyzer(BaseUpstreamAnalyzer):
    """
    Detects upstream changes in stash-box linked tags.

    Compares tag data from configured stash-box endpoints against
    local Stash data, using stored snapshots for 3-way diffing.
    """

    type = "upstream_tag_changes"

    @property
    def entity_type(self) -> str:
        return "tag"

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        return await self.stash.get_tags_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client: StashBoxClient, stashbox_id: str) -> Optional[dict]:
        return await stashbox_client.get_tag(stashbox_id)

    def _build_local_data(self, entity: dict) -> dict:
        return _build_local_tag_data(entity)

    def _normalize_upstream(self, raw_data: dict) -> dict:
        return normalize_upstream_tag(raw_data)

    def _get_default_fields(self) -> set[str]:
        return DEFAULT_TAG_FIELDS

    def _get_field_labels(self) -> dict[str, str]:
        return TAG_FIELD_LABELS

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        return diff_tag_fields(local_data, upstream_data, snapshot, enabled_fields)

"""
Upstream Studio Changes Analyzer

Detects changes in stash-box linked studios by comparing upstream
data against local Stash data using 3-way diffing with stored snapshots.
"""

import logging
from typing import Optional

from .base_upstream import BaseUpstreamAnalyzer
from stashbox_client import StashBoxClient
from upstream_field_mapper import (
    normalize_upstream_studio,
    diff_studio_fields,
    DEFAULT_STUDIO_FIELDS,
    STUDIO_FIELD_LABELS,
)

logger = logging.getLogger(__name__)


def _build_local_studio_data(studio: dict, endpoint: str = "") -> dict:
    """Extract comparable field values from a local Stash studio.

    For parent_studio, resolves the parent's stashbox ID for the given endpoint
    so it can be compared against the upstream stashbox UUID. If the parent isn't
    linked to this endpoint, parent_studio stays None — a numeric local ID can't
    be meaningfully compared against a stash-box UUID.
    """
    parent = studio.get("parent_studio")
    parent_studio_val = None
    parent_studio_name = None
    if parent and parent.get("id"):
        parent_studio_name = parent.get("name")
        # Resolve parent's stashbox ID for this endpoint
        for sid in parent.get("stash_ids", []):
            if sid["endpoint"] == endpoint:
                parent_studio_val = sid["stash_id"]
                break

    return {
        "name": studio.get("name"),
        "url": studio.get("url") or None,
        "parent_studio": parent_studio_val,
        "_parent_studio_name": parent_studio_name,
    }


class UpstreamStudioAnalyzer(BaseUpstreamAnalyzer):
    """
    Detects upstream changes in stash-box linked studios.

    Compares studio data from configured stash-box endpoints against
    local Stash data, using stored snapshots for 3-way diffing.
    """

    type = "upstream_studio_changes"
    logic_version = 3  # v3: drop numeric ID fallback for unlinked parent studios

    @property
    def entity_type(self) -> str:
        return "studio"

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        self._current_endpoint = endpoint
        return await self.stash.get_studios_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client: StashBoxClient, stashbox_id: str) -> Optional[dict]:
        return await stashbox_client.get_studio(stashbox_id)

    def _build_local_data(self, entity: dict) -> dict:
        return _build_local_studio_data(entity, self._current_endpoint)

    def _normalize_upstream(self, raw_data: dict) -> dict:
        return normalize_upstream_studio(raw_data)

    def _get_default_fields(self) -> set[str]:
        return DEFAULT_STUDIO_FIELDS

    def _get_field_labels(self) -> dict[str, str]:
        return STUDIO_FIELD_LABELS

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        return diff_studio_fields(local_data, upstream_data, snapshot, enabled_fields)

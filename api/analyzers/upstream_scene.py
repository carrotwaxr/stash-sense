"""Upstream Scene Change Analyzer.

Detects field changes between local scenes and their stash-box counterparts.
Handles both simple fields (title, date, etc.) and relational fields
(studio, performers, tags) with set-based comparison.
"""

import logging
from typing import Optional

from .base_upstream import BaseUpstreamAnalyzer
from config import get_stashbox_shortname
from upstream_field_mapper import (
    normalize_upstream_scene,
    diff_scene_fields,
    DEFAULT_SCENE_FIELDS,
    SCENE_FIELD_LABELS,
)

logger = logging.getLogger(__name__)


def _has_scene_changes(result: dict) -> bool:
    """Check if a scene diff result has any actual changes."""
    if result.get("changes"):
        return True
    if result.get("studio_change") is not None:
        return True
    pc = result.get("performer_changes", {})
    if pc.get("added") or pc.get("removed") or pc.get("alias_changed"):
        return True
    tc = result.get("tag_changes", {})
    if tc.get("added") or tc.get("removed"):
        return True
    return False


class UpstreamSceneAnalyzer(BaseUpstreamAnalyzer):
    """Detect upstream changes to scenes linked to stash-box endpoints."""

    type = "upstream_scene_changes"
    _current_endpoint: str = ""

    @property
    def entity_type(self) -> str:
        return "scene"

    async def _process_endpoint(
        self, endpoint: str, api_key: str, incremental: bool,
        skip_local_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        """Store the current endpoint before processing for stash_id filtering."""
        self._current_endpoint = endpoint
        return await super()._process_endpoint(endpoint, api_key, incremental, skip_local_ids)

    async def _get_local_entities(self, endpoint: str) -> list[dict]:
        return await self.stash.get_scenes_for_endpoint(endpoint)

    async def _get_upstream_entity(self, stashbox_client, stashbox_id: str) -> Optional[dict]:
        return await stashbox_client.get_scene(stashbox_id)

    def _build_local_data(self, entity: dict) -> dict:
        """Build comparable data from a local Stash scene.

        Maps local entity IDs to stashbox IDs via stash_ids for comparison
        with upstream data (which uses stashbox IDs natively).
        Filters stash_ids by the current endpoint to avoid cross-endpoint mismatches.
        """
        endpoint = self._current_endpoint
        performers = []
        for p in (entity.get("performers") or []):
            perf_stash_id = None
            for sid in (p.get("stash_ids") or []):
                if sid["endpoint"] == endpoint:
                    perf_stash_id = sid["stash_id"]
                    break
            if perf_stash_id:
                performers.append({"id": perf_stash_id, "name": p.get("name"), "as": None})

        tags = []
        for t in (entity.get("tags") or []):
            for sid in (t.get("stash_ids") or []):
                if sid["endpoint"] == endpoint:
                    tags.append({"id": sid["stash_id"], "name": t.get("name")})
                    break

        studio = None
        local_studio = entity.get("studio")
        if local_studio:
            for sid in (local_studio.get("stash_ids") or []):
                if sid["endpoint"] == endpoint:
                    studio = {"id": sid["stash_id"], "name": local_studio.get("name")}
                    break

        urls = entity.get("urls") or []
        if isinstance(urls, str):
            urls = [urls] if urls else []

        return {
            "title": entity.get("title") or "",
            "date": entity.get("date") or "",
            "details": entity.get("details") or "",
            "director": entity.get("director") or "",
            "code": entity.get("code") or "",
            "urls": urls,
            "studio": studio,
            "performers": performers,
            "tags": tags,
        }

    def _normalize_upstream(self, raw_data: dict) -> dict:
        return normalize_upstream_scene(raw_data)

    def _get_default_fields(self) -> set[str]:
        return DEFAULT_SCENE_FIELDS

    def _get_field_labels(self) -> dict[str, str]:
        return SCENE_FIELD_LABELS

    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        """Override: returns scene diff dict if changes exist, empty list if not.

        The base class checks `if not changes:` - so we return [] (falsy)
        when there are no changes, and the full dict (truthy) when there are.
        The _build_recommendation_details override handles the dict format.
        """
        result = diff_scene_fields(local_data, upstream_data, snapshot, enabled_fields)
        if not _has_scene_changes(result):
            return []
        return result

    def _build_recommendation_details(
        self,
        endpoint: str,
        endpoint_name: str,
        stash_box_id: str,
        local_entity: dict,
        updated_at: Optional[str],
        changes,
    ) -> dict:
        """Build scene-specific recommendation details.

        The `changes` param is a dict from diff_scene_fields (not a list like other entities).
        """
        # Extract current local entity IDs so the UI can merge (not replace) on apply
        current_performer_ids = [
            str(p["id"]) for p in (local_entity.get("performers") or [])
        ]
        current_tag_ids = [
            str(t["id"]) for t in (local_entity.get("tags") or [])
        ]
        current_studio_id = None
        local_studio = local_entity.get("studio")
        if local_studio:
            current_studio_id = str(local_studio["id"])

        details = {
            "endpoint": endpoint,
            "endpoint_name": endpoint_name,
            "stash_box_id": stash_box_id,
            "scene_id": str(local_entity["id"]),
            "scene_name": local_entity.get("title", ""),
            "upstream_updated_at": updated_at,
            "current_performer_ids": current_performer_ids,
            "current_tag_ids": current_tag_ids,
            "current_studio_id": current_studio_id,
        }
        details["changes"] = changes.get("changes", [])
        details["studio_change"] = changes.get("studio_change")
        details["performer_changes"] = changes.get("performer_changes")
        details["tag_changes"] = changes.get("tag_changes")
        return details

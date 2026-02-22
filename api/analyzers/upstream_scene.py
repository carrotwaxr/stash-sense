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
    _performer_name_lookup: dict[str, str] | None = None
    _tag_name_lookup: dict[str, str] | None = None
    _studio_name_lookup: dict[str, str] | None = None

    @property
    def entity_type(self) -> str:
        return "scene"

    async def _build_name_lookups(self):
        """Build name→local_id lookup dicts for auto-matching added entities."""
        if self._performer_name_lookup is not None:
            return  # Already built for this run

        # Performers: name + aliases
        self._performer_name_lookup = {}
        all_performers = await self.stash.get_all_performers()
        for p in all_performers:
            pid = str(p["id"])
            name = (p.get("name") or "").strip().lower()
            if name:
                self._performer_name_lookup[name] = pid
            for alias in (p.get("alias_list") or []):
                alias_lower = alias.strip().lower()
                if alias_lower:
                    # Don't overwrite a primary name match
                    self._performer_name_lookup.setdefault(alias_lower, pid)

        # Tags: name only (allTags doesn't include aliases)
        self._tag_name_lookup = {}
        all_tags = await self.stash.get_all_tags()
        for t in all_tags:
            name = (t.get("name") or "").strip().lower()
            if name:
                self._tag_name_lookup[name] = str(t["id"])

        # Studios: name + aliases
        self._studio_name_lookup = {}
        all_studios = await self.stash.get_all_studios()
        for s in all_studios:
            sid = str(s["id"])
            name = (s.get("name") or "").strip().lower()
            if name:
                self._studio_name_lookup[name] = sid
            for alias in (s.get("aliases") or []):
                alias_lower = alias.strip().lower()
                if alias_lower:
                    self._studio_name_lookup.setdefault(alias_lower, sid)

        logger.info(
            f"Name lookups built: {len(self._performer_name_lookup)} performer names, "
            f"{len(self._tag_name_lookup)} tags, {len(self._studio_name_lookup)} studios"
        )

    async def _process_endpoint(
        self, endpoint: str, api_key: str, incremental: bool,
        skip_local_ids: set[str] | None = None,
    ) -> tuple[int, int]:
        """Store the current endpoint before processing for stash_id filtering."""
        self._current_endpoint = endpoint
        await self._build_name_lookups()
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
        current_performers = [
            {"id": str(p["id"]), "name": p.get("name", "")}
            for p in (local_entity.get("performers") or [])
        ]
        current_tags = [
            {"id": str(t["id"]), "name": t.get("name", "")}
            for t in (local_entity.get("tags") or [])
        ]
        current_studio_id = None
        current_studio = None
        local_studio = local_entity.get("studio")
        if local_studio:
            current_studio_id = str(local_studio["id"])
            current_studio = {"id": current_studio_id, "name": local_studio.get("name", "")}

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
            "current_performers": current_performers,
            "current_tags": current_tags,
            "current_studio": current_studio,
        }
        details["changes"] = changes.get("changes", [])
        details["studio_change"] = changes.get("studio_change")
        details["performer_changes"] = changes.get("performer_changes")
        details["tag_changes"] = changes.get("tag_changes")

        # Enrich added entities with local matches for auto-linking
        pc = details.get("performer_changes") or {}
        if pc.get("added") and self._performer_name_lookup:
            for perf in pc["added"]:
                match_id = self._performer_name_lookup.get(
                    (perf.get("name") or "").strip().lower()
                )
                if not match_id:
                    for alias in (perf.get("aliases") or []):
                        match_id = self._performer_name_lookup.get(alias.strip().lower())
                        if match_id:
                            break
                if match_id:
                    perf["local_match"] = {"id": match_id}

        tc = details.get("tag_changes") or {}
        if tc.get("added") and self._tag_name_lookup:
            for tag in tc["added"]:
                name = (tag.get("name") or "").strip().lower()
                match_id = self._tag_name_lookup.get(name)
                if match_id:
                    tag["local_match"] = {"id": match_id}

        sc = details.get("studio_change")
        if sc and sc.get("upstream") and self._studio_name_lookup:
            upstream_studio = sc["upstream"]
            match_id = self._studio_name_lookup.get(
                (upstream_studio.get("name") or "").strip().lower()
            )
            if not match_id:
                for alias in (upstream_studio.get("aliases") or []):
                    match_id = self._studio_name_lookup.get(alias.strip().lower())
                    if match_id:
                        break
            if match_id:
                upstream_studio["local_match"] = {"id": match_id}

        return details

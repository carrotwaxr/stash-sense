"""
Base Upstream Analyzer

Generic base class for analyzers that detect upstream changes by comparing
data from stash-box endpoints against local Stash data using 3-way diffing
with stored snapshots.

Subclasses implement entity-specific methods (fetching, normalization, diffing)
while the base handles the common workflow: iterate endpoints, process entities,
manage watermarks, handle dismissals, and upsert recommendations.
"""

import logging
from abc import abstractmethod
from typing import Optional

from .base import BaseAnalyzer, AnalysisResult
from config import get_stashbox_shortname

logger = logging.getLogger(__name__)


class BaseUpstreamAnalyzer(BaseAnalyzer):
    """
    Base class for upstream change detection analyzers.

    Subclasses must implement the abstract methods to specify:
    - What entity type they handle (performer, tag, studio, etc.)
    - How to fetch local and upstream entities
    - How to build comparable data dicts from local entities
    - How to normalize upstream data
    - What fields to monitor by default
    - Human-readable field labels
    """

    @property
    @abstractmethod
    def entity_type(self) -> str:
        """Return the entity type string, e.g. 'performer', 'tag', 'studio'."""
        ...

    @abstractmethod
    def _get_local_entities(self, endpoint: str) -> list[dict]:
        """Fetch local entities linked to the given stash-box endpoint.

        This is called as a coroutine (async). Must return a list of entity dicts,
        each containing 'id' and 'stash_ids' fields.
        """
        ...

    @abstractmethod
    async def _get_upstream_entity(self, stashbox_client, stashbox_id: str) -> Optional[dict]:
        """Fetch a single entity from a stash-box endpoint by its stash_box_id.

        Args:
            stashbox_client: A StashBoxClient instance.
            stashbox_id: The entity's stash-box ID.

        Returns the raw upstream entity dict, or None if not found.
        """
        ...

    @abstractmethod
    def _build_local_data(self, entity: dict) -> dict:
        """Extract comparable field values from a local entity into a flat dict.

        The dict keys should use the canonical field names used by the diff engine.
        """
        ...

    @abstractmethod
    def _normalize_upstream(self, raw_data: dict) -> dict:
        """Normalize raw upstream entity data into a comparable dict.

        The dict keys should use the same canonical field names as _build_local_data.
        """
        ...

    @abstractmethod
    def _get_default_fields(self) -> set[str]:
        """Return the default set of field names to monitor."""
        ...

    @abstractmethod
    def _get_field_labels(self) -> dict[str, str]:
        """Return a mapping of field name to human-readable label."""
        ...

    @abstractmethod
    def _diff_fields(
        self,
        local_data: dict,
        upstream_data: dict,
        snapshot: Optional[dict],
        enabled_fields: set[str],
    ) -> list[dict]:
        """Compute the 3-way diff between local, upstream, and snapshot data.

        Returns a list of change dicts, each with keys:
            field, field_label, local_value, upstream_value,
            previous_upstream_value, merge_type
        """
        ...

    def _is_upstream_deleted(self, upstream: dict) -> bool:
        """Check if the upstream entity is deleted or merged.

        Default implementation checks 'deleted' and 'merged_into_id' fields.
        Subclasses can override for different deletion indicators.
        """
        return upstream.get("deleted") or upstream.get("merged_into_id")

    def _get_upstream_updated_at(self, upstream: dict) -> Optional[str]:
        """Extract the updated_at timestamp from upstream data.

        Default implementation reads the 'updated' field.
        Subclasses can override if the timestamp field has a different name.
        """
        return upstream.get("updated")

    def _build_recommendation_details(
        self,
        endpoint: str,
        endpoint_name: str,
        stash_box_id: str,
        local_entity: dict,
        updated_at: Optional[str],
        changes: list[dict],
    ) -> dict:
        """Build the details dict for a recommendation.

        Default implementation includes common fields. Subclasses can override
        to add entity-specific details (e.g., performer_image_path).
        """
        return {
            "endpoint": endpoint,
            "endpoint_name": endpoint_name,
            "stash_box_id": stash_box_id,
            f"{self.entity_type}_id": local_entity["id"],
            f"{self.entity_type}_name": local_entity.get("name", ""),
            "upstream_updated_at": updated_at,
            "changes": changes,
        }

    def _create_stashbox_client(self, endpoint: str, api_key: str):
        """Create a StashBoxClient for the given endpoint.

        Subclasses can override to control which StashBoxClient class is used,
        enabling test patching at the subclass module level.
        """
        from stashbox_client import StashBoxClient
        return StashBoxClient(endpoint, api_key)

    def _build_local_lookup(self, entities: list[dict], endpoint: str) -> dict[str, dict]:
        """Build a lookup dict mapping stash_box_id to local entity.

        Default implementation iterates stash_ids on each entity.
        Subclasses can override if entities use a different linking structure.
        """
        local_lookup = {}
        for entity in entities:
            for sid in entity.get("stash_ids", []):
                if sid["endpoint"] == endpoint:
                    local_lookup[sid["stash_id"]] = entity
        return local_lookup

    async def run(self, incremental: bool = True) -> AnalysisResult:
        """Run the upstream change detection analysis.

        Iterates over all configured stash-box endpoints, processes each one,
        and aggregates results.
        """
        connections = await self.stash.get_stashbox_connections()

        settings = self.rec_db.get_settings(self.type)
        endpoint_config = {}
        if settings and settings.config:
            endpoint_config = settings.config.get("endpoints", {})

        total_processed = 0
        total_created = 0
        errors = []

        logger.info(
            f"Starting upstream {self.entity_type} analysis (incremental={incremental}), "
            f"{len(connections)} endpoints"
        )

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
                logger.error(error_msg, exc_info=True)
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
        local_entities = await self._get_local_entities(endpoint)

        if not local_entities:
            return 0, 0

        local_lookup = self._build_local_lookup(local_entities, endpoint)

        logger.warning(
            f"Endpoint {endpoint}: {len(local_lookup)} linked {self.entity_type}s to check"
        )
        self.set_items_total(len(local_lookup))

        watermark = None
        if incremental:
            wm = self.rec_db.get_watermark(f"{self.type}:{endpoint}")
            if wm and wm.get("last_cursor"):
                watermark = wm["last_cursor"]

        enabled_fields = self.rec_db.get_enabled_fields(endpoint, self.entity_type)
        if enabled_fields is None:
            enabled_fields = self._get_default_fields()

        sbc = self._create_stashbox_client(endpoint, api_key)
        latest_updated_at = None
        created = 0
        processed = 0
        skipped = 0

        for i, (stash_box_id, local_entity) in enumerate(local_lookup.items()):
            local_id = local_entity["id"]

            if self.rec_db.is_permanently_dismissed(self.type, self.entity_type, local_id):
                skipped += 1
                continue

            try:
                up = await self._get_upstream_entity(sbc, stash_box_id)
            except Exception as e:
                logger.warning(
                    f"Failed to fetch {self.entity_type} {stash_box_id}: {e}"
                )
                continue

            if not up:
                continue

            updated_at = self._get_upstream_updated_at(up)

            if latest_updated_at is None or (updated_at and updated_at > latest_updated_at):
                latest_updated_at = updated_at

            # In incremental mode, skip entities not updated since last run
            if watermark and updated_at and updated_at <= watermark:
                skipped += 1
                continue

            if self._is_upstream_deleted(up):
                continue

            normalized = self._normalize_upstream(up)

            snapshot_row = self.rec_db.get_upstream_snapshot(
                entity_type=self.entity_type,
                endpoint=endpoint,
                stash_box_id=stash_box_id,
            )
            snapshot = snapshot_row["upstream_data"] if snapshot_row else None

            local_data = self._build_local_data(local_entity)

            changes = self._diff_fields(
                local_data, normalized, snapshot, enabled_fields
            )

            processed += 1

            if (i + 1) % 50 == 0:
                logger.warning(
                    f"Progress: {i + 1}/{len(local_lookup)} checked, "
                    f"{processed} compared, {created} new recs"
                )
                self.update_progress(processed, created)

            if not changes:
                # No differences: safe to update snapshot baseline
                self.rec_db.upsert_upstream_snapshot(
                    entity_type=self.entity_type,
                    local_entity_id=local_id,
                    endpoint=endpoint,
                    stash_box_id=stash_box_id,
                    upstream_data=normalized,
                    upstream_updated_at=updated_at or "",
                )
                # Auto-resolve any stale pending recommendation
                stale = self.rec_db.get_recommendation_by_target(
                    self.type, self.entity_type, local_id, status="pending"
                )
                if stale:
                    self.rec_db.resolve_recommendation(
                        stale.id, action="auto_resolved",
                        details={"reason": "no_differences"},
                    )
                continue

            endpoint_name = get_stashbox_shortname(endpoint)
            details = self._build_recommendation_details(
                endpoint=endpoint,
                endpoint_name=endpoint_name,
                stash_box_id=stash_box_id,
                local_entity=local_entity,
                updated_at=updated_at,
                changes=changes,
            )

            # Check for existing pending recommendation
            existing_pending = self.rec_db.get_recommendation_by_target(
                self.type, self.entity_type, local_id, status="pending"
            )

            if existing_pending:
                self.rec_db.update_recommendation_details(existing_pending.id, details)
            else:
                # Check for dismissed recommendation that can be reopened
                existing_dismissed = self.rec_db.get_recommendation_by_target(
                    self.type, self.entity_type, local_id, status="dismissed"
                )

                if existing_dismissed:
                    self.rec_db.undismiss(self.type, self.entity_type, local_id)
                    self.rec_db.reopen_recommendation(existing_dismissed.id, details)
                    created += 1
                else:
                    rec_id = self.create_recommendation(
                        target_type=self.entity_type,
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

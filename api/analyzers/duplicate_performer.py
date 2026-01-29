"""
Duplicate Performer Analyzer

Detects performers that share the same stash_id, indicating duplicates
that should be merged.

Ported from: stash-plugins/scripts/duplicate-performer-finder/
"""

from .base import BaseAnalyzer, AnalysisResult


def get_total_content_count(performer: dict) -> int:
    """Calculate total content count for a performer."""
    return (
        (performer.get("scene_count") or 0)
        + (performer.get("image_count") or 0)
        + (performer.get("gallery_count") or 0)
    )


class DuplicatePerformerAnalyzer(BaseAnalyzer):
    """
    Finds performers that share the same stash_id for the same endpoint.

    This indicates duplicate entries that should be merged.
    """

    type = "duplicate_performer"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        """
        Run duplicate performer analysis.

        Note: This analyzer doesn't support incremental mode well since
        we need to compare all performers against each other. For now,
        it always runs a full scan.
        """
        # Fetch all performers
        performers = await self.stash.get_all_performers()

        # Find duplicates (same stash_id + endpoint)
        duplicates = self._find_duplicates(performers)

        # Create recommendations
        created = 0
        for (endpoint, stash_id), group in duplicates.items():
            # Sort by content count - suggest keeping the one with most content
            sorted_performers = sorted(
                group,
                key=get_total_content_count,
                reverse=True,
            )

            # Use first performer's ID as the target (anchor)
            target_id = sorted_performers[0]["id"]

            # Build details with suggestion
            performer_details = []
            for i, p in enumerate(sorted_performers):
                performer_details.append({
                    "id": p["id"],
                    "name": p["name"],
                    "scene_count": p.get("scene_count", 0),
                    "image_count": p.get("image_count", 0),
                    "gallery_count": p.get("gallery_count", 0),
                    "total_content": get_total_content_count(p),
                    "is_suggested_keeper": i == 0,
                    "image_path": p.get("image_path"),
                })

            rec_id = self.create_recommendation(
                target_type="performer",
                target_id=target_id,
                details={
                    "endpoint": endpoint,
                    "stash_id": stash_id,
                    "performers": performer_details,
                    "suggested_keeper_id": target_id,
                },
                confidence=1.0,  # Deterministic - same stash_id is certain
            )

            if rec_id:
                created += 1

        return AnalysisResult(
            items_processed=len(performers),
            recommendations_created=created,
        )

    def _find_duplicates(
        self, performers: list[dict]
    ) -> dict[tuple[str, str], list[dict]]:
        """
        Find performers that share the same stash_id for the same endpoint.

        Returns:
            Dict mapping (endpoint, stash_id) tuples to lists of duplicate performers.
            Only includes groups with 2+ distinct performers.
        """
        buckets: dict[tuple[str, str], list[dict]] = {}
        seen: dict[tuple[str, str], set[str]] = {}

        for performer in performers:
            stash_ids = performer.get("stash_ids") or []
            performer_id = performer.get("id")

            for sid in stash_ids:
                key = (sid["endpoint"], sid["stash_id"])
                if key not in buckets:
                    buckets[key] = []
                    seen[key] = set()

                # Only add if this performer isn't already in this bucket
                if performer_id not in seen[key]:
                    buckets[key].append(performer)
                    seen[key].add(performer_id)

        # Filter to only actual duplicates (2+ distinct performers)
        return {k: v for k, v in buckets.items() if len(v) >= 2}

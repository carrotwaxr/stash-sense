"""Field mapping and 3-way diff engine for upstream entity sync.

Maps fields between stash-box schema and local Stash schema,
and computes 3-way diffs (local vs upstream vs previous snapshot)
to detect meaningful upstream changes.

Supports a registry pattern so new entity types can register their
field configurations alongside performers.
"""

import re
from typing import Optional


# All default monitored performer field names (using local Stash field names)
DEFAULT_PERFORMER_FIELDS: set[str] = {
    "name",
    "disambiguation",
    "aliases",
    "gender",
    "birthdate",
    "death_date",
    "ethnicity",
    "country",
    "eye_color",
    "hair_color",
    "height",
    "cup_size",
    "band_size",
    "waist_size",
    "hip_size",
    "breast_type",
    "tattoos",
    "piercings",
    "career_start_year",
    "career_end_year",
    "urls",
    "details",
}

# Merge type for each field, controls how diffs and merges are handled
FIELD_MERGE_TYPES: dict[str, str] = {
    "name": "name",
    "disambiguation": "simple",
    "aliases": "alias_list",
    "gender": "simple",
    "birthdate": "simple",
    "death_date": "simple",
    "ethnicity": "simple",
    "country": "simple",
    "eye_color": "simple",
    "hair_color": "simple",
    "height": "simple",
    "cup_size": "simple",
    "band_size": "simple",
    "waist_size": "simple",
    "hip_size": "simple",
    "breast_type": "simple",
    "tattoos": "text",
    "piercings": "text",
    "career_start_year": "simple",
    "career_end_year": "simple",
    "urls": "alias_list",
    "details": "text",
}

# Human-readable labels for each field
FIELD_LABELS: dict[str, str] = {
    "name": "Name",
    "disambiguation": "Disambiguation",
    "aliases": "Aliases",
    "gender": "Gender",
    "birthdate": "Birthdate",
    "death_date": "Death Date",
    "ethnicity": "Ethnicity",
    "country": "Country",
    "eye_color": "Eye Color",
    "hair_color": "Hair Color",
    "height": "Height",
    "cup_size": "Cup Size",
    "band_size": "Band Size",
    "waist_size": "Waist Size",
    "hip_size": "Hip Size",
    "breast_type": "Breast Type",
    "tattoos": "Tattoos",
    "piercings": "Piercings",
    "career_start_year": "Career Start Year",
    "career_end_year": "Career End Year",
    "urls": "URLs",
    "details": "Details",
}

# ==================== Entity Field Config Registry ====================
#
# Registry of field configurations keyed by entity type.
# Each entry contains:
#   - default_fields: set of field names monitored by default
#   - labels: dict mapping field name to human-readable label
#   - merge_types: dict mapping field name to merge type string
#
# New entity types can be registered via register_entity_fields().
ENTITY_FIELD_CONFIGS: dict[str, dict] = {
    "performer": {
        "default_fields": DEFAULT_PERFORMER_FIELDS,
        "labels": FIELD_LABELS,
        "merge_types": FIELD_MERGE_TYPES,
    },
}


def get_field_config(entity_type: str) -> dict:
    """Get field configuration for an entity type.

    Returns a dict with keys: default_fields, labels, merge_types.
    Raises KeyError if the entity type is not registered.
    """
    if entity_type not in ENTITY_FIELD_CONFIGS:
        raise KeyError(f"Unknown entity type: {entity_type!r}. "
                       f"Registered types: {list(ENTITY_FIELD_CONFIGS.keys())}")
    return ENTITY_FIELD_CONFIGS[entity_type]


def register_entity_fields(entity_type: str, config: dict):
    """Register field configuration for a new entity type.

    Args:
        entity_type: The entity type string (e.g. 'tag', 'studio').
        config: Dict with keys 'default_fields', 'labels', 'merge_types'.
    """
    required_keys = {"default_fields", "labels", "merge_types"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    ENTITY_FIELD_CONFIGS[entity_type] = config


# ==================== Tag Field Config ====================

DEFAULT_TAG_FIELDS: set[str] = {
    "name",
    "description",
    "aliases",
}

TAG_FIELD_MERGE_TYPES: dict[str, str] = {
    "name": "name",
    "description": "text",
    "aliases": "alias_list",
}

TAG_FIELD_LABELS: dict[str, str] = {
    "name": "Name",
    "description": "Description",
    "aliases": "Aliases",
}

# Register tag fields in the entity config registry
ENTITY_FIELD_CONFIGS["tag"] = {
    "default_fields": DEFAULT_TAG_FIELDS,
    "labels": TAG_FIELD_LABELS,
    "merge_types": TAG_FIELD_MERGE_TYPES,
}


def normalize_upstream_tag(upstream: dict) -> dict:
    """Convert stash-box tag data to normalized dict using local Stash field names.

    Handles:
    - Direct field mappings (name, description)
    - Aliases normalized from None to empty list
    - Category stored for reference but not diffed
    """
    result = {}

    for field_name in ("name", "description"):
        if field_name in upstream:
            result[field_name] = upstream[field_name]

    # Aliases: normalize None to empty list
    if "aliases" in upstream:
        result["aliases"] = upstream["aliases"] if upstream["aliases"] is not None else []

    return result


def diff_tag_fields(
    local: dict,
    upstream: dict,
    snapshot: Optional[dict],
    enabled_fields: set[str],
) -> list[dict]:
    """Convenience wrapper: 3-way diff using tag field config."""
    return diff_fields(
        local=local,
        upstream=upstream,
        snapshot=snapshot,
        enabled_fields=enabled_fields,
        merge_types=TAG_FIELD_MERGE_TYPES,
        labels=TAG_FIELD_LABELS,
    )


# ==================== Studio Field Config ====================

DEFAULT_STUDIO_FIELDS: set[str] = {
    "name",
    "urls",
    "parent_studio",
}

STUDIO_FIELD_MERGE_TYPES: dict[str, str] = {
    "name": "name",
    "urls": "alias_list",
    "parent_studio": "simple",
}

STUDIO_FIELD_LABELS: dict[str, str] = {
    "name": "Name",
    "urls": "URLs",
    "parent_studio": "Parent Studio",
}

# Register studio fields in the entity config registry
ENTITY_FIELD_CONFIGS["studio"] = {
    "default_fields": DEFAULT_STUDIO_FIELDS,
    "labels": STUDIO_FIELD_LABELS,
    "merge_types": STUDIO_FIELD_MERGE_TYPES,
}


def normalize_upstream_studio(upstream: dict) -> dict:
    """Convert stash-box studio data to normalized dict using local Stash field names.

    Handles:
    - Direct name mapping
    - URLs extracted from [{url, type}] objects to plain string list
    - Parent studio extracted as StashBox UUID (resolved to local ID at action time)
    """
    result = {}

    if "name" in upstream:
        result["name"] = upstream["name"]

    # URLs: extract url strings from [{url, ...}] objects, same as performers
    if "urls" in upstream:
        raw_urls = upstream["urls"]
        if raw_urls is None:
            result["urls"] = []
        else:
            result["urls"] = [
                u.get("url") if isinstance(u, dict) else u
                for u in raw_urls
            ]

    # Parent: extract StashBox UUID and name
    if "parent" in upstream:
        parent = upstream["parent"]
        result["parent_studio"] = parent["id"] if parent else None
        result["_parent_studio_name"] = parent["name"] if parent else None

    return result


def diff_studio_fields(
    local: dict,
    upstream: dict,
    snapshot: Optional[dict],
    enabled_fields: set[str],
) -> list[dict]:
    """Convenience wrapper: 3-way diff using studio field config.

    Enriches parent_studio changes with display-friendly names so the UI
    can show "Studio Name" instead of raw stashbox UUIDs.
    """
    changes = diff_fields(
        local=local,
        upstream=upstream,
        snapshot=snapshot,
        enabled_fields=enabled_fields,
        merge_types=STUDIO_FIELD_MERGE_TYPES,
        labels=STUDIO_FIELD_LABELS,
    )

    # Enrich parent_studio changes with display names
    for change in changes:
        if change["field"] == "parent_studio":
            change["local_display"] = local.get("_parent_studio_name") or change["local_value"]
            change["upstream_display"] = upstream.get("_parent_studio_name") or change["upstream_value"]

    return changes


# ==================== Performer-Specific Mapping Constants ====================

# Fields that map directly from upstream to local with the same name
_DIRECT_FIELDS = [
    "name",
    "disambiguation",
    "gender",
    "country",
    "eye_color",
    "hair_color",
    "height",
    "cup_size",
    "band_size",
    "waist_size",
    "hip_size",
    "breast_type",
    "career_start_year",
    "career_end_year",
    "death_date",
    "details",
    "ethnicity",
]

# Fields that have different names upstream vs local: upstream_name -> local_name
_RENAMED_FIELDS = {
    "birth_date": "birthdate",
}


def _format_body_modifications(mods: Optional[list[dict]]) -> Optional[str]:
    """Format a list of body modification dicts to a text string.

    Each modification has 'location' and 'description' fields.
    Output format: "Location: Description; ..." with description omitted if empty.
    Returns None if the input is None.
    """
    if mods is None:
        return None
    parts = []
    for mod in mods:
        location = mod.get("location", "")
        description = mod.get("description", "")
        if description:
            parts.append(f"{location}: {description}")
        else:
            parts.append(location)
    return "; ".join(parts)


def parse_measurements(measurements: Optional[str]) -> dict:
    """Parse a Stash measurements string into individual components.

    Handles formats like "38F-24-35", "34DD-26-36", "38-24-35", "38F", etc.
    Returns dict with keys: band_size, cup_size, waist_size, hip_size (any may be None).
    """
    result = {"band_size": None, "cup_size": None, "waist_size": None, "hip_size": None}
    if not measurements or not measurements.strip():
        return result

    parts = measurements.strip().split("-", 2)

    # Parse bust part (first segment): e.g. "38F", "34DD", "38"
    if parts[0]:
        bust_match = re.match(r"^(\d+)([A-Za-z]+)?$", parts[0].strip())
        if bust_match:
            result["band_size"] = int(bust_match.group(1))
            if bust_match.group(2):
                result["cup_size"] = bust_match.group(2).upper()

    # Waist (second segment)
    if len(parts) > 1 and parts[1].strip():
        try:
            result["waist_size"] = int(parts[1].strip())
        except ValueError:
            pass

    # Hip (third segment)
    if len(parts) > 2 and parts[2].strip():
        try:
            result["hip_size"] = int(parts[2].strip())
        except ValueError:
            pass

    return result


def parse_career_length(career_length: Optional[str]) -> dict:
    """Parse a Stash career_length string into start/end years.

    Handles formats like "2010-2023", "2010-", "2010", etc.
    Returns dict with keys: career_start_year, career_end_year (may be None).
    """
    result = {"career_start_year": None, "career_end_year": None}
    if not career_length or not career_length.strip():
        return result

    parts = career_length.strip().split("-", 1)
    if parts[0].strip():
        try:
            result["career_start_year"] = int(parts[0].strip())
        except ValueError:
            pass
    if len(parts) > 1 and parts[1].strip():
        try:
            result["career_end_year"] = int(parts[1].strip())
        except ValueError:
            pass

    return result


def normalize_upstream_performer(upstream: dict) -> dict:
    """Convert stash-box performer data to normalized dict using local Stash field names.

    Handles:
    - Direct field mappings (same name upstream and local)
    - Renamed fields (birth_date -> birthdate, is_favorite -> favorite)
    - Body modifications (tattoos/piercings) formatted to text
    - URLs extracted from [{url, type}] objects to plain string list
    - Aliases normalized from None to empty list
    """
    result = {}

    # Direct mappings
    for field_name in _DIRECT_FIELDS:
        if field_name in upstream:
            result[field_name] = upstream[field_name]

    # Renamed fields
    for upstream_name, local_name in _RENAMED_FIELDS.items():
        if upstream_name in upstream:
            result[local_name] = upstream[upstream_name]

    # Aliases: normalize None to empty list
    if "aliases" in upstream:
        result["aliases"] = upstream["aliases"] if upstream["aliases"] is not None else []

    # Tattoos and piercings: format from body modification objects to text
    if "tattoos" in upstream:
        result["tattoos"] = _format_body_modifications(upstream["tattoos"])

    if "piercings" in upstream:
        result["piercings"] = _format_body_modifications(upstream["piercings"])

    # URLs: extract url strings from [{url, type}] objects
    if "urls" in upstream:
        raw_urls = upstream["urls"]
        if raw_urls is None:
            result["urls"] = []
        else:
            result["urls"] = [u["url"] for u in raw_urls]

    return result


def _is_empty(value) -> bool:
    """Check if a value is semantically empty (None, empty string, 0, empty list)."""
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    if isinstance(value, (int, float)) and value == 0:
        return True
    if isinstance(value, list) and len(value) == 0:
        return True
    return False


def _values_equal(local_value, upstream_value, merge_type: str) -> bool:
    """Compare two field values for equality.

    - alias_list: case-insensitive set comparison
    - strings: case-insensitive comparison (eliminates BROWN vs Brown false positives)
    - both empty: treats None, "", 0, and [] as equivalent empty values
    - all other types: standard equality
    """
    # Treat all forms of empty as equal (None vs 0, "" vs None, etc.)
    if _is_empty(local_value) and _is_empty(upstream_value):
        return True
    if merge_type == "alias_list":
        local_set = {
            v.lower().rstrip("/") if isinstance(v, str) else v
            for v in (local_value or [])
        }
        upstream_set = {
            v.lower().rstrip("/") if isinstance(v, str) else v
            for v in (upstream_value or [])
        }
        return local_set == upstream_set
    if isinstance(local_value, str) and isinstance(upstream_value, str):
        return local_value.lower() == upstream_value.lower()
    return local_value == upstream_value


def diff_fields(
    local: dict,
    upstream: dict,
    snapshot: Optional[dict],
    enabled_fields: set[str],
    merge_types: dict[str, str],
    labels: dict[str, str],
) -> list[dict]:
    """Generic 3-way diff between local, upstream, and previous snapshot.

    For each enabled field:
    - Skip if local == upstream (already in sync)
    - If snapshot exists: skip if upstream == snapshot (user intentionally
      set local differently; upstream hasn't changed)
    - If snapshot is None (first run): flag all differences

    Args:
        local: Local entity data dict.
        upstream: Upstream entity data dict.
        snapshot: Previous upstream snapshot dict, or None for first run.
        enabled_fields: Set of field names to compare.
        merge_types: Dict mapping field name to merge type string.
        labels: Dict mapping field name to human-readable label.

    Returns a list of change dicts with keys:
        field, field_label, local_value, upstream_value,
        previous_upstream_value, merge_type
    """
    changes = []

    for field_name in sorted(enabled_fields):
        merge_type = merge_types.get(field_name, "simple")

        local_value = local.get(field_name)
        upstream_value = upstream.get(field_name)

        # Skip if already in sync
        if _values_equal(local_value, upstream_value, merge_type):
            continue

        # If we have a snapshot, check if upstream actually changed
        if snapshot is not None:
            previous_upstream_value = snapshot.get(field_name)
            if _values_equal(upstream_value, previous_upstream_value, merge_type):
                # Upstream hasn't changed since snapshot; user set local differently on purpose
                continue
        else:
            previous_upstream_value = None

        changes.append({
            "field": field_name,
            "field_label": labels.get(field_name, field_name),
            "local_value": local_value,
            "upstream_value": upstream_value,
            "previous_upstream_value": previous_upstream_value,
            "merge_type": merge_type,
        })

    return changes


def diff_performer_fields(
    local: dict,
    upstream: dict,
    snapshot: Optional[dict],
    enabled_fields: set[str],
) -> list[dict]:
    """Convenience wrapper: 3-way diff using performer field config.

    Delegates to the generic diff_fields() with performer-specific
    merge types and labels.
    """
    return diff_fields(
        local=local,
        upstream=upstream,
        snapshot=snapshot,
        enabled_fields=enabled_fields,
        merge_types=FIELD_MERGE_TYPES,
        labels=FIELD_LABELS,
    )


# ==================== Scene Field Config ====================

DEFAULT_SCENE_FIELDS: set[str] = {
    "title", "date", "details", "director", "code", "urls",
    "studio", "performers", "tags",
}

SCENE_FIELD_LABELS: dict[str, str] = {
    "title": "Title",
    "date": "Date",
    "details": "Details",
    "director": "Director",
    "code": "Code",
    "urls": "URLs",
    "studio": "Studio",
    "performers": "Performers",
    "tags": "Tags",
}

SCENE_FIELD_MERGE_TYPES: dict[str, str] = {
    "title": "name",
    "date": "simple",
    "details": "text",
    "director": "simple",
    "code": "simple",
    "urls": "alias_list",
    "studio": "simple",
    "performers": "performer_list",
    "tags": "id_list",
}

SCENE_SIMPLE_FIELDS = {"title", "date", "details", "director", "code", "urls"}
SCENE_RELATIONAL_FIELDS = {"studio", "performers", "tags"}


def normalize_upstream_scene(upstream: dict) -> dict:
    """Normalize a StashBox scene to canonical field names."""
    urls = []
    for u in (upstream.get("urls") or []):
        if isinstance(u, dict):
            urls.append(u.get("url", ""))
        else:
            urls.append(str(u))

    performers = []
    for pa in (upstream.get("performers") or []):
        perf = pa.get("performer", {})
        performers.append({
            "id": perf.get("id"),
            "name": perf.get("name"),
            "as": pa.get("as"),
        })

    tags = []
    for t in (upstream.get("tags") or []):
        tags.append({"id": t.get("id"), "name": t.get("name")})

    studio = upstream.get("studio")
    if studio:
        studio = {"id": studio.get("id"), "name": studio.get("name")}

    return {
        "title": upstream.get("title") or "",
        "date": upstream.get("date") or "",
        "details": upstream.get("details") or "",
        "director": upstream.get("director") or "",
        "code": upstream.get("code") or "",
        "urls": urls,
        "studio": studio,
        "performers": performers,
        "tags": tags,
    }


def diff_scene_fields(
    local: dict,
    upstream: dict,
    snapshot: dict | None,
    enabled_fields: set[str],
) -> dict:
    """Compute changes between local and upstream scene data.

    Returns a dict with:
      - changes: list of simple field diffs (same format as other entities)
      - studio_change: dict or None (old/new studio)
      - performer_changes: {added: [], removed: [], alias_changed: []}
      - tag_changes: {added: [], removed: []}
    """
    # Simple field diffs (reuse existing diff_fields)
    simple_changes = diff_fields(
        local=local,
        upstream=upstream,
        snapshot=snapshot,
        enabled_fields=enabled_fields & SCENE_SIMPLE_FIELDS,
        merge_types=SCENE_FIELD_MERGE_TYPES,
        labels=SCENE_FIELD_LABELS,
    )

    # Studio diff
    studio_change = None
    if "studio" in enabled_fields:
        local_studio = local.get("studio")
        upstream_studio = upstream.get("studio")
        if _studio_changed(local_studio, upstream_studio, snapshot):
            studio_change = {
                "local": local_studio,
                "upstream": upstream_studio,
            }

    # Performer diff (set comparison by stashbox ID)
    performer_changes = {"added": [], "removed": [], "alias_changed": []}
    if "performers" in enabled_fields:
        performer_changes = _diff_performers(
            local.get("performers", []),
            upstream.get("performers", []),
            snapshot.get("performers", []) if snapshot else None,
        )

    # Tag diff (set comparison by stashbox ID)
    tag_changes = {"added": [], "removed": []}
    if "tags" in enabled_fields:
        tag_changes = _diff_tags(
            local.get("tags", []),
            upstream.get("tags", []),
            snapshot.get("tags", []) if snapshot else None,
        )

    return {
        "changes": simple_changes,
        "studio_change": studio_change,
        "performer_changes": performer_changes,
        "tag_changes": tag_changes,
    }


def _studio_changed(local_studio, upstream_studio, snapshot):
    """Check if studio changed (3-way aware)."""
    local_id = local_studio.get("id") if local_studio else None
    upstream_id = upstream_studio.get("id") if upstream_studio else None
    if local_id == upstream_id:
        return False
    if snapshot is not None:
        snapshot_studio = snapshot.get("studio")
        snapshot_id = snapshot_studio.get("id") if snapshot_studio else None
        if upstream_id == snapshot_id:
            return False  # user intentionally set local differently
    return True


def _diff_performers(local_perfs, upstream_perfs, snapshot_perfs):
    """Set-based performer diff with alias tracking."""
    local_by_id = {p["id"]: p for p in local_perfs}
    upstream_by_id = {p["id"]: p for p in upstream_perfs}

    # 3-way: only flag changes if upstream differs from snapshot
    if snapshot_perfs is not None:
        snapshot_by_id = {p["id"]: p for p in snapshot_perfs}
        upstream_ids = set(upstream_by_id.keys())
        snapshot_ids = set(snapshot_by_id.keys())
        if upstream_ids == snapshot_ids:
            # Check alias changes too
            aliases_same = all(
                upstream_by_id[pid].get("as") == snapshot_by_id[pid].get("as")
                for pid in upstream_ids
            )
            if aliases_same:
                return {"added": [], "removed": [], "alias_changed": []}

    added = []
    removed = []
    alias_changed = []

    for pid, up_perf in upstream_by_id.items():
        if pid not in local_by_id:
            added.append(up_perf)
        else:
            # Check alias change
            local_alias = local_by_id[pid].get("as")
            upstream_alias = up_perf.get("as")
            if local_alias != upstream_alias:
                alias_changed.append({
                    **up_perf,
                    "local_alias": local_alias,
                    "upstream_alias": upstream_alias,
                })

    for pid, local_perf in local_by_id.items():
        if pid not in upstream_by_id:
            removed.append(local_perf)

    return {"added": added, "removed": removed, "alias_changed": alias_changed}


def _diff_tags(local_tags, upstream_tags, snapshot_tags):
    """Set-based tag diff."""
    local_ids = {t["id"] for t in local_tags}
    upstream_ids = {t["id"] for t in upstream_tags}

    # 3-way: only flag if upstream set changed from snapshot
    if snapshot_tags is not None:
        snapshot_ids = {t["id"] for t in snapshot_tags}
        if upstream_ids == snapshot_ids:
            return {"added": [], "removed": []}

    upstream_by_id = {t["id"]: t for t in upstream_tags}
    local_by_id = {t["id"]: t for t in local_tags}

    added = [upstream_by_id[tid] for tid in upstream_ids - local_ids]
    removed = [local_by_id[tid] for tid in local_ids - upstream_ids]

    return {"added": added, "removed": removed}


# Register scene fields in the entity config registry
ENTITY_FIELD_CONFIGS["scene"] = {
    "default_fields": DEFAULT_SCENE_FIELDS,
    "labels": SCENE_FIELD_LABELS,
    "merge_types": SCENE_FIELD_MERGE_TYPES,
}

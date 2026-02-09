"""Field mapping and 3-way diff engine for upstream performer sync.

Maps fields between stash-box schema and local Stash schema,
and computes 3-way diffs (local vs upstream vs previous snapshot)
to detect meaningful upstream changes.
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
    "favorite",
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
    "favorite": "simple",
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
    "favorite": "Favorite",
}

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
    "is_favorite": "favorite",
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
            v.lower() if isinstance(v, str) else v
            for v in (local_value or [])
        }
        upstream_set = {
            v.lower() if isinstance(v, str) else v
            for v in (upstream_value or [])
        }
        return local_set == upstream_set
    if isinstance(local_value, str) and isinstance(upstream_value, str):
        return local_value.lower() == upstream_value.lower()
    return local_value == upstream_value


def diff_performer_fields(
    local: dict,
    upstream: dict,
    snapshot: Optional[dict],
    enabled_fields: set[str],
) -> list[dict]:
    """Compute 3-way diff between local, upstream, and previous snapshot.

    For each enabled field:
    - Skip if local == upstream (already in sync)
    - If snapshot exists: skip if upstream == snapshot (user intentionally
      set local differently; upstream hasn't changed)
    - If snapshot is None (first run): flag all differences

    Returns a list of change dicts with keys:
        field, field_label, local_value, upstream_value,
        previous_upstream_value, merge_type
    """
    changes = []

    for field_name in sorted(enabled_fields):
        merge_type = FIELD_MERGE_TYPES.get(field_name, "simple")

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
            "field_label": FIELD_LABELS.get(field_name, field_name),
            "local_value": local_value,
            "upstream_value": upstream_value,
            "previous_upstream_value": previous_upstream_value,
            "merge_type": merge_type,
        })

    return changes

# Upstream Performer Sync Design

## Overview

New recommendation type (`upstream_performer_changes`) that detects changes in stash-box linked performers and presents field-level merge controls to the user. Scans configured stash-box endpoints for upstream updates, compares against local Stash data using a 3-way diff (local vs upstream vs previous snapshot), and generates recommendations with inline merge UI.

**Scope:** Performers only. Infrastructure designed to extend to Scenes, Studios, and Tags later.

## Architecture

### Flow

1. User triggers "Check for Upstream Updates" from the recommendations dashboard
2. `UpstreamPerformerAnalyzer` queries local Stash via GraphQL for all performers linked to each enabled stash-box endpoint (using `stash_ids_endpoint` filter)
3. Analyzer queries stash-box sorted by `UPDATED_AT DESC`, paginating with rate limiting until it passes the watermark timestamp (incremental)
4. 3-way diff engine compares upstream fields against local fields, using stored snapshots to distinguish intentional local differences from actual upstream changes
5. Creates/updates recommendations of type `upstream_performer_changes` with per-field diff in `details` JSON
6. UI renders inline diff cards with per-field merge controls
7. User applies selected changes via Stash's `performerUpdate` mutation with client-side validation

### Key Infrastructure

- **`upstream_snapshots` table** - stores last-seen upstream state per performer per endpoint
- **`upstream_field_config` table** - per-endpoint, per-field monitoring toggles
- **New stash-box GraphQL client** - queries performer data from stash-box endpoints (existing client only talks to local Stash)
- **Existing `RateLimiter`** with `Priority.LOW` for scan operations

## Data Model

### New Table: `upstream_snapshots`

```sql
CREATE TABLE upstream_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entity_type TEXT NOT NULL,          -- 'performer' (later: 'scene', 'studio', 'tag')
    local_entity_id TEXT NOT NULL,      -- Stash performer ID
    endpoint TEXT NOT NULL,             -- e.g. 'https://stashdb.org/graphql'
    stash_box_id TEXT NOT NULL,         -- UUID on stash-box
    upstream_data JSON NOT NULL,        -- Full snapshot of tracked fields
    upstream_updated_at TEXT NOT NULL,   -- stash-box's updated_at timestamp
    fetched_at TEXT DEFAULT (datetime('now')),
    UNIQUE(entity_type, endpoint, stash_box_id)
);
CREATE INDEX idx_upstream_entity ON upstream_snapshots(entity_type, endpoint);
CREATE INDEX idx_upstream_stash_box_id ON upstream_snapshots(stash_box_id);
```

### New Table: `upstream_field_config`

```sql
CREATE TABLE upstream_field_config (
    endpoint TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    field_name TEXT NOT NULL,
    enabled INTEGER DEFAULT 1,
    PRIMARY KEY (endpoint, entity_type, field_name)
);
```

Pre-populated with all performer fields enabled by default on first run.

### Modified Table: `dismissed_targets`

Add `permanent` column:

```sql
ALTER TABLE dismissed_targets ADD COLUMN permanent INTEGER DEFAULT 0;
```

- `permanent = 0` (default): "Dismiss this update" - can be resurfaced if upstream changes again after the dismissal
- `permanent = 1`: "Never show updates for this performer" - permanent ignore

### Existing Tables Used

- **`recommendations`** - type = `'upstream_performer_changes'`, unique constraint on `(type, target_type, target_id)`
- **`analysis_watermarks`** - `last_cursor` stores latest `updated_at` timestamp seen per endpoint
- **`recommendation_settings`** - `config` JSON stores per-endpoint settings

### Recommendation `details` JSON

```json
{
    "endpoint": "https://stashdb.org/graphql",
    "endpoint_name": "stashdb.org",
    "stash_box_id": "abc-123-def",
    "performer_id": "42",
    "performer_name": "Current Local Name",
    "performer_image_path": "/performer/42/image",
    "upstream_updated_at": "2026-01-15T10:30:00Z",
    "changes": [
        {
            "field": "name",
            "field_label": "Name",
            "local_value": "Jane Doe",
            "upstream_value": "Jane Smith",
            "previous_upstream_value": "Jane Doe",
            "merge_type": "name"
        },
        {
            "field": "aliases",
            "field_label": "Aliases",
            "local_value": ["JD", "Janey"],
            "upstream_value": ["JD", "Jane D.", "Janey Doe"],
            "previous_upstream_value": ["JD"],
            "merge_type": "alias_list"
        },
        {
            "field": "height",
            "field_label": "Height",
            "local_value": 165,
            "upstream_value": 168,
            "previous_upstream_value": 165,
            "merge_type": "simple"
        }
    ]
}
```

### Merge Types

| Type | UI Control | Used For |
|------|-----------|----------|
| `name` | 5-option radio: keep / accept upstream / accept + demote local to alias / keep + add upstream as alias / custom edit | `name` field |
| `alias_list` | Union checkbox list with source highlighting | `aliases`, `urls` |
| `simple` | 3-option radio: keep local / accept upstream / custom edit | Most scalar fields |
| `text` | 3-option radio + editable textarea | `details`, `tattoos`, `piercings` |

## Analyzer Logic

### `UpstreamPerformerAnalyzer`

Extends `BaseAnalyzer` with `type = "upstream_performer_changes"`.

**Step 1 - Gather local performers per endpoint:**

```
For each enabled stash-box endpoint:
  Query local Stash: findPerformers(
    performer_filter: {stash_ids_endpoint: {endpoint: url, modifier: NOT_NULL}}
  )
  Paginate through all results
  Build lookup: {stash_box_id -> local_performer_data}
  Skip performers with no stash-box IDs (already filtered by query)
```

**Step 2 - Incremental upstream fetch:**

```
Load watermark from analysis_watermarks (type: 'upstream_performer_changes')
Query stash-box: queryPerformers(sort: UPDATED_AT, direction: DESC, per_page: 25)
Paginate with rate limiting (Priority.LOW) until:
  - performer.updated < watermark (nothing new beyond this point)
  - OR all results exhausted (first run baseline)
For each fetched performer:
  If stash_box_id in local lookup → candidate for diff
  Store/update upstream_snapshots with fetched data
Update watermark to latest updated_at seen
```

Note: stash-box supports sort by `UPDATED_AT` but not filter by it, so we paginate and stop at the watermark.

**Step 3 - Diff and create recommendations:**

```
For each candidate performer:
  Load enabled fields from upstream_field_config
  Load previous snapshot from upstream_snapshots
  Compare upstream vs local for each enabled field

  3-way diff logic:
    If no previous snapshot (first run):
      Flag all differences as potential updates
    If previous snapshot exists:
      Skip fields where upstream == previous snapshot
        (means user intentionally set local differently)
      Flag fields where upstream != previous snapshot
        (actual upstream change)

  If any meaningful differences:
    Check dismissed_targets:
      permanent = 1 → skip entirely
      permanent = 0, but upstream changed since dismissal → new recommendation
      permanent = 0, upstream unchanged → skip
    Check existing pending recommendation:
      Exists → update details JSON with latest diff
      Not exists → create new recommendation
```

### 3-Way Diff Interpretation

| Upstream vs Snapshot | Local vs Upstream | Action |
|---------------------|-------------------|--------|
| Unchanged | Different | Skip - user intentionally set local value |
| Changed | Different (local matches old upstream) | Recommend update - clear upstream change |
| Changed | Different (local differs from both) | Flag as conflict - both sides changed |
| Changed | Same | Skip - already in sync |

## Stash-Box GraphQL Client

New client class (or methods on existing client) for querying stash-box endpoints directly.

### Required Queries

**Query performers (paginated, sorted by updated_at):**
```graphql
query QueryPerformers($input: PerformerQueryInput!) {
  queryPerformers(input: $input) {
    count
    performers {
      id
      name
      disambiguation
      aliases
      gender
      birth_date
      death_date
      ethnicity
      country
      eye_color
      hair_color
      height
      cup_size
      band_size
      waist_size
      hip_size
      breast_type
      career_start_year
      career_end_year
      tattoos { location description }
      piercings { location description }
      urls { url type }
      is_favorite
      deleted
      merged_into_id
      created
      updated
    }
  }
}
```

Variables:
```json
{
  "input": {
    "page": 1,
    "per_page": 25,
    "sort": "UPDATED_AT",
    "direction": "DESC"
  }
}
```

### Rate Limiting

Uses existing `RateLimiter` with `Priority.LOW` for scanning. Per-endpoint delay is configurable (default 0.5s). Respects 429 responses with exponential backoff.

## Stash ↔ Stash-Box Field Mapping

| Stash-box Field | Stash Field | Normalization |
|-----------------|-------------|---------------|
| `name` | `name` | Direct |
| `disambiguation` | `disambiguation` | Direct |
| `aliases` | `alias_list` | Direct (both string arrays) |
| `gender` | `gender` | Enum → string normalization |
| `birth_date` | `birthdate` | String format alignment |
| `death_date` | `death_date` | Direct |
| `ethnicity` | `ethnicity` | Enum → free text |
| `country` | `country` | Direct |
| `eye_color` | `eye_color` | Enum → free text |
| `hair_color` | `hair_color` | Enum → free text |
| `height` | `height_cm` | Both integers in cm |
| `cup_size` | `fake_tits` | Part of measurements mapping |
| `band_size` | (measurements) | Stash-box separate, Stash combined |
| `waist_size` | (measurements) | Stash-box separate, Stash combined |
| `hip_size` | (measurements) | Stash-box separate, Stash combined |
| `breast_type` | `fake_tits` | Enum mapping |
| `career_start_year` | `career_length` | Integer vs free text - parse where possible |
| `career_end_year` | `career_length` | Integer vs free text - parse where possible |
| `tattoos` | `tattoos` | `[BodyModification]` → formatted text |
| `piercings` | `piercings` | `[BodyModification]` → formatted text |
| `urls` | `urls` | Stash-box has `{url, type}`, Stash has plain strings |
| `is_favorite` | `favorite` | Direct boolean |

## UI Design

### Recommendation Card (List View)

Summary card showing performer name, endpoint, number of changed fields, and time since upstream update. "Review" button expands to detail view. "Dismiss" dropdown offers "Dismiss this update" or "Never show for this performer."

### Detail View (Inline Expansion)

Each changed field rendered as a row within the recommendation card:

- **Simple fields**: Radio buttons (Keep local / Accept upstream / Custom edit)
- **Name field**: 5-option radio (Keep local / Accept upstream / Accept upstream + demote local to alias / Keep local + add upstream as alias / Custom edit)
- **Alias/URL lists**: Union checkbox list with source highlighting (local-only, upstream-only, both). Add custom button.
- **Text fields**: Radio buttons + editable textarea pre-filled with selected option

### Validation (Client-Side, Before Apply)

1. **Name uniqueness**: Query all local performers, check proposed name + disambiguation is unique (case-insensitive)
2. **Alias self-conflict**: Proposed aliases can't match the performer's own chosen name (case-insensitive)
3. **Alias deduplication**: No duplicate aliases within the list (case-insensitive)
4. **Death date**: Must be >= birthdate if both present
5. Validation errors shown inline with the offending field highlighted
6. "Apply" button blocked until validation passes

### Validation (Server-Side Error Handling)

| Error | UI Response |
|-------|------------|
| `NameExistsError` | Show conflicting performer name + link, suggest adding disambiguation |
| `DuplicateAliasError` | Highlight offending alias, let user uncheck it |
| Unexpected error | Show raw error message, keep form open for retry |

### Partial Apply

User can accept some field changes and skip others:
- Only changed fields included in `PerformerUpdateInput` mutation
- After successful partial apply, update snapshot
- If un-applied fields still differ, keep recommendation open with remaining fields
- If all fields resolved, mark recommendation as resolved

### Settings UI

New section in recommendations dashboard settings:

- Per-endpoint enable/disable toggles with rate limit delay configuration
- Per-field monitoring checkboxes for each entity type
- Endpoints auto-discovered from Stash's `getStashBoxConnections()`

## Settings & Configuration

### Per-Endpoint Config

Stored in `recommendation_settings.config` JSON:

```json
{
    "endpoints": {
        "https://stashdb.org/graphql": {
            "enabled": true,
            "rate_limit_delay": 0.5,
            "entity_types": ["performer"]
        }
    }
}
```

### Default Monitored Fields (Performers)

All enabled by default: `name`, `disambiguation`, `aliases`, `gender`, `birthdate`, `death_date`, `ethnicity`, `country`, `eye_color`, `hair_color`, `height`, `measurements`, `tattoos`, `piercings`, `career_start_year`, `career_end_year`, `urls`, `details`, `is_favorite`

Explicitly excluded: `scene_count` (constantly changing), `age` (computed from birthdate), `images` (separate concern)

## Error Handling & Edge Cases

| Scenario | Handling |
|----------|----------|
| Stash-box endpoint unreachable | Log error, mark analysis run as failed, show retry in UI |
| Rate limited (429) | Exponential backoff via RateLimiter, resume from cursor |
| Stash-box performer deleted | Flag in recommendation: "Deleted upstream", option to dismiss |
| Stash-box performer merged | `merged_into_id` set - follow redirect to new ID, update snapshot |
| Local performer deleted before resolution | Check existence before showing resolve UI, auto-dismiss if gone |
| Multiple endpoints for same performer | Each endpoint generates independent recommendations |
| Pending rec exists, new scan finds more changes | Update existing recommendation's details JSON |
| Dismissed rec, upstream changes again | Soft dismiss: create new rec. Permanent dismiss: skip. |
| First run (no snapshots) | All differences flagged as potential updates; snapshot baseline established |

## Future Extensions

- **Other entity types**: Same infrastructure supports Scenes, Studios, Tags by adding new entity types to `upstream_snapshots` and `upstream_field_config`
- **Periodic scheduling**: `recommendation_settings` already has `interval_hours` and `next_run_at` columns
- **Bulk apply**: "Accept all upstream changes" button for users who trust their stash-box
- **Image sync**: Separate feature for syncing performer images
- **Bi-directional sync**: Submit local changes upstream (requires stash-box edit permissions)

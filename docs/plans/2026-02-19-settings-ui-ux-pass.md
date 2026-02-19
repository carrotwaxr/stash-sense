# Settings Page UI/UX Pass

**Date:** 2026-02-19
**Ticket:** refactor: settings page UI/UX pass - consistent components, human-friendly intervals

## Problem

The plugin settings page has accumulated visual inconsistencies:
- Job Schedules uses raw `<input type="number">` with browser-native spinners and inline styles
- Interval values displayed as raw hours (168 = 1 week) — meaningless at a glance
- Three different coding patterns across Sidecar Settings, Job Schedules, and Upstream Sync sections
- Inline styles scattered throughout `stash-sense-settings.js` instead of CSS classes
- Inconsistent save patterns: auto-save vs explicit Save button

## Design Decisions

- **Server-side interval definitions**: `JobDefinition` gets `allowed_intervals` and `description` fields. The `queue_types` API serializes them to the frontend. Single source of truth.
- **Tiered interval presets**: Light/Network jobs get frequent options (6h–2w), heavy/slow-changing jobs get infrequent options (1d–1mo).
- **Native `<select>` for interval picker**: Styled to match dark theme. Accessible, zero JS overhead, options come from server.
- **Auto-save everywhere**: All settings auto-save with 500ms debounce, except upstream field config (batch checkbox operation keeps explicit Save).
- **No new component abstractions**: Use existing CSS classes consistently + add a few new ones. No UI library needed.
- **Defaults unchanged**: 24h for DB update + upstream, 168h for duplicates + fingerprints.

## Server Changes

### job_models.py

Add predefined interval tiers:

```python
INTERVALS_FREQUENT = [
    (6, "Every 6 hours"), (12, "Every 12 hours"),
    (24, "Every day"), (48, "Every 2 days"), (72, "Every 3 days"),
    (168, "Every week"), (336, "Every 2 weeks"),
]

INTERVALS_INFREQUENT = [
    (24, "Every day"), (48, "Every 2 days"), (72, "Every 3 days"),
    (168, "Every week"), (336, "Every 2 weeks"), (720, "Every month"),
]
```

Extend `JobDefinition`:
- `description: str` — human-readable description of what the job does
- `allowed_intervals: list[tuple[int, str]]` — valid interval options for this job type

Job descriptions:
- Database Update: "Checks for updated face recognition data"
- Upstream Performer Changes: "Detects field changes from stash-box sources"
- Duplicate Performer Detection: "Finds performers that may be duplicates"
- Duplicate Scene File Detection: "Finds scenes with identical file fingerprints"
- Duplicate Scene Detection: "Finds scenes that may be the same content"
- Fingerprint Generation: "Generates face recognition fingerprints for scenes"

Interval assignments:
- `INTERVALS_FREQUENT`: Database Update, Upstream Performer Changes
- `INTERVALS_INFREQUENT`: All three duplicate detectors, Fingerprint Generation

### queue_router.py

Serialize new fields in `queue_types` response:
```json
{
  "type_id": "database_update",
  "display_name": "Database Update",
  "description": "Checks for updated face recognition data",
  "allowed_intervals": [
    {"hours": 6, "label": "Every 6 hours"},
    {"hours": 12, "label": "Every 12 hours"},
    ...
  ]
}
```

## Frontend Changes

### New CSS classes (stash-sense.css)

- `.ss-select` — styled native `<select>` matching dark theme (same border/bg/font as `.ss-number-input input`)
- `.ss-setting-hint` — 13px secondary-color text (replaces 4+ inline style occurrences)
- `.ss-setting-row-vertical` — variant of `.ss-setting-row` with `flex-direction: column`
- `.ss-setting-row-header` — flex row with space-between for compound row headers
- `.ss-upstream-fields-wrapper` — wrapper for expandable field config area
- Fix `.ss-number-input` spinner rules: change from `opacity: 1` to `display: none` / `-webkit-appearance: none`

### Schedule row layout

```
┌─────────────────────────────────────────────────────────────┐
│  Database Update                    [toggle]  [Every day ▾] │
│  Checks for updated face recognition data                   │
└─────────────────────────────────────────────────────────────┘
```

- Description is static (what the job does), not echoing the interval
- `<select>` disabled when toggle is off
- Auto-saves on toggle or dropdown change via `debouncedSave` pattern

### renderSchedulesCategory() refactor

- Replace all inline `styles: {}` with CSS classes
- Replace `<input type="number">` with `<select>` populated from `allowed_intervals`
- Remove per-row Save button, use debounced auto-save
- Use `.ss-setting-row`, `.ss-setting-info`, `.ss-setting-control` consistently

### renderEndpointFieldConfig() cleanup

- Inline style on row wrapper → `.ss-setting-row .ss-setting-row-vertical`
- Inline flex header → `.ss-setting-row-header`
- "Show Fields" button → `.ss-btn .ss-btn-sm`
- Fields wrapper → `.ss-upstream-fields-wrapper`
- Loading/help text → `.ss-setting-hint`
- Keep explicit Save for batch field config (not auto-save)
- No behavioral changes

### Other inline style cleanup

- Description/help text throughout → `.ss-setting-hint`
- Remove all remaining `style:` attributes in `stash-sense-settings.js`

## Not in Scope

- StashBox provider cards at top of settings page
- Operations tab styling
- API behavior changes beyond serializing new metadata fields

## Testing

- **Unit**: `JobDefinition` serialization includes `description` and `allowed_intervals` in `queue_types` response
- **Unit**: Schedule update API accepts hour values from predefined lists
- **Manual**: Visual verification — no inline styles, consistent alignment, dropdown works, auto-save works
- **Manual**: Responsive layout at 768px breakpoint
- **Manual**: Toggle disables/enables dropdown

## Key Files

- `api/job_models.py` — `JobDefinition` extensions, interval tier constants
- `api/queue_router.py` — serialize new fields
- `plugin/stash-sense-settings.js` — full refactor of schedule + upstream sync rendering
- `plugin/stash-sense.css` — new classes, spinner fix

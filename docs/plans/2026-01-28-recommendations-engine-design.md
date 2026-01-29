# Stash Sense: Recommendations Engine Design

**Date:** 2026-01-28
**Status:** Design Complete

---

## Overview

The recommendations engine extends the Stash Sense sidecar to analyze user libraries and surface actionable recommendations for library curation. It absorbs existing standalone scripts (duplicate-performer-finder, scene-file-deduper) and adds new analysis types powered by the face recognition database and Stash GraphQL API.

---

## Architecture

### Two-Database Model

The sidecar maintains two separate SQLite databases:

| Database | Purpose | Access | Lifecycle |
|----------|---------|--------|-----------|
| `performers.db` | Face embeddings, identity graph, stashbox mappings | Read-only | Distributed, user downloads updates |
| `stash_sense.db` | Recommendations, analysis state, user settings | Read-write | User-local, persists across face DB updates |

This separation ensures:
- Face DB stays distributable and easily updatable
- User-specific state (dismissed recommendations, preferences) persists across updates
- Clear separation of concerns

```
Sidecar Process
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│  ┌─────────────────────┐    ┌─────────────────────────────┐    │
│  │ performers.db       │    │ stash_sense.db              │    │
│  │ (read-only)         │    │ (read-write)                │    │
│  │                     │    │                             │    │
│  │ - face embeddings   │    │ - recommendations           │    │
│  │ - identity graph    │    │ - analysis runs             │    │
│  │ - stashbox mappings │    │ - user settings             │    │
│  │ - performer metadata│    │ - dismissed targets         │    │
│  └─────────┬───────────┘    └──────────────┬──────────────┘    │
│            │                               │                    │
│            └───────────┬───────────────────┘                    │
│                        ▼                                        │
│              ┌─────────────────┐                                │
│              │ Analysis Engine │                                │
│              │                 │                                │
│              │ - Face matching │◄─── Query face DB              │
│              │ - Cross-linking │◄─── Query identity graph       │
│              │ - Stash queries │◄─── GraphQL to user's Stash    │
│              │                 │                                │
│              │ Writes to ──────┼───► stash_sense.db             │
│              └─────────────────┘                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## Recommendation Types

| Analysis Type | Detection Logic | Resolution Actions |
|---------------|-----------------|-------------------|
| `duplicate_performer` | Same stash_id across multiple local performers | Merge (pick keeper), Dismiss |
| `duplicate_scene_files` | Scene has multiple files | Delete files (pick keeper), Dismiss |
| `missing_markers` | Scene has sprite but no markers | Generate markers, Dismiss |
| **Performer/Scene Issues:** | | |
| `unidentified_scene` | Scene has detected faces but no performers tagged | Add performer(s), Dismiss |
| `wrong_performer` | Tagged performer's face not detected in scene | Remove performer, Dismiss |
| `missing_performer` | Detected face matches known performer not tagged | Add performer, Dismiss |
| `extra_performer` | Performer tagged but their face never appears | Remove performer, Dismiss |
| **Cross-linking:** | | |
| `missing_stashbox_link` | Performer exists in stash-box X (per identity graph) but user lacks that link | Add stash-id, Dismiss |
| `stashbox_updates` | Stash-box entity `updated_at` > local sync time | Pull updates, Dismiss |

### Performer/Scene Analysis

The performer/scene issues all derive from the same analysis:
1. Fetch scene sprite sheet
2. Run face detection
3. Query face DB for matches
4. Compare detected performers to tagged performers
5. Generate appropriate recommendation type based on discrepancies

### Cross-linking via Identity Graph

The `missing_stashbox_link` recommendation leverages the identity graph in `performers.db`:
- Our DB tracks which stash-box endpoints each performer exists in (StashDB, ThePornDB, PMVStash, etc.)
- User has configured multiple stash-boxes in their Stash
- Analysis finds performers linked to one endpoint but not others they exist in
- Recommendation: "This performer exists in ThePornDB too, want to add that stash-id?"

---

## Database Schema (`stash_sense.db`)

```sql
-- Schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY
);
INSERT INTO schema_version (version) VALUES (1);

-- Core recommendations table
CREATE TABLE recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,  -- 'duplicate_performer', 'missing_markers', etc.
    status TEXT NOT NULL DEFAULT 'pending',  -- 'pending', 'dismissed', 'resolved'

    -- Polymorphic target (what entity this is about)
    target_type TEXT NOT NULL,  -- 'scene', 'performer', 'studio', 'file'
    target_id TEXT NOT NULL,    -- Stash entity ID

    -- Type-specific payload (JSON)
    details JSON NOT NULL,  -- varies by type

    -- Resolution tracking
    resolution_action TEXT,     -- what action was taken: 'merged', 'deleted', 'linked', etc.
    resolution_details JSON,    -- action-specific data
    resolved_at TEXT,

    -- Analysis metadata
    confidence REAL,            -- 0-1 where applicable, NULL for deterministic
    source_analysis_id INTEGER REFERENCES analysis_runs(id),

    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now')),

    UNIQUE(type, target_type, target_id)  -- one active recommendation per type/target
);

CREATE INDEX idx_rec_status ON recommendations(status);
CREATE INDEX idx_rec_type ON recommendations(type);
CREATE INDEX idx_rec_target ON recommendations(target_type, target_id);

-- Track analysis runs (for debugging, incremental analysis, rate limiting)
CREATE TABLE analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    type TEXT NOT NULL,           -- 'duplicate_performer', 'scene_face_check', etc.
    status TEXT NOT NULL,         -- 'running', 'completed', 'failed'
    started_at TEXT NOT NULL,
    completed_at TEXT,

    -- Progress tracking
    items_total INTEGER,
    items_processed INTEGER,
    recommendations_created INTEGER DEFAULT 0,

    -- For incremental runs
    cursor TEXT,                  -- bookmark for resuming (e.g. last scene_id processed)

    error_message TEXT
);

CREATE INDEX idx_analysis_type_status ON analysis_runs(type, status);

-- User preferences per recommendation type
CREATE TABLE recommendation_settings (
    type TEXT PRIMARY KEY,        -- recommendation type
    enabled INTEGER DEFAULT 1,    -- run this analysis?
    auto_dismiss_threshold REAL,  -- auto-dismiss below this confidence
    notify INTEGER DEFAULT 1,     -- show in dashboard?
    interval_hours INTEGER,       -- how often to run
    last_run_at TEXT,
    next_run_at TEXT,

    -- Type-specific config (JSON)
    config JSON                   -- e.g. {"min_face_confidence": 0.7} for face checks
);

-- Dismissed targets (don't re-recommend)
CREATE TABLE dismissed_targets (
    type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    dismissed_at TEXT DEFAULT (datetime('now')),
    reason TEXT,                  -- user can note why they dismissed
    PRIMARY KEY (type, target_type, target_id)
);

-- Track what we've already analyzed (for incremental analysis)
CREATE TABLE analysis_watermarks (
    type TEXT PRIMARY KEY,
    last_completed_at TEXT,
    last_cursor TEXT,              -- for incremental: "scenes updated after X"
    last_stash_updated_at TEXT     -- Stash's max updated_at we've seen
);
```

### Details JSON Examples

```json
// duplicate_performer
{
  "endpoint": "https://stashdb.org",
  "stash_id": "abc-123-uuid",
  "performers": [
    {"id": "1", "name": "Angela White", "scene_count": 45},
    {"id": "2", "name": "Angela White (duplicate)", "scene_count": 2}
  ],
  "suggested_keeper": "1"
}

// missing_performer
{
  "detected_stashdb_id": "abc-123-uuid",
  "detected_name": "Angela White",
  "face_box": {"x": 120, "y": 80, "width": 100, "height": 120},
  "frame_index": 5
}

// missing_stashbox_link
{
  "performer_name": "Angela White",
  "existing_links": ["stashdb:abc-123"],
  "missing_endpoint": "theporndb",
  "known_id": "def-456"
}
```

---

## API Endpoints

### Existing (Face Recognition)

```
POST /identify              - Identify faces in image
POST /identify/scene        - Identify performers in scene via sprite
GET  /health                - Health check
GET  /database/info         - Face DB stats
```

### Recommendations

```
GET  /recommendations                      - List recommendations (filterable)
     ?type=duplicate_performer
     ?status=pending
     ?target_type=scene
     ?limit=50&offset=0

GET  /recommendations/{id}                 - Get single recommendation with full details
POST /recommendations/{id}/resolve         - Mark resolved with action taken
     {"action": "merged", "details": {"kept_id": "1", "merged_ids": ["2", "3"]}}
POST /recommendations/{id}/dismiss         - Dismiss (won't re-recommend)
     {"reason": "These are actually different people"}
```

### Analysis

```
GET  /analysis/types                       - List available analysis types + enabled status
POST /analysis/{type}/run                  - Trigger analysis run (async, returns run_id)
GET  /analysis/runs                        - List recent analysis runs
GET  /analysis/runs/{id}                   - Get run status/progress
```

### Scheduler

```
GET  /scheduler/status                     - Get scheduler state, next run times
POST /scheduler/pause                      - Pause all background analysis
POST /scheduler/resume                     - Resume background analysis
```

### Configuration

```
GET  /settings                             - Get all recommendation settings
PUT  /settings/{type}                      - Update settings for a type
     {"enabled": true, "interval_hours": 12, "config": {"min_confidence": 0.7}}
```

### Stash Connection

```
GET  /stash/status                         - Test Stash GraphQL connection
POST /stash/sync                           - Trigger full library analysis
```

---

## Background Job Scheduler

The sidecar runs an internal scheduler for continuous library analysis:

```python
from apscheduler.schedulers.asyncio import AsyncIOScheduler

scheduler = AsyncIOScheduler()

# Default schedules (user-configurable)
DEFAULT_SCHEDULES = {
    "duplicate_performer":    {"interval_hours": 24, "incremental": True},
    "duplicate_scene_files":  {"interval_hours": 24, "incremental": True},
    "scene_face_check":       {"interval_hours": 6,  "incremental": True},
    "missing_markers":        {"interval_hours": 12, "incremental": True},
    "missing_stashbox_link":  {"interval_hours": 24, "incremental": False},
    "stashbox_updates":       {"interval_hours": 4,  "incremental": True},
}
```

### Scheduler Behavior

- Respects rate limits (configurable delay between Stash API calls)
- Runs incrementally where possible (only items added/modified since last run)
- Pauses automatically if Stash is unreachable
- Exposes status via API
- User can pause/resume via API or UI

### Incremental Analysis

For efficiency, most analyzers support incremental mode:
1. Track `last_stash_updated_at` watermark
2. Query Stash for entities with `updated_at > watermark`
3. Only analyze new/modified items
4. Update watermark on completion

---

## Plugin UI

The Stash plugin provides a dedicated page at `/plugins/stash-sense`:

```
/plugins/stash-sense
├── Dashboard (landing)
│   ├── Summary cards: "12 duplicate performers", "8 scenes need markers", etc.
│   ├── Scheduler status: "Last scan: 2h ago, Next: in 4h"
│   └── Quick actions: "Run full analysis", "Pause scheduler"
│
├── Tabs by recommendation type
│   ├── Performers
│   │   ├── Duplicates (merge UI with content counts)
│   │   └── Missing stash-box links (cross-link UI)
│   ├── Scenes
│   │   ├── Face mismatches (sprite overlay showing detected vs tagged)
│   │   ├── Missing markers (generate button)
│   │   └── Duplicate files (file picker with size/quality info)
│   └── Stash-box Sync
│       └── Pending updates (diff view, pull button)
│
└── Settings
    ├── Analysis toggles (enable/disable each type)
    ├── Schedule config (frequency per type)
    ├── Confidence thresholds
    └── Stash-box endpoints to monitor
```

### On-Demand Actions

Contextual actions remain on the pages they affect:
- "Identify Performers" button stays on scene page
- "Find on StashDB" stays on performer page

The recommendations dashboard is for review/bulk actions on issues discovered by background analysis.

---

## Absorbing Existing Scripts

The `stash-plugins/scripts/` tools get absorbed into the sidecar:

| Existing Script | Becomes | Migration |
|-----------------|---------|-----------|
| `duplicate-performer-finder/` | `analyzers/duplicate_performer.py` | Port `find_duplicates()`, `group_by_endpoint()` |
| `scene-file-deduper/` | `analyzers/duplicate_scene_files.py` | Port `get_multi_file_scenes()` query |

### Analyzer Pattern

```python
# api/analyzers/base.py
class BaseAnalyzer:
    """Base class for all analyzers."""

    type: str  # recommendation type this analyzer produces

    def __init__(self, stash: StashClient, face_db: PerformerDatabase, rec_db: RecommendationsDB):
        self.stash = stash
        self.face_db = face_db
        self.rec_db = rec_db

    async def run(self, incremental: bool = True) -> AnalysisResult:
        """Run analysis, create recommendations, return summary."""
        raise NotImplementedError


# api/analyzers/duplicate_performer.py
class DuplicatePerformerAnalyzer(BaseAnalyzer):
    """Ported from scripts/duplicate-performer-finder"""

    type = "duplicate_performer"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        performers = await self.stash.get_all_performers()
        duplicates = self._find_duplicates(performers)

        created = 0
        for (endpoint, stash_id), group in duplicates.items():
            if not self.rec_db.is_dismissed(self.type, "performer", group[0]["id"]):
                self.rec_db.create_recommendation(
                    type=self.type,
                    target_type="performer",
                    target_id=group[0]["id"],
                    details={"endpoint": endpoint, "stash_id": stash_id, "performers": group}
                )
                created += 1

        return AnalysisResult(items_processed=len(performers), recommendations_created=created)
```

---

## Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python | ML libraries native, existing codebase, FastAPI is capable |
| API Framework | FastAPI | Already in use, async support, good performance |
| Scheduler | APScheduler | Lightweight, async-native, no external deps |
| GraphQL Client | gql | Popular, async support via httpx |
| Database | SQLite | Simple, no external service, sufficient for single-user |

---

## Key Flows

### Face Mismatch Analysis

1. Query Stash for scenes (incremental: `updated_at > watermark`)
2. For each scene:
   - Fetch sprite sheet
   - Run face detection
   - Query `performers.db` for matches
   - Get tagged performers from Stash
   - Compare detected vs tagged
   - Create appropriate recommendations (unidentified, wrong, missing, extra)
3. Update watermark

### Cross-linking Analysis

1. Query Stash for user's performers with stash-box links
2. For each performer:
   - Look up in `performers.db` identity graph
   - Check which endpoints our DB says they exist in
   - Compare to user's linked endpoints
   - Create `missing_stashbox_link` for gaps
3. Recommendations enable easy "Add this stash-id" action

### Database Update Flow

1. User downloads new `performers.db`
2. Sidecar detects change (file watcher or restart)
3. Reloads face DB connection
4. `stash_sense.db` untouched - all user state preserved
5. New analysis runs may find new cross-links with updated identity graph

---

## Future Considerations

- **Bulk actions**: "Dismiss all low-confidence face mismatches"
- **Export**: Generate reports of library issues
- **Webhooks**: Notify external systems of new recommendations
- **Priority scoring**: Surface most impactful recommendations first
- **Integration with peek-stash-browser**: Share recommendations data

---

*This document captures the design as validated through brainstorming session 2026-01-28.*

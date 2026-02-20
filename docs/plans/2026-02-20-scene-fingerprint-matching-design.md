# Automated Scene Identification via Stash-Box Fingerprints

**Date**: 2026-02-20
**Status**: Design
**Ticket**: feat: automated scene identification via stash-box fingerprints

## Summary

Automated background job that matches local Stash scenes to stash-box entries using fingerprint lookups (MD5, OSHASH, PHASH). Results surface as recommendations with quality scoring, dismiss/accept actions, and "Accept All" for high-confidence matches. Mirrors Stash's Tagger UI functionality but adds scheduled background scraping, persistent dismissal tracking, and quality thresholds.

## Background

Stash's built-in Tagger UI is functionally perfect for manual scene identification but requires the user to sit through each match interactively. Stash's `metadataIdentify` batch mode is too coarse — it either skips multi-match scenes entirely or blindly takes the first result, with no quality scoring or dismissal memory.

This feature fills the gap: automated, scheduled, quality-aware scene identification that accumulates results over time and lets users batch-accept high-confidence matches while reviewing ambiguous ones.

## Architecture

### Data Flow

1. **Scan phase** (NETWORK resource job): Query Stash for all local scenes with their file fingerprints. For each scene, determine which configured stash-box endpoints it's missing a `stash_id` for.

2. **Match phase**: For scenes missing a stash_id, batch-submit their fingerprints to that stash-box's `findScenesBySceneFingerprints` GraphQL API (up to 40 scenes per batch, matching Stash's own batching).

3. **Score & filter phase**: For each candidate match, compute a quality score based on fingerprint match count/percentage, fingerprint types, duration agreement, and community vote count. Store qualifying matches as recommendations.

4. **Dismiss-aware deduplication**: Before creating a recommendation, check if this specific `(local_scene_id, stashbox_scene_id)` pair has been dismissed. If so, skip it. A different stashbox scene matching the same local scene is a new recommendation — this handles the scenario where a wrong match is dismissed but a correct one appears later.

### Why Not BaseUpstreamAnalyzer

`BaseUpstreamAnalyzer` is designed for 3-way diffs on entities that already have a stash_id link. This analyzer finds scenes that *don't* have that link yet. It extends `BaseAnalyzer` directly, similar to the duplicate detection analyzers.

## Quality Scoring

Each candidate match is scored on multiple signals:

### Fingerprint Match Signals

- **Exact hash match** (MD5 or OSHASH): Highest confidence — same file content. If either matches, almost certainly correct.
- **PHASH match**: Lower confidence, quality depends on Hamming distance. Distance 0 is strong, 1-8 progressively weaker. We don't control the stash-box server's distance threshold, but we can apply our own stricter filtering on results.
- **Match count**: How many of the local scene's fingerprints appear in the stash-box result (e.g., 2/3 or 3/3).

### Duration Agreement

Compare local file duration against the stash-box scene's fingerprint durations (each fingerprint submission includes a duration value). Close agreement (+/- a few seconds) reinforces the match. Large discrepancy is a warning signal.

### Community Confidence

Net vote count from stash-box's fingerprint voting system (submissions minus reports). Higher net votes = more community validation of this fingerprint association.

### High-Confidence Classification

A match is flagged `high_confidence` (eligible for "Accept All") when it meets **both** user-configurable thresholds:

- **Minimum matching fingerprints** (default: 2): At least N fingerprints in agreement
- **Minimum match percentage** (default: 66%): At least this percentage of the local scene's fingerprints match

**Additional rule**: If a local scene has multiple candidate matches from the same endpoint, none qualify for Accept All — ambiguity requires human judgment.

### User-Configurable Settings

| Setting | Default | Description |
|---------|---------|-------------|
| `min_matching_fingerprints` | 2 | Minimum fingerprint matches for high-confidence |
| `min_match_percentage` | 66 | Minimum % of local fingerprints that must match |

These are stored via the existing user settings system (`getUserSetting`/`setUserSetting`).

## Dismissal Mechanics

### Pair-Based Dismissal

Dismissals are keyed on the `(local_scene_id, stashbox_scene_id)` pair, not just the local scene. This enables the critical scenario:

1. Scan finds scene A matches stashbox scene X (wrong match)
2. User dismisses it
3. Next scan: stashbox scene X is still returned for scene A → pair is dismissed, skip
4. Community fixes the fingerprint data; now scene A matches stashbox scene Y instead
5. Scene Y is a different stashbox_scene_id → new recommendation surfaces

### Implementation

Uses the existing `dismissed_targets` table with `type="scene_fingerprint_match"`. The `target_id` is the local scene ID. The paired stashbox scene ID is stored in the recommendation's `details` JSON and checked during the dismiss comparison. The `is_dismissed()` check is extended to compare against the details field for this recommendation type.

## Recommendation Schema

### Type

`scene_fingerprint_match` — added to the recommendation type registry alongside existing types.

### Details JSON Structure

```json
{
  "local_scene_id": "123",
  "endpoint": "https://stashdb.org/graphql",
  "stashbox_scene_id": "abc-uuid",
  "stashbox_scene_title": "Scene Title",
  "stashbox_studio": "Studio Name",
  "stashbox_performers": ["Performer A", "Performer B"],
  "stashbox_date": "2024-01-15",
  "stashbox_cover_url": "https://...",
  "matching_fingerprints": [
    {"algorithm": "MD5", "hash": "abc123", "duration": 1834, "submissions": 5},
    {"algorithm": "OSHASH", "hash": "def456", "duration": 1834, "submissions": 3}
  ],
  "total_local_fingerprints": 3,
  "match_count": 2,
  "match_percentage": 66.7,
  "duration_local": 1835,
  "duration_remote": 1834,
  "high_confidence": true,
  "phash_distance": null
}
```

Enough metadata is stored to render the UI without round-trips back to the stash-box API. Title, studio, performers, date, and cover URL are all captured at scan time.

## Job Queue Integration

### Job Definition

```python
"scene_fingerprint_match": JobDefinition(
    type_id="scene_fingerprint_match",
    display_name="Scene Fingerprint Matching",
    description="Match local scenes to stash-box entries via fingerprints",
    resource=ResourceType.NETWORK,
    default_priority=JobPriority.NORMAL,
    supports_incremental=True,
    schedulable=True,
    default_interval_hours=168,  # weekly
    allowed_intervals=((24, "Daily"), (72, "Every 3 days"), (168, "Weekly"), (336, "Every 2 weeks")),
)
```

NETWORK resource because the bottleneck is stash-box API calls, not CPU or GPU.

### Incremental Strategy

Full scan queries every local scene for fingerprints and stash_ids — expensive with large libraries. Incremental runs use a watermark based on `scene.updated_at` from Stash, so only scenes added or modified since the last run are re-checked. A scene getting a new file (re-encode, remux) updates its timestamp and triggers re-evaluation.

### Batching

The `findScenesBySceneFingerprints` API accepts up to 40 scenes per batch. For a 10K scene library where ~3K scenes lack a stash_id for an endpoint, that's ~75 API calls per endpoint — manageable with rate limiting.

### Per-Endpoint Iteration

The analyzer iterates each configured stash-box endpoint independently. A scene might be matched on StashDB but not yet on ThePornDB — each endpoint gets its own pass.

## Action Flow

### Accept (Single Match)

1. Update the local scene in Stash: add `stash_id` entry `{endpoint, stash_id}` via `update_scene`
2. Mark the recommendation as resolved with `action: "accepted"`
3. (Future) Optionally submit the local scene's fingerprints back to stash-box as a community contribution

### Accept All

Operates on all `high_confidence` recommendations, filterable by endpoint. Same action as single accept, repeated in batch. API accepts a filter: `{"high_confidence": true, "endpoint": "..."}`.

### Dismiss

Stores the `(local_scene_id, stashbox_scene_id)` pair in dismissed_targets. On re-scan, the analyzer skips this exact pair but allows new matches for the same local scene.

### Fingerprint Submission (Deferred)

Submitting local fingerprints back to stash-box after accepting a match mirrors Stash's Tagger behavior and contributes to community data quality. Deferred to a follow-up — requires stash-box API key permissions and adds complexity.

## New Code Required

### Stash Client

`get_scenes_with_fingerprints(updated_after: str = None) -> list[dict]`
- Returns scenes with: id, title, stash_ids, files[].fingerprints (MD5, OSHASH, PHASH), files[].duration
- Supports filtering by updated_at for incremental runs
- May need a new GraphQL query since existing scene queries don't include file fingerprints

### StashBox Client

`find_scenes_by_fingerprints(fingerprint_sets: list[list[dict]]) -> list[list[dict]]`
- Wrapper around `findScenesBySceneFingerprints` batch GraphQL query
- Input: list of fingerprint-sets (one set per local scene)
- Output: list of match-lists (one list per local scene, each containing matched stashbox scenes with their fingerprint/metadata)

### Analyzer

`api/analyzers/scene_fingerprint_match.py`
- Extends `BaseAnalyzer`
- Implements the scan → match → score → recommend pipeline
- Per-endpoint iteration with incremental watermark support

### Router Endpoints

`POST /recommendations/actions/accept-fingerprint-match`
- Input: `{recommendation_id, scene_id, stash_id, endpoint}`
- Adds stash_id to local scene, resolves recommendation

`POST /recommendations/actions/accept-all-fingerprint-matches`
- Input: `{endpoint?: string}` (optional filter)
- Batch accepts all high-confidence matches

### Plugin Backend Proxy

New modes: `rec_accept_fingerprint_match`, `rec_accept_all_fingerprint_matches`

### Plugin UI

- New dashboard card: `scene_fingerprint_match` with count + Accept All button
- List view: local scene info + stashbox match info + fingerprint summary + actions
- Detail view: full fingerprint breakdown, duration comparison, per-fingerprint vote counts
- Settings: min_matching_fingerprints and min_match_percentage controls

## Plugin UI Design

### Dashboard Card

- Icon: fingerprint or link icon
- Title: "Scene Fingerprint Matches"
- Subtitle: "N pending (M high-confidence)"
- Quick action: "Accept All High-Confidence (M)" button
- Click: navigate to list view

### List View

Each row shows:
- Local scene: title (or filename), duration
- Arrow / link indicator
- Stashbox match: title, studio, performers (truncated), date
- Fingerprint badge: "2/3 match" with color (green = all match, yellow = partial, red = single)
- High-confidence indicator (checkmark badge)
- Accept / Dismiss buttons

### Detail View

- Side-by-side: local scene info (left) vs stashbox scene info (right)
- Fingerprint table: algorithm, hash, match status, duration, community votes
- Duration comparison bar
- Accept / Dismiss with optional dismiss reason

## Testing

### Unit Tests

- Quality scoring logic: various fingerprint combinations → correct scores and high_confidence flags
- Dismiss pair matching: same pair dismissed → skip, different stashbox scene → allow
- Incremental watermark: only scenes after watermark are processed

### Integration Tests

- Analyzer end-to-end with mocked Stash and StashBox clients
- Accept action: verify stash_id added to scene and recommendation resolved
- Accept All: verify only high_confidence matches are accepted
- Re-scan after dismiss: dismissed pair skipped, new match surfaces

## Documentation

None needed beyond this design document. The feature is user-facing through the existing recommendations UI patterns.

## Key Files

| File | Change |
|------|--------|
| `api/analyzers/scene_fingerprint_match.py` | New analyzer |
| `api/stash_client_unified.py` | New fingerprint query method |
| `api/stashbox_client.py` | New batch fingerprint lookup method |
| `api/recommendations_router.py` | Accept/Accept All endpoints |
| `api/recommendations_db.py` | Dismiss pair comparison for this type |
| `api/job_models.py` | Job registration |
| `plugin/stash-sense-recommendations.js` | Dashboard card, list/detail views |
| `plugin/stash_sense_backend.py` | New proxy modes |
| `plugin/stash-sense.css` | Fingerprint match styling |

## Open Questions

- **Fingerprint submission back to stash-box**: Include in v1 or defer? Adds community value but requires API key permissions and error handling for permission failures.
- **Cover image display**: Should we show the stashbox scene's cover image in the match UI? Helpful for visual confirmation but requires proxying images through the sidecar (CSP restrictions from Stash UI).
- **Scene metadata preview**: When accepting a match, should we also offer to pull in the stashbox scene's metadata (title, date, performers, tags) like the Tagger does, or just link the stash_id and let the upstream scene change detector handle field sync on the next run?

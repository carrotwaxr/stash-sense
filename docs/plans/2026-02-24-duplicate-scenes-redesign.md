# Duplicate Scenes Detection Redesign

**Date:** 2026-02-24
**Status:** Draft

## Problem

The current duplicate scenes detection generates 22,753 recommendations from ~24K scenes — nearly all of them false positives. Three root causes:

1. **Face candidate explosion**: The SQL self-join on `scene_fingerprint_faces` creates candidate pairs for ANY two scenes sharing a single identified performer. Popular performers appearing in 500+ scenes generate 100K+ pairs each, totaling 4,095,469 face candidates.

2. **Broken face scoring**: `face_signature_similarity()` computes a score based on average proportion differences across the **union** of all performers in both scenes. Two scenes with completely different performers can score 50%+ because their proportion distributions are numerically similar. 85% of recommendations (19,256) are face-only with zero metadata support.

3. **Missing plugin UI**: No list card renderer or detail view for `duplicate_scenes` type — users see raw fallback text and "Unknown recommendation type."

### Current Signal Breakdown (Production Data)

| Signal Type | Count | Avg Confidence | Quality |
|---|---|---|---|
| face_only | 19,256 | 56.3% | Junk — no corroboration |
| face+metadata | 3,298 | 63.8% | Mixed — some real dupes |
| stashbox | 185 | 100% | All legitimate |
| metadata_only | 14 | 52.6% | Low volume |

## Goals

- Catch duplicates that Stash's built-in phash dedup misses (different intros/outros, shorter cuts, re-encodes from different sources)
- Produce dramatically fewer, higher-quality recommendations
- Provide actionable UI: side-by-side comparison with merge/delete/dismiss
- Persistent dismissals prevent repeat false positives (key UX advantage over Stash)

## Design

### Phase 1: Candidate Generation

Replace the three current candidate sources with:

#### Source 1: Stash-box ID grouping (keep as-is)
Scenes sharing `(endpoint, stash_id)`. ~188 candidates. All legitimate duplicates.

#### Source 2: Phash similarity (NEW — replaces face candidates)
Query all scene phashes from Stash via `get_scenes_with_fingerprints()`. Stash computes a 64-bit DCT-based perceptual hash per file, already available for all scenes.

- Convert hex phash strings to 64-bit integers
- Find all scene pairs with Hamming distance ≤ 10 bits
- ~24K scenes × 8 bytes = 192KB in memory
- All-pairs comparison using vectorized bitwise XOR + popcount runs in seconds on CPU
- No video decoding or GPU required — uses data Stash already computed

This replaces the 4M+ face candidate source with a content-similarity signal that's free to compute.

#### Source 3: Metadata intersection (keep as-is)
Scenes sharing `(studio_id, performer_id)` pairs. ~143K candidates. Same logic as today.

#### Removed: Face fingerprint candidates
Face identity moves from candidate generation to **scoring only**. Having the same performers is corroboration, not evidence of duplication.

### Phase 2: Scoring

#### Tier 1: Stash-box ID Match → 100%
Identical stash-box ID on the same endpoint. Authoritative. No further scoring needed.

#### Tier 2: Strong phash match (Hamming ≤ 4 bits) → 85-95%
Visual content is nearly identical. Highest non-authoritative signal.

Bonuses:
- Same date: +3
- Performer overlap: +3
- Same studio: +2
- Duration ratio ≥ 0.3: +2

Cap: 95%

#### Tier 3: Moderate phash match (Hamming 5-10 bits) + corroboration → 60-84%
Phash is close but not definitive. Requires at least one corroborating signal:

- Face corroboration: ≥1 shared performer with >10% proportion in both scenes
- Metadata corroboration: same studio + ≥50% performer overlap, or exact performer match
- Date match: same release date

Score = phash_base (scaled by Hamming distance) + corroboration_bonus. Cap: 84%.

#### Tier 4: Metadata + face only (no phash match) → 50-70%
Catches cases phash misses entirely (different intros, shorter cuts from different sources).

Requires at minimum:
- Same date + exact performer match → 45 points (strongest metadata signal)
- Same date + partial performer overlap (Jaccard ≥ 0.5) → 35 points
- Exact performer match without date → 25 points
- Same studio → 10 points (mild boost)

Face corroboration (requires ≥1 actually shared performer):
- All performers match with proportions within 10% → +15
- ≥50% performer overlap with proportions within 20% → +10

Duration penalty (only subtracts, never adds):
- Ratio < 0.15 (e.g., 3min vs 45min) → -20 points
- Ratio 0.15-0.30 → -10 points
- Ratio ≥ 0.30 → no penalty (handles web release 50m vs DVD 25m)

Cap: 70%

#### Hard Rules (all tiers)
- Face score = 0 when zero shared performers (Jaccard = 0)
- No single non-authoritative signal scores above 70% alone
- Minimum 50% confidence to create any recommendation
- Duration differences never add confidence, only subtract at extremes

### Face Scoring Fix

The `face_signature_similarity()` function must be rewritten:

```python
def face_signature_similarity(fp_a, fp_b):
    shared = set(fp_a.faces.keys()) & set(fp_b.faces.keys())
    shared.discard("unknown")

    if not shared:
        return 0.0, "No shared performers"

    # Only score based on actually shared performers
    proportion_diffs = []
    for pid in shared:
        diff = abs(fp_a.faces[pid].proportion - fp_b.faces[pid].proportion)
        proportion_diffs.append(diff)

    avg_diff = sum(proportion_diffs) / len(proportion_diffs)

    # Jaccard similarity of performer sets (excluding unknown)
    all_a = set(fp_a.faces.keys()) - {"unknown"}
    all_b = set(fp_b.faces.keys()) - {"unknown"}
    jaccard = len(all_a & all_b) / len(all_a | all_b) if (all_a | all_b) else 0

    # Score: Jaccard drives the base, proportion similarity is a bonus
    base = jaccard * 60.0  # 0-60 based on cast overlap
    proportion_bonus = max(0, 15.0 * (1.0 - avg_diff * 4.0))  # 0-15 for close proportions

    score = min(base + proportion_bonus, 75.0)
    return score, f"{len(shared)} shared performers (Jaccard {jaccard:.0%}), {avg_diff:.1%} avg proportion diff"
```

Key changes:
- Returns 0 when no performers are actually shared
- Score is based on Jaccard similarity (what fraction of the cast overlaps)
- Proportion difference only matters for shared performers
- Capped at 75 (can't create a recommendation alone at 50% threshold without extreme cast overlap)

### Plugin UI

#### List Card (`renderRecommendationCard`)
Add a `duplicate_scenes` handler showing:
- Scene A title and Scene B title
- Confidence badge (color-coded: green ≥80%, yellow ≥60%, gray ≥50%)
- Primary signal indicator (stashbox / phash / metadata)

#### Detail View (`renderDuplicateScenesDetail`)
Side-by-side comparison:

```
┌─────────────────────┬─────────────────────┐
│  Scene A            │  Scene B            │
│  [thumbnail]        │  [thumbnail]        │
│  Title (→ Stash)    │  Title (→ Stash)    │
│  Studio: X          │  Studio: X          │
│  Performers: A, B   │  Performers: A, B   │
│  Date: 2024-01-15   │  Date: 2024-01-15   │
│  Duration: 45:30    │  Duration: 23:15    │
│  Files: 1080p 4.2GB │  Files: 720p 1.8GB  │
│  ○ Keep this scene  │  ○ Keep this scene  │
└─────────────────────┴─────────────────────┘

Confidence: 92% — High confidence duplicate
Signals: Same studio + Exact performer match + Same date

[ Merge Scenes ]  [ Delete A ]  [ Delete B ]  [ Dismiss ]
```

Both scenes are fetched from Stash at render time (we store only scene IDs in recommendation details).

#### Actions

**Merge Scenes**: Calls Stash `sceneMerge` mutation:
```graphql
mutation SceneMerge($input: SceneMergeInput!) {
  sceneMerge(input: $input) { count }
}
```
- `destination`: selected keeper scene ID
- `source`: [other scene ID]
- `play_history: true`
- `o_history: true`
- Tags, performers, stash IDs consolidate onto keeper. Files transfer.
- Resolves recommendation with `action: "merged"`.

**Delete Scene A / B**: Calls `sceneDestroy` mutation with confirmation dialog. Resolves recommendation with `action: "deleted"`.

**Dismiss**: Existing dismiss flow. Persistent — won't resurface.

**Open in Stash**: Direct links to both scenes (`/scenes/{id}`).

### API Changes

Add to `stash_client_unified.py`:
- `merge_scenes(source_ids, destination_id)` — wraps `sceneMerge` mutation
- `delete_scene(scene_id)` — wraps `sceneDestroy` mutation (if not already present)
- `get_scene_details(scene_id)` — fetch full scene data for detail view (title, studio, performers, date, duration, files, cover image)

Add to `recommendations_router.py`:
- `POST /recommendations/{id}/merge-scenes` — merge scenes and resolve recommendation
- `POST /recommendations/{id}/delete-scene` — delete a scene and resolve recommendation

### Data Migration

When this ships:
1. Clear all existing `duplicate_scenes` recommendations (they're all junk from the broken scoring)
2. Bump the analyzer's `logic_version` to force re-analysis
3. Clear `duplicate_candidates` table

### Future Enhancements (not in scope)

- Field-level conflict resolution UI during merge (let user pick title from A vs B)
- Multi-point phash fingerprinting (sample phash at 10%, 25%, 50%, 75%, 90% of video duration to catch scenes with different intros)
- Cluster-based dedup (group 3+ scenes as duplicates of each other instead of pairwise)

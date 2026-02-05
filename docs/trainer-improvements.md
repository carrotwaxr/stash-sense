# Stash-Sense Trainer Improvement Recommendations

**Generated:** 2026-02-05
**Updated:** 2026-02-05 (post-identification improvements)

## Executive Summary

Benchmark analysis revealed key bottlenecks limiting identification accuracy. We've implemented identification logic improvements in stash-sense that improved **precision by 75%** (17.8% → 31.1%). However, further gains require improvements to the training data in stash-sense-trainer.

**Key findings:**
- **29.1% of performers have only 1 face** - the biggest source of false positives
- **Multi-signal (body/tattoo) provides 0% benefit** - coverage exists but signals aren't discriminative enough
- **480p video underperforms severely** - 12.5% accuracy vs 37.7% for 1080p

## What Was Fixed in stash-sense (Identification Logic)

These changes are already implemented and tested:

| Change | Impact |
|--------|--------|
| `min_appearances=2` | Requires 2+ face matches per performer |
| `min_unique_frames=2` | Must appear in 2+ different frames |
| `min_confidence=0.35` | Filters low-quality matches |
| Stronger "found by both" boost | 0.15 → 0.25 for hybrid mode |
| Cluster-based max performers | Limits over-reporting |

**Result:** Precision improved 75%, FP ratio dropped from 81.4% to 68.9%

## What Needs Improvement in stash-sense-trainer

### Priority 1: More Faces Per Performer (HIGHEST IMPACT)

**Current state:**
| Face Count | Performers | Percentage | Status |
|------------|------------|------------|--------|
| 1 face | 25,494 | 29.1% | **Critical** |
| 2-3 faces | 25,348 | 28.9% | Low |
| 4-5 faces | 10,658 | 12.2% | Acceptable |
| 6-10 faces | 17,326 | 19.8% | Good |
| 11+ faces | 8,825 | 10.1% | Excellent |

**Why this matters:**
- Single-face performers have no angle/expression diversity
- Can't cross-validate matches
- Higher false positive risk

**Action:** Run enrichment with ALL face sources enabled, prioritizing:
1. Performers with only 1 face (25,494 performers)
2. High scene-count performers (most likely to appear in user libraries)

**Target:** Reduce 1-face performers from 29.1% to <15%

### Priority 2: Implement `--min-faces` Filter in DB Optimize

Add parameter to exclude low-face-count performers from production index:

```bash
# Only include performers with 2+ faces
python -m trainer db optimize --min-faces 2
```

**Implementation:**
- Filter during export, not deletion from source DB
- Allows re-running with different thresholds
- Source data stays intact

**Impact estimates:**
- `--min-faces 2`: Removes 25,494 performers (29.1%), keeps 62,157
- `--min-faces 3`: Removes 50,842 performers (58.0%), keeps 36,809

### Priority 3: Investigate Multi-Signal Ineffectiveness (LOWER PRIORITY)

**Current coverage:**
| Signal | DB Coverage | Test Set Coverage |
|--------|-------------|-------------------|
| Body proportions | 29.6% | 36.6% |
| Tattoo detection | 39.9% | 62.5% |
| Either signal | 44.0% | 72.0% |

**The puzzle:** Even with 72% coverage in test scenes, multi-signal provides **0% accuracy improvement**.

**Possible causes to investigate:**
1. Body proportions may not be discriminative enough (many performers have similar ratios)
2. Tattoo detection may have too many false positives/negatives
3. The weighting/fusion logic may need tuning
4. Signals may only help for edge cases not represented in test set

**Recommendation:** De-prioritize until face coverage is improved. More faces will have higher impact.

## Benchmark Results Summary

### After Identification Improvements

| Metric | Old Code | Improved Code | Change |
|--------|----------|---------------|--------|
| Precision | 17.8% | **31.1%** | **+75%** |
| FP Ratio | 81.4% | **68.9%** | **-12.5%** |
| Accuracy | 38.6% | 33.6% | -5% |

### By Resolution

| Resolution | Accuracy | Avg Faces Detected |
|------------|----------|-------------------|
| 480p | 12.5% | 5.0 |
| 720p | 26.5% | 7.3 |
| 1080p | 37.7% | 8.3 |
| 4K | 50.0% | 5.0 |

**Note:** 480p detects 40% fewer faces than 1080p. No trainer-side fix - users should use higher resolution video.

## Quick Start: Trainer Session

Copy this to start a new stash-sense-trainer session:

```
I need to improve the face database for stash-sense identification.

Current problems:
1. 29.1% of performers (25,494) have only 1 face - causes false positives
2. Need to run face enrichment with all sources enabled
3. Need to implement --min-faces parameter for db optimize

Priority actions:
1. Run enrichment focusing on 1-face performers first
2. Add --min-faces parameter to db optimize command
3. Re-export database with --min-faces 2

See docs/plans/trainer-improvements.md for full details (if it exists)
or reference the analysis in stash-sense/docs/trainer-improvements.md
```

## SQL Queries for Analysis

### Find 1-face performers by popularity
```sql
SELECT p.id, p.canonical_name, p.face_count, p.scene_count
FROM performers p
WHERE p.face_count = 1
ORDER BY p.scene_count DESC NULLS LAST
LIMIT 1000;
```

### Count performers needing enrichment
```sql
SELECT
    CASE
        WHEN face_count = 0 THEN '0 faces'
        WHEN face_count = 1 THEN '1 face'
        WHEN face_count <= 3 THEN '2-3 faces'
        ELSE '4+ faces'
    END as bucket,
    COUNT(*) as count
FROM performers
GROUP BY bucket;
```

## Success Metrics

After trainer improvements, re-run benchmark to verify:

| Metric | Current | Target |
|--------|---------|--------|
| 1-face performers | 29.1% | <15% |
| Precision | 31.1% | >40% |
| Accuracy | 33.6% | >40% |
| FP ratio | 68.9% | <60% |

## Files Reference

- Identification logic: `stash-sense/api/main.py` (frequency_based_matching, hybrid_matching)
- Benchmark framework: `stash-sense/api/benchmark/`
- Investigation results: `stash-sense/api/benchmark_results/investigation/`

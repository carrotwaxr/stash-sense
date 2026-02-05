# Identification Benchmark Framework Design

**Date:** 2026-02-04
**Status:** Ready for implementation

---

## Overview

Comprehensive benchmark framework to diagnose, tune, and validate the performer identification system. Tests against real Stash scenes with known performers to find optimal parameters and evaluate multi-signal impact.

**Goals:**
1. Diagnose failure patterns (DB coverage? wrong matches? noise?)
2. A/B test parameters to find optimal settings
3. Evaluate multi-signal (body + tattoo) impact
4. Re-validate earlier findings against enriched DB (100k → 407k faces)

---

## 1. Test Scene Selection

### Selection Criteria

Query Stash for scenes meeting:
- Resolution: 480p or higher (width >= 854 or height >= 480)
- Has stashdb.org stash_id (confirms proper StashDB match)
- 2+ performers tagged
- All tagged performers have stashdb.org stash_ids

### Stratified Sampling

Sample across difficulty levels:
- **Resolution tiers:** 480p, 720p, 1080p, 4K
- **Performer count:** 2, 3, 4+
- **DB coverage:** "well-covered" (all performers 5+ faces) vs "sparse" (some <5)

### Data Structure

```python
@dataclass
class ExpectedPerformer:
    stashdb_id: str
    name: str
    faces_in_db: int
    has_body_data: bool
    has_tattoo_data: bool

@dataclass
class TestScene:
    scene_id: str
    stashdb_id: str
    title: str
    resolution: str  # "480p", "720p", "1080p", "4k"
    width: int
    height: int
    duration_sec: float
    expected_performers: list[ExpectedPerformer]
    db_coverage_tier: str  # "well-covered", "sparse"
```

**Target:** 100-200 test scenes with known ground truth.

---

## 2. Parameter Grid

### Parameters to Test

| Parameter | Values | Rationale |
|-----------|--------|-----------|
| `matching_mode` | frequency, hybrid | Skip cluster (known issues) |
| `max_distance` | 0.5, 0.6, 0.7, 0.8 | Precision vs recall trade-off |
| `min_face_size` | 40, 60, 80 | Quality vs coverage trade-off |
| `use_multi_signal` | true, false | Evaluate body+tattoo impact |
| `num_frames` | 40 | Fixed (proven optimal) |

### Fixed Parameters

```python
num_frames = 40
start_offset_pct = 0.05
end_offset_pct = 0.95
min_face_confidence = 0.5
top_k = 5  # Get more candidates for analysis
cluster_threshold = 0.6
```

### Execution Strategy

```
Phase 1: Baseline
  - Run (frequency, 0.7, 40, multi-signal=true) on ALL scenes
  - Establish ground truth results

Phase 2: Distance tuning
  - Test max_distance (0.5, 0.6, 0.8) on 30% sample

Phase 3: Face size tuning
  - Test min_face_size (60, 80) on 30% sample

Phase 4: Multi-signal evaluation
  - Compare multi-signal on vs off on ALL scenes

Phase 5: Mode comparison
  - Test hybrid mode on scenes where frequency struggled
```

---

## 3. Metrics

### Per-Scene Metrics

```python
@dataclass
class SceneResult:
    scene_id: str
    params: dict

    # Ground truth comparison
    true_positives: int       # Expected performers correctly identified
    false_negatives: int      # Expected performers missed
    false_positives: int      # Detected performers not in expected list

    # Ranking quality
    expected_in_top_1: int    # Expected performers ranked #1
    expected_in_top_3: int    # Expected performers in top 3

    # Score analysis
    correct_match_scores: list[float]
    incorrect_match_scores: list[float]
    score_gap: float          # Avg correct score - avg incorrect score

    # Detection stats
    faces_detected: int
    faces_after_filter: int
    persons_clustered: int

    # Timing
    elapsed_sec: float
```

### Per-Performer Analysis

```python
@dataclass
class PerformerResult:
    stashdb_id: str
    name: str
    faces_in_db: int
    has_body_data: bool
    has_tattoo_data: bool

    # Outcome
    was_found: bool
    rank_if_found: int
    confidence_if_found: float
    distance_if_found: float

    # Context for failures
    who_beat_them: list[tuple[str, float]]  # (name, score) of higher-ranked
    best_match_for_missed: Optional[str]    # Who did we match instead?
```

### Aggregate Metrics

```python
@dataclass
class AggregateMetrics:
    total_scenes: int
    total_expected: int

    # Accuracy
    accuracy: float                    # true_positives / total_expected
    precision: float                   # true_positives / (true_positives + false_positives)
    recall: float                      # true_positives / total_expected

    # By resolution
    accuracy_by_resolution: dict[str, float]

    # By DB coverage
    accuracy_by_coverage: dict[str, float]

    # By face count
    accuracy_by_face_count: dict[str, float]  # "1-2", "3-5", "6+"
```

---

## 4. Pattern Detection

### Failure Analysis

For each missed performer, classify why:

1. **Insufficient DB coverage** - Performer has <3 faces in DB
2. **Similar performer won** - Higher-ranked match looks similar, has more faces
3. **Not detected** - No face detected in any frame for this performer
4. **Low confidence** - Detected but below threshold
5. **Multi-signal penalty** - Body/tattoo mismatch lowered score

### False Positive Analysis

For each wrong match, classify:

1. **Background person** - Crew, extra, or bystander detected
2. **Similar appearance** - Looks like expected performer
3. **DB pollution** - Wrong face in performer's DB entry
4. **Low-quality detection** - Blurry/partial face matched poorly

### Correlation Analysis

- Accuracy vs `faces_in_db` (find minimum threshold)
- Accuracy vs resolution
- False positive rate vs `max_distance`
- Multi-signal impact by performer characteristics

---

## 5. Iterative Testing

### Round Structure

```
Round 1: Broad Exploration
  - Baseline on all scenes
  - Each parameter variation on 30% sample
  - Identify parameters with >2% accuracy impact

Round 2: Focused Refinement
  - Finer granularity on impactful parameters
  - e.g., if 0.6 beat 0.7, test 0.55, 0.65
  - Drop parameters with no impact

Round 3: Combination Testing
  - Test promising combinations together
  - Interaction effects between parameters

Round 4: Validation
  - Winning config on ALL scenes
  - Statistical significance check
  - Final recommendation
```

### Stopping Criteria

- No parameter change improves accuracy by >1%
- Or consistent winners across resolution tiers
- Maximum 6 rounds

### Checkpointing

Save state after each round:
```python
@dataclass
class BenchmarkState:
    round_num: int
    scenes: list[TestScene]
    results_by_round: dict[int, list[SceneResult]]
    current_best_params: dict
    current_best_accuracy: float
    parameters_eliminated: list[str]
```

---

## 6. Output and Reporting

### Console Progress

```
=== Round 1: Baseline ===
Testing 150 scenes with params: frequency, dist=0.7, face=40, multi=true
[################....] 80/150 scenes | 73.2% accuracy | ETA 5m
```

### Summary Report

```
=== Benchmark Summary ===
Scenes tested: 150
Total expected performers: 342
Overall accuracy: 76.3% (261/342)

By Resolution:
  480p:  68.2% (45/66)
  720p:  74.1% (83/112)
  1080p: 82.3% (121/147)
  4K:    70.6% (12/17)

By DB Coverage:
  Well-covered (5+ faces): 84.2%
  Sparse (<5 faces):       51.3%

Multi-signal Impact:
  Face-only:        74.1%
  Face+body+tattoo: 76.3% (+2.2%)
```

### CSV Exports

| File | Contents |
|------|----------|
| `scene_results.csv` | Per-scene metrics, all parameters |
| `performer_results.csv` | Per-performer outcomes with DB stats |
| `false_positives.csv` | Every wrong match with context |
| `parameter_comparison.csv` | A/B results by parameter |
| `failure_patterns.csv` | Classified failure reasons |

### Final Recommendation

```json
{
  "recommended_config": {
    "matching_mode": "frequency",
    "max_distance": 0.65,
    "min_face_size": 50,
    "use_multi_signal": true,
    "use_body": true,
    "use_tattoo": true
  },
  "accuracy": 0.791,
  "baseline_accuracy": 0.741,
  "improvement": 0.050,
  "validated_on": {
    "scenes": 150,
    "performers": 342
  },
  "notes": [
    "Multi-signal provides +2.2% on well-covered performers",
    "max_distance=0.65 reduces false positives without hurting recall",
    "480p still lags behind 1080p by ~14%"
  ]
}
```

---

## 7. Re-validation of Earlier Findings

### Original Test Scenes

Re-run on documented scenes from `face-detection-tuning.md`:
- Scene 13938 (1080p, 2 performers)
- Scene 30835 (1080p, 2 performers)
- Scene 16342 (1080p, 2 performers)
- Scene 26367 (1080p, 2 performers)

### Findings to Re-test

| Finding | Original Result | Re-test |
|---------|-----------------|---------|
| 480p accuracy | ~40% | ? |
| 1080p accuracy | ~70% | ? |
| Frequency mode | 70% (14/20) | ? |
| Cluster mode | 70% but unstable | ? |
| 1-face performers | 55-65% confidence | ? |

### Before/After Report

```
=== Re-validation Report ===

Database Growth:
  Before: ~100k faces, ~X performers
  After:  407k faces, 132k performers

Finding: "480p achieves ~40% accuracy"
  Before: 40%
  After:  ??%
  Change: +??%

Finding: "1-face performers match poorly"
  Before: 55-65% confidence
  After:  Average faces/performer now: Y
          1-face performers: Z% of DB
          Accuracy: ??%
```

---

## 8. Code Architecture

### File Structure

```
api/
  benchmark/
    __init__.py
    runner.py           # Main orchestration
    scene_selector.py   # Query Stash, stratified sampling
    test_executor.py    # Run identify with parameters
    analyzer.py         # Compute metrics, patterns
    reporter.py         # Generate reports
    models.py           # Data classes
    config.py           # Parameter grids, thresholds

  benchmark_results/    # Output (gitignored)
    round_1/
    round_2/
    ...
    final_report.md
    recommended_config.json
    checkpoint.json
```

### Key Classes

```python
class BenchmarkRunner:
    """Orchestrates iterative benchmark."""
    def run(self, max_rounds: int = 4) -> BenchmarkReport
    def run_round(self, round_num: int, param_grid: dict) -> RoundResults
    def save_checkpoint(self)
    def load_checkpoint(self) -> Optional[BenchmarkState]

class SceneSelector:
    """Queries Stash for test scenes."""
    def select_scenes(self, min_count: int = 100) -> list[TestScene]
    def get_performer_db_coverage(self, stashdb_ids: list[str]) -> dict
    def stratify_scenes(self, scenes: list) -> dict[str, list[TestScene]]

class TestExecutor:
    """Runs identification."""
    def identify_scene(self, scene: TestScene, params: dict) -> SceneResult
    def run_batch(self, scenes: list, params: dict) -> list[SceneResult]

class Analyzer:
    """Computes metrics and patterns."""
    def compare_to_ground_truth(self, result, scene) -> PerformerResults
    def find_failure_patterns(self, results) -> FailureReport
    def compare_parameters(self, results_a, results_b) -> Comparison
    def compute_aggregate_metrics(self, results) -> AggregateMetrics

class Reporter:
    """Generates outputs."""
    def print_progress(self, current, total, metrics)
    def generate_summary(self, results) -> str
    def export_csvs(self, results, output_dir)
    def generate_final_report(self, state) -> str
```

### CLI Interface

```bash
# Full benchmark
python -m benchmark.runner

# Quick test (20 scenes, 2 rounds)
python -m benchmark.runner --quick

# Resume from checkpoint
python -m benchmark.runner --resume

# Specific scenes only
python -m benchmark.runner --scenes 13938,30835,16342

# Skip to specific round
python -m benchmark.runner --resume --start-round 3
```

---

## 9. Implementation Notes

### Rate Limiting

Use existing rate limiter to avoid overwhelming Stash:
- 2-3 concurrent scene identifications max
- Progress bar with ETA

### Error Handling

- Skip scenes that fail extraction (log error, continue)
- Retry transient failures up to 3 times
- Save partial results on interrupt (Ctrl+C)

### Performance Estimates

- ~150 scenes × 40 frames × ~2 sec/scene = ~5 min per full pass
- Full benchmark (4 rounds, parameter variations) = ~30-60 min
- Quick mode (20 scenes, 2 rounds) = ~5 min

---

## 10. Success Criteria

The benchmark is complete when we have:

1. **Clear parameter recommendations** with validated accuracy numbers
2. **Failure pattern analysis** showing where to invest effort
3. **Multi-signal evaluation** - does it help? By how much? When?
4. **Re-validation results** - did DB enrichment improve accuracy?
5. **Actionable next steps** - more faces? Better parameters? Algorithm changes?

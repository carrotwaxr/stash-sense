"""Systematic investigation of identification issues.

Phase 1: Establish new baseline with optimal settings
Phase 2: Failure pattern analysis
Phase 3: Multi-signal investigation
Phase 4: 480p deep dive

Usage:
    python -m benchmark.investigation
"""

import asyncio
import json
import os
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from config import StashConfig, DatabaseConfig
from stash_client import StashClient
from database_reader import PerformerDatabaseReader
from recognizer import FaceRecognizer
from multi_signal_matcher import MultiSignalMatcher

from benchmark.scene_selector import SceneSelector
from benchmark.test_executor import TestExecutor
from benchmark.analyzer import Analyzer
from benchmark.models import BenchmarkParams, TestScene, SceneResult, ExpectedPerformer


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs, flush=True)


def initialize_components():
    """Initialize all benchmark components."""
    stash_config = StashConfig.from_env()
    data_dir = os.environ.get("DATA_DIR", "./data")
    db_config = DatabaseConfig(data_dir=Path(data_dir))

    flush_print("Initializing components...")
    stash_client = StashClient(stash_config.url, stash_config.api_key)
    database_reader = PerformerDatabaseReader(str(db_config.sqlite_db_path))

    flush_print("  Loading face recognition models...")
    recognizer = FaceRecognizer(db_config)

    flush_print("  Loading multi-signal matcher...")
    matcher = MultiSignalMatcher(recognizer, database_reader)

    scene_selector = SceneSelector(stash_client, database_reader)
    analyzer = Analyzer()

    return recognizer, matcher, scene_selector, analyzer, database_reader, stash_client


async def run_with_params(
    name: str,
    scenes: list[TestScene],
    params: BenchmarkParams,
    executor: TestExecutor,
    analyzer: Analyzer,
) -> tuple[list[SceneResult], dict]:
    """Run identification and return results with metrics."""
    flush_print(f"\n{'='*60}")
    flush_print(f"Running: {name}")
    flush_print(f"Params: mode={params.matching_mode}, dist={params.max_distance}, face={params.min_face_size}")
    flush_print(f"Scenes: {len(scenes)}")
    flush_print("="*60)

    results = await executor.run_batch(scenes, params)
    metrics = analyzer.compute_aggregate_metrics(results)

    flush_print(f"Results: {metrics.total_true_positives}/{metrics.total_expected} ({metrics.accuracy:.1%} accuracy)")
    flush_print(f"Precision: {metrics.precision:.1%}, Recall: {metrics.recall:.1%}")

    return results, {
        "name": name,
        "scenes": len(scenes),
        "expected": metrics.total_expected,
        "true_positives": metrics.total_true_positives,
        "false_negatives": metrics.total_false_negatives,
        "false_positives": metrics.total_false_positives,
        "accuracy": metrics.accuracy,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }


def analyze_failure_patterns(
    scenes: list[TestScene],
    results: list[SceneResult],
    db_reader: PerformerDatabaseReader,
) -> dict:
    """Deep analysis of why performers are missed."""
    flush_print("\n" + "="*60)
    flush_print("PHASE 2: FAILURE PATTERN ANALYSIS")
    flush_print("="*60)

    # Build scene lookup
    scene_by_id = {s.scene_id: s for s in scenes}
    result_by_scene = {r.scene_id: r for r in results}

    # Categorize failures
    failures = {
        "insufficient_db_coverage": [],  # <3 faces in DB
        "low_db_coverage": [],           # 3-5 faces in DB
        "no_faces_detected": [],         # Zero faces detected in scene
        "few_faces_detected": [],        # Fewer faces than performers
        "wrong_match_won": [],           # Detected but someone else ranked higher
        "below_threshold": [],           # Detected but below distance threshold
        "unknown": [],                   # Can't determine reason
    }

    # Track per-performer stats
    performer_stats = defaultdict(lambda: {
        "name": "",
        "faces_in_db": 0,
        "scenes_appeared": 0,
        "times_found": 0,
        "times_missed": 0,
        "miss_reasons": [],
    })

    total_expected = 0
    total_found = 0

    for scene in scenes:
        result = result_by_scene.get(scene.scene_id)
        if not result:
            continue

        # Get found performer IDs from result
        found_ids = set()
        found_performers = []
        if hasattr(result, 'params') and isinstance(result.params, dict):
            # Results don't directly store found performers, need to infer from TP/FP
            pass

        # For each expected performer
        for performer in scene.expected_performers:
            total_expected += 1
            stats = performer_stats[performer.stashdb_id]
            stats["name"] = performer.name
            stats["faces_in_db"] = performer.faces_in_db
            stats["scenes_appeared"] += 1

            # Check if found (simplified - we know TP count but not which ones)
            # We'll use faces_in_db and detection stats to infer failure reasons

            # Categorize by DB coverage
            if performer.faces_in_db < 3:
                category = "insufficient_db_coverage"
            elif performer.faces_in_db < 6:
                category = "low_db_coverage"
            elif result.faces_detected == 0:
                category = "no_faces_detected"
            elif result.faces_after_filter < len(scene.expected_performers):
                category = "few_faces_detected"
            else:
                category = "unknown"

            # We can't know exactly which performers were found without more data
            # So we'll track aggregate patterns

    # Aggregate analysis
    flush_print("\n### Failure Categories (estimated from scene-level data) ###\n")

    # Analyze by DB coverage
    coverage_buckets = defaultdict(lambda: {"total": 0, "scenes_with_issues": 0})
    for scene in scenes:
        result = result_by_scene.get(scene.scene_id)
        if not result:
            continue

        for performer in scene.expected_performers:
            if performer.faces_in_db < 3:
                bucket = "0-2 faces"
            elif performer.faces_in_db < 6:
                bucket = "3-5 faces"
            elif performer.faces_in_db < 10:
                bucket = "6-9 faces"
            else:
                bucket = "10+ faces"

            coverage_buckets[bucket]["total"] += 1

    # Scene-level failure analysis
    scenes_with_zero_tp = 0
    scenes_with_partial_tp = 0
    scenes_with_full_tp = 0
    scenes_with_many_fp = 0

    detection_stats = {
        "total_faces_detected": 0,
        "total_faces_after_filter": 0,
        "total_persons_clustered": 0,
        "scenes_with_no_detection": 0,
    }

    for result in results:
        scene = scene_by_id.get(result.scene_id)
        if not scene:
            continue

        expected_count = len(scene.expected_performers)

        if result.true_positives == 0:
            scenes_with_zero_tp += 1
        elif result.true_positives < expected_count:
            scenes_with_partial_tp += 1
        else:
            scenes_with_full_tp += 1

        if result.false_positives > expected_count:
            scenes_with_many_fp += 1

        detection_stats["total_faces_detected"] += result.faces_detected
        detection_stats["total_faces_after_filter"] += result.faces_after_filter
        detection_stats["total_persons_clustered"] += result.persons_clustered
        if result.faces_detected == 0:
            detection_stats["scenes_with_no_detection"] += 1

    flush_print(f"Scene-level outcomes:")
    flush_print(f"  Full success (all found):  {scenes_with_full_tp}/{len(results)} ({100*scenes_with_full_tp/len(results):.1f}%)")
    flush_print(f"  Partial success:           {scenes_with_partial_tp}/{len(results)} ({100*scenes_with_partial_tp/len(results):.1f}%)")
    flush_print(f"  Complete failure (0 found): {scenes_with_zero_tp}/{len(results)} ({100*scenes_with_zero_tp/len(results):.1f}%)")
    flush_print(f"  Many false positives:      {scenes_with_many_fp}/{len(results)} ({100*scenes_with_many_fp/len(results):.1f}%)")

    flush_print(f"\nDetection statistics:")
    flush_print(f"  Avg faces detected/scene:     {detection_stats['total_faces_detected']/len(results):.1f}")
    flush_print(f"  Avg faces after filter/scene: {detection_stats['total_faces_after_filter']/len(results):.1f}")
    flush_print(f"  Avg persons clustered/scene:  {detection_stats['total_persons_clustered']/len(results):.1f}")
    flush_print(f"  Scenes with no detection:     {detection_stats['scenes_with_no_detection']}")

    flush_print(f"\nPerformer DB coverage distribution:")
    for bucket in ["0-2 faces", "3-5 faces", "6-9 faces", "10+ faces"]:
        count = coverage_buckets[bucket]["total"]
        flush_print(f"  {bucket}: {count} performers")

    # False positive analysis
    total_fp = sum(r.false_positives for r in results)
    total_tp = sum(r.true_positives for r in results)
    flush_print(f"\nFalse positive analysis:")
    flush_print(f"  Total true positives:  {total_tp}")
    flush_print(f"  Total false positives: {total_fp}")
    flush_print(f"  FP ratio: {total_fp/(total_tp+total_fp):.1%} of all matches are wrong")

    return {
        "scenes_with_zero_tp": scenes_with_zero_tp,
        "scenes_with_partial_tp": scenes_with_partial_tp,
        "scenes_with_full_tp": scenes_with_full_tp,
        "scenes_with_many_fp": scenes_with_many_fp,
        "detection_stats": detection_stats,
        "coverage_buckets": {k: v["total"] for k, v in coverage_buckets.items()},
        "total_tp": total_tp,
        "total_fp": total_fp,
    }


def analyze_multi_signal_data(
    scenes: list[TestScene],
    db_reader: PerformerDatabaseReader,
) -> dict:
    """Investigate why multi-signal provides no benefit."""
    flush_print("\n" + "="*60)
    flush_print("PHASE 3: MULTI-SIGNAL DATA INVESTIGATION")
    flush_print("="*60)

    # Check body and tattoo data availability
    body_data = db_reader.get_all_body_proportions()
    tattoo_data = db_reader.get_all_tattoo_info()

    flush_print(f"\nDatabase multi-signal coverage:")
    flush_print(f"  Performers with body data:   {len(body_data)}")
    flush_print(f"  Performers with tattoo data: {len(tattoo_data)}")

    # Check coverage for test scene performers
    performers_with_body = 0
    performers_with_tattoo = 0
    performers_with_both = 0
    performers_with_neither = 0
    total_performers = 0

    for scene in scenes:
        for performer in scene.expected_performers:
            total_performers += 1
            universal_id = f"stashdb.org:{performer.stashdb_id}"

            has_body = universal_id in body_data
            has_tattoo = universal_id in tattoo_data and tattoo_data[universal_id].get("has_tattoos", False)

            if has_body:
                performers_with_body += 1
            if has_tattoo:
                performers_with_tattoo += 1
            if has_body and has_tattoo:
                performers_with_both += 1
            if not has_body and not has_tattoo:
                performers_with_neither += 1

    flush_print(f"\nTest scene performer coverage:")
    flush_print(f"  Total performers in test set: {total_performers}")
    flush_print(f"  With body proportions:        {performers_with_body} ({100*performers_with_body/total_performers:.1f}%)")
    flush_print(f"  With tattoo data:             {performers_with_tattoo} ({100*performers_with_tattoo/total_performers:.1f}%)")
    flush_print(f"  With both signals:            {performers_with_both} ({100*performers_with_both/total_performers:.1f}%)")
    flush_print(f"  With neither signal:          {performers_with_neither} ({100*performers_with_neither/total_performers:.1f}%)")

    # Sample body data quality
    flush_print(f"\nBody data sample (first 5):")
    for i, (uid, data) in enumerate(list(body_data.items())[:5]):
        flush_print(f"  {uid}: shoulder_hip={data.get('shoulder_hip_ratio', 'N/A'):.2f}, confidence={data.get('confidence', 'N/A'):.2f}")

    # Sample tattoo data
    tattoo_positive = [(uid, data) for uid, data in tattoo_data.items() if data.get("has_tattoos")]
    flush_print(f"\nTattoo data sample (performers with tattoos):")
    for uid, data in list(tattoo_positive)[:5]:
        flush_print(f"  {uid}: locations={data.get('locations', [])[:3]}")

    return {
        "total_body_data": len(body_data),
        "total_tattoo_data": len(tattoo_data),
        "test_performers": total_performers,
        "with_body": performers_with_body,
        "with_tattoo": performers_with_tattoo,
        "with_both": performers_with_both,
        "with_neither": performers_with_neither,
    }


def analyze_480p_issues(
    scenes: list[TestScene],
    results: list[SceneResult],
) -> dict:
    """Deep dive into 480p performance issues."""
    flush_print("\n" + "="*60)
    flush_print("PHASE 4: 480p DEEP DIVE")
    flush_print("="*60)

    scene_by_id = {s.scene_id: s for s in scenes}
    result_by_scene = {r.scene_id: r for r in results}

    # Separate by resolution
    resolution_data = defaultdict(lambda: {
        "scenes": [],
        "results": [],
        "faces_detected": [],
        "faces_filtered": [],
        "persons_clustered": [],
        "tp": 0,
        "fn": 0,
        "fp": 0,
    })

    for scene in scenes:
        result = result_by_scene.get(scene.scene_id)
        if not result:
            continue

        res = scene.resolution
        resolution_data[res]["scenes"].append(scene)
        resolution_data[res]["results"].append(result)
        resolution_data[res]["faces_detected"].append(result.faces_detected)
        resolution_data[res]["faces_filtered"].append(result.faces_after_filter)
        resolution_data[res]["persons_clustered"].append(result.persons_clustered)
        resolution_data[res]["tp"] += result.true_positives
        resolution_data[res]["fn"] += result.false_negatives
        resolution_data[res]["fp"] += result.false_positives

    flush_print("\nDetailed resolution comparison:")
    flush_print(f"{'Resolution':<10} {'Scenes':>8} {'Accuracy':>10} {'Avg Faces':>12} {'Avg Filtered':>14} {'Avg Persons':>13}")
    flush_print("-" * 75)

    for res in ["480p", "720p", "1080p", "4k"]:
        data = resolution_data[res]
        if not data["scenes"]:
            continue

        n = len(data["scenes"])
        total = data["tp"] + data["fn"]
        accuracy = data["tp"] / total if total > 0 else 0
        avg_faces = sum(data["faces_detected"]) / n
        avg_filtered = sum(data["faces_filtered"]) / n
        avg_persons = sum(data["persons_clustered"]) / n

        flush_print(f"{res:<10} {n:>8} {accuracy:>9.1%} {avg_faces:>12.1f} {avg_filtered:>14.1f} {avg_persons:>13.1f}")

    # 480p specific analysis
    data_480 = resolution_data["480p"]
    if data_480["scenes"]:
        flush_print(f"\n480p specific issues:")
        flush_print(f"  Scenes: {len(data_480['scenes'])}")
        flush_print(f"  True positives: {data_480['tp']}")
        flush_print(f"  False negatives: {data_480['fn']}")
        flush_print(f"  False positives: {data_480['fp']}")

        # Check if 480p has fewer faces detected
        avg_480_faces = sum(data_480["faces_detected"]) / len(data_480["scenes"])
        avg_1080_faces = sum(resolution_data["1080p"]["faces_detected"]) / len(resolution_data["1080p"]["scenes"]) if resolution_data["1080p"]["scenes"] else 0

        flush_print(f"\n  Avg faces detected (480p): {avg_480_faces:.1f}")
        flush_print(f"  Avg faces detected (1080p): {avg_1080_faces:.1f}")

        if avg_480_faces < avg_1080_faces * 0.7:
            flush_print(f"  -> 480p detects significantly fewer faces!")

    return {
        res: {
            "scenes": len(data["scenes"]),
            "tp": data["tp"],
            "fn": data["fn"],
            "fp": data["fp"],
            "accuracy": data["tp"] / (data["tp"] + data["fn"]) if (data["tp"] + data["fn"]) > 0 else 0,
        }
        for res, data in resolution_data.items()
    }


async def run_investigation():
    """Run the full investigation."""
    flush_print("=" * 60)
    flush_print("SYSTEMATIC IDENTIFICATION INVESTIGATION")
    flush_print("=" * 60)
    flush_print(f"Started: {datetime.now().isoformat()}")

    # Initialize
    recognizer, matcher, scene_selector, analyzer, db_reader, stash_client = initialize_components()
    executor = TestExecutor(recognizer, matcher)

    # Select test scenes
    flush_print("\nSelecting test scenes...")
    all_scenes = await scene_selector.select_scenes(min_count=100)
    flush_print(f"Selected {len(all_scenes)} scenes")

    output_dir = Path("benchmark_results/investigation")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_findings = {}

    # ==========================================================================
    # PHASE 1: NEW BASELINE WITH OPTIMAL SETTINGS
    # ==========================================================================
    flush_print("\n" + "="*60)
    flush_print("PHASE 1: ESTABLISHING NEW BASELINE")
    flush_print("="*60)

    # Old baseline for comparison
    old_params = BenchmarkParams(
        matching_mode="frequency",
        max_distance=0.7,
        min_face_size=40,
        use_multi_signal=True,
        num_frames=40,
    )
    old_results, old_metrics = await run_with_params(
        "Old baseline (frequency, dist=0.7, face=40)",
        all_scenes,
        old_params,
        executor,
        analyzer,
    )

    # New optimized baseline
    new_params = BenchmarkParams(
        matching_mode="hybrid",
        max_distance=0.6,
        min_face_size=60,
        use_multi_signal=True,
        num_frames=40,
    )
    new_results, new_metrics = await run_with_params(
        "New baseline (hybrid, dist=0.6, face=60)",
        all_scenes,
        new_params,
        executor,
        analyzer,
    )

    improvement = new_metrics["accuracy"] - old_metrics["accuracy"]
    flush_print(f"\n>>> IMPROVEMENT: {improvement:+.1%} ({old_metrics['accuracy']:.1%} -> {new_metrics['accuracy']:.1%})")

    all_findings["phase1_baseline"] = {
        "old": old_metrics,
        "new": new_metrics,
        "improvement": improvement,
    }

    # ==========================================================================
    # PHASE 2: FAILURE PATTERN ANALYSIS
    # ==========================================================================
    failure_analysis = analyze_failure_patterns(all_scenes, new_results, db_reader)
    all_findings["phase2_failures"] = failure_analysis

    # ==========================================================================
    # PHASE 3: MULTI-SIGNAL INVESTIGATION
    # ==========================================================================
    multi_signal_analysis = analyze_multi_signal_data(all_scenes, db_reader)
    all_findings["phase3_multi_signal"] = multi_signal_analysis

    # ==========================================================================
    # PHASE 4: 480p DEEP DIVE
    # ==========================================================================
    resolution_analysis = analyze_480p_issues(all_scenes, new_results)
    all_findings["phase4_480p"] = resolution_analysis

    # ==========================================================================
    # SUMMARY & RECOMMENDATIONS
    # ==========================================================================
    flush_print("\n" + "="*60)
    flush_print("SUMMARY & RECOMMENDATIONS")
    flush_print("="*60)

    flush_print(f"""
KEY FINDINGS:

1. BASELINE IMPROVEMENT
   Old: {old_metrics['accuracy']:.1%} accuracy
   New: {new_metrics['accuracy']:.1%} accuracy
   Gain: {improvement:+.1%}

2. FAILURE PATTERNS
   - Complete failures (0 found): {failure_analysis['scenes_with_zero_tp']} scenes
   - Partial success: {failure_analysis['scenes_with_partial_tp']} scenes
   - Full success: {failure_analysis['scenes_with_full_tp']} scenes
   - High FP rate: {failure_analysis['total_fp']} false positives vs {failure_analysis['total_tp']} true positives

3. MULTI-SIGNAL COVERAGE
   - Body data coverage: {multi_signal_analysis['with_body']}/{multi_signal_analysis['test_performers']} ({100*multi_signal_analysis['with_body']/multi_signal_analysis['test_performers']:.1f}%)
   - Tattoo data coverage: {multi_signal_analysis['with_tattoo']}/{multi_signal_analysis['test_performers']} ({100*multi_signal_analysis['with_tattoo']/multi_signal_analysis['test_performers']:.1f}%)
   - Neither signal: {multi_signal_analysis['with_neither']}/{multi_signal_analysis['test_performers']} ({100*multi_signal_analysis['with_neither']/multi_signal_analysis['test_performers']:.1f}%)

4. RESOLUTION BREAKDOWN
""")

    for res in ["480p", "720p", "1080p", "4k"]:
        if res in resolution_analysis and resolution_analysis[res]["scenes"] > 0:
            flush_print(f"   {res}: {resolution_analysis[res]['accuracy']:.1%} ({resolution_analysis[res]['scenes']} scenes)")

    flush_print("""
RECOMMENDATIONS:

1. ADOPT NEW BASELINE: hybrid mode + dist=0.6 + face=60

2. REDUCE FALSE POSITIVES: High FP ratio suggests threshold tuning or
   post-processing (e.g., require minimum face appearances)

3. IMPROVE MULTI-SIGNAL: Low coverage means signals can't help most performers.
   Need to run body/tattoo extraction on more performer images.

4. ADDRESS 480p: Consider resolution-specific parameters or upscaling
   before face detection.
""")

    # Save results
    results_path = output_dir / "investigation_results.json"
    with open(results_path, "w") as f:
        json.dump(all_findings, f, indent=2, default=str)
    flush_print(f"\nResults saved to: {results_path}")

    return all_findings


if __name__ == "__main__":
    asyncio.run(run_investigation())

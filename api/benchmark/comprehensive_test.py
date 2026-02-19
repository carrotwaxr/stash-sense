"""Comprehensive benchmark test to answer key identification questions.

This script runs a series of benchmarks to:
1. Establish baseline accuracy with current parameters
2. Test parameter variations (distance, face size)
3. Evaluate multi-signal impact (face-only vs face+body+tattoo)
4. Re-validate findings on original test scenes
5. Analyze failure patterns by resolution, DB coverage, etc.

Usage:
    python -m benchmark.comprehensive_test
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

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
from benchmark.models import BenchmarkParams, TestScene


@dataclass
class TestResult:
    """Result of a single test configuration."""
    name: str
    params: dict
    scenes_tested: int
    performers_expected: int
    true_positives: int
    false_negatives: int
    false_positives: int
    accuracy: float
    precision: float
    recall: float
    avg_time_per_scene: float
    accuracy_by_resolution: dict
    accuracy_by_coverage: dict


def initialize_components():
    """Initialize all benchmark components."""
    stash_config = StashConfig.from_env()
    data_dir = os.environ.get("DATA_DIR", "./data")
    db_config = DatabaseConfig(data_dir=Path(data_dir))

    print("Initializing components...")
    stash_client = StashClient(stash_config.url, stash_config.api_key)
    database_reader = PerformerDatabaseReader(str(db_config.sqlite_db_path))

    print("  Loading face recognition models...")
    recognizer = FaceRecognizer(db_config)

    print("  Loading multi-signal matcher...")
    matcher = MultiSignalMatcher(recognizer, database_reader)

    scene_selector = SceneSelector(stash_client, database_reader)
    analyzer = Analyzer()

    return recognizer, matcher, scene_selector, analyzer, database_reader


async def run_test_config(
    name: str,
    scenes: list[TestScene],
    params: BenchmarkParams,
    executor: TestExecutor,
    analyzer: Analyzer,
) -> TestResult:
    """Run a single test configuration and return results."""
    print(f"\n--- Testing: {name} ---")
    print(f"    Params: distance={params.max_distance}, face_size={params.min_face_size}, multi_signal={params.use_multi_signal}")
    print(f"    Scenes: {len(scenes)}")

    start_time = time.time()
    results = await executor.run_batch(scenes, params)
    elapsed = time.time() - start_time

    # Compute metrics
    metrics = analyzer.compute_aggregate_metrics(results)
    accuracy_by_res = analyzer.compute_accuracy_by_resolution(scenes, results)
    accuracy_by_cov = analyzer.compute_accuracy_by_coverage(scenes, results)

    result = TestResult(
        name=name,
        params=params.to_dict(),
        scenes_tested=metrics.total_scenes,
        performers_expected=metrics.total_expected,
        true_positives=metrics.total_true_positives,
        false_negatives=metrics.total_false_negatives,
        false_positives=metrics.total_false_positives,
        accuracy=metrics.accuracy,
        precision=metrics.precision,
        recall=metrics.recall,
        avg_time_per_scene=elapsed / len(scenes) if scenes else 0,
        accuracy_by_resolution=accuracy_by_res,
        accuracy_by_coverage=accuracy_by_cov,
    )

    print(f"    Results: {metrics.total_true_positives}/{metrics.total_expected} ({metrics.accuracy:.1%} accuracy)")
    print(f"    By resolution: {accuracy_by_res}")
    print(f"    Time: {elapsed:.1f}s ({result.avg_time_per_scene:.1f}s/scene)")

    return result


async def run_comprehensive_benchmark():
    """Run the full comprehensive benchmark suite."""
    print("=" * 60)
    print("COMPREHENSIVE IDENTIFICATION BENCHMARK")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Initialize
    recognizer, matcher, scene_selector, analyzer, db_reader = initialize_components()
    executor = TestExecutor(recognizer, matcher)

    # Get DB stats
    db_stats = db_reader.get_stats()
    print("\nDatabase stats:")
    print(f"  Performers: {db_stats['performer_count']}")
    print(f"  Performers with faces: {db_stats['performers_with_faces']}")
    print(f"  Total faces: {db_stats['total_faces']}")

    # Select test scenes (aim for 100 for statistical significance)
    print("\nSelecting test scenes...")
    all_scenes = await scene_selector.select_scenes(min_count=100)
    print(f"Selected {len(all_scenes)} scenes")

    # Stratify for analysis
    stratified = scene_selector.stratify_scenes(all_scenes)
    print("By resolution:")
    for tier, scenes in stratified.items():
        print(f"  {tier}: {len(scenes)} scenes")

    results = []
    output_dir = Path("benchmark_results/comprehensive")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==========================================================================
    # TEST 1: BASELINE (current default parameters)
    # ==========================================================================
    baseline_params = BenchmarkParams(
        matching_mode="frequency",
        max_distance=0.7,
        min_face_size=40,
        use_multi_signal=True,
        num_frames=40,
    )
    baseline = await run_test_config(
        "Baseline (dist=0.7, face=40, multi=true)",
        all_scenes,
        baseline_params,
        executor,
        analyzer,
    )
    results.append(baseline)

    # ==========================================================================
    # TEST 2: DISTANCE VARIATIONS
    # ==========================================================================
    sample_30pct = scene_selector.sample_stratified(all_scenes, int(len(all_scenes) * 0.3), seed=42)

    for distance in [0.5, 0.6, 0.8]:
        params = BenchmarkParams(
            matching_mode="frequency",
            max_distance=distance,
            min_face_size=40,
            use_multi_signal=True,
            num_frames=40,
        )
        result = await run_test_config(
            f"Distance test (dist={distance})",
            sample_30pct,
            params,
            executor,
            analyzer,
        )
        results.append(result)

    # ==========================================================================
    # TEST 3: FACE SIZE VARIATIONS
    # ==========================================================================
    for face_size in [60, 80]:
        params = BenchmarkParams(
            matching_mode="frequency",
            max_distance=0.7,
            min_face_size=face_size,
            use_multi_signal=True,
            num_frames=40,
        )
        result = await run_test_config(
            f"Face size test (face={face_size})",
            sample_30pct,
            params,
            executor,
            analyzer,
        )
        results.append(result)

    # ==========================================================================
    # TEST 4: MULTI-SIGNAL EVALUATION (face-only vs face+body+tattoo)
    # ==========================================================================
    no_multi_params = BenchmarkParams(
        matching_mode="frequency",
        max_distance=0.7,
        min_face_size=40,
        use_multi_signal=False,
        num_frames=40,
    )
    no_multi = await run_test_config(
        "Face-only (multi=false)",
        all_scenes,
        no_multi_params,
        executor,
        analyzer,
    )
    results.append(no_multi)

    # ==========================================================================
    # TEST 5: HYBRID MODE COMPARISON
    # ==========================================================================
    hybrid_params = BenchmarkParams(
        matching_mode="hybrid",
        max_distance=0.7,
        min_face_size=40,
        use_multi_signal=True,
        num_frames=40,
    )
    hybrid = await run_test_config(
        "Hybrid mode",
        sample_30pct,
        hybrid_params,
        executor,
        analyzer,
    )
    results.append(hybrid)

    # ==========================================================================
    # TEST 6: RE-VALIDATION OF ORIGINAL TEST SCENES
    # ==========================================================================
    original_scene_ids = ["13938", "30835", "16342", "26367"]
    original_scenes = [s for s in all_scenes if s.scene_id in original_scene_ids]

    if original_scenes:
        original_test = await run_test_config(
            "Original test scenes (re-validation)",
            original_scenes,
            baseline_params,
            executor,
            analyzer,
        )
        results.append(original_test)
    else:
        print("\nNote: Original test scenes not found in selected scenes")

    # ==========================================================================
    # GENERATE REPORT
    # ==========================================================================
    print("\n" + "=" * 60)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 60)

    # Summary table
    print("\n### Test Results Summary ###\n")
    print(f"{'Test Name':<45} {'Scenes':>7} {'Accuracy':>10} {'Precision':>10} {'Recall':>10}")
    print("-" * 90)
    for r in results:
        print(f"{r.name:<45} {r.scenes_tested:>7} {r.accuracy:>9.1%} {r.precision:>9.1%} {r.recall:>9.1%}")

    # Multi-signal comparison
    print("\n### Multi-Signal Impact ###")
    multi_on = next((r for r in results if "Baseline" in r.name), None)
    multi_off = next((r for r in results if "Face-only" in r.name), None)
    if multi_on and multi_off:
        diff = multi_on.accuracy - multi_off.accuracy
        print(f"Face+Body+Tattoo: {multi_on.accuracy:.1%}")
        print(f"Face-only:        {multi_off.accuracy:.1%}")
        print(f"Improvement:      {diff:+.1%}")

    # Distance comparison
    print("\n### Distance Threshold Impact ###")
    dist_results = [r for r in results if "Distance test" in r.name]
    for r in dist_results:
        print(f"max_distance={r.params['max_distance']}: {r.accuracy:.1%}")

    # Face size comparison
    print("\n### Face Size Impact ###")
    face_results = [r for r in results if "Face size" in r.name]
    for r in face_results:
        print(f"min_face_size={r.params['min_face_size']}: {r.accuracy:.1%}")

    # Resolution breakdown (from baseline)
    print("\n### Accuracy by Resolution (Baseline) ###")
    if baseline.accuracy_by_resolution:
        for res, acc in sorted(baseline.accuracy_by_resolution.items()):
            print(f"  {res}: {acc:.1%}")

    # Coverage breakdown
    print("\n### Accuracy by DB Coverage (Baseline) ###")
    if baseline.accuracy_by_coverage:
        for cov, acc in sorted(baseline.accuracy_by_coverage.items()):
            print(f"  {cov}: {acc:.1%}")

    # Save results
    results_dict = {
        "timestamp": datetime.now().isoformat(),
        "db_stats": db_stats,
        "total_scenes": len(all_scenes),
        "tests": [asdict(r) for r in results],
    }

    results_path = output_dir / "comprehensive_results.json"
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Generate markdown report
    report_path = output_dir / "comprehensive_report.md"
    with open(report_path, "w") as f:
        f.write("# Comprehensive Benchmark Report\n\n")
        f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"**Database:** {db_stats['total_faces']} faces, {db_stats['performer_count']} performers\n\n")
        f.write(f"**Test Scenes:** {len(all_scenes)}\n\n")

        f.write("## Results Summary\n\n")
        f.write("| Test | Scenes | Accuracy | Precision | Recall |\n")
        f.write("|------|--------|----------|-----------|--------|\n")
        for r in results:
            f.write(f"| {r.name} | {r.scenes_tested} | {r.accuracy:.1%} | {r.precision:.1%} | {r.recall:.1%} |\n")

        f.write("\n## Key Findings\n\n")

        if multi_on and multi_off:
            diff = multi_on.accuracy - multi_off.accuracy
            f.write("### Multi-Signal Impact\n")
            f.write(f"- Face+Body+Tattoo: **{multi_on.accuracy:.1%}**\n")
            f.write(f"- Face-only: {multi_off.accuracy:.1%}\n")
            f.write(f"- Improvement: **{diff:+.1%}**\n\n")

        f.write("### Distance Threshold\n")
        for r in dist_results:
            f.write(f"- max_distance={r.params['max_distance']}: {r.accuracy:.1%}\n")

        f.write("\n### Resolution Breakdown\n")
        if baseline.accuracy_by_resolution:
            for res, acc in sorted(baseline.accuracy_by_resolution.items()):
                f.write(f"- {res}: {acc:.1%}\n")

    print(f"Report saved to: {report_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_comprehensive_benchmark())

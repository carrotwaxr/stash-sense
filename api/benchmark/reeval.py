"""Re-evaluation benchmark after face recognition database rebuild.

Runs a structured parameter sweep against the new database (107K performers,
277K faces, proper alignment + normalization) to find optimal settings.

6 Phases:
1. Baseline with current defaults
2. Independent parameter sweeps
3. Best-of combination
4. Original scene re-validation
5. Gallery benchmark
6. Image benchmark

Usage:
    cd api
    python -m benchmark.reeval
"""

import asyncio
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv
load_dotenv()

from config import StashConfig, DatabaseConfig
from stash_client import StashClient
from database_reader import PerformerDatabaseReader
from recognizer import FaceRecognizer
from multi_signal_matcher import MultiSignalMatcher

from benchmark.scene_selector import SceneSelector, STASHDB_ENDPOINT
from benchmark.test_executor import TestExecutor
from benchmark.analyzer import Analyzer
from benchmark.models import BenchmarkParams, TestScene, SceneResult


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class SweepResult:
    """Result of a single parameter sweep configuration."""
    name: str
    param_name: str
    param_value: object
    params: dict
    metrics: dict
    scene_results: list[dict] = field(default_factory=list)


@dataclass
class GalleryBenchmarkResult:
    """Result from benchmarking a single gallery."""
    gallery_id: str
    gallery_title: str
    total_images: int
    expected_performers: list[str]  # stashdb_ids
    found_performers: list[str]
    true_positives: int
    false_positives: int
    false_negatives: int
    max_distance: float
    details: list[dict] = field(default_factory=list)


@dataclass
class ImageBenchmarkResult:
    """Result from benchmarking a single image."""
    image_id: str
    expected_performers: list[str]  # stashdb_ids
    found_performers: list[str]
    correct: bool
    max_distance: float
    details: list[dict] = field(default_factory=list)


# ============================================================================
# Initialization
# ============================================================================

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

    return recognizer, matcher, scene_selector, analyzer, database_reader, stash_client, stash_config


# ============================================================================
# Helpers
# ============================================================================

def compute_metrics(scenes: list[TestScene], results: list[SceneResult], analyzer: Analyzer) -> dict:
    """Compute aggregate metrics from scene results."""
    agg = analyzer.compute_aggregate_metrics(results)
    by_res = analyzer.compute_accuracy_by_resolution(scenes, results)
    by_cov = analyzer.compute_accuracy_by_coverage(scenes, results)
    return {
        "total_scenes": agg.total_scenes,
        "total_expected": agg.total_expected,
        "true_positives": agg.total_true_positives,
        "false_positives": agg.total_false_positives,
        "false_negatives": agg.total_false_negatives,
        "accuracy": agg.accuracy,
        "precision": agg.precision,
        "recall": agg.recall,
        "accuracy_by_resolution": by_res,
        "accuracy_by_coverage": by_cov,
    }


def print_metrics(name: str, metrics: dict):
    """Print a summary line for a test configuration."""
    print(f"  {name:<55} acc={metrics['accuracy']:.1%}  prec={metrics['precision']:.1%}  "
          f"TP={metrics['true_positives']}  FP={metrics['false_positives']}  FN={metrics['false_negatives']}")


def print_table_header():
    """Print sweep results table header."""
    print(f"  {'Configuration':<55} {'Accuracy':>8}  {'Precision':>9}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-'*55} {'-'*8}  {'-'*9}  {'-'*4}  {'-'*4}  {'-'*4}")


def print_table_row(name: str, metrics: dict):
    """Print a table row."""
    print(f"  {name:<55} {metrics['accuracy']:>7.1%}  {metrics['precision']:>8.1%}  "
          f"{metrics['true_positives']:>4}  {metrics['false_positives']:>4}  {metrics['false_negatives']:>4}")


async def run_config(
    name: str,
    scenes: list[TestScene],
    params: BenchmarkParams,
    executor: TestExecutor,
    analyzer: Analyzer,
) -> tuple[dict, list[SceneResult]]:
    """Run a configuration and return metrics + raw results."""
    start = time.time()
    results = await executor.run_batch(scenes, params)
    elapsed = time.time() - start
    metrics = compute_metrics(scenes, results, analyzer)
    metrics["elapsed_sec"] = elapsed
    metrics["avg_time_per_scene"] = elapsed / len(scenes) if scenes else 0
    return metrics, results


def get_stashdb_id_for_performer(performer_data: dict) -> Optional[str]:
    """Extract StashDB ID from a Stash performer's stash_ids."""
    for sid in performer_data.get("stash_ids", []):
        if sid.get("endpoint") == STASHDB_ENDPOINT:
            return sid.get("stash_id")
    return None


# ============================================================================
# Phase 1: Baseline
# ============================================================================

async def phase1_baseline(
    scenes: list[TestScene],
    executor: TestExecutor,
    analyzer: Analyzer,
) -> tuple[dict, list[SceneResult]]:
    """Phase 1: Establish baseline with current defaults."""
    print("\n" + "=" * 70)
    print("PHASE 1: BASELINE (current defaults)")
    print("=" * 70)

    params = BenchmarkParams()  # All defaults
    metrics, results = await run_config("Baseline", scenes, params, executor, analyzer)

    print_table_header()
    print_table_row("Baseline (defaults)", metrics)

    if metrics["accuracy_by_resolution"]:
        print("\n  By resolution:")
        for res, acc in sorted(metrics["accuracy_by_resolution"].items()):
            print(f"    {res}: {acc:.1%}")

    if metrics["accuracy_by_coverage"]:
        print("\n  By coverage:")
        for cov, acc in sorted(metrics["accuracy_by_coverage"].items()):
            print(f"    {cov}: {acc:.1%}")

    return metrics, results


# ============================================================================
# Phase 2: Independent Parameter Sweeps
# ============================================================================

async def phase2_sweeps(
    scenes: list[TestScene],
    executor: TestExecutor,
    analyzer: Analyzer,
) -> dict[str, list[SweepResult]]:
    """Phase 2: Run independent parameter sweeps."""
    print("\n" + "=" * 70)
    print("PHASE 2: INDEPENDENT PARAMETER SWEEPS")
    print("=" * 70)

    sweeps: dict[str, list[SweepResult]] = {}

    # --- max_distance sweep ---
    sweep_name = "max_distance"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for val in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        params = BenchmarkParams(max_distance=val)
        name = f"max_distance={val}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=val,
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # --- min_face_size sweep ---
    sweep_name = "min_face_size"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for val in [30, 40, 60, 80, 100]:
        params = BenchmarkParams(min_face_size=val)
        name = f"min_face_size={val}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=val,
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # --- fusion weights sweep ---
    sweep_name = "fusion_weights"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for fn_w, af_w in [(0.7, 0.3), (0.6, 0.4), (0.5, 0.5), (0.4, 0.6), (0.3, 0.7)]:
        params = BenchmarkParams(facenet_weight=fn_w, arcface_weight=af_w)
        name = f"fn={fn_w}/af={af_w}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=(fn_w, af_w),
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # --- matching_mode sweep ---
    sweep_name = "matching_mode"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for val in ["frequency", "hybrid"]:
        params = BenchmarkParams(matching_mode=val)
        name = f"matching_mode={val}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=val,
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # --- min_appearances sweep ---
    sweep_name = "min_appearances"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for val in [1, 2, 3]:
        params = BenchmarkParams(min_appearances=val)
        name = f"min_appearances={val}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=val,
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # --- min_unique_frames sweep ---
    sweep_name = "min_unique_frames"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for val in [1, 2, 3]:
        params = BenchmarkParams(min_unique_frames=val)
        name = f"min_unique_frames={val}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=val,
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # --- min_confidence sweep ---
    sweep_name = "min_confidence"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for val in [0.2, 0.3, 0.35, 0.4, 0.5]:
        params = BenchmarkParams(min_confidence=val)
        name = f"min_confidence={val}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=val,
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # --- num_frames sweep ---
    sweep_name = "num_frames"
    print(f"\n--- Sweep: {sweep_name} ---")
    print_table_header()
    sweep_results = []
    for val in [20, 40, 60, 80]:
        params = BenchmarkParams(num_frames=val)
        name = f"num_frames={val}"
        metrics, _ = await run_config(name, scenes, params, executor, analyzer)
        print_table_row(name, metrics)
        sweep_results.append(SweepResult(
            name=name, param_name=sweep_name, param_value=val,
            params=params.to_dict(), metrics=metrics,
        ))
    sweeps[sweep_name] = sweep_results

    # Print best-per-sweep summary
    print("\n--- Best per sweep ---")
    for sweep_name, results in sweeps.items():
        best = max(results, key=lambda r: (r.metrics["accuracy"], r.metrics["precision"]))
        print(f"  {sweep_name:<25} best: {best.name:<30} acc={best.metrics['accuracy']:.1%}  prec={best.metrics['precision']:.1%}")

    return sweeps


# ============================================================================
# Phase 3: Best-of Combination
# ============================================================================

def pick_best_value(sweep_results: list[SweepResult]) -> object:
    """Pick the best value from a sweep by accuracy then precision."""
    best = max(sweep_results, key=lambda r: (r.metrics["accuracy"], r.metrics["precision"]))
    return best.param_value


async def phase3_best_combination(
    scenes: list[TestScene],
    executor: TestExecutor,
    analyzer: Analyzer,
    sweeps: dict[str, list[SweepResult]],
    baseline_metrics: dict,
) -> tuple[dict, BenchmarkParams]:
    """Phase 3: Combine best values from each sweep."""
    print("\n" + "=" * 70)
    print("PHASE 3: BEST-OF COMBINATION")
    print("=" * 70)

    # Pick best from each sweep
    best_max_distance = pick_best_value(sweeps["max_distance"])
    best_min_face_size = pick_best_value(sweeps["min_face_size"])
    best_fusion = pick_best_value(sweeps["fusion_weights"])
    best_matching_mode = pick_best_value(sweeps["matching_mode"])
    best_min_appearances = pick_best_value(sweeps["min_appearances"])
    best_min_unique_frames = pick_best_value(sweeps["min_unique_frames"])
    best_min_confidence = pick_best_value(sweeps["min_confidence"])
    best_num_frames = pick_best_value(sweeps["num_frames"])

    fn_w, af_w = best_fusion

    best_params = BenchmarkParams(
        max_distance=best_max_distance,
        min_face_size=best_min_face_size,
        facenet_weight=fn_w,
        arcface_weight=af_w,
        matching_mode=best_matching_mode,
        min_appearances=best_min_appearances,
        min_unique_frames=best_min_unique_frames,
        min_confidence=best_min_confidence,
        num_frames=best_num_frames,
    )

    print("  Combined best params:")
    print(f"    max_distance     = {best_max_distance}")
    print(f"    min_face_size    = {best_min_face_size}")
    print(f"    fusion_weights   = fn={fn_w}/af={af_w}")
    print(f"    matching_mode    = {best_matching_mode}")
    print(f"    min_appearances  = {best_min_appearances}")
    print(f"    min_unique_frames= {best_min_unique_frames}")
    print(f"    min_confidence   = {best_min_confidence}")
    print(f"    num_frames       = {best_num_frames}")

    metrics, _ = await run_config("Best combination", scenes, best_params, executor, analyzer)

    print()
    print_table_header()
    print_table_row("Baseline (defaults)", baseline_metrics)
    print_table_row("Best combination", metrics)

    diff = metrics["accuracy"] - baseline_metrics["accuracy"]
    print(f"\n  Improvement over baseline: {diff:+.1%}")

    return metrics, best_params


# ============================================================================
# Phase 4: Original Scene Re-validation
# ============================================================================

async def phase4_original_scenes(
    all_scenes: list[TestScene],
    executor: TestExecutor,
    analyzer: Analyzer,
    scene_selector: SceneSelector,
    best_params: BenchmarkParams,
) -> dict:
    """Phase 4: Re-run original 8 test scenes with both default and best params."""
    print("\n" + "=" * 70)
    print("PHASE 4: ORIGINAL SCENE RE-VALIDATION")
    print("=" * 70)

    original_1080p_ids = {"13938", "30835", "16342", "26367"}
    original_480p_ids = {"1414", "377", "1551", "3283"}
    all_original_ids = original_1080p_ids | original_480p_ids

    # Find scenes - some may already be in our set, others need to be loaded
    existing_ids = {s.scene_id for s in all_scenes}
    original_scenes = [s for s in all_scenes if s.scene_id in all_original_ids]

    missing_ids = all_original_ids - existing_ids
    if missing_ids:
        print(f"  Loading {len(missing_ids)} additional original scenes: {missing_ids}")
        # Query them individually from Stash
        for scene_id in missing_ids:
            try:
                query = f"""
                query {{
                    findScene(id: "{scene_id}") {{
                        id
                        title
                        files {{ width height duration }}
                        stash_ids {{ endpoint stash_id }}
                        performers {{
                            name
                            stash_ids {{ endpoint stash_id }}
                        }}
                    }}
                }}
                """
                response = scene_selector.stash_client._query(query)
                scene_data = response.get("findScene")
                if scene_data and scene_selector._is_valid_test_scene(scene_data):
                    test_scene = await scene_selector._convert_to_test_scene(scene_data)
                    original_scenes.append(test_scene)
                else:
                    print(f"    Scene {scene_id} not valid for benchmarking (missing performers/stash_ids)")
            except Exception as e:
                print(f"    Failed to load scene {scene_id}: {e}")

    if not original_scenes:
        print("  No original scenes available, skipping phase 4")
        return {}

    print(f"  Testing {len(original_scenes)} original scenes")

    # Run with defaults
    default_params = BenchmarkParams()
    default_metrics, default_results = await run_config("Defaults", original_scenes, default_params, executor, analyzer)

    # Run with best params
    best_metrics, best_results = await run_config("Best params", original_scenes, best_params, executor, analyzer)

    print()
    print_table_header()
    print_table_row("Original scenes (defaults)", default_metrics)
    print_table_row("Original scenes (best params)", best_metrics)

    # Per-scene breakdown
    print("\n  Per-scene breakdown:")
    default_by_scene = {r.scene_id: r for r in default_results}
    best_by_scene = {r.scene_id: r for r in best_results}

    for scene in sorted(original_scenes, key=lambda s: s.scene_id):
        dr = default_by_scene.get(scene.scene_id)
        br = best_by_scene.get(scene.scene_id)
        res = scene.resolution
        expected = len(scene.expected_performers)
        d_tp = dr.true_positives if dr else 0
        b_tp = br.true_positives if br else 0
        print(f"    Scene {scene.scene_id:>6} ({res:>5}) "
              f"expected={expected}  default_TP={d_tp}  best_TP={b_tp}")

    return {
        "scenes_tested": len(original_scenes),
        "default_metrics": default_metrics,
        "best_metrics": best_metrics,
    }


# ============================================================================
# Phase 5: Gallery Benchmark
# ============================================================================

async def find_benchmark_galleries(stash_client: StashClient, stash_config) -> list[dict]:
    """Find galleries with 2+ tagged performers who have StashDB IDs."""
    print("  Querying Stash for galleries with tagged performers...")

    # Find galleries with performer tags - paginate through
    query = """
    query FindGalleries($page: Int!, $per_page: Int!) {
        findGalleries(
            filter: { page: $page, per_page: $per_page, sort: "random" }
        ) {
            count
            galleries {
                id
                title
                image_count
                performers {
                    id
                    name
                    stash_ids { endpoint stash_id }
                }
            }
        }
    }
    """

    candidates = []
    page = 1
    per_page = 25
    max_pages = 40  # Cap search

    while len(candidates) < 10 and page <= max_pages:
        data = stash_client._query(query, {"page": page, "per_page": per_page})
        galleries = data.get("findGalleries", {}).get("galleries", [])

        if not galleries:
            break

        for g in galleries:
            performers = g.get("performers", [])
            # All performers need StashDB IDs
            stashdb_performers = []
            for p in performers:
                sid = get_stashdb_id_for_performer(p)
                if sid:
                    stashdb_performers.append({"id": p["id"], "name": p["name"], "stashdb_id": sid})

            if len(stashdb_performers) >= 2 and g.get("image_count", 0) >= 3:
                candidates.append({
                    "id": g["id"],
                    "title": g.get("title", f"Gallery {g['id']}"),
                    "image_count": g["image_count"],
                    "performers": stashdb_performers,
                })

            if len(candidates) >= 10:
                break

        page += 1

    print(f"  Found {len(candidates)} qualifying galleries")
    return candidates


async def phase5_gallery_benchmark(
    recognizer: FaceRecognizer,
    stash_client: StashClient,
    stash_config,
) -> dict:
    """Phase 5: Benchmark gallery identification."""
    print("\n" + "=" * 70)
    print("PHASE 5: GALLERY BENCHMARK")
    print("=" * 70)

    galleries = await find_benchmark_galleries(stash_client, stash_config)
    if not galleries:
        print("  No qualifying galleries found, skipping")
        return {}

    stash_url = stash_config.url.rstrip("/")
    api_key = stash_config.api_key

    distance_thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    all_results = {}

    for threshold in distance_thresholds:
        print(f"\n  --- max_distance={threshold} ---")
        gallery_results = []

        for gal in galleries:
            expected_ids = {p["stashdb_id"] for p in gal["performers"]}
            found_ids = set()
            details = []

            # Fetch gallery images
            from stash_client_unified import StashClientUnified
            unified = StashClientUnified(stash_url, api_key)
            gallery_data = await unified.get_gallery_by_id(gal["id"])

            if not gallery_data or not gallery_data.get("images"):
                print(f"    Gallery {gal['id']}: no images")
                continue

            images = gallery_data["images"]
            performer_appearances = defaultdict(list)
            performer_info = {}

            headers = {"ApiKey": api_key} if api_key else {}
            async with httpx.AsyncClient(timeout=30.0) as client:
                for img in images:
                    img_url = img.get("paths", {}).get("image")
                    if not img_url:
                        continue
                    try:
                        resp = await client.get(img_url, headers=headers)
                        resp.raise_for_status()
                        from embeddings import load_image
                        image = load_image(resp.content)
                        results = recognizer.recognize_image(
                            image, top_k=5, max_distance=threshold,
                        )
                        for result in results:
                            if result.matches:
                                best = result.matches[0]
                                pid = best.stashdb_id
                                performer_appearances[pid].append({
                                    "image_id": img["id"],
                                    "distance": best.combined_score,
                                })
                                if pid not in performer_info or best.combined_score < performer_info[pid]["distance"]:
                                    performer_info[pid] = {
                                        "name": best.name,
                                        "distance": best.combined_score,
                                    }
                    except Exception:
                        pass  # Skip failed images

            # Apply gallery aggregation filter: 2+ appearances OR single < 0.4
            for pid, appearances in performer_appearances.items():
                image_count = len(set(a["image_id"] for a in appearances))
                best_dist = min(a["distance"] for a in appearances)
                if image_count >= 2 or best_dist < 0.4:
                    found_ids.add(pid)
                    details.append({
                        "stashdb_id": pid,
                        "name": performer_info.get(pid, {}).get("name", "Unknown"),
                        "image_count": image_count,
                        "best_distance": best_dist,
                    })

            tp = len(expected_ids & found_ids)
            fp = len(found_ids - expected_ids)
            fn = len(expected_ids - found_ids)

            gallery_results.append(GalleryBenchmarkResult(
                gallery_id=gal["id"],
                gallery_title=gal["title"][:40],
                total_images=len(images),
                expected_performers=[p["stashdb_id"] for p in gal["performers"]],
                found_performers=list(found_ids),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                max_distance=threshold,
                details=details,
            ))

            status = "OK" if fn == 0 and fp == 0 else f"FN={fn} FP={fp}"
            print(f"    Gallery {gal['id']:>6} ({len(images):>3} imgs, {len(expected_ids)} expected): "
                  f"TP={tp} FP={fp} FN={fn}  [{status}]")

        total_tp = sum(r.true_positives for r in gallery_results)
        total_fp = sum(r.false_positives for r in gallery_results)
        total_fn = sum(r.false_negatives for r in gallery_results)
        total_expected = total_tp + total_fn
        acc = total_tp / total_expected if total_expected > 0 else 0
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

        print(f"  Totals: TP={total_tp} FP={total_fp} FN={total_fn}  acc={acc:.1%}  prec={prec:.1%}")
        all_results[threshold] = {
            "galleries": [asdict(r) for r in gallery_results],
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
            "accuracy": acc,
            "precision": prec,
        }

    return all_results


# ============================================================================
# Phase 6: Image Benchmark
# ============================================================================

async def find_benchmark_images(stash_client: StashClient) -> list[dict]:
    """Find images with 1+ tagged performers who have StashDB IDs."""
    print("  Querying Stash for tagged images...")

    query = """
    query FindImages($page: Int!, $per_page: Int!) {
        findImages(
            filter: { page: $page, per_page: $per_page, sort: "random" }
        ) {
            count
            images {
                id
                title
                paths { image }
                performers {
                    id
                    name
                    stash_ids { endpoint stash_id }
                }
            }
        }
    }
    """

    candidates = []
    page = 1
    per_page = 25
    max_pages = 40

    while len(candidates) < 20 and page <= max_pages:
        data = stash_client._query(query, {"page": page, "per_page": per_page})
        images = data.get("findImages", {}).get("images", [])

        if not images:
            break

        for img in images:
            performers = img.get("performers", [])
            stashdb_performers = []
            for p in performers:
                sid = get_stashdb_id_for_performer(p)
                if sid:
                    stashdb_performers.append({"id": p["id"], "name": p["name"], "stashdb_id": sid})

            if len(stashdb_performers) >= 1 and img.get("paths", {}).get("image"):
                candidates.append({
                    "id": img["id"],
                    "title": img.get("title", f"Image {img['id']}"),
                    "image_url": img["paths"]["image"],
                    "performers": stashdb_performers,
                })

            if len(candidates) >= 20:
                break

        page += 1

    print(f"  Found {len(candidates)} qualifying images")
    return candidates


async def phase6_image_benchmark(
    recognizer: FaceRecognizer,
    stash_client: StashClient,
    stash_config,
) -> dict:
    """Phase 6: Benchmark single image identification."""
    print("\n" + "=" * 70)
    print("PHASE 6: IMAGE BENCHMARK")
    print("=" * 70)

    images = await find_benchmark_images(stash_client)
    if not images:
        print("  No qualifying images found, skipping")
        return {}

    stash_url = stash_config.url.rstrip("/")
    api_key = stash_config.api_key

    distance_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    all_results = {}

    for threshold in distance_thresholds:
        print(f"\n  --- max_distance={threshold} ---")
        image_results = []

        headers = {"ApiKey": api_key} if api_key else {}
        async with httpx.AsyncClient(timeout=30.0) as client:
            for img_data in images:
                expected_ids = {p["stashdb_id"] for p in img_data["performers"]}
                found_ids = set()
                details = []

                try:
                    resp = await client.get(img_data["image_url"], headers=headers)
                    resp.raise_for_status()
                    from embeddings import load_image
                    image = load_image(resp.content)
                    results = recognizer.recognize_image(
                        image, top_k=5, max_distance=threshold,
                    )

                    for result in results:
                        if result.matches:
                            best = result.matches[0]
                            found_ids.add(best.stashdb_id)
                            details.append({
                                "stashdb_id": best.stashdb_id,
                                "name": best.name,
                                "distance": best.combined_score,
                            })
                except Exception as e:
                    print(f"    Image {img_data['id']}: error - {e}")
                    continue

                tp_ids = expected_ids & found_ids
                correct = len(tp_ids) > 0

                image_results.append(ImageBenchmarkResult(
                    image_id=img_data["id"],
                    expected_performers=list(expected_ids),
                    found_performers=list(found_ids),
                    correct=correct,
                    max_distance=threshold,
                    details=details,
                ))

                expected_names = [p["name"] for p in img_data["performers"]]
                found_names = [d["name"] for d in details]
                status = "HIT" if correct else "MISS"
                print(f"    Image {img_data['id']:>6}: expected={expected_names}  "
                      f"found={found_names}  [{status}]")

        total = len(image_results)
        correct_count = sum(1 for r in image_results if r.correct)
        acc = correct_count / total if total > 0 else 0

        # Also compute TP/FP/FN across all images
        total_tp = 0
        total_fp = 0
        total_fn = 0
        for r in image_results:
            expected = set(r.expected_performers)
            found = set(r.found_performers)
            total_tp += len(expected & found)
            total_fp += len(found - expected)
            total_fn += len(expected - found)

        total_expected = total_tp + total_fn
        performer_acc = total_tp / total_expected if total_expected > 0 else 0
        prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0

        print(f"  Image accuracy (any hit): {acc:.1%} ({correct_count}/{total})")
        print(f"  Performer accuracy: {performer_acc:.1%}  precision: {prec:.1%}  "
              f"TP={total_tp} FP={total_fp} FN={total_fn}")

        all_results[threshold] = {
            "images": [asdict(r) for r in image_results],
            "image_accuracy": acc,
            "performer_accuracy": performer_acc,
            "precision": prec,
            "total_tp": total_tp,
            "total_fp": total_fp,
            "total_fn": total_fn,
        }

    return all_results


# ============================================================================
# Report Generation
# ============================================================================

def generate_report(
    db_stats: dict,
    num_scenes: int,
    baseline_metrics: dict,
    sweeps: dict[str, list[SweepResult]],
    best_combination_metrics: dict,
    best_params: BenchmarkParams,
    phase4_results: dict,
    gallery_results: dict,
    image_results: dict,
    output_path: Path,
):
    """Generate markdown report."""
    lines = [
        "# Face Recognition Re-evaluation Report",
        "",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Database:** {db_stats.get('total_faces', '?')} faces, "
        f"{db_stats.get('performer_count', '?')} performers",
        f"**Test Scenes:** {num_scenes}",
        "",
        "## Phase 1: Baseline",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Accuracy | {baseline_metrics['accuracy']:.1%} |",
        f"| Precision | {baseline_metrics['precision']:.1%} |",
        f"| TP / Expected | {baseline_metrics['true_positives']} / {baseline_metrics['total_expected']} |",
        f"| False Positives | {baseline_metrics['false_positives']} |",
        "",
    ]

    if baseline_metrics.get("accuracy_by_resolution"):
        lines.append("### By Resolution")
        lines.append("")
        lines.append("| Resolution | Accuracy |")
        lines.append("|------------|----------|")
        for res, acc in sorted(baseline_metrics["accuracy_by_resolution"].items()):
            lines.append(f"| {res} | {acc:.1%} |")
        lines.append("")

    # Phase 2: Sweeps
    lines.append("## Phase 2: Parameter Sweeps")
    lines.append("")

    for sweep_name, results in sweeps.items():
        lines.append(f"### {sweep_name}")
        lines.append("")
        lines.append("| Config | Accuracy | Precision | TP | FP | FN |")
        lines.append("|--------|----------|-----------|----|----|-----|")
        for r in results:
            m = r.metrics
            lines.append(f"| {r.name} | {m['accuracy']:.1%} | {m['precision']:.1%} | "
                        f"{m['true_positives']} | {m['false_positives']} | {m['false_negatives']} |")
        best = max(results, key=lambda r: (r.metrics["accuracy"], r.metrics["precision"]))
        lines.append(f"\n**Best:** {best.name} (acc={best.metrics['accuracy']:.1%})")
        lines.append("")

    # Phase 3: Best combination
    lines.append("## Phase 3: Best-of Combination")
    lines.append("")
    lines.append("| Metric | Baseline | Best Combo | Delta |")
    lines.append("|--------|----------|------------|-------|")
    for metric in ["accuracy", "precision"]:
        b = baseline_metrics[metric]
        c = best_combination_metrics[metric]
        lines.append(f"| {metric.title()} | {b:.1%} | {c:.1%} | {c-b:+.1%} |")
    lines.append("")
    lines.append("**Best parameters:**")
    lines.append("```json")
    lines.append(json.dumps(best_params.to_dict(), indent=2))
    lines.append("```")
    lines.append("")

    # Phase 4: Original scenes
    if phase4_results:
        lines.append("## Phase 4: Original Scene Re-validation")
        lines.append("")
        dm = phase4_results.get("default_metrics", {})
        bm = phase4_results.get("best_metrics", {})
        if dm and bm:
            lines.append("| Config | Accuracy | Precision | TP | FP | FN |")
            lines.append("|--------|----------|-----------|----|----|-----|")
            lines.append(f"| Default params | {dm.get('accuracy',0):.1%} | {dm.get('precision',0):.1%} | "
                        f"{dm.get('true_positives',0)} | {dm.get('false_positives',0)} | {dm.get('false_negatives',0)} |")
            lines.append(f"| Best params | {bm.get('accuracy',0):.1%} | {bm.get('precision',0):.1%} | "
                        f"{bm.get('true_positives',0)} | {bm.get('false_positives',0)} | {bm.get('false_negatives',0)} |")
        lines.append("")

    # Phase 5: Gallery
    if gallery_results:
        lines.append("## Phase 5: Gallery Benchmark")
        lines.append("")
        lines.append("| max_distance | Accuracy | Precision | TP | FP | FN |")
        lines.append("|-------------|----------|-----------|----|----|-----|")
        for threshold, data in sorted(gallery_results.items()):
            lines.append(f"| {threshold} | {data['accuracy']:.1%} | {data['precision']:.1%} | "
                        f"{data['total_tp']} | {data['total_fp']} | {data['total_fn']} |")
        lines.append("")

    # Phase 6: Image
    if image_results:
        lines.append("## Phase 6: Image Benchmark")
        lines.append("")
        lines.append("| max_distance | Image Accuracy | Performer Accuracy | Precision | TP | FP | FN |")
        lines.append("|-------------|---------------|-------------------|-----------|----|----|-----|")
        for threshold, data in sorted(image_results.items()):
            lines.append(f"| {threshold} | {data['image_accuracy']:.1%} | {data['performer_accuracy']:.1%} | "
                        f"{data['precision']:.1%} | {data['total_tp']} | {data['total_fp']} | {data['total_fn']} |")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")
    lines.append("Based on the sweep results, the recommended default parameters are:")
    lines.append("")
    lines.append("```python")
    d = best_params.to_dict()
    for k, v in d.items():
        lines.append(f"    {k} = {v!r}")
    lines.append("```")
    lines.append("")

    report = "\n".join(lines)
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")
    return report


# ============================================================================
# Main
# ============================================================================

async def run_reeval():
    """Run the full re-evaluation benchmark."""
    print("=" * 70)
    print("FACE RECOGNITION RE-EVALUATION BENCHMARK")
    print(f"Date: {datetime.now().isoformat()}")
    print("=" * 70)

    # Initialize
    recognizer, matcher, scene_selector, analyzer, db_reader, stash_client, stash_config = initialize_components()
    executor = TestExecutor(recognizer, matcher)

    # DB stats
    db_stats = db_reader.get_stats()
    print(f"\nDatabase: {db_stats['performer_count']} performers, "
          f"{db_stats.get('performers_with_faces', '?')} with faces, "
          f"{db_stats['total_faces']} faces")

    # Select test scenes
    print("\nSelecting 20 stratified test scenes...")
    all_scenes = await scene_selector.select_scenes(min_count=20)
    print(f"Selected {len(all_scenes)} scenes")

    stratified = scene_selector.stratify_scenes(all_scenes)
    for tier, scenes in stratified.items():
        if scenes:
            print(f"  {tier}: {len(scenes)} scenes")

    # Output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path("benchmark_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()

    # Phase 1: Baseline
    baseline_metrics, baseline_results = await phase1_baseline(all_scenes, executor, analyzer)

    # Phase 2: Parameter sweeps
    sweeps = await phase2_sweeps(all_scenes, executor, analyzer)

    # Phase 3: Best combination
    best_combo_metrics, best_params = await phase3_best_combination(
        all_scenes, executor, analyzer, sweeps, baseline_metrics,
    )

    # Phase 4: Original scene re-validation
    phase4_results = await phase4_original_scenes(
        all_scenes, executor, analyzer, scene_selector, best_params,
    )

    # Phase 5: Gallery benchmark
    gallery_results = await phase5_gallery_benchmark(recognizer, stash_client, stash_config)

    # Phase 6: Image benchmark
    image_results = await phase6_image_benchmark(recognizer, stash_client, stash_config)

    total_elapsed = time.time() - total_start
    print(f"\n{'=' * 70}")
    print(f"TOTAL TIME: {total_elapsed / 60:.1f} minutes")
    print(f"{'=' * 70}")

    # Save JSON results
    json_path = output_dir / f"reeval_{timestamp}.json"
    json_data = {
        "timestamp": datetime.now().isoformat(),
        "db_stats": db_stats,
        "num_scenes": len(all_scenes),
        "total_elapsed_sec": total_elapsed,
        "baseline": baseline_metrics,
        "sweeps": {
            name: [{"name": r.name, "param_name": r.param_name,
                     "param_value": str(r.param_value), "params": r.params,
                     "metrics": r.metrics}
                    for r in results]
            for name, results in sweeps.items()
        },
        "best_combination": {
            "params": best_params.to_dict(),
            "metrics": best_combo_metrics,
        },
        "phase4_original_scenes": phase4_results,
        "phase5_galleries": gallery_results,
        "phase6_images": image_results,
    }
    json_path.write_text(json.dumps(json_data, indent=2, default=str))
    print(f"JSON results saved to: {json_path}")

    # Generate markdown report
    report_path = output_dir / "reeval_report.md"
    generate_report(
        db_stats=db_stats,
        num_scenes=len(all_scenes),
        baseline_metrics=baseline_metrics,
        sweeps=sweeps,
        best_combination_metrics=best_combo_metrics,
        best_params=best_params,
        phase4_results=phase4_results,
        gallery_results=gallery_results,
        image_results=image_results,
        output_path=report_path,
    )

    return json_data


if __name__ == "__main__":
    asyncio.run(run_reeval())

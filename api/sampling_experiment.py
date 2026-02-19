"""Frame sampling strategy A/B experiment.

Compares 7 frame sampling strategies to determine if more frames,
smarter distribution, or oversample-and-filter produces better
identification results, and at what compute cost.

Usage:
    cd /home/carrot/code/stash-sense/api

    # Auto-select 6 diverse scenes
    python sampling_experiment.py

    # Specify scenes manually
    python sampling_experiment.py --scenes "13938,30835,16342,26367,12345,67890"

    # Test subset of strategies
    python sampling_experiment.py --strategies "uniform-40,weighted-40,oversample-to-40"
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SamplingStrategy:
    """Definition of a sampling strategy to test."""

    name: str
    extract_frames: int  # Number of frames to extract from video
    recognize_frames: int  # Number of frames to run recognition on
    distribution: str  # "uniform" or "weighted"
    is_oversample: bool = False  # If True, extract many, filter to recognize_frames


@dataclass
class PhaseTimings:
    """Timing breakdown for each phase of identification."""

    extraction_sec: float = 0.0
    detection_sec: float = 0.0
    recognition_sec: float = 0.0
    matching_sec: float = 0.0

    @property
    def total_sec(self) -> float:
        return (
            self.extraction_sec
            + self.detection_sec
            + self.recognition_sec
            + self.matching_sec
        )


@dataclass
class StrategySceneResult:
    """Result of running one strategy on one scene."""

    strategy_name: str
    scene_id: str
    scene_duration_sec: float
    frames_extracted: int
    frames_recognized: int
    faces_detected: int
    faces_recognized: int
    true_positives: int
    false_negatives: int
    false_positives: int
    expected_count: int
    timings: PhaseTimings
    found_ids: list[str] = field(default_factory=list)
    expected_ids: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        total = self.true_positives + self.false_negatives
        return self.true_positives / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        total = self.true_positives + self.false_positives
        return self.true_positives / total if total > 0 else 0.0


@dataclass
class StrategyAggregate:
    """Aggregated results for one strategy across all scenes."""

    strategy_name: str
    extract_frames: int
    recognize_frames: int
    scenes_tested: int
    total_tp: int
    total_fp: int
    total_fn: int
    avg_faces_detected: float
    avg_extraction_sec: float
    avg_detection_sec: float
    avg_recognition_sec: float
    avg_matching_sec: float
    avg_total_sec: float
    scene_results: list[StrategySceneResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        total = self.total_tp + self.total_fn
        return self.total_tp / total if total > 0 else 0.0

    @property
    def precision(self) -> float:
        total = self.total_tp + self.total_fp
        return self.total_tp / total if total > 0 else 0.0


# =============================================================================
# STRATEGIES
# =============================================================================

STRATEGIES = [
    SamplingStrategy("uniform-40", 40, 40, "uniform"),
    SamplingStrategy("uniform-80", 80, 80, "uniform"),
    SamplingStrategy("uniform-120", 120, 120, "uniform"),
    SamplingStrategy("weighted-40", 40, 40, "weighted"),
    SamplingStrategy("weighted-80", 80, 80, "weighted"),
    SamplingStrategy("oversample-to-40", 120, 40, "uniform", is_oversample=True),
    SamplingStrategy("oversample-to-60", 120, 60, "uniform", is_oversample=True),
]


# =============================================================================
# CORE FUNCTIONS
# =============================================================================


async def extract_frames_with_timestamps(
    stream_url: str,
    timestamps: list[float],
    api_key: Optional[str],
) -> list:
    """Extract frames at specific timestamps using the same pattern as extract_frames().

    Args:
        stream_url: HTTP URL to video stream
        timestamps: List of timestamps in seconds
        api_key: Stash API key

    Returns:
        List of ExtractedFrame objects (successful extractions only)
    """
    from frame_extractor import (
        FrameExtractionConfig,
        extract_frame_async,
    )

    config = FrameExtractionConfig()
    semaphore = asyncio.Semaphore(config.max_concurrent_extractions)

    async def extract_with_semaphore(ts, idx):
        async with semaphore:
            return await extract_frame_async(stream_url, ts, idx, api_key, config)

    tasks = [extract_with_semaphore(ts, idx) for idx, ts in enumerate(timestamps)]
    results = await asyncio.gather(*tasks)
    return [f for f in results if f is not None]


async def run_strategy_on_scene(
    strategy: SamplingStrategy,
    scene,  # TestScene
    recognizer,  # FaceRecognizer
    stash_url: str,
    api_key: str,
) -> StrategySceneResult:
    """Run a single strategy on a single scene.

    Phases:
    1. Compute timestamps & extract frames (ffmpeg)
    2. Detect faces (RetinaFace)
    3. If oversample: score & filter frames; then recognize (FaceNet512 + ArcFace)
    4. Match (frequency_based_matching)

    Returns:
        StrategySceneResult with metrics and timings
    """
    from frame_extractor import (
        build_stash_stream_url,
        calculate_extraction_timestamps,
        calculate_weighted_timestamps,
        score_frame_by_faces,
        FrameExtractionConfig,
    )
    from matching import MatchingConfig
    from recognizer import RecognitionResult
    from main import frequency_based_matching

    timings = PhaseTimings()
    stream_url = build_stash_stream_url(stash_url, scene.scene_id)
    duration_sec = scene.duration_sec

    # --- Phase 1: Compute timestamps & extract frames ---
    t0 = time.time()

    if strategy.distribution == "weighted":
        timestamps = calculate_weighted_timestamps(
            duration_sec, num_frames=strategy.extract_frames
        )
    else:
        config = FrameExtractionConfig(num_frames=strategy.extract_frames)
        timestamps = calculate_extraction_timestamps(duration_sec, config)

    frames = await extract_frames_with_timestamps(stream_url, timestamps, api_key)
    timings.extraction_sec = time.time() - t0

    # --- Phase 2: Detect faces ---
    t1 = time.time()

    # For oversample strategies, detect on ALL frames first, then filter
    frame_faces: list[tuple] = []  # (frame, faces_list)
    total_faces = 0

    for frame in frames:
        faces = recognizer.generator.detect_faces(frame.image, min_confidence=0.8)
        frame_faces.append((frame, faces))
        total_faces += len(faces)

    # If oversample: score and keep top N frames by face quality
    if strategy.is_oversample and len(frame_faces) > strategy.recognize_frames:
        scored = [
            (score_frame_by_faces(faces), frame, faces)
            for frame, faces in frame_faces
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        frame_faces = [(frame, faces) for _, frame, faces in scored[: strategy.recognize_frames]]

    timings.detection_sec = time.time() - t1

    # --- Phase 3: Recognize faces ---
    t2 = time.time()

    match_config = MatchingConfig(
        query_k=100,
        facenet_weight=0.6,
        arcface_weight=0.4,
        max_results=10,
        max_distance=0.7,
    )

    all_results: list[tuple[int, RecognitionResult]] = []
    faces_recognized = 0

    for frame, faces in frame_faces:
        for face in faces:
            # Apply minimum face size filter
            if face.bbox["w"] < 40 or face.bbox["h"] < 40:
                continue

            matches, _, emb = recognizer.recognize_face_v2(face, match_config)
            result = RecognitionResult(face=face, matches=matches, embedding=emb)
            all_results.append((frame.frame_index, result))
            faces_recognized += 1

    timings.recognition_sec = time.time() - t2

    # --- Phase 4: Matching ---
    t3 = time.time()

    persons = frequency_based_matching(
        all_results,
        top_k=5,
        min_appearances=2,
        min_unique_frames=2,
        max_distance=0.7,
        min_confidence=0.35,
    )

    timings.matching_sec = time.time() - t3

    # --- Compare to ground truth ---
    expected_ids = {p.stashdb_id for p in scene.expected_performers}
    found_ids = set()
    for person in persons:
        stashdb_id = person.best_match.stashdb_id
        if ":" in stashdb_id:
            stashdb_id = stashdb_id.split(":")[-1]
        found_ids.add(stashdb_id)

    tp = len(expected_ids & found_ids)
    fn = len(expected_ids - found_ids)
    fp = len(found_ids - expected_ids)

    return StrategySceneResult(
        strategy_name=strategy.name,
        scene_id=scene.scene_id,
        scene_duration_sec=duration_sec,
        frames_extracted=len(frames),
        frames_recognized=len(frame_faces),
        faces_detected=total_faces,
        faces_recognized=faces_recognized,
        true_positives=tp,
        false_negatives=fn,
        false_positives=fp,
        expected_count=len(expected_ids),
        timings=timings,
        found_ids=sorted(found_ids),
        expected_ids=sorted(expected_ids),
    )


async def select_diverse_scenes(
    stash_client,
    database_reader,
    count: int = 6,
) -> list:
    """Auto-select scenes with duration diversity.

    Selects:
    - 2 short scenes (< 15 min)
    - 2 medium scenes (15-45 min)
    - 2 long scenes (> 45 min)

    All 720p+ resolution, all with 2+ performers with StashDB IDs.

    Args:
        stash_client: StashClient instance
        database_reader: PerformerDatabaseReader instance
        count: Total scenes to select (default 6)

    Returns:
        List of TestScene objects
    """
    from benchmark.scene_selector import SceneSelector

    selector = SceneSelector(stash_client, database_reader)

    # Fetch a pool of candidate scenes
    print("  Fetching candidate scenes from Stash...")
    all_scenes = await selector.select_scenes(min_count=200)

    # Filter for 720p+
    hd_scenes = [
        s for s in all_scenes if max(s.width, s.height) >= 720
    ]
    print(f"  Found {len(hd_scenes)} HD scenes (720p+) out of {len(all_scenes)} total")

    # Bucket by duration
    short = [s for s in hd_scenes if s.duration_sec < 900]  # < 15 min
    medium = [s for s in hd_scenes if 900 <= s.duration_sec <= 2700]  # 15-45 min
    long = [s for s in hd_scenes if s.duration_sec > 2700]  # > 45 min

    print(f"  Duration buckets: {len(short)} short, {len(medium)} medium, {len(long)} long")

    import random

    random.seed(42)  # Reproducible
    per_bucket = count // 3

    selected = []
    for bucket_name, bucket in [("short", short), ("medium", medium), ("long", long)]:
        sample_size = min(per_bucket, len(bucket))
        if sample_size > 0:
            sampled = random.sample(bucket, sample_size)
            selected.extend(sampled)
            for s in sampled:
                dur_min = s.duration_sec / 60
                print(
                    f"    {bucket_name}: scene {s.scene_id} "
                    f"({dur_min:.1f}min, {s.width}x{s.height}, "
                    f"{len(s.expected_performers)} performers)"
                )
        else:
            print(f"    WARNING: No {bucket_name} scenes available")

    return selected


async def fetch_scene_data(
    scene_ids: list[str],
    stash_client,
    database_reader,
) -> list:
    """Fetch TestScene objects for specific scene IDs.

    Args:
        scene_ids: List of Stash scene ID strings
        stash_client: StashClient instance
        database_reader: PerformerDatabaseReader instance

    Returns:
        List of TestScene objects
    """
    from benchmark.scene_selector import SceneSelector

    selector = SceneSelector(stash_client, database_reader)

    scenes = []
    for scene_id in scene_ids:
        query = f"""
        query {{
            findScene(id: "{scene_id}") {{
                id
                title
                files {{
                    width
                    height
                    duration
                }}
                stash_ids {{
                    endpoint
                    stash_id
                }}
                performers {{
                    name
                    stash_ids {{
                        endpoint
                        stash_id
                    }}
                }}
            }}
        }}
        """
        result = stash_client._query(query)
        scene_data = result.get("findScene")
        if not scene_data:
            print(f"  WARNING: Scene {scene_id} not found, skipping")
            continue

        if not scene_data.get("files"):
            print(f"  WARNING: Scene {scene_id} has no files, skipping")
            continue

        # Build a format compatible with _convert_to_test_scene
        # Add stash_ids field if not present
        if selector._is_valid_test_scene(scene_data):
            test_scene = await selector._convert_to_test_scene(scene_data)
            dur_min = test_scene.duration_sec / 60
            print(
                f"  Scene {scene_id}: {dur_min:.1f}min, "
                f"{test_scene.width}x{test_scene.height}, "
                f"{len(test_scene.expected_performers)} performers"
            )
            scenes.append(test_scene)
        else:
            print(
                f"  WARNING: Scene {scene_id} doesn't meet criteria "
                f"(needs 720p+, 2+ performers with StashDB IDs), skipping"
            )

    return scenes


def aggregate_strategy_results(
    results: list[StrategySceneResult],
) -> StrategyAggregate:
    """Aggregate results for one strategy across all scenes.

    Args:
        results: List of per-scene results for one strategy

    Returns:
        StrategyAggregate with totals and averages
    """
    if not results:
        return StrategyAggregate(
            strategy_name="unknown",
            extract_frames=0,
            recognize_frames=0,
            scenes_tested=0,
            total_tp=0,
            total_fp=0,
            total_fn=0,
            avg_faces_detected=0,
            avg_extraction_sec=0,
            avg_detection_sec=0,
            avg_recognition_sec=0,
            avg_matching_sec=0,
            avg_total_sec=0,
        )

    n = len(results)
    return StrategyAggregate(
        strategy_name=results[0].strategy_name,
        extract_frames=results[0].frames_extracted,
        recognize_frames=results[0].frames_recognized,
        scenes_tested=n,
        total_tp=sum(r.true_positives for r in results),
        total_fp=sum(r.false_positives for r in results),
        total_fn=sum(r.false_negatives for r in results),
        avg_faces_detected=sum(r.faces_detected for r in results) / n,
        avg_extraction_sec=sum(r.timings.extraction_sec for r in results) / n,
        avg_detection_sec=sum(r.timings.detection_sec for r in results) / n,
        avg_recognition_sec=sum(r.timings.recognition_sec for r in results) / n,
        avg_matching_sec=sum(r.timings.matching_sec for r in results) / n,
        avg_total_sec=sum(r.timings.total_sec for r in results) / n,
        scene_results=results,
    )


# =============================================================================
# EXPERIMENT RUNNER
# =============================================================================


async def run_experiment(
    scenes: list,
    strategies: list[SamplingStrategy],
    recognizer,
    stash_url: str,
    api_key: str,
) -> list[StrategyAggregate]:
    """Run all strategies on all scenes.

    Args:
        scenes: List of TestScene objects
        strategies: List of SamplingStrategy objects to test
        recognizer: FaceRecognizer instance
        stash_url: Stash base URL
        api_key: Stash API key

    Returns:
        List of StrategyAggregate results, one per strategy
    """
    aggregates = []

    for si, strategy in enumerate(strategies):
        print(f"\n{'='*60}")
        print(f"Strategy {si+1}/{len(strategies)}: {strategy.name}")
        print(
            f"  Extract: {strategy.extract_frames}, "
            f"Recognize: {strategy.recognize_frames}, "
            f"Distribution: {strategy.distribution}"
            f"{', oversample+filter' if strategy.is_oversample else ''}"
        )
        print(f"{'='*60}")

        scene_results = []
        for sci, scene in enumerate(scenes):
            dur_min = scene.duration_sec / 60
            print(
                f"  [{sci+1}/{len(scenes)}] Scene {scene.scene_id} "
                f"({dur_min:.1f}min, {len(scene.expected_performers)} performers)...",
                end="",
                flush=True,
            )

            try:
                result = await run_strategy_on_scene(
                    strategy, scene, recognizer, stash_url, api_key
                )
                scene_results.append(result)

                tp = result.true_positives
                expected = result.expected_count
                print(
                    f" {tp}/{expected} "
                    f"(+{result.false_positives}FP) "
                    f"[{result.timings.total_sec:.1f}s]"
                )
            except Exception as e:
                print(f" ERROR: {type(e).__name__}: {str(e)[:80]}")

        agg = aggregate_strategy_results(scene_results)
        aggregates.append(agg)

        print(
            f"  => {strategy.name}: "
            f"{agg.accuracy:.1%} accuracy, "
            f"{agg.precision:.1%} precision, "
            f"TP={agg.total_tp} FP={agg.total_fp} FN={agg.total_fn}, "
            f"avg {agg.avg_total_sec:.1f}s/scene"
        )

    return aggregates


# =============================================================================
# OUTPUT
# =============================================================================


def print_results_table(aggregates: list[StrategyAggregate]) -> None:
    """Print formatted comparison table."""
    print(f"\n{'='*130}")
    print("RESULTS SUMMARY")
    print(f"{'='*130}")

    header = (
        f"{'Strategy':<20} | {'Frames':>9} | {'Accuracy':>8} | {'Precision':>9} | "
        f"{'TP':>3} | {'FP':>3} | {'FN':>3} | {'Avg Faces':>9} | "
        f"{'Extract':>7} | {'Detect':>7} | {'Recog':>7} | {'Total':>7}"
    )
    print(header)
    print("-" * 130)

    for agg in aggregates:
        frames_str = f"{agg.extract_frames}/{agg.recognize_frames}"
        print(
            f"{agg.strategy_name:<20} | {frames_str:>9} | "
            f"{agg.accuracy:>7.1%} | {agg.precision:>8.1%} | "
            f"{agg.total_tp:>3} | {agg.total_fp:>3} | {agg.total_fn:>3} | "
            f"{agg.avg_faces_detected:>9.1f} | "
            f"{agg.avg_extraction_sec:>6.1f}s | "
            f"{agg.avg_detection_sec:>6.1f}s | "
            f"{agg.avg_recognition_sec:>6.1f}s | "
            f"{agg.avg_total_sec:>6.1f}s"
        )

    # Key findings
    print(f"\n{'='*60}")
    print("KEY FINDINGS")
    print(f"{'='*60}")

    if aggregates:
        best_accuracy = max(aggregates, key=lambda a: a.accuracy)
        best_precision = max(aggregates, key=lambda a: a.precision)
        best_efficiency = max(
            aggregates,
            key=lambda a: a.accuracy / a.avg_total_sec if a.avg_total_sec > 0 else 0,
        )

        print(f"- Best accuracy:    {best_accuracy.strategy_name} ({best_accuracy.accuracy:.1%})")
        print(f"- Best precision:   {best_precision.strategy_name} ({best_precision.precision:.1%})")
        print(
            f"- Best efficiency:  {best_efficiency.strategy_name} "
            f"({best_efficiency.accuracy:.1%} in {best_efficiency.avg_total_sec:.1f}s avg)"
        )

        # Compare uniform-40 baseline to others
        baseline = next((a for a in aggregates if a.strategy_name == "uniform-40"), None)
        if baseline:
            print(f"\nBaseline (uniform-40): {baseline.accuracy:.1%} accuracy, {baseline.avg_total_sec:.1f}s avg")
            for agg in aggregates:
                if agg.strategy_name == "uniform-40":
                    continue
                acc_diff = agg.accuracy - baseline.accuracy
                time_diff = agg.avg_total_sec - baseline.avg_total_sec
                print(
                    f"  vs {agg.strategy_name}: "
                    f"{acc_diff:+.1%} accuracy, "
                    f"{time_diff:+.1f}s time"
                )


def save_results(
    aggregates: list[StrategyAggregate],
    scenes: list,
    output_dir: str = "benchmark_results",
) -> str:
    """Save results to JSON file.

    Args:
        aggregates: List of StrategyAggregate results
        scenes: List of TestScene objects used
        output_dir: Directory to save results

    Returns:
        Path to saved JSON file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"sampling_experiment_{timestamp}.json"
    filepath = Path(output_dir) / filename

    data = {
        "timestamp": datetime.now().isoformat(),
        "scenes": [
            {
                "scene_id": s.scene_id,
                "title": s.title,
                "duration_sec": s.duration_sec,
                "resolution": s.resolution,
                "width": s.width,
                "height": s.height,
                "expected_performers": [
                    {"stashdb_id": p.stashdb_id, "name": p.name}
                    for p in s.expected_performers
                ],
            }
            for s in scenes
        ],
        "strategies": [],
    }

    for agg in aggregates:
        strategy_data = {
            "name": agg.strategy_name,
            "extract_frames": agg.extract_frames,
            "recognize_frames": agg.recognize_frames,
            "scenes_tested": agg.scenes_tested,
            "total_tp": agg.total_tp,
            "total_fp": agg.total_fp,
            "total_fn": agg.total_fn,
            "accuracy": agg.accuracy,
            "precision": agg.precision,
            "avg_faces_detected": agg.avg_faces_detected,
            "avg_extraction_sec": agg.avg_extraction_sec,
            "avg_detection_sec": agg.avg_detection_sec,
            "avg_recognition_sec": agg.avg_recognition_sec,
            "avg_matching_sec": agg.avg_matching_sec,
            "avg_total_sec": agg.avg_total_sec,
            "scene_results": [
                {
                    "scene_id": r.scene_id,
                    "scene_duration_sec": r.scene_duration_sec,
                    "frames_extracted": r.frames_extracted,
                    "frames_recognized": r.frames_recognized,
                    "faces_detected": r.faces_detected,
                    "faces_recognized": r.faces_recognized,
                    "true_positives": r.true_positives,
                    "false_negatives": r.false_negatives,
                    "false_positives": r.false_positives,
                    "expected_count": r.expected_count,
                    "accuracy": r.accuracy,
                    "precision": r.precision,
                    "found_ids": r.found_ids,
                    "expected_ids": r.expected_ids,
                    "timings": {
                        "extraction_sec": r.timings.extraction_sec,
                        "detection_sec": r.timings.detection_sec,
                        "recognition_sec": r.timings.recognition_sec,
                        "matching_sec": r.timings.matching_sec,
                        "total_sec": r.timings.total_sec,
                    },
                }
                for r in agg.scene_results
            ],
        }
        data["strategies"].append(strategy_data)

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return str(filepath)


# =============================================================================
# MAIN
# =============================================================================


async def main():
    parser = argparse.ArgumentParser(description="Frame sampling strategy A/B experiment")
    parser.add_argument(
        "--scenes",
        type=str,
        default=None,
        help="Comma-separated scene IDs (default: auto-select 6 diverse scenes)",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=None,
        help="Comma-separated strategy names to test (default: all 7)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Directory for JSON output (default: benchmark_results)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("FRAME SAMPLING EXPERIMENT")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")

    # Initialize components (same pattern as comprehensive_test.py)
    from config import StashConfig, DatabaseConfig
    from stash_client import StashClient
    from database_reader import PerformerDatabaseReader
    from recognizer import FaceRecognizer

    stash_config = StashConfig.from_env()
    data_dir = os.environ.get("DATA_DIR", "./data")
    db_config = DatabaseConfig(data_dir=Path(data_dir))

    print("\nInitializing components...")
    stash_client = StashClient(stash_config.url, stash_config.api_key)
    database_reader = PerformerDatabaseReader(str(db_config.sqlite_db_path))

    print("  Loading face recognition models...")
    recognizer = FaceRecognizer(db_config)

    # Select strategies
    if args.strategies:
        strategy_names = [s.strip() for s in args.strategies.split(",")]
        strategies = [s for s in STRATEGIES if s.name in strategy_names]
        unknown = set(strategy_names) - {s.name for s in strategies}
        if unknown:
            print(f"  WARNING: Unknown strategies ignored: {unknown}")
            print(f"  Available: {[s.name for s in STRATEGIES]}")
    else:
        strategies = STRATEGIES

    print(f"\nStrategies: {len(strategies)}")
    for s in strategies:
        print(f"  {s.name}: extract={s.extract_frames}, recognize={s.recognize_frames}, dist={s.distribution}")

    # Select scenes
    print("\nSelecting scenes...")
    if args.scenes:
        scene_ids = [s.strip() for s in args.scenes.split(",")]
        scenes = await fetch_scene_data(scene_ids, stash_client, database_reader)
    else:
        scenes = await select_diverse_scenes(stash_client, database_reader)

    if not scenes:
        print("ERROR: No valid scenes found. Exiting.")
        return

    # Count duration buckets
    short = sum(1 for s in scenes if s.duration_sec < 900)
    medium = sum(1 for s in scenes if 900 <= s.duration_sec <= 2700)
    long = sum(1 for s in scenes if s.duration_sec > 2700)
    print(f"\nScenes: {len(scenes)} ({short} short, {medium} medium, {long} long)")

    # Run experiment
    print(f"\n{'='*60}")
    print(f"Running {len(strategies)} strategies x {len(scenes)} scenes")
    print(f"{'='*60}")

    experiment_start = time.time()
    aggregates = await run_experiment(
        scenes, strategies, recognizer, stash_config.url, stash_config.api_key
    )
    experiment_elapsed = time.time() - experiment_start

    # Print results
    print_results_table(aggregates)

    print(f"\nTotal experiment time: {experiment_elapsed:.1f}s ({experiment_elapsed/60:.1f}min)")

    # Save results
    filepath = save_results(aggregates, scenes, args.output_dir)
    print(f"Results saved to: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())

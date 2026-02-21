"""Tattoo detection baseline benchmark.

Captures tattoo detection results from the live sidecar for later comparison
after PyTorch-to-ONNX model conversion. Calls the sidecar's /identify/scene
endpoint with use_tattoo=true and records per-scene tattoo-specific outputs.

Modes:
    Capture baseline (default):
        python -m benchmark.tattoo_benchmark capture --scenes 100,200,300

    Compare two baselines:
        python -m benchmark.tattoo_benchmark compare baseline_a.json baseline_b.json

Usage:
    cd api
    source ../.venv/bin/activate
    python -m benchmark.tattoo_benchmark capture --scenes 100,200,300 -o baseline_pytorch.json
    python -m benchmark.tattoo_benchmark compare baseline_pytorch.json baseline_onnx.json
"""

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SIDECAR_URL = "http://10.0.0.4:6960"
DEFAULT_STASH_URL = "http://10.0.0.4:6969"
DEFAULT_NUM_FRAMES = 30
DEFAULT_OUTPUT_DIR = Path("benchmark_results")
REQUEST_TIMEOUT = 300.0  # 5 min per scene (tattoo + face pipeline can be slow)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class PersonRecord:
    """Per-person data extracted from one scene result."""

    person_id: int
    frame_count: int
    signals_used: list[str]
    tattoos_detected: int

    # Best match info
    best_match_stashdb_id: Optional[str]
    best_match_name: Optional[str]
    best_match_distance: Optional[float]
    best_match_confidence: Optional[float]

    # All match IDs (for cross-run alignment)
    all_match_ids: list[str]


@dataclass
class SceneRecord:
    """Full benchmark record for one scene."""

    scene_id: str
    success: bool
    error: Optional[str]

    # High-level counts
    frames_analyzed: int
    frames_requested: int
    faces_detected: int
    faces_after_filter: int
    multi_signal_used: bool

    # Person-level detail
    persons: list[PersonRecord]

    # Timing
    timing: Optional[dict]

    # Wall-clock time for the HTTP call
    wall_time_sec: float


@dataclass
class BenchmarkRun:
    """Top-level container for a complete benchmark run."""

    timestamp: str
    sidecar_url: str
    stash_url: str
    num_frames: int
    scene_ids: list[str]
    scenes: list[SceneRecord]

    # Aggregate stats
    total_scenes: int
    successful_scenes: int
    total_persons: int
    persons_with_tattoos: int
    total_tattoo_detections: int
    total_wall_time_sec: float


# ---------------------------------------------------------------------------
# API key loading
# ---------------------------------------------------------------------------


def _load_api_key() -> str:
    """Load STASH_API_KEY from environment or api/.env file."""
    key = os.environ.get("STASH_API_KEY", "")
    if key:
        return key

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("STASH_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")

    return ""


# ---------------------------------------------------------------------------
# Scene discovery (if no explicit IDs given)
# ---------------------------------------------------------------------------


async def discover_scene_ids(
    stash_url: str,
    api_key: str,
    count: int = 20,
) -> list[str]:
    """Query Stash for random scenes with 2+ tagged performers.

    This mirrors the existing benchmark's scene selection criteria:
    scenes must have at least 2 performers with StashDB IDs.
    """
    query = """
    query FindScenes($page: Int!, $per_page: Int!) {
        findScenes(
            filter: { page: $page, per_page: $per_page, sort: "random" }
            scene_filter: { performer_count: { modifier: GREATER_THAN, value: 1 } }
        ) {
            scenes {
                id
                performers {
                    stash_ids { endpoint stash_id }
                }
            }
        }
    }
    """

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["ApiKey"] = api_key

    scene_ids: list[str] = []
    page = 1
    max_pages = 20

    async with httpx.AsyncClient(timeout=30.0) as client:
        while len(scene_ids) < count and page <= max_pages:
            payload = {
                "query": query,
                "variables": {"page": page, "per_page": 25},
            }
            resp = await client.post(
                f"{stash_url}/graphql", json=payload, headers=headers
            )
            resp.raise_for_status()
            data = resp.json().get("data", {})
            scenes = data.get("findScenes", {}).get("scenes", [])

            if not scenes:
                break

            for s in scenes:
                performers = s.get("performers", [])
                has_stash_ids = any(
                    p.get("stash_ids") for p in performers
                )
                if has_stash_ids:
                    scene_ids.append(s["id"])
                if len(scene_ids) >= count:
                    break
            page += 1

    return scene_ids


# ---------------------------------------------------------------------------
# Capture one scene
# ---------------------------------------------------------------------------


async def capture_scene(
    client: httpx.AsyncClient,
    sidecar_url: str,
    stash_url: str,
    api_key: str,
    scene_id: str,
    num_frames: int,
) -> SceneRecord:
    """Call /identify/scene and extract tattoo-relevant fields."""
    payload = {
        "stash_url": stash_url,
        "scene_id": scene_id,
        "api_key": api_key,
        "num_frames": num_frames,
        "use_tattoo": True,
        "use_body": False,
        "use_multi_signal": True,
    }

    t0 = time.time()
    try:
        resp = await client.post(
            f"{sidecar_url}/identify/scene",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        wall_time = time.time() - t0

        if resp.status_code != 200:
            return SceneRecord(
                scene_id=scene_id,
                success=False,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
                frames_analyzed=0,
                frames_requested=num_frames,
                faces_detected=0,
                faces_after_filter=0,
                multi_signal_used=False,
                persons=[],
                timing=None,
                wall_time_sec=wall_time,
            )

        data = resp.json()

    except Exception as exc:
        wall_time = time.time() - t0
        return SceneRecord(
            scene_id=scene_id,
            success=False,
            error=str(exc)[:300],
            frames_analyzed=0,
            frames_requested=num_frames,
            faces_detected=0,
            faces_after_filter=0,
            multi_signal_used=False,
            persons=[],
            timing=None,
            wall_time_sec=wall_time,
        )

    # Parse persons
    persons: list[PersonRecord] = []
    for p in data.get("persons", []):
        best = p.get("best_match") or {}
        all_matches = p.get("all_matches", [])

        persons.append(
            PersonRecord(
                person_id=p.get("person_id", 0),
                frame_count=p.get("frame_count", 0),
                signals_used=p.get("signals_used", []),
                tattoos_detected=p.get("tattoos_detected", 0),
                best_match_stashdb_id=best.get("stashdb_id"),
                best_match_name=best.get("name"),
                best_match_distance=best.get("distance"),
                best_match_confidence=best.get("confidence"),
                all_match_ids=[m.get("stashdb_id", "") for m in all_matches],
            )
        )

    return SceneRecord(
        scene_id=scene_id,
        success=True,
        error=None,
        frames_analyzed=data.get("frames_analyzed", 0),
        frames_requested=data.get("frames_requested", num_frames),
        faces_detected=data.get("faces_detected", 0),
        faces_after_filter=data.get("faces_after_filter", 0),
        multi_signal_used=data.get("multi_signal_used", False),
        persons=persons,
        timing=data.get("timing"),
        wall_time_sec=wall_time,
    )


# ---------------------------------------------------------------------------
# Capture all scenes
# ---------------------------------------------------------------------------


async def run_capture(
    sidecar_url: str,
    stash_url: str,
    api_key: str,
    scene_ids: list[str],
    num_frames: int,
    output_path: Path,
) -> BenchmarkRun:
    """Run the full capture pass over the given scenes."""
    print(f"Tattoo Detection Baseline Capture")
    print(f"  Sidecar:    {sidecar_url}")
    print(f"  Stash:      {stash_url}")
    print(f"  Scenes:     {len(scene_ids)}")
    print(f"  Num frames: {num_frames}")
    print(f"  Output:     {output_path}")
    print()

    scenes: list[SceneRecord] = []
    t_total = time.time()

    async with httpx.AsyncClient() as client:
        for i, scene_id in enumerate(scene_ids):
            label = f"[{i + 1}/{len(scene_ids)}]"
            print(f"  {label} Scene {scene_id} ... ", end="", flush=True)

            record = await capture_scene(
                client, sidecar_url, stash_url, api_key, scene_id, num_frames
            )
            scenes.append(record)

            if record.success:
                tattoo_persons = [
                    p for p in record.persons if p.tattoos_detected > 0
                ]
                total_tats = sum(p.tattoos_detected for p in record.persons)
                print(
                    f"OK  {record.wall_time_sec:.1f}s  "
                    f"faces={record.faces_detected}  "
                    f"persons={len(record.persons)}  "
                    f"tattoo_persons={len(tattoo_persons)}  "
                    f"tattoo_detections={total_tats}"
                )
            else:
                print(f"FAIL  {record.error}")

    total_wall = time.time() - t_total

    # Compute aggregates
    successful = [s for s in scenes if s.success]
    all_persons = [p for s in successful for p in s.persons]
    tattoo_persons = [p for p in all_persons if p.tattoos_detected > 0]
    total_tattoos = sum(p.tattoos_detected for p in all_persons)

    run = BenchmarkRun(
        timestamp=datetime.now().isoformat(),
        sidecar_url=sidecar_url,
        stash_url=stash_url,
        num_frames=num_frames,
        scene_ids=scene_ids,
        scenes=scenes,
        total_scenes=len(scene_ids),
        successful_scenes=len(successful),
        total_persons=len(all_persons),
        persons_with_tattoos=len(tattoo_persons),
        total_tattoo_detections=total_tattoos,
        total_wall_time_sec=total_wall,
    )

    # Print summary
    print()
    print(f"Summary")
    print(f"  Scenes:              {run.total_scenes}")
    print(f"  Successful:          {run.successful_scenes}")
    print(f"  Persons:             {run.total_persons}")
    print(f"  Persons w/ tattoos:  {run.persons_with_tattoos}")
    print(f"  Total tattoo dets:   {run.total_tattoo_detections}")
    print(f"  Total wall time:     {run.total_wall_time_sec:.1f}s")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(run), indent=2, default=str))
    print(f"\nSaved to {output_path}")

    return run


# ---------------------------------------------------------------------------
# Compare two baselines
# ---------------------------------------------------------------------------


def load_run(path: Path) -> dict:
    """Load a benchmark run from JSON."""
    return json.loads(path.read_text())


def compare_runs(path_a: Path, path_b: Path) -> None:
    """Compare two benchmark runs and report tattoo detection agreement.

    For each scene present in both runs, compares per-person tattoo
    detection counts and reports agreement rates.
    """
    run_a = load_run(path_a)
    run_b = load_run(path_b)

    print(f"Tattoo Detection Comparison")
    print(f"  Run A: {path_a.name}  ({run_a['timestamp']})")
    print(f"  Run B: {path_b.name}  ({run_b['timestamp']})")
    print()

    # Index scenes by scene_id
    scenes_a = {s["scene_id"]: s for s in run_a["scenes"]}
    scenes_b = {s["scene_id"]: s for s in run_b["scenes"]}

    common_ids = sorted(set(scenes_a.keys()) & set(scenes_b.keys()))
    only_a = sorted(set(scenes_a.keys()) - set(scenes_b.keys()))
    only_b = sorted(set(scenes_b.keys()) - set(scenes_a.keys()))

    print(f"  Scenes in both:     {len(common_ids)}")
    if only_a:
        print(f"  Only in A:          {len(only_a)}  ({', '.join(only_a[:5])}{'...' if len(only_a) > 5 else ''})")
    if only_b:
        print(f"  Only in B:          {len(only_b)}  ({', '.join(only_b[:5])}{'...' if len(only_b) > 5 else ''})")
    print()

    if not common_ids:
        print("No common scenes to compare.")
        return

    # Per-scene comparison
    total_persons_compared = 0
    tattoo_count_exact_matches = 0
    tattoo_presence_agreements = 0

    # Track timing differences
    timing_diffs = []

    print(f"{'Scene':>8}  {'Persons A':>9}  {'Persons B':>9}  {'Tats A':>6}  {'Tats B':>6}  {'Agree':>5}  {'Status'}")
    print(f"{'':->8}  {'':->9}  {'':->9}  {'':->6}  {'':->6}  {'':->5}  {'':->20}")

    for scene_id in common_ids:
        sa = scenes_a[scene_id]
        sb = scenes_b[scene_id]

        if not sa.get("success") or not sb.get("success"):
            status = "SKIP (error)"
            print(f"{scene_id:>8}  {'---':>9}  {'---':>9}  {'---':>6}  {'---':>6}  {'---':>5}  {status}")
            continue

        persons_a = sa.get("persons", [])
        persons_b = sb.get("persons", [])

        # Match persons by best_match_stashdb_id
        persons_a_by_id = {}
        for p in persons_a:
            sid = p.get("best_match_stashdb_id")
            if sid:
                persons_a_by_id[sid] = p

        persons_b_by_id = {}
        for p in persons_b:
            sid = p.get("best_match_stashdb_id")
            if sid:
                persons_b_by_id[sid] = p

        matched_performer_ids = set(persons_a_by_id.keys()) & set(persons_b_by_id.keys())

        scene_presence = 0

        for pid in matched_performer_ids:
            pa = persons_a_by_id[pid]
            pb = persons_b_by_id[pid]

            tats_a = pa.get("tattoos_detected", 0)
            tats_b = pb.get("tattoos_detected", 0)
            has_a = tats_a > 0
            has_b = tats_b > 0

            total_persons_compared += 1

            # Presence agreement: both say tattoos or both say no tattoos
            if has_a == has_b:
                tattoo_presence_agreements += 1
                scene_presence += 1

            # Exact count agreement
            if tats_a == tats_b:
                tattoo_count_exact_matches += 1

        total_tats_a = sum(p.get("tattoos_detected", 0) for p in persons_a)
        total_tats_b = sum(p.get("tattoos_detected", 0) for p in persons_b)

        matched_count = len(matched_performer_ids)
        agree_pct = (
            f"{scene_presence}/{matched_count}"
            if matched_count > 0
            else "n/a"
        )

        status = "OK" if matched_count > 0 else "no matched persons"
        print(
            f"{scene_id:>8}  "
            f"{len(persons_a):>9}  "
            f"{len(persons_b):>9}  "
            f"{total_tats_a:>6}  "
            f"{total_tats_b:>6}  "
            f"{agree_pct:>5}  "
            f"{status}"
        )

        # Timing comparison
        timing_a = sa.get("timing") or {}
        timing_b = sb.get("timing") or {}
        if timing_a.get("total_ms") and timing_b.get("total_ms"):
            timing_diffs.append({
                "scene_id": scene_id,
                "a_ms": timing_a["total_ms"],
                "b_ms": timing_b["total_ms"],
                "diff_ms": timing_b["total_ms"] - timing_a["total_ms"],
            })

    # Aggregate summary
    print()
    print(f"Aggregate Comparison")
    print(f"  Persons compared:          {total_persons_compared}")

    if total_persons_compared > 0:
        presence_rate = tattoo_presence_agreements / total_persons_compared
        exact_rate = tattoo_count_exact_matches / total_persons_compared
        print(f"  Tattoo presence agreement: {tattoo_presence_agreements}/{total_persons_compared} ({presence_rate:.1%})")
        print(f"  Exact count agreement:     {tattoo_count_exact_matches}/{total_persons_compared} ({exact_rate:.1%})")

    # Timing summary
    if timing_diffs:
        avg_a = sum(t["a_ms"] for t in timing_diffs) / len(timing_diffs)
        avg_b = sum(t["b_ms"] for t in timing_diffs) / len(timing_diffs)
        avg_diff = avg_b - avg_a
        print()
        print(f"Timing Comparison (across {len(timing_diffs)} scenes)")
        print(f"  Avg total time A:  {avg_a:.0f}ms")
        print(f"  Avg total time B:  {avg_b:.0f}ms")
        print(f"  Avg difference:    {avg_diff:+.0f}ms ({avg_diff / avg_a * 100:+.1f}%)" if avg_a > 0 else "")

    # Aggregate tattoo counts
    common_ids_set = set(common_ids)
    total_a_tats = sum(
        sum(p.get("tattoos_detected", 0) for p in s.get("persons", []))
        for s in run_a["scenes"] if s.get("success") and s["scene_id"] in common_ids_set
    )
    total_b_tats = sum(
        sum(p.get("tattoos_detected", 0) for p in s.get("persons", []))
        for s in run_b["scenes"] if s.get("success") and s["scene_id"] in common_ids_set
    )
    print()
    print(f"Total tattoo detections (common scenes)")
    print(f"  Run A: {total_a_tats}")
    print(f"  Run B: {total_b_tats}")
    if total_a_tats > 0:
        print(f"  Ratio: {total_b_tats / total_a_tats:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tattoo detection baseline benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Mode: capture (default) or compare
    sub = parser.add_subparsers(dest="mode")

    # --- Capture mode ---
    cap = sub.add_parser("capture", help="Capture baseline results (default)")
    cap.add_argument(
        "--scenes",
        type=str,
        help="Comma-separated scene IDs (e.g. '100,200,300'). "
        "If omitted, discovers scenes from Stash randomly.",
    )
    cap.add_argument(
        "--count",
        type=int,
        default=20,
        help="Number of scenes to discover if --scenes not given (default: 20)",
    )
    cap.add_argument(
        "--num-frames",
        type=int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Frames to extract per scene (default: {DEFAULT_NUM_FRAMES})",
    )
    cap.add_argument(
        "--sidecar-url",
        type=str,
        default=DEFAULT_SIDECAR_URL,
        help=f"Sidecar API URL (default: {DEFAULT_SIDECAR_URL})",
    )
    cap.add_argument(
        "--stash-url",
        type=str,
        default=DEFAULT_STASH_URL,
        help=f"Stash URL (default: {DEFAULT_STASH_URL})",
    )
    cap.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON path (default: benchmark_results/tattoo_baseline_<timestamp>.json)",
    )

    # --- Compare mode ---
    cmp = sub.add_parser("compare", help="Compare two baseline files")
    cmp.add_argument("file_a", type=str, help="First baseline JSON file")
    cmp.add_argument("file_b", type=str, help="Second baseline JSON file")

    args = parser.parse_args(argv)

    # Default to capture mode if no subcommand given
    if args.mode is None:
        args.mode = "capture"
        # Re-parse with capture defaults
        args = cap.parse_args(argv or [])
        args.mode = "capture"

    return args


async def async_main(argv: list[str] = None) -> int:
    args = parse_args(argv)

    if args.mode == "compare":
        compare_runs(Path(args.file_a), Path(args.file_b))
        return 0

    # Capture mode
    api_key = _load_api_key()
    if not api_key:
        print("WARNING: No STASH_API_KEY found. Requests may fail if Stash requires auth.")

    sidecar_url = args.sidecar_url.rstrip("/")
    stash_url = args.stash_url.rstrip("/")

    # Resolve scene IDs
    if args.scenes:
        scene_ids = [s.strip() for s in args.scenes.split(",") if s.strip()]
    else:
        print(f"No --scenes provided, discovering {args.count} scenes from Stash...")
        scene_ids = await discover_scene_ids(stash_url, api_key, count=args.count)
        if not scene_ids:
            print("ERROR: Could not discover any scenes from Stash.")
            return 1
        print(f"Discovered {len(scene_ids)} scene IDs: {', '.join(scene_ids[:10])}{'...' if len(scene_ids) > 10 else ''}")

    # Output path
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = DEFAULT_OUTPUT_DIR / f"tattoo_baseline_{timestamp}.json"

    await run_capture(
        sidecar_url=sidecar_url,
        stash_url=stash_url,
        api_key=api_key,
        scene_ids=scene_ids,
        num_frames=args.num_frames,
        output_path=output_path,
    )

    return 0


def main(argv: list[str] = None) -> int:
    return asyncio.run(async_main(argv))


if __name__ == "__main__":
    sys.exit(main())

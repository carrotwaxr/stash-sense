"""CLI for the face recognition failure analysis (issue #139).

In the container:

    docker exec stash-sense python -m benchmark.failure build-set
    docker exec stash-sense python -m benchmark.failure run
    docker exec stash-sense python -m benchmark.failure report /data/benchmark/runs/<run_id> --attribution 0.55
    docker exec stash-sense python -m benchmark.failure replay /data/benchmark/runs/<run_id> --matching gallery

Detection test set:

    docker exec stash-sense python -m benchmark.failure detect-sample
    python -m benchmark.failure label --dir <copy of /data/benchmark/detection>
    docker exec stash-sense python -m benchmark.failure detect-score
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


def _classify_params(args):
    from benchmark.failure.classify import ClassifyParams
    return ClassifyParams(
        attribution_max_distance=args.attribution,
        usable_min_size=args.usable_size,
        profile_yaw=args.profile_yaw,
    )


def _add_classify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--attribution", type=float, default=0.65,
                   help="max fused distance to attribute a scene face to an expected performer (default 0.65)")
    p.add_argument("--usable-size", type=int, default=80,
                   help="short side in px below which a face counts as small (default 80)")
    p.add_argument("--profile-yaw", type=float, default=45.0,
                   help="|yaw| in degrees above which a face counts as profile (default 45)")


def _add_matching_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--matching", choices=["frequency", "gallery"],
                   help="scene aggregation: frequency (production default) or gallery (#148)")
    p.add_argument("--max-distance", type=float, help="override max_distance for accepting a match")
    p.add_argument("--keep-fraction", type=float, help="gallery: share of a cluster's best faces in the score")
    p.add_argument("--min-margin", type=float, help="gallery: required lead over the runner-up")
    p.add_argument("--candidate-k", type=int, help="gallery: per-face matches that nominate candidates")
    p.add_argument("--candidate-max-distance", type=float, help="gallery: per-face cutoff for nominating")


def _matching_overrides(args) -> dict:
    return {
        "matching": args.matching,
        "max_distance": args.max_distance,
        "gallery_keep_fraction": args.keep_fraction,
        "gallery_min_margin": args.min_margin,
        "gallery_candidate_k": args.candidate_k,
        "gallery_candidate_max_distance": args.candidate_max_distance,
    }


def build_parser() -> argparse.ArgumentParser:
    from benchmark.failure.source import default_detection_dir, default_runs_dir, default_set_path

    parser = argparse.ArgumentParser(prog="python -m benchmark.failure", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("build-set", help="sample the benchmark scenes from Stash and freeze them to JSON")
    p.add_argument("--count", type=int, default=200)
    p.add_argument("--seed", type=int, default=139)
    p.add_argument("--out", type=Path, default=default_set_path())
    p.add_argument("--force", action="store_true", help="overwrite an existing set")

    p = sub.add_parser("run", help="run the benchmark, then write report.md and summary.json")
    p.add_argument("--set", dest="set_path", type=Path, default=default_set_path())
    p.add_argument("--runs-dir", type=Path, default=default_runs_dir())
    p.add_argument("--resume", type=Path, help="continue an interrupted run directory")
    p.add_argument("--limit", type=int, help="only the first N scenes (smoke test)")
    p.add_argument("--scenes", help="comma-separated scene ids to run")
    p.add_argument("--num-frames", type=int, help="override the num_frames setting")
    p.add_argument("--no-probe", action="store_true", help="skip the keyframe probe that splits frame choice from detection")
    p.add_argument("--probe-stream", action="store_true", help="allow the probe to decode over Stash's HTTP stream")
    p.add_argument("--probe-max-frames", type=int, default=400)
    _add_matching_args(p)
    _add_classify_args(p)

    p = sub.add_parser("replay", help="re-run matching on a run's cached embeddings (no video or GPU)")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--runs-dir", type=Path, help="where to write the new run (default: beside the source)")
    p.add_argument("--scenes", help="comma-separated scene ids to replay")
    _add_matching_args(p)
    _add_classify_args(p)

    p = sub.add_parser("report", help="re-score a run's raw JSON (no GPU needed)")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--suffix", default="", help="write report<suffix>.md instead of overwriting report.md")
    _add_classify_args(p)

    p = sub.add_parser("detect-sample", help="extract SD frames and pre-draw proposal boxes for labeling")
    p.add_argument("--set", dest="set_path", type=Path, default=default_set_path())
    p.add_argument("--dir", type=Path, default=default_detection_dir())
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--seed", type=int, default=139)
    p.add_argument("--resolutions", default="SD", help="comma-separated tiers (default SD)")
    p.add_argument("--force", action="store_true")

    p = sub.add_parser("label", help="serve the labeling page for a detection set")
    p.add_argument("--dir", type=Path, default=default_detection_dir())
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8139)

    p = sub.add_parser("detect-score", help="score the production detector against the labels")
    p.add_argument("--dir", type=Path, default=default_detection_dir())
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--det-size", type=int, help="override the detection_size setting")
    p.add_argument("--channel-order", choices=["rgb", "bgr"], default="bgr",
                   help="bgr = production (what RetinaFace expects); rgb = production before #147")
    return parser


async def _build_set(args) -> int:
    from benchmark.failure.scene_set import build_set
    from stash_client_unified import StashClientUnified

    if args.out.exists() and not args.force:
        print(f"{args.out} exists; the set is frozen so runs stay comparable. Pass --force to replace it.")
        return 1
    client = StashClientUnified(os.environ.get("STASH_URL", ""), os.environ.get("STASH_API_KEY", ""))
    bench, available = await build_set(client, args.count, args.seed)
    bench.save(args.out)
    picked: dict[str, int] = {}
    for s in bench.scenes:
        picked[s.stratum] = picked.get(s.stratum, 0) + 1
    print(f"{'Stratum':<22} {'available':>9} {'picked':>7}")
    for key in sorted(available):
        print(f"{key:<22} {available[key]:>9} {picked.get(key, 0):>7}")
    performers = sum(len(s.expected) for s in bench.scenes)
    print(f"\n{len(bench.scenes)} scenes, {performers} expected performers -> {args.out}")
    return 0


async def _run(args) -> int:
    from benchmark.failure.runner import init_settings, resolve_run_params, run

    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    settings_error = init_settings(data_dir)
    if settings_error:
        print(f"Settings not loaded ({settings_error}); using face_config defaults")
    params = resolve_run_params(num_frames=args.num_frames, **_matching_overrides(args))
    await run(
        args.set_path, args.runs_dir,
        run_params=params,
        classify_params=_classify_params(args),
        probe=not args.no_probe,
        probe_stream=args.probe_stream,
        probe_max_frames=args.probe_max_frames,
        limit=args.limit,
        scene_ids=[s.strip() for s in args.scenes.split(",")] if args.scenes else None,
        resume=args.resume,
        settings_error=settings_error,
    )
    return 0


def _replay(args) -> int:
    from benchmark.failure.replay import replay

    replay(
        args.run_dir, args.runs_dir,
        overrides=_matching_overrides(args),
        classify_params=_classify_params(args),
        scene_ids=[s.strip() for s in args.scenes.split(",")] if args.scenes else None,
    )
    return 0


def _report(args) -> int:
    from benchmark.failure.runner import write_report

    meta_path = args.run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    report, summary = write_report(args.run_dir, _classify_params(args), meta, args.suffix)
    print(f"Report: {report}")
    if summary["verdict"]["next_step"]:
        print(summary["verdict"]["next_step"])
    return 0


def _detect_sample(args) -> int:
    from benchmark.failure.detection.sample import sample_frames
    from benchmark.failure.scene_set import BenchmarkSet

    path = sample_frames(
        BenchmarkSet.load(args.set_path), args.dir,
        count=args.count, seed=args.seed,
        resolutions=tuple(r.strip() for r in args.resolutions.split(",")),
        stash_url=os.environ.get("STASH_URL", ""), api_key=os.environ.get("STASH_API_KEY", ""),
        force=args.force,
    )
    print(f"Proposals: {path}")
    return 0


def _detect_score(args) -> int:
    from benchmark.failure.detection.score import score
    from benchmark.failure.runner import init_settings, resolve_run_params

    init_settings(Path(os.environ.get("DATA_DIR", "./data")))
    det_size = args.det_size or resolve_run_params().detection_size
    report = score(args.dir, det_size=det_size, iou_threshold=args.iou, channel_order=args.channel_order)
    print(f"Report: {report}")
    return 0


def main(argv=None) -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    args = build_parser().parse_args(argv)
    if args.command == "build-set":
        return asyncio.run(_build_set(args))
    if args.command == "run":
        return asyncio.run(_run(args))
    if args.command == "report":
        return _report(args)
    if args.command == "replay":
        return _replay(args)
    if args.command == "detect-sample":
        return _detect_sample(args)
    if args.command == "label":
        from benchmark.failure.detection.labeler import serve
        serve(args.dir, host=args.host, port=args.port)
        return 0
    if args.command == "detect-score":
        return _detect_score(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())

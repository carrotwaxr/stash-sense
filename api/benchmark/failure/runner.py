"""Benchmark run: every scene in the set, raw JSON per scene, then the report.

Raw records land in ``<run_dir>/raw/<scene_id>.json`` as each scene finishes,
so an interrupted run resumes with ``--resume <run_dir>``. Each scene's face
embeddings go to ``<run_dir>/emb/<scene_id>.npz`` for ``replay``.
"""

import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmark.failure.classify import ClassifyParams, classify_scene
from benchmark.failure.pipeline import RunParams
from benchmark.failure.report import render_markdown, summarise
from benchmark.failure.scene_set import BenchmarkSet

PROBE_REASONS = ("no_face_detected", "small_or_profile")

SCREENSHOT_QUERY = "query($id: ID!) { findScene(id: $id) { paths { screenshot } } }"


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(path)


def load_records(run_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted((run_dir / "raw").glob("*.json"))]


def write_report(run_dir: Path, params: ClassifyParams, meta: dict, suffix: str = "") -> tuple[Path, dict]:
    """Summarise the raw records in run_dir; returns (report path, summary)."""
    summary = summarise(load_records(run_dir), params, meta)
    write_json(run_dir / f"summary{suffix}.json", summary)
    report = run_dir / f"report{suffix}.md"
    report.write_text(render_markdown(summary))
    return report, summary


def init_settings(data_dir: Path, tier: Optional[str] = None) -> Optional[str]:
    """Load the sidecar's saved settings so the run uses what production uses.

    ``tier`` replaces the detected hardware tier for defaults, so a CPU-only
    machine can run with production's frame count and detection size.
    Returns an error string when settings can't be loaded (defaults apply).
    """
    try:
        from hardware import init_hardware
        from recommendations_db import RecommendationsDB
        from settings import init_settings as _init

        detected = init_hardware(str(data_dir)).tier
        _init(RecommendationsDB(data_dir / "stash_sense.db"), tier or detected)
        return None
    except Exception as e:  # a missing DB shouldn't block a research run
        return f"{type(e).__name__}: {e}"


def resolve_run_params(**overrides) -> RunParams:
    """RunParams from live settings, then any explicit CLI overrides."""
    params = RunParams()
    try:
        from settings import get_setting
        params = replace(
            params,
            num_frames=int(get_setting("num_frames")),
            detection_size=int(get_setting("detection_size")),
            max_concurrent_extractions=int(get_setting("frame_extraction_concurrency")),
        )
    except (RuntimeError, KeyError):
        pass
    return replace(params, **{k: v for k, v in overrides.items() if v is not None})


async def run(
    set_path: Path,
    runs_dir: Path,
    *,
    run_params: RunParams,
    classify_params: ClassifyParams,
    probe: bool = True,
    probe_stream: bool = False,
    probe_max_frames: int = 400,
    limit: Optional[int] = None,
    scene_ids: Optional[list[str]] = None,
    resume: Optional[Path] = None,
    settings_error: Optional[str] = None,
) -> Path:
    from config import DatabaseConfig
    from database_reader import PerformerDatabaseReader
    from recognizer import FaceRecognizer
    from stash_client_unified import StashClientUnified

    from benchmark.failure.gallery import GalleryStore
    from benchmark.failure.pipeline import run_scene
    from benchmark.failure.probe import probe_scene
    from benchmark.failure.source import resolve_video_source

    bench = BenchmarkSet.load(set_path)
    scenes = bench.scenes
    if scene_ids:
        wanted = set(scene_ids)
        scenes = [s for s in scenes if s.scene_id in wanted]
    if limit:
        scenes = scenes[:limit]

    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    stash_url = os.environ.get("STASH_URL", "").rstrip("/")
    api_key = os.environ.get("STASH_API_KEY", "")
    if not stash_url:
        raise SystemExit("STASH_URL is not set")

    if resume:
        run_dir = Path(resume)
        meta = json.loads((run_dir / "meta.json").read_text())
    else:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = Path(runs_dir) / run_id
        manifest = data_dir / "manifest.json"
        meta = {
            "run_id": run_id,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "db_version": json.loads(manifest.read_text()).get("version") if manifest.exists() else None,
            "set_path": str(set_path),
            "set_seed": bench.seed,
            "set_created_at": bench.created_at,
            "run_params": run_params.to_dict(),
            "probe": probe,
            "settings_error": settings_error,
        }
        write_json(run_dir / "meta.json", meta)
    print(f"Run directory: {run_dir}", flush=True)

    db_config = DatabaseConfig(data_dir=data_dir)
    recognizer = FaceRecognizer(db_config)
    db_reader = recognizer.db_reader or PerformerDatabaseReader(str(db_config.sqlite_db_path))
    store = GalleryStore(db_reader, recognizer.facenet_index, recognizer.arcface_index, recognizer.faces)
    stash = StashClientUnified(stash_url, api_key)

    done = {p.stem for p in (run_dir / "raw").glob("*.json")}
    todo = [s for s in scenes if s.scene_id not in done]
    print(f"{len(scenes)} scenes in scope, {len(done)} already done, {len(todo)} to run", flush=True)

    for i, scene in enumerate(todo, 1):
        t0 = time.time()
        galleries = [store.get(e.stashdb_id) for e in scene.expected]
        try:
            shot = await stash._execute(SCREENSHOT_QUERY, {"id": scene.scene_id})
            screenshot_url = ((shot.get("findScene") or {}).get("paths") or {}).get("screenshot")
            record = await run_scene(scene, recognizer, galleries, run_params,
                                     stash_url=stash_url, api_key=api_key, screenshot_url=screenshot_url,
                                     embeddings_path=run_dir / "emb" / f"{scene.scene_id}.npz")
        except Exception as e:
            record = {"scene_id": scene.scene_id, "stratum": scene.stratum, "status": "error",
                      "error": f"{type(e).__name__}: {e}"}

        if record["status"] == "ok":
            needs = [o.stashdb_id for o in classify_scene(record, classify_params)
                     if not o.found and o.reason in PROBE_REASONS]
            if needs and probe:
                source = resolve_video_source(scene.path, scene.scene_id, stash_url, api_key)
                if source.local or probe_stream:
                    try:
                        by_id = {g.stashdb_id: g for g in galleries}
                        record["probe"] = await probe_scene(record, needs, recognizer, by_id, run_params,
                                                            classify_params, source=source,
                                                            max_frames=probe_max_frames)
                    except Exception as e:
                        record["probe_error"] = f"{type(e).__name__}: {e}"
                else:
                    record["probe_error"] = "skipped: no local file (pass --probe-stream to decode over HTTP)"

        write_json(run_dir / "raw" / f"{scene.scene_id}.json", record)
        if record["status"] == "ok":
            outs = classify_scene(record, classify_params)
            found = sum(o.found for o in outs)
            misses = ", ".join(f"{o.name}: {o.reason}" for o in outs if not o.found)
            print(f"[{i}/{len(todo)}] scene {scene.scene_id} ({scene.stratum}, {record['source']}) "
                  f"found {found}/{len(outs)} in {time.time() - t0:.0f}s" + (f" | {misses}" if misses else ""),
                  flush=True)
        else:
            print(f"[{i}/{len(todo)}] scene {scene.scene_id} FAILED: {record['error']}", flush=True)

    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir / "meta.json", meta)
    report, summary = write_report(run_dir, classify_params, meta)
    print(f"\nReport: {report}\nSummary JSON: {run_dir / 'summary.json'}", flush=True)
    if summary["verdict"]["next_step"]:
        print(summary["verdict"]["next_step"], flush=True)
    return report

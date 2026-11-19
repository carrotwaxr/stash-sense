"""Replay a run's scene matching from its cached embeddings (#148).

``run`` stores every recognised face's embeddings in ``<run_dir>/emb/``.
Replay re-queries the index and re-runs scene aggregation from them, so
matching changes can be compared on exactly the same faces in seconds,
without video or a GPU. The result is a new run directory with the usual
``raw/``, ``summary.json`` and ``report.md``.

Distances only reproduce against the face DB the source run used; replay
records both versions and warns when they differ.
"""

import json
import os
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from benchmark.failure.classify import ClassifyParams
from benchmark.failure.pipeline import RunParams, aggregate, load_embeddings, match_config_for
from benchmark.failure.runner import load_records, write_json, write_report

# Records copied from the source run, minus what aggregation rewrites.
REWRITTEN = ("faces", "persons", "aggregation_ms")


def source_params(meta: dict) -> RunParams:
    """The source run's pipeline settings; fields newer than the run take defaults."""
    known = RunParams.__dataclass_fields__
    return RunParams(**{k: v for k, v in (meta.get("run_params") or {}).items() if k in known})


def replay(
    src: Path,
    runs_dir: Optional[Path],
    *,
    overrides: dict,
    classify_params: ClassifyParams,
    scene_ids: Optional[list[str]] = None,
) -> Path:
    from config import DatabaseConfig
    from database_reader import PerformerDatabaseReader
    from recognizer import FaceRecognizer, RecognitionResult

    from benchmark.failure.gallery import GalleryStore

    src = Path(src)
    src_meta = json.loads((src / "meta.json").read_text())
    params = replace(source_params(src_meta), **{k: v for k, v in overrides.items() if v is not None})

    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    manifest = data_dir / "manifest.json"
    db_version = json.loads(manifest.read_text()).get("version") if manifest.exists() else None
    if db_version != src_meta.get("db_version"):
        print(f"WARNING: face DB {db_version} differs from the source run's {src_meta.get('db_version')}; "
              "distances and matches won't be comparable", flush=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-replay"
    run_dir = Path(runs_dir or src.parent) / run_id
    meta = {
        "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "replay_of": src_meta.get("run_id"),
        "db_version": db_version,
        "source_db_version": src_meta.get("db_version"),
        "set_path": src_meta.get("set_path"),
        "set_seed": src_meta.get("set_seed"),
        "set_created_at": src_meta.get("set_created_at"),
        "run_params": params.to_dict(),
        "probe": src_meta.get("probe"),
    }
    write_json(run_dir / "meta.json", meta)
    print(f"Replaying {src} -> {run_dir}", flush=True)

    db_config = DatabaseConfig(data_dir=data_dir)
    recognizer = FaceRecognizer(db_config)
    db_reader = recognizer.db_reader or PerformerDatabaseReader(str(db_config.sqlite_db_path))
    store = GalleryStore(db_reader, recognizer.facenet_index, recognizer.arcface_index, recognizer.faces)
    match_config = match_config_for(params)

    records = load_records(src)
    if scene_ids:
        wanted = set(scene_ids)
        records = [r for r in records if r["scene_id"] in wanted]

    missing = 0
    t_start = time.time()
    for rec in records:
        out = {k: v for k, v in rec.items() if k not in REWRITTEN}
        emb_path = src / "emb" / f"{rec['scene_id']}.npz"
        if rec.get("status") == "ok" and not emb_path.exists():
            missing += 1
            out.update(status="error", error="no cached embeddings (run predates the cache)")
        elif rec.get("status") == "ok":
            recognised = []
            for fi, ts, face, emb, eligible in load_embeddings(emb_path):
                matches, _, _ = recognizer.recognize_face_v2(face, match_config, embedding=emb)
                recognised.append((fi, ts, RecognitionResult(face=face, matches=matches, embedding=emb), eligible))
            galleries = [store.get(e["stashdb_id"]) for e in rec["expected"]]
            aggregate(out, recognizer, recognised, galleries, params)
        write_json(run_dir / "raw" / f"{rec['scene_id']}.json", out)

    if missing:
        print(f"WARNING: {missing} scenes had no cached embeddings and count as errors", flush=True)
    meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(run_dir / "meta.json", meta)
    report, summary = write_report(run_dir, classify_params, meta)
    print(f"Replayed {len(records)} scenes in {time.time() - t_start:.0f}s\nReport: {report}", flush=True)
    print_comparison(src, summary)
    return run_dir


def print_comparison(src: Path, summary: dict) -> None:
    """Recall, precision and aggregation time: source run vs replay."""
    src_summary_path = src / "summary.json"
    if not src_summary_path.exists():
        return
    old = json.loads(src_summary_path.read_text())
    rows = [("source", old), ("replay", summary)]
    print(f"\n{'':<8} {'found':>6} {'recall':>8} {'FPs':>5} {'precision':>10} {'agg ms (median)':>16}")
    for label, s in rows:
        t = s["totals"]
        agg = (s.get("timing") or {}).get("median_aggregation_ms")
        print(f"{label:<8} {t['found']:>6} {_pct(t['recall']):>8} {t['false_positives']:>5} "
              f"{_pct(t['precision']):>10} {'n/a' if agg is None else f'{agg:.1f}':>16}")


def _pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"

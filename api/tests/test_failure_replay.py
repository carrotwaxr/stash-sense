"""Benchmark embedding cache and replay (#148)."""

from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.failure.pipeline import (
    RunParams,
    load_embeddings,
    match_config_for,
    save_embeddings,
)
from benchmark.failure.replay import source_params
from benchmark.failure.report import render_markdown, summarise


def _recognised(i, eligible=True, embedding=True, ts=1.5, yaw=10.0):
    face = SimpleNamespace(bbox={"x": i, "y": 2, "w": 80, "h": 90}, confidence=0.9, yaw=yaw)
    emb = SimpleNamespace(facenet=np.full(4, i, np.float32), arcface=np.full(4, -i, np.float32)) if embedding else None
    return (i, ts, SimpleNamespace(face=face, embedding=emb), eligible)


def test_embeddings_round_trip(tmp_path):
    path = tmp_path / "emb" / "1.npz"
    rows = [_recognised(1), _recognised(2, eligible=False, ts=None, yaw=None), _recognised(3, embedding=False)]
    save_embeddings(path, rows)
    assert path.exists() and not list(path.parent.glob("*.tmp*"))

    loaded = load_embeddings(path)
    assert [r[0] for r in loaded] == [1, 2]  # the face without an embedding is dropped
    _frame, ts, face, emb, eligible = loaded[0]
    assert ts == 1.5 and eligible and face.yaw == pytest.approx(10.0) and face.image is None
    assert face.bbox == {"x": 1, "y": 2, "w": 80, "h": 90} and face.confidence == pytest.approx(0.9)
    assert np.array_equal(emb.facenet, np.full(4, 1)) and np.array_equal(emb.arcface, np.full(4, -1))
    assert loaded[1][1] is None and loaded[1][2].yaw is None and not loaded[1][4]


def test_source_params_ignore_unknown_and_default_new_fields():
    params = source_params({"run_params": {"num_frames": 40, "dropped_field": 1}})
    assert params.num_frames == 40 and params.matching == "frequency"


def test_gallery_mode_widens_per_face_matches():
    freq = match_config_for(RunParams())
    gal = match_config_for(RunParams(matching="gallery", gallery_candidate_k=12, gallery_candidate_max_distance=0.8))
    assert gal.max_results == 12 and gal.max_distance == 0.8
    assert freq.max_distance == RunParams().max_distance


def test_summary_reports_aggregation_timing():
    recs = [{"scene_id": str(i), "status": "ok", "stratum": "2020+/1080p", "era": "2020+", "resolution": "1080p",
             "expected": [], "faces": [], "persons": [], "aggregation_ms": ms, "elapsed_sec": 8.0}
            for i, ms in enumerate([2.0, 4.0, 9.0])]
    summary = summarise(recs)
    assert summary["timing"] == {"median_aggregation_ms": 4.0, "max_aggregation_ms": 9.0, "median_scene_sec": 8.0}
    assert "median 4.0 ms" in render_markdown(summary)

"""Tests for the face recognition failure analysis (benchmark/failure, #139)."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.failure.classify import ClassifyParams, classify_scene, false_positives, stage_for
from benchmark.failure.gallery import GalleryStore, PerformerGallery, distance_to
from benchmark.failure.probe import evenly_pick, keyframe_command, parse_pts, summarise_probe
from benchmark.failure.report import render_markdown, summarise
from benchmark.failure.scene_set import (
    BenchmarkScene,
    BenchmarkSet,
    ExpectedPerformer,
    era_for,
    resolution_for,
    scene_from_stash,
    stratified_sample,
)
from benchmark.failure.source import resolve_video_source

STASHDB = "https://stashdb.org/graphql"


# ---------------------------------------------------------------- scene set

@pytest.mark.parametrize("date,era", [
    ("2009-12-31", "pre-2010"), ("2010-01-01", "2010-2014"), ("2014", "2010-2014"),
    ("2019-06-01", "2015-2019"), ("2024-02-02", "2020+"), (None, None), ("", None), ("n/a", None),
])
def test_era_for(date, era):
    assert era_for(date) == era


@pytest.mark.parametrize("w,h,res", [
    (640, 480, "SD"), (720, 576, "SD"), (1280, 720, "720p"), (720, 1280, "720p"),
    (1920, 1080, "1080p"), (2560, 1440, "1080p"), (3840, 2160, "4K"), (0, 0, None),
])
def test_resolution_uses_short_side(w, h, res):
    assert resolution_for(w, h) == res


def _stash_scene(**over):
    row = {
        "id": "7", "title": "t", "date": "2016-03-04",
        "files": [{"path": "/data/a.mp4", "width": 1920, "height": 1080, "duration": 1200.0}],
        "stash_ids": [{"endpoint": STASHDB, "stash_id": "scene-uuid"}],
        "performers": [{"name": "A", "stash_ids": [{"endpoint": STASHDB, "stash_id": "pa"}]}],
    }
    row.update(over)
    return row


def test_scene_from_stash_qualifies():
    s = scene_from_stash(_stash_scene())
    assert s.scene_id == "7" and s.era == "2015-2019" and s.resolution == "1080p"
    assert [p.stashdb_id for p in s.expected] == ["pa"]
    assert s.stratum == "2015-2019 / 1080p"


@pytest.mark.parametrize("over", [
    {"stash_ids": []},
    {"files": []},
    {"date": None},
    {"performers": []},
    {"performers": [{"name": "A", "stash_ids": [{"endpoint": STASHDB, "stash_id": "pa"}]},
                    {"name": "B", "stash_ids": [{"endpoint": "https://fansdb.cc/graphql", "stash_id": "x"}]}]},
    {"files": [{"path": "/a", "width": 1920, "height": 1080, "duration": 0}]},
])
def test_scene_from_stash_rejects(over):
    assert scene_from_stash(_stash_scene(**over)) is None


def _scene(i, era="2020+", res="1080p", expected=("p1",)):
    return BenchmarkScene(scene_id=str(i), title="", date="2020-01-01", era=era, resolution=res,
                          width=1920, height=1080, duration=600.0, path=f"/data/{i}.mp4",
                          expected=[ExpectedPerformer(e, e.upper()) for e in expected])


def test_stratified_sample_is_even_and_redistributes():
    pool = ([_scene(i, "2020+", "1080p") for i in range(50)]
            + [_scene(100 + i, "pre-2010", "SD") for i in range(3)]
            + [_scene(200 + i, "2010-2014", "720p") for i in range(50)])
    picked = stratified_sample(pool, 30, seed=1)
    counts = {}
    for s in picked:
        counts[s.stratum] = counts.get(s.stratum, 0) + 1
    assert len(picked) == 30
    assert counts["pre-2010 / SD"] == 3  # all it has
    assert abs(counts["2020+ / 1080p"] - counts["2010-2014 / 720p"]) <= 1


def test_stratified_sample_deterministic_and_capped():
    pool = [_scene(i) for i in range(10)]
    assert [s.scene_id for s in stratified_sample(pool, 5, 7)] == [s.scene_id for s in stratified_sample(pool, 5, 7)]
    assert len(stratified_sample(pool, 50, 7)) == 10


def test_set_round_trip(tmp_path):
    bench = BenchmarkSet(seed=3, created_at="x", scenes=[_scene(1, expected=("a", "b"))])
    bench.save(tmp_path / "s.json")
    loaded = BenchmarkSet.load(tmp_path / "s.json")
    assert loaded.scenes[0].expected[1].stashdb_id == "b"
    assert loaded.seed == 3


# ---------------------------------------------------------------- source

class _Lib:
    def __init__(self, mapping):
        self.mapping = mapping

    def to_local(self, path):
        if path not in self.mapping:
            raise ValueError("outside map")
        return self.mapping[path]


def test_source_prefers_local_file():
    src = resolve_video_source("/data/a.mp4", "7", "http://stash:9999", "key",
                               library=_Lib({"/data/a.mp4": "/media/a.mp4"}), exists=lambda p: True)
    assert src.local and src.input == "/media/a.mp4" and src.api_key is None


def test_source_falls_back_to_stream():
    src = resolve_video_source("/other/a.mp4", "7", "http://stash:9999/", "key",
                               library=_Lib({}), exists=lambda p: True)
    assert not src.local and src.input == "http://stash:9999/scene/7/stream" and src.api_key == "key"
    missing = resolve_video_source("/data/a.mp4", "7", "http://s", "k",
                                   library=_Lib({"/data/a.mp4": "/media/a.mp4"}), exists=lambda p: False)
    assert not missing.local


# ---------------------------------------------------------------- gallery

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_distance_to_is_min_fused_cosine():
    g = PerformerGallery("p", 1, facenet=np.stack([_unit([1, 0]), _unit([0, 1])]),
                         arcface=np.stack([_unit([1, 0]), _unit([0, 1])]))
    assert distance_to(np.array([2.0, 0]), np.array([1.0, 0]), g) == pytest.approx(0.0, abs=1e-6)
    # FaceNet matches face 0 exactly, ArcFace is orthogonal to it: 0.5*0 + 0.5*1
    assert distance_to(np.array([1.0, 0]), np.array([0, 1.0]), g, 0.5, 0.5) == pytest.approx(0.5, abs=1e-6)
    assert distance_to(np.array([1.0, 0]), np.array([1.0, 0]), PerformerGallery("q", None)) is None


class _FakeIndex:
    def __init__(self, vectors):
        self.vectors = vectors

    def get_vectors(self, ids):
        return [self.vectors[i] for i in ids]


class _FakeDB:
    def get_performer_by_stashbox_id(self, endpoint, sid):
        return SimpleNamespace(id=5, canonical_name="Pat") if sid == "pa" else None

    def get_stashbox_ids(self, pid):
        return [SimpleNamespace(stashbox_performer_id="pa"), SimpleNamespace(stashbox_performer_id="tpdb-1")]

    def get_faces(self, pid):
        return [SimpleNamespace(facenet_index=0, arcface_index=0), SimpleNamespace(facenet_index=1, arcface_index=1)]


def test_gallery_store_loads_ids_and_skips_invisible_slots():
    idx = _FakeIndex({0: [1.0, 0.0], 1: [0.0, 2.0]})
    store = GalleryStore(_FakeDB(), idx, idx, faces_mapping=["stashdb.org:pa", None])
    g = store.get("pa")
    assert g.in_db and g.face_count == 1 and g.name == "Pat"
    assert g.is_match("tpdb-1") and g.is_match("pa") and not g.is_match("other")
    assert store.get("pa") is g
    assert not store.get("nobody").in_db


# ---------------------------------------------------------------- classify

def _face(dist, top=None, w=120, h=120, yaw=0.0, eligible=True, frame=0):
    return {"frame": frame, "timestamp": 1.0, "x": 0, "y": 0, "w": w, "h": h, "det_score": 0.9,
            "yaw": yaw, "eligible": eligible, "dist": dist,
            "top": [{"stashdb_id": s, "name": s, "distance": d} for s, d in (top or [])]}


def _record(faces, persons=(), expected=("pa", "pb"), not_in_db=(), probe=None):
    rec = {
        "scene_id": "1", "stratum": "2020+ / SD", "era": "2020+", "resolution": "SD", "status": "ok",
        "expected": [{"stashdb_id": e, "name": e.upper(), "in_db": e not in not_in_db, "ids": [e, f"alt-{e}"]}
                     for e in expected],
        "faces": faces,
        "persons": [{"frame_count": n, "best": {"stashdb_id": s, "name": s, "distance": 0.3}} for s, n in persons],
    }
    if probe:
        rec["probe"] = probe
    return rec


def _reasons(rec, **kw):
    return {o.stashdb_id: (o.found, o.reason) for o in classify_scene(rec, ClassifyParams(**kw))}


def test_found_needs_multi_frame_person_and_accepts_alt_ids():
    rec = _record([], persons=[("alt-pa", 3), ("pb", 1)])
    out = {o.stashdb_id: o for o in classify_scene(rec)}
    assert out["pa"].found
    assert not out["pb"].found and out["pb"].shown_as_single_frame


def test_not_in_db_beats_everything():
    rec = _record([_face({"pa": 0.2, "pb": 0.9})], not_in_db=("pa",))
    assert _reasons(rec)["pa"] == (False, "not_in_db")


def test_no_face_detected_when_nothing_attributed():
    rec = _record([_face({"pa": 0.9, "pb": 0.95})])
    assert _reasons(rec)["pa"] == (False, "no_face_detected")


def test_attribution_goes_to_the_closest_expected_performer():
    rec = _record([_face({"pa": 0.5, "pb": 0.4}, top=[("x", 0.45)])])
    r = _reasons(rec)
    assert r["pa"] == (False, "no_face_detected")
    assert r["pb"] == (False, "wrong_match")


def test_dropped_by_aggregation_when_top1_on_an_eligible_face():
    rec = _record([_face({"pa": 0.3, "pb": 0.9}, top=[("pa", 0.3)], w=50, h=50)])
    assert _reasons(rec)["pa"] == (False, "dropped_by_aggregation")


def test_top1_on_ineligible_face_does_not_count_as_aggregation():
    rec = _record([_face({"pa": 0.3, "pb": 0.9}, top=[("pa", 0.3)], w=30, h=30, eligible=False)])
    assert _reasons(rec)["pa"] == (False, "small_or_profile")


@pytest.mark.parametrize("face_kw", [{"w": 60, "h": 120}, {"yaw": 60.0}, {"yaw": -50.0}])
def test_small_or_profile(face_kw):
    rec = _record([_face({"pa": 0.4, "pb": 0.9}, top=[("zz", 0.35)], **face_kw)])
    assert _reasons(rec)["pa"] == (False, "small_or_profile")


def test_wrong_match_records_winner_and_rank():
    rec = _record([
        _face({"pa": 0.45, "pb": 0.9}, top=[("zz", 0.32), ("pa", 0.45)]),
        _face({"pa": 0.55, "pb": 0.9}, top=[("yy", 0.3)]),
    ])
    o = [x for x in classify_scene(rec) if x.stashdb_id == "pa"][0]
    assert o.reason == "wrong_match" and o.stage == "matching"
    assert o.winner["stashdb_id"] == "zz" and o.rank_on_best_face == 2
    assert o.best_distance == 0.45 and o.attributed_faces == 2 and o.usable_faces == 2


def test_attribution_threshold_moves_reason():
    rec = _record([_face({"pa": 0.7, "pb": 0.95}, top=[("zz", 0.3)])])
    assert _reasons(rec)["pa"][1] == "no_face_detected"
    assert _reasons(rec, attribution_max_distance=0.75)["pa"][1] == "wrong_match"


def test_each_expected_performer_gets_exactly_one_outcome_and_misses_one_reason():
    rec = _record([_face({"pa": 0.3, "pb": 0.9}, top=[("zz", 0.3)])], persons=[("zz", 4)],
                  expected=("pa", "pb", "pc"), not_in_db=("pc",))
    outs = classify_scene(rec)
    assert len(outs) == 3
    assert all((o.reason is None) == o.found for o in outs)


def test_probe_sets_stage():
    probe = {"pa": {"recoverable": True}, "pb": {"recoverable": False}}
    rec = _record([_face({"pa": 0.9, "pb": 0.9})], probe=probe)
    stages = {o.stashdb_id: o.stage for o in classify_scene(rec)}
    assert stages == {"pa": "frame_choice", "pb": "detection_or_absent"}


@pytest.mark.parametrize("reason,rec,stage", [
    ("not_in_db", None, "db_coverage"), ("wrong_match", None, "matching"),
    ("dropped_by_aggregation", None, "matching"), ("no_face_detected", None, "unprobed"),
    ("small_or_profile", False, "size_or_pose_throughout"), ("small_or_profile", True, "frame_choice"),
    (None, None, None),
])
def test_stage_for(reason, rec, stage):
    assert stage_for(reason, rec) == stage


def test_false_positives_ignore_expected_alt_ids_and_single_frames():
    rec = _record([], persons=[("alt-pa", 3), ("zz", 2), ("yy", 1)])
    assert [fp["stashdb_id"] for fp in false_positives(rec)] == ["zz"]


# ---------------------------------------------------------------- probe

def test_parse_pts_reads_showinfo_lines_in_order():
    stderr = (
        "Input #0, mov...\n"
        "[Parsed_showinfo_0 @ 0x1] n:   0 pts:      0 pts_time:0       pos: 48 fmt:yuv420p\n"
        "[Parsed_showinfo_0 @ 0x1] n:   1 pts: 256000 pts_time:10.01   pos: 99\n"
        "frame=    2 fps=0.0 q=3.0\n"
    )
    assert parse_pts(stderr) == [0.0, 10.01]


def test_evenly_pick():
    assert evenly_pick(3, 10) == [0, 1, 2]
    picks = evenly_pick(100, 5)
    assert picks[0] == 0 and picks[-1] == 99 and len(picks) == 5
    assert evenly_pick(9, 1) == [4]


def test_keyframe_command_only_sends_key_for_streams():
    local = keyframe_command("/media/a.mp4", "/tmp/%06d.jpg", None)
    assert "-headers" not in local and local.index("-skip_frame") < local.index("-i")
    assert "-vsync" in local
    remote = keyframe_command("http://s/scene/1/stream", "/tmp/%06d.jpg", "k")
    assert remote[remote.index("-headers") + 1] == "ApiKey: k\r\n"


def test_summarise_probe_needs_usable_correct_top1():
    rec = _record([])
    faces = [
        _face({"pa": 0.3, "pb": 0.9}, top=[("pa", 0.3)], w=50, h=50),        # too small
        _face({"pa": 0.35, "pb": 0.9}, top=[("pa", 0.35)], yaw=70.0),        # profile
        _face({"pa": 0.4, "pb": 0.9}, top=[("zz", 0.3)]),                    # wrong
    ]
    ids = {"pa", "alt-pa"}
    p = ClassifyParams()
    assert summarise_probe(faces, rec, "pa", ids, p, 3)["recoverable"] is False
    faces.append(_face({"pa": 0.33, "pb": 0.9}, top=[("alt-pa", 0.33)], frame=9))
    v = summarise_probe(faces, rec, "pa", ids, p, 4)
    assert v["recoverable"] and v["best_distance"] == 0.33 and v["matching_usable_faces"] == 1


# ---------------------------------------------------------------- report

def _records():
    hit = _record([_face({"pa": 0.3, "pb": 0.9}, top=[("pa", 0.3)])], persons=[("pa", 5)])
    small = _record([_face({"pa": 0.9, "pb": 0.4}, top=[("zz", 0.3)], w=40, h=40)], persons=[("pa", 5)],
                    probe={"pb": {"recoverable": True}})
    small["scene_id"], small["resolution"], small["stratum"] = "2", "1080p", "2020+ / 1080p"
    err = {"scene_id": "3", "stratum": "2020+ / SD", "status": "error", "error": "boom"}
    return [hit, small, err]


def test_summarise_counts_reasons_stages_and_verdict():
    s = summarise(_records(), meta={"run_id": "r1"})
    assert s["scenes"] == {"total": 3, "ok": 2, "errors": 1}
    assert s["totals"]["expected"] == 4 and s["totals"]["found"] == 2
    assert s["reasons"]["small_or_profile"] == 1 and s["reasons"]["no_face_detected"] == 1
    assert s["stages"]["frame_choice"] == 1 and s["stages"]["unprobed"] == 1
    assert s["verdict"]["stage"] == "frame_choice" and "#140" in s["verdict"]["next_step"]
    assert set(s["by_resolution"]) == {"SD", "1080p"}
    assert s["by_resolution"]["SD"]["recall"] == 0.5
    assert s["errors"] == [{"scene_id": "3", "error": "boom"}]
    assert sum(1 for o in s["outcomes"] if not o["found"]) == sum(s["reasons"].values())
    json.dumps(s)  # serialisable


def test_render_markdown_has_the_sections():
    md = render_markdown(summarise(_records(), meta={"run_id": "r1", "db_version": "2026.02.18"}))
    for heading in ("## Verdict", "## Miss reasons", "### By resolution", "### By stratum",
                    "## Attribution sensitivity", "## Scenes that failed"):
        assert heading in md
    assert "2026.02.18" in md and "—" not in md


def test_summarise_empty():
    s = summarise([])
    assert s["verdict"]["stage"] is None and s["totals"]["recall"] is None
    render_markdown(s)


# ---------------------------------------------------------------- runner / CLI

def test_write_report_rescoring(tmp_path):
    from benchmark.failure.runner import write_json, write_report
    for r in _records():
        write_json(tmp_path / "raw" / f"{r['scene_id']}.json", r)
    report, summary = write_report(tmp_path, ClassifyParams(attribution_max_distance=0.95), {"run_id": "x"}, "-loose")
    assert report.name == "report-loose.md" and (tmp_path / "summary-loose.json").exists()
    assert summary["classify_params"]["attribution_max_distance"] == 0.95


def test_resolve_run_params_overrides_win():
    from benchmark.failure.runner import resolve_run_params
    assert resolve_run_params(num_frames=12).num_frames == 12
    assert resolve_run_params(num_frames=None).num_frames > 0


def test_cli_parses_every_command():
    from benchmark.failure.__main__ import build_parser
    p = build_parser()
    assert p.parse_args(["build-set", "--count", "10"]).count == 10
    a = p.parse_args(["run", "--limit", "3", "--no-probe", "--attribution", "0.6"])
    assert a.limit == 3 and a.no_probe and a.attribution == 0.6
    assert p.parse_args(["report", "/tmp/r"]).suffix == ""
    assert p.parse_args(["label", "--port", "9000"]).port == 9000
    assert p.parse_args(["detect-sample", "--resolutions", "SD,720p"]).resolutions == "SD,720p"
    a = p.parse_args(["detect-score", "--iou", "0.4", "--channel-order", "bgr"])
    assert a.iou == 0.4 and a.channel_order == "bgr"


# ---------------------------------------------------------------- shared pipeline step

def test_recognize_detected_faces_keeps_frame_indices_and_embeddings():
    from scene_matcher import recognize_detected_faces

    calls = []

    class Gen:
        def get_embeddings_batch(self, images):
            return [f"emb-{img}" for img in images]

    class Rec:
        generator = Gen()

        def recognize_face_v2(self, face, config, embedding=None):
            calls.append(embedding)
            return [f"match-{face.image}"], None, embedding

    faces = [(3, SimpleNamespace(image="a")), (-1, SimpleNamespace(image="b"))]
    out = recognize_detected_faces(Rec(), faces, match_config="cfg")
    assert [fi for fi, _ in out] == [3, -1]
    assert out[1][1].embedding == "emb-b" and out[1][1].matches == ["match-b"]
    assert calls == ["emb-a", "emb-b"]
    assert recognize_detected_faces(Rec(), [], "cfg") == []


# ---------------------------------------------------------------- run_scene end to end (fake models)

@pytest.mark.asyncio
async def test_run_scene_through_production_aggregation(monkeypatch):
    """Two frames, Pat's face in both, a tiny face in one: Pat is found via
    the real clustered_frequency_matching, the tiny face is recorded but
    never reaches the matcher."""
    import frame_extractor
    from benchmark.failure import pipeline
    from embeddings import FaceEmbedding
    from recognizer import PerformerMatch

    pat_fn, pat_af = _unit([1, 0, 0]), _unit([1, 0, 0])

    def det(w, image_tag):
        return SimpleNamespace(image=image_tag, bbox={"x": 0, "y": 0, "w": w, "h": w}, confidence=0.9, yaw=5.0)

    class Gen:
        def detect_faces(self, image, min_confidence):
            return [det(150, "pat"), det(20, "tiny")] if image == "f0" else [det(150, "pat")]

        def get_embeddings_batch(self, images):
            return [FaceEmbedding(facenet=pat_fn if i == "pat" else _unit([0, 1, 0]),
                                  arcface=pat_af if i == "pat" else _unit([0, 1, 0])) for i in images]

    class Rec:
        generator = Gen()

        def recognize_face_v2(self, face, config, embedding=None):
            if face.image != "pat":
                return [], None, embedding
            m = PerformerMatch(universal_id="stashdb.org:pa", stashdb_id="pa", name="Pat", country=None,
                               image_url=None, facenet_distance=0.2, arcface_distance=0.2, combined_score=0.2)
            return [m], None, embedding

    async def fake_extract(source, duration, api_key, config):
        frames = [SimpleNamespace(frame_index=i, timestamp_sec=10.0 * i, image=f"f{i}") for i in range(2)]
        return SimpleNamespace(frames=frames, errors=[])

    monkeypatch.setattr(frame_extractor, "extract_frames", fake_extract)
    gallery = PerformerGallery("pa", 5, name="Pat", stashbox_ids={"pa"},
                               facenet=np.stack([pat_fn]), arcface=np.stack([pat_af]))
    scene = _scene(1, expected=("pa",))
    params = pipeline.RunParams(num_frames=2, use_screenshot=False)
    record = await pipeline.run_scene(scene, Rec(), [gallery], params,
                                      stash_url="http://stash", api_key="k", screenshot_url=None)

    assert record["status"] == "ok" and record["frames_extracted"] == 2 and record["source"] == "stream"
    assert len(record["faces"]) == 3
    assert [f["eligible"] for f in record["faces"]].count(False) == 1
    pat_faces = [f for f in record["faces"] if f["eligible"]]
    assert all(f["dist"]["pa"] == pytest.approx(0.0, abs=1e-3) for f in pat_faces)
    assert record["persons"][0]["best"]["stashdb_id"] == "pa" and record["persons"][0]["frame_count"] == 2
    [outcome] = classify_scene(record)
    assert outcome.found
    json.dumps(record)

"""Run one benchmark scene through the production identification pipeline.

Steps match ``POST /identify/scene`` in the chosen matching mode: even frame sampling,
RetinaFace detection, size filter, batch embedding + index match
(``recognize_detected_faces``), the upscaled screenshot, then
the scene aggregation (``clustered_frequency_matching``, or
``gallery_cluster_matching`` when ``matching="gallery"``, #148). Two
deliberate differences:

* No tagged-performer boost: the plugin sends the scene's tagged performers,
  which here *are* the answer key.
* No body/tattoo re-ranking: this benchmark measures the face pipeline.

Faces under the size filter are embedded too, but only for diagnostics;
they never reach the matcher.
"""

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np

import face_config

TOP_N_RECORDED = 5


@dataclass(frozen=True)
class RunParams:
    """Pipeline settings for a run, resolved once and saved with the results."""
    num_frames: int = face_config.NUM_FRAMES
    start_offset_pct: float = face_config.START_OFFSET_PCT
    end_offset_pct: float = face_config.END_OFFSET_PCT
    min_face_size: int = face_config.MIN_FACE_SIZE
    min_face_confidence: float = face_config.MIN_FACE_CONFIDENCE
    max_distance: float = face_config.MAX_DISTANCE
    top_k: int = face_config.TOP_K
    cluster_threshold: float = face_config.CLUSTER_THRESHOLD
    facenet_weight: float = face_config.FACENET_WEIGHT
    arcface_weight: float = face_config.ARCFACE_WEIGHT
    detection_size: int = 640
    max_concurrent_extractions: int = 8
    use_screenshot: bool = True
    matching: str = "frequency"  # or "gallery" (#148)
    gallery_candidate_k: int = face_config.GALLERY_CANDIDATE_K
    gallery_candidate_max_distance: float = face_config.GALLERY_CANDIDATE_MAX_DISTANCE
    gallery_keep_fraction: float = face_config.GALLERY_KEEP_FRACTION
    gallery_min_margin: float = face_config.GALLERY_MIN_MARGIN

    def to_dict(self) -> dict:
        return asdict(self)


def face_record(frame_idx: int, timestamp: Optional[float], result, eligible: bool,
                galleries: list, params: RunParams) -> dict:
    """Flatten one recognised face into the JSON shape classify.py reads."""
    from benchmark.failure.gallery import distance_to

    face = result.face
    emb = result.embedding
    dist = {}
    for g in galleries:
        d = distance_to(emb.facenet, emb.arcface, g, params.facenet_weight, params.arcface_weight) if emb else None
        dist[g.stashdb_id] = None if d is None else round(d, 4)
    return {
        "frame": frame_idx,
        "timestamp": None if timestamp is None else round(timestamp, 2),
        "x": int(face.bbox["x"]), "y": int(face.bbox["y"]),
        "w": int(face.bbox["w"]), "h": int(face.bbox["h"]),
        "det_score": round(float(face.confidence), 3),
        "yaw": None if face.yaw is None else round(float(face.yaw), 1),
        "eligible": eligible,
        "top": [
            {"stashdb_id": m.stashdb_id, "name": m.name, "distance": round(float(m.combined_score), 4)}
            for m in result.matches[:TOP_N_RECORDED]
        ],
        "dist": dist,
    }


def person_record(person) -> dict:
    best = person.best_match
    return {
        "frame_count": person.frame_count,
        "best": None if best is None else {
            "stashdb_id": best.stashdb_id, "name": best.name,
            "distance": None if best.distance is None else round(float(best.distance), 4),
        },
    }


def expected_record(gallery, name: str) -> dict:
    return {
        "stashdb_id": gallery.stashdb_id,
        "name": name,
        "in_db": gallery.in_db,
        "performer_id": gallery.performer_id,
        "faces_in_db": gallery.face_count,
        "ids": sorted(gallery.stashbox_ids),
    }


def match_config_for(params: RunParams):
    from matching import MatchingConfig
    # Same as the endpoint; max_results is wider so the record shows top 5.
    max_results = max(params.top_k * 2, TOP_N_RECORDED)
    max_distance = params.max_distance
    if params.matching == "gallery":
        max_results = max(max_results, params.gallery_candidate_k)
        max_distance = max(max_distance, params.gallery_candidate_max_distance)
    return MatchingConfig(
        query_k=100,
        facenet_weight=params.facenet_weight,
        arcface_weight=params.arcface_weight,
        max_results=max_results,
        max_distance=max_distance,
    )


def recognise_frames(recognizer, frames: list, params: RunParams) -> list[tuple]:
    """Detect and recognise faces on (frame_index, timestamp, image) triples.

    Returns (frame_index, timestamp, RecognitionResult, eligible) tuples.
    """
    from scene_matcher import recognize_detected_faces

    detected, meta = [], []
    for frame_idx, ts, image in frames:
        for face in recognizer.generator.detect_faces(image, min_confidence=params.min_face_confidence):
            eligible = face.bbox["w"] >= params.min_face_size and face.bbox["h"] >= params.min_face_size
            detected.append((frame_idx, face))
            meta.append((ts, eligible))
    results = recognize_detected_faces(recognizer, detected, match_config_for(params))
    return [(fi, ts, r, el) for (fi, r), (ts, el) in zip(results, meta)]


def production_persons(recognizer, recognised: list[tuple], params: RunParams) -> list:
    """Scene aggregation exactly as the endpoint runs it (frequency or gallery mode)."""
    from scene_matcher import clustered_frequency_matching, gallery_cluster_matching

    eligible = [(fi, r) for fi, _ts, r, el in recognised if el]
    if params.matching == "gallery":
        return gallery_cluster_matching(
            eligible,
            recognizer,
            cluster_threshold=params.cluster_threshold,
            top_k=params.top_k,
            max_distance=params.max_distance,
            min_margin=params.gallery_min_margin,
            keep_fraction=params.gallery_keep_fraction,
            candidate_k=params.gallery_candidate_k,
            facenet_weight=params.facenet_weight,
            arcface_weight=params.arcface_weight,
            scene_performer_stashdb_ids=None,
        )
    if params.matching != "frequency":
        raise ValueError(f"unknown matching mode {params.matching!r}")
    return clustered_frequency_matching(
        eligible,
        recognizer,
        cluster_threshold=params.cluster_threshold,
        top_k=params.top_k,
        max_distance=params.max_distance,
        scene_performer_stashdb_ids=None,
    )


async def run_scene(scene, recognizer, galleries, params: RunParams, *,
                    stash_url: str, api_key: str, screenshot_url: Optional[str],
                    embeddings_path: Path | None = None) -> dict:
    """Process one BenchmarkScene and return its raw JSON record."""
    from benchmark.failure.source import resolve_video_source
    from frame_extractor import FrameExtractionConfig, extract_frames

    t0 = time.time()
    record = {
        "scene_id": scene.scene_id, "title": scene.title, "stratum": scene.stratum,
        "era": scene.era, "resolution": scene.resolution,
        "width": scene.width, "height": scene.height, "duration": scene.duration,
        "expected": [expected_record(g, e.name) for g, e in zip(galleries, scene.expected)],
        "status": "ok", "error": None,
    }
    source = resolve_video_source(scene.path, scene.scene_id, stash_url, api_key)
    record["source"] = "local" if source.local else "stream"

    config = FrameExtractionConfig(
        num_frames=params.num_frames,
        start_offset_pct=params.start_offset_pct,
        end_offset_pct=params.end_offset_pct,
        min_face_size=params.min_face_size,
        min_face_confidence=params.min_face_confidence,
        max_concurrent_extractions=params.max_concurrent_extractions,
    )
    extraction = await extract_frames(source.input, scene.duration, source.api_key, config)
    record["frames_requested"] = params.num_frames
    record["frames_extracted"] = len(extraction.frames)
    if not extraction.frames:
        record["status"] = "error"
        record["error"] = "no frames extracted" + (f": {extraction.errors[0]}" if extraction.errors else "")
        record["elapsed_sec"] = round(time.time() - t0, 1)
        return record

    frames = [(f.frame_index, f.timestamp_sec, f.image) for f in extraction.frames]
    recognised = recognise_frames(recognizer, frames, params)

    if params.use_screenshot and screenshot_url:
        from identification_router import fetch_screenshot_image
        try:
            image = await fetch_screenshot_image(screenshot_url, api_key, scene.width or 1920)
        except Exception as e:  # production logs and carries on, so do we
            image = None
            record["screenshot_error"] = str(e)
        if image is not None:
            recognised += recognise_frames(recognizer, [(-1, None, image)], params)

    if embeddings_path is not None:
        save_embeddings(embeddings_path, recognised)
    aggregate(record, recognizer, recognised, galleries, params)
    record["elapsed_sec"] = round(time.time() - t0, 1)
    return record


def aggregate(record: dict, recognizer, recognised: list[tuple], galleries: list, params: RunParams) -> None:
    """Run scene aggregation and fill the record's faces, persons and timing."""
    import identification_router  # the matchers import it lazily; keep that out of the timing

    t0 = time.perf_counter()
    persons = production_persons(recognizer, recognised, params)
    record["aggregation_ms"] = round((time.perf_counter() - t0) * 1000, 1)
    record["faces"] = [face_record(fi, ts, r, el, galleries, params) for fi, ts, r, el in recognised]
    record["persons"] = [person_record(p) for p in persons]


def save_embeddings(path: Path, recognised: list[tuple]) -> None:
    """Store each recognised face's embeddings, so matching can be replayed without video.

    Rows follow the record's ``faces`` order. Faces without an embedding get
    zero rows; ``load_embeddings`` drops them.
    """
    n = len(recognised)
    dim = next((len(r.embedding.facenet) for _, _, r, _ in recognised if r.embedding is not None), 512)
    facenet = np.zeros((n, dim), dtype=np.float32)
    arcface = np.zeros((n, dim), dtype=np.float32)
    has = np.zeros(n, dtype=bool)
    for i, (_, _, r, _) in enumerate(recognised):
        if r.embedding is not None:
            facenet[i], arcface[i], has[i] = r.embedding.facenet, r.embedding.arcface, True
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(
        tmp,
        frame=np.array([fi for fi, _, _, _ in recognised], dtype=np.int32),
        timestamp=np.array([np.nan if ts is None else ts for _, ts, _, _ in recognised], dtype=np.float64),
        bbox=np.array([[r.face.bbox[k] for k in ("x", "y", "w", "h")] for _, _, r, _ in recognised],
                      dtype=np.int32).reshape(n, 4),
        det_score=np.array([r.face.confidence for _, _, r, _ in recognised], dtype=np.float32),
        yaw=np.array([np.nan if r.face.yaw is None else r.face.yaw for _, _, r, _ in recognised], dtype=np.float32),
        eligible=np.array([el for _, _, _, el in recognised], dtype=bool),
        has_embedding=has,
        facenet=facenet,
        arcface=arcface,
    )
    tmp.replace(path)


def load_embeddings(path: Path) -> list[tuple]:
    """Read ``save_embeddings`` output as (frame, timestamp, DetectedFace, FaceEmbedding, eligible).

    The DetectedFace has no image: replay only needs the box, score and yaw.
    """
    from embeddings import DetectedFace, FaceEmbedding

    z = np.load(path)
    out = []
    for i in range(len(z["frame"])):
        if not z["has_embedding"][i]:
            continue
        x, y, w, h = (int(v) for v in z["bbox"][i])
        ts, yaw = float(z["timestamp"][i]), float(z["yaw"][i])
        face = DetectedFace(image=None, bbox={"x": x, "y": y, "w": w, "h": h},
                            confidence=float(z["det_score"][i]), yaw=None if np.isnan(yaw) else yaw)
        out.append((int(z["frame"][i]), None if np.isnan(ts) else ts, face,
                    FaceEmbedding(facenet=z["facenet"][i], arcface=z["arcface"][i]), bool(z["eligible"][i])))
    return out

"""Sample frames from benchmark scenes and pre-draw proposal boxes.

Frames are spread evenly over the scenes of the chosen resolutions, at
seeded uniform-random timestamps between 5% and 95% of each scene. Each
frame gets proposals from a permissive detector (low threshold, large input)
so the labeler mostly accepts or deletes boxes rather than drawing them.
"""

import json
import math
import os
import random
from datetime import UTC, datetime
from pathlib import Path

from . import FORMAT_VERSION, FRAMES_DIR, PROPOSALS_FILE

PROPOSAL_DET_THRESH = 0.2
PROPOSAL_DET_SIZE = 1280
START_PCT = 0.05
END_PCT = 0.95


def _default_extract_frame():
    from frame_extractor import FrameExtractionConfig, extract_frame_sync

    config = FrameExtractionConfig()

    def extract(input, timestamp, api_key):
        return extract_frame_sync(input, timestamp, api_key, config)

    return extract


def build_proposer():
    """High-recall proposal detector: buffalo_sc RetinaFace at 1280, thresh 0.2.

    Returns ``propose(rgb_image) -> [{x, y, w, h, score}]``.
    """
    import onnxruntime as ort
    from insightface.app import FaceAnalysis

    cuda = "CUDAExecutionProvider" in ort.get_available_providers()
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if cuda else ["CPUExecutionProvider"]
    app = FaceAnalysis(
        name="buffalo_sc",
        root=os.environ.get("INSIGHTFACE_HOME", "~/.insightface"),
        allowed_modules=["detection"],
        providers=providers,
    )
    app.prepare(
        ctx_id=0 if cuda else -1,
        det_thresh=PROPOSAL_DET_THRESH,
        det_size=(PROPOSAL_DET_SIZE, PROPOSAL_DET_SIZE),
    )

    def propose(image):
        # InsightFace detectors expect BGR (cv2 order); frames are RGB.
        faces = app.get(image[:, :, ::-1].copy())
        return [_box_from_xyxy(f.bbox, float(f.det_score), image) for f in faces]

    return propose


def _box_from_xyxy(bbox, score, image) -> dict:
    """Clamp an x1,y1,x2,y2 box to the image and return integer x,y,w,h."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in bbox[:4])
    x1, x2 = max(0.0, min(x1, width)), max(0.0, min(x2, width))
    y1, y2 = max(0.0, min(y1, height)), max(0.0, min(y2, height))
    x, y = round(x1), round(y1)
    return {"x": x, "y": y, "w": max(0, round(x2) - x), "h": max(0, round(y2) - y),
            "score": round(score, 4)}


def _plan(scenes, count: int, rng: random.Random) -> list[tuple]:
    """(scene, timestamp) pairs: ceil(count/n) per scene, in scene order.

    Every scene's timestamps are drawn up front so the plan for a seed does
    not depend on which extractions later fail.
    """
    if not scenes or count <= 0:
        return []
    per_scene = math.ceil(count / len(scenes))
    plan = []
    for scene in scenes:
        lo, hi = scene.duration * START_PCT, scene.duration * END_PCT
        for _ in range(per_scene):
            plan.append((scene, round(rng.uniform(lo, hi), 3)))
    return plan


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, path)


def sample_frames(
    bench_set,
    out_dir: Path,
    *,
    count: int = 300,
    seed: int = 139,
    resolutions: tuple[str, ...] = ("SD",),
    stash_url: str = "",
    api_key: str = "",
    extract_frame=None,
    propose=None,
    resolve_source=None,
    force: bool = False,
) -> Path:
    """Extract up to ``count`` frames with proposals; return proposals.json.

    Refuses to overwrite an existing proposals.json unless ``force``; never
    touches labels.json. Failed extractions are skipped, and later scenes'
    spare slots make up for them where they can.
    """
    from PIL import Image

    out_dir = Path(out_dir)
    proposals_path = out_dir / PROPOSALS_FILE
    if proposals_path.exists() and not force:
        raise FileExistsError(
            f"{proposals_path} already exists; pass force to re-sample "
            "(labels.json is kept either way)"
        )

    scenes = [s for s in bench_set.scenes if s.resolution in resolutions]
    if not scenes:
        raise ValueError(f"no benchmark scenes with resolution in {resolutions}")

    if resolve_source is None:
        from ..source import resolve_video_source as resolve_source
    if extract_frame is None:
        extract_frame = _default_extract_frame()
    if propose is None:
        propose = build_proposer()

    frames_dir = out_dir / FRAMES_DIR
    frames_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    plan = _plan(scenes, count, rng)
    print(f"Sampling {count} frames from {len(scenes)} scenes "
          f"({', '.join(resolutions)}) into {out_dir}", flush=True)

    sources = {}
    frames = []
    seen = set()
    skipped = 0
    for scene, timestamp in plan:
        if len(frames) >= count:
            break
        name = f"{scene.scene_id}_{round(timestamp * 1000)}.jpg"
        if name in seen:
            continue
        if scene.scene_id not in sources:
            sources[scene.scene_id] = resolve_source(scene.path, scene.scene_id, stash_url, api_key)
        source = sources[scene.scene_id]
        # extract_frame_sync logs and returns None on any ffmpeg failure.
        image = extract_frame(source.input, timestamp, source.api_key)
        if image is None:
            skipped += 1
            print(f"  scene {scene.scene_id} @ {timestamp:.1f}s: no frame, skipped", flush=True)
            continue

        Image.fromarray(image).save(frames_dir / name, quality=95)
        proposals = propose(image)
        seen.add(name)
        height, width = image.shape[:2]
        frames.append({
            "file": name,
            "scene_id": scene.scene_id,
            "timestamp": timestamp,
            "width": int(width),
            "height": int(height),
            "resolution": scene.resolution,
            "proposals": proposals,
        })
        print(f"  [{len(frames)}/{count}] scene {scene.scene_id} @ {timestamp:.1f}s: "
              f"{len(proposals)} proposals", flush=True)

    _write_json(proposals_path, {
        "version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "proposal_detector": {"det_thresh": PROPOSAL_DET_THRESH, "det_size": PROPOSAL_DET_SIZE},
        "frames": frames,
    })
    print(f"Wrote {len(frames)} frames ({skipped} skipped) to {proposals_path}", flush=True)
    return proposals_path

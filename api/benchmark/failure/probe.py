"""Frame-choice probe: could better frames have found a missed performer?

For performers missed as ``no_face_detected`` or ``small_or_profile``, decode
the whole scene's keyframes (cheap: no inter-frame decoding) and look for a
moment where their face is usable (large and frontal) *and* the matcher's
top-1 for it is them within max_distance. If such a frame exists, a smarter
frame picker would have had something to work with; if not, the loss sits
with detection, or the performer's face is never usable in this scene.
"""

import re
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np

from benchmark.failure.classify import ClassifyParams, attributed_faces, is_usable_face

PTS_RE = re.compile(r"pts_time:\s*([0-9.]+)")


def keyframe_command(input_path: str, out_pattern: str, api_key: Optional[str]) -> list[str]:
    cmd = ["ffmpeg", "-hide_banner", "-nostdin", "-loglevel", "info", "-skip_frame", "nokey"]
    if api_key:
        cmd += ["-headers", f"ApiKey: {api_key}\r\n"]
    # -vsync 0 rather than -fps_mode: the image ships ffmpeg 4.4.
    cmd += ["-i", input_path, "-an", "-sn", "-vsync", "0",
            "-vf", "showinfo", "-q:v", "3", "-f", "image2", out_pattern]
    return cmd


def parse_pts(stderr: str) -> list[float]:
    """showinfo prints one line per output frame, in order."""
    return [float(m.group(1)) for line in stderr.splitlines()
            if "Parsed_showinfo" in line and (m := PTS_RE.search(line))]


def evenly_pick(n: int, k: int) -> list[int]:
    """k indices spread over range(n), all of them when n <= k."""
    if n <= k:
        return list(range(n))
    return sorted({int(round(i * (n - 1) / (k - 1))) for i in range(k)}) if k > 1 else [n // 2]


async def decode_keyframes(input_path: str, api_key: Optional[str], max_frames: int,
                           timeout: float = 1800.0) -> list[tuple[float, np.ndarray]]:
    """All keyframes (timestamp, RGB image), thinned evenly to max_frames."""
    from PIL import Image
    from media_exec import media_job_command, run_command

    with tempfile.TemporaryDirectory(prefix="ss-probe-") as tmp:
        cmd = media_job_command(keyframe_command(input_path, f"{tmp}/%06d.jpg", api_key))
        result = await run_command(cmd, timeout=timeout)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg keyframe decode failed: {result.stderr.decode(errors='replace')[-300:]}")
        files = sorted(Path(tmp).glob("*.jpg"))
        pts = parse_pts(result.stderr.decode(errors="replace"))
        out = []
        for i in evenly_pick(len(files), max_frames):
            ts = pts[i] if i < len(pts) else None
            out.append((ts, np.array(Image.open(files[i]).convert("RGB"))))
        return out


def summarise_probe(faces: list[dict], scene: dict, stashdb_id: str, ids: set[str],
                    p: ClassifyParams, keyframes: int) -> dict:
    """Probe verdict for one performer from the keyframe face records."""
    view = {"faces": faces, "expected": scene["expected"]}
    mine = attributed_faces(view, stashdb_id, p)
    usable = [f for f in mine if is_usable_face(f, p)]
    hits = [f for f in usable if f["top"] and f["top"][0]["stashdb_id"] in ids
            and f["top"][0]["distance"] <= p.max_distance]
    best = min(hits, key=lambda f: f["top"][0]["distance"]) if hits else None
    return {
        "recoverable": bool(hits),
        "keyframes": keyframes,
        "attributed_faces": len(mine),
        "usable_faces": len(usable),
        "matching_usable_faces": len(hits),
        "best_timestamp": best["timestamp"] if best else None,
        "best_distance": best["top"][0]["distance"] if best else None,
    }


async def probe_scene(scene: dict, needs_probe: list[str], recognizer, galleries: dict,
                      run_params, classify_params: ClassifyParams, *, source,
                      max_frames: int = 400) -> dict:
    """Return {stashdb_id: verdict} for the performers that need probing."""
    from benchmark.failure.pipeline import face_record, recognise_frames

    keyframes = await decode_keyframes(source.input, source.api_key, max_frames)
    frames = [(i, ts, img) for i, (ts, img) in enumerate(keyframes)]
    recognised = recognise_frames(recognizer, frames, run_params)
    gallery_list = [galleries[e["stashdb_id"]] for e in scene["expected"]]
    faces = [face_record(fi, ts, r, el, gallery_list, run_params) for fi, ts, r, el in recognised]
    out = {}
    for exp in scene["expected"]:
        sid = exp["stashdb_id"]
        if sid in needs_probe:
            ids = set(exp.get("ids") or []) | {sid}
            out[sid] = summarise_probe(faces, scene, sid, ids, classify_params, len(keyframes))
    return out

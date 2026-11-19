"""Score the production face detector against the hand-labeled frames.

Only reviewed frames count. Predictions are matched one-to-one to labeled
boxes, greedily by IoU (highest first, ties broken by label then prediction
order). Recall is broken down by the labeled face's short side, and "usable
recall" only credits a match when the prediction also passes the production
size filter, i.e. the face would reach recognition.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import face_config

from . import FRAMES_DIR, SCORES_DIR, load_proposals
from .labeler import load_labels

SIZE_BUCKETS = ("<40", "40-79", "80+")
MISSED_IN_MARKDOWN = 50


def iou(a: dict, b: dict) -> float:
    ix = max(0.0, min(a["x"] + a["w"], b["x"] + b["w"]) - max(a["x"], b["x"]))
    iy = max(0.0, min(a["y"] + a["h"], b["y"] + b["h"]) - max(a["y"], b["y"]))
    inter = ix * iy
    union = max(0, a["w"]) * max(0, a["h"]) + max(0, b["w"]) * max(0, b["h"]) - inter
    return inter / union if union > 0 else 0.0


def greedy_match(labels: list[dict], preds: list[dict], threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one (label index, prediction index, IoU) pairs with IoU >= threshold."""
    candidates = []
    for li, label in enumerate(labels):
        for pi, pred in enumerate(preds):
            v = iou(label, pred)
            if v >= threshold and v > 0:
                candidates.append((-v, li, pi))
    candidates.sort()
    used_l, used_p, pairs = set(), set(), []
    for neg, li, pi in candidates:
        if li in used_l or pi in used_p:
            continue
        used_l.add(li)
        used_p.add(pi)
        pairs.append((li, pi, -neg))
    return pairs


def size_bucket(b: dict) -> str:
    short = min(b["w"], b["h"])
    if short < 40:
        return "<40"
    if short < 80:
        return "40-79"
    return "80+"


def _clamp(b: dict, width: int, height: int) -> dict:
    x1, y1 = max(0, min(b["x"], width)), max(0, min(b["y"], height))
    x2, y2 = max(0, min(b["x"] + b["w"], width)), max(0, min(b["y"] + b["h"], height))
    return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}


def build_production_detector(det_size: int, min_confidence: float):
    """The sidecar's detector at an explicit det_size.

    Uses FaceEmbeddingGenerator.detect_faces unchanged, but builds and
    prepares its analyzer here, the same way embeddings.py's face_analyzer
    property does (buffalo_sc, INSIGHTFACE_HOME root, same providers and
    ctx_id). The property would read det_size from settings instead, so the
    analyzer is injected into ``_face_analyzer`` before first use.

    Returns ``detect(rgb_image) -> [{x, y, w, h, score}]``; detect_faces
    hands the analyzer BGR.
    """
    import os

    from insightface.app import FaceAnalysis

    from embeddings import FaceEmbeddingGenerator

    generator = FaceEmbeddingGenerator()
    analyzer = FaceAnalysis(
        name="buffalo_sc",
        root=os.environ.get("INSIGHTFACE_HOME", "~/.insightface"),
        providers=generator._ort_providers(),
    )
    analyzer.prepare(
        ctx_id=0 if generator.device == "cuda" else -1,
        det_size=(det_size, det_size),
    )
    generator._face_analyzer = analyzer

    def detect(image):
        return [
            {**face.bbox, "score": float(face.confidence)}
            for face in generator.detect_faces(image, min_confidence=min_confidence)
        ]

    return detect


def _ratio(num: int, den: int):
    return num / den if den else None


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def score(
    out_dir: Path,
    *,
    detect=None,
    det_size: int = 640,
    min_confidence: float = face_config.MIN_FACE_CONFIDENCE,
    min_face_size: int = face_config.MIN_FACE_SIZE,
    iou_threshold: float = 0.5,
    channel_order: str = "bgr",
) -> Path:
    """Run the detector on reviewed frames; write report.md + results.json.

    ``channel_order`` is what RetinaFace is fed. ``detect`` takes RGB frames
    and, like production's detect_faces since #147, hands the detector BGR,
    which is what InsightFace expects (blobFromImage with swapRB=True). So
    "bgr" is production; "rgb" pre-swaps the frame so the detector sees RGB,
    which is how production ran before #147.

    Returns the report.md path under ``out_dir/scores/<UTC timestamp>/``.
    """
    import numpy as np
    from PIL import Image

    out_dir = Path(out_dir)
    frames = load_proposals(out_dir)["frames"]
    labels = load_labels(out_dir)["frames"]
    reviewed = [f for f in frames if (labels.get(f["file"]) or {}).get("reviewed")]
    if not reviewed:
        raise ValueError(f"no reviewed frames in {out_dir / 'labels.json'}; label some first")

    if channel_order not in ("rgb", "bgr"):
        raise ValueError(f"channel_order must be 'rgb' or 'bgr', not {channel_order!r}")
    if detect is None:
        detect = build_production_detector(det_size, min_confidence)
    if channel_order == "rgb":
        production_detect = detect

        def detect(image):
            return production_detect(np.ascontiguousarray(image[:, :, ::-1]))

    n_labels = n_preds = n_matched = n_usable = 0
    by_size = {k: {"labels": 0, "matched": 0, "usable": 0} for k in SIZE_BUCKETS}
    missed, false_positives, per_frame = [], [], []

    for i, frame in enumerate(reviewed, 1):
        with Image.open(out_dir / FRAMES_DIR / frame["file"]) as im:
            image = np.asarray(im.convert("RGB"))
        height, width = image.shape[:2]
        gt = [{k: b[k] for k in ("x", "y", "w", "h")} for b in labels[frame["file"]]["boxes"]]
        raw = [p for p in detect(image) if p.get("score", 1.0) >= min_confidence]
        clamped = [_clamp(p, width, height) for p in raw]
        pairs = greedy_match(gt, clamped, iou_threshold)
        matched_by_label = {li: pi for li, pi, _ in pairs}
        matched_preds = {pi for _, pi, _ in pairs}

        frame_usable = 0
        for li, b in enumerate(gt):
            bucket = size_bucket(b)
            by_size[bucket]["labels"] += 1
            pi = matched_by_label.get(li)
            if pi is None:
                missed.append({"file": frame["file"], "box": b, "size": bucket})
                continue
            by_size[bucket]["matched"] += 1
            # The production size filter sees the detector's own box.
            if raw[pi]["w"] >= min_face_size and raw[pi]["h"] >= min_face_size:
                by_size[bucket]["usable"] += 1
                frame_usable += 1
        for pi, p in enumerate(raw):
            if pi not in matched_preds:
                false_positives.append({"file": frame["file"], "box": p})

        n_labels += len(gt)
        n_preds += len(raw)
        n_matched += len(pairs)
        n_usable += frame_usable
        per_frame.append({"file": frame["file"], "labels": len(gt), "predictions": len(raw),
                          "matched": len(pairs), "usable": frame_usable})
        print(f"  [{i}/{len(reviewed)}] {frame['file']}: {len(pairs)}/{len(gt)} found, "
              f"{len(raw)} predicted", flush=True)

    for v in by_size.values():
        v["recall"] = _ratio(v["matched"], v["labels"])
        v["usable_recall"] = _ratio(v["usable"], v["labels"])

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_dir / SCORES_DIR / stamp
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_dir = out_dir / SCORES_DIR / f"{stamp}-{suffix}"
    run_dir.mkdir(parents=True)

    results = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "det_size": det_size,
        "min_confidence": min_confidence,
        "min_face_size": min_face_size,
        "iou_threshold": iou_threshold,
        "channel_order": channel_order,
        "reviewed_frames": len(reviewed),
        "labels": n_labels,
        "predictions": n_preds,
        "matched": n_matched,
        "usable": n_usable,
        "recall": _ratio(n_matched, n_labels),
        "precision": _ratio(n_matched, n_preds),
        "usable_recall": _ratio(n_usable, n_labels),
        "recall_by_size": by_size,
        "missed": missed,
        "false_positives": false_positives,
        "frames": per_frame,
    }
    (run_dir / "results.json").write_text(json.dumps(results, indent=2))
    report = run_dir / "report.md"
    report.write_text(_markdown(results))
    print(f"Recall {_pct(results['recall'])}, precision {_pct(results['precision'])}, "
          f"usable recall {_pct(results['usable_recall'])}; report: {report}", flush=True)
    return report


def _markdown(r: dict) -> str:
    lines = [
        "# Face detector recall on labeled frames",
        "",
        f"- Detector det_size: {r['det_size']}, fed {r['channel_order'].upper()} frames",
        f"- Min confidence: {r['min_confidence']}",
        f"- IoU threshold: {r['iou_threshold']}",
        f"- Production size filter: w and h >= {r['min_face_size']} px",
        f"- Reviewed frames: {r['reviewed_frames']}",
        f"- Labeled faces: {r['labels']}; predictions: {r['predictions']}; matched: {r['matched']}",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Recall | {_pct(r['recall'])} ({r['matched']}/{r['labels']}) |",
        f"| Precision | {_pct(r['precision'])} ({r['matched']}/{r['predictions']}) |",
        f"| Usable recall | {_pct(r['usable_recall'])} ({r['usable']}/{r['labels']}) |",
        "",
        "Usable recall credits a labeled face only when the matching detection also passes",
        "the production size filter, so the face would go on to recognition.",
        "",
        "## Recall by labeled face size (short side)",
        "",
        "| Size | Labeled | Matched | Recall | Usable recall |",
        "|---|---|---|---|---|",
    ]
    for name, v in r["recall_by_size"].items():
        lines.append(f"| {name} | {v['labels']} | {v['matched']} | {_pct(v['recall'])} "
                     f"| {_pct(v['usable_recall'])} |")
    missed = r["missed"]
    lines += ["", f"## Missed labeled faces ({len(missed)})", ""]
    if missed:
        lines += ["| File | x | y | w | h | Size |", "|---|---|---|---|---|---|"]
        for m in missed[:MISSED_IN_MARKDOWN]:
            b = m["box"]
            lines.append(f"| {m['file']} | {b['x']} | {b['y']} | {b['w']} | {b['h']} | {m['size']} |")
        if len(missed) > MISSED_IN_MARKDOWN:
            lines += ["", f"...and {len(missed) - MISSED_IN_MARKDOWN} more; the full list is in results.json."]
    else:
        lines.append("None.")
    lines.append("")
    return "\n".join(lines)

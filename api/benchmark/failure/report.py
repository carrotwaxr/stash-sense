"""Turn raw scene records into the summary JSON and the markdown report."""

from collections import Counter
from dataclasses import replace
from statistics import median
from typing import Optional

from benchmark.failure.classify import (
    REASONS,
    STAGES,
    ClassifyParams,
    classify_scene,
    false_positives,
)
from benchmark.failure.scene_set import ERAS, RESOLUTIONS

SENSITIVITY_THRESHOLDS = (0.55, 0.65, 0.75)

# Stages that can win the verdict; "unprobed" is a gap in the data, not a cause.
VERDICT_STAGES = ["frame_choice", "matching", "detection_or_absent", "size_or_pose_throughout", "db_coverage"]

NEXT_STEP = {
    "frame_choice": "Frame choice loses the most: pick up #140 (pose-guided frame selection).",
    "matching": "Matching loses the most: look at scene aggregation and the embedding models before frame selection.",
    "detection_or_absent": "Detection (or the performer's face never being on screen) loses the most: check the detection test set's recall before anything else.",
    "size_or_pose_throughout": "Performers whose face is never large and frontal in the scene lose the most: frame choice can't fix these; consider pose-robust embeddings.",
    "db_coverage": "Missing performers in the face DB lose the most: widen the trainer's coverage.",
}


def _rate(num: int, den: int) -> Optional[float]:
    return None if den == 0 else num / den


def _group(outcomes: list, key) -> dict:
    groups: dict[str, list] = {}
    for o in outcomes:
        groups.setdefault(key(o), []).append(o)
    return groups


def _row(outcomes: list, scene_ids: set[str]) -> dict:
    found = sum(o.found for o in outcomes)
    reasons = Counter(o.reason for o in outcomes if not o.found)
    return {
        "scenes": len(scene_ids),
        "expected": len(outcomes),
        "found": found,
        "recall": _rate(found, len(outcomes)),
        "reasons": {r: reasons.get(r, 0) for r in REASONS},
    }


def summarise(records: list[dict], params: ClassifyParams = ClassifyParams(), meta: Optional[dict] = None) -> dict:
    ok = [r for r in records if r.get("status") == "ok"]
    errors = [{"scene_id": r["scene_id"], "error": r.get("error")} for r in records if r.get("status") != "ok"]

    outcomes = [o for r in ok for o in classify_scene(r, params)]
    fps = [fp for r in ok for fp in false_positives(r, params)]
    found = sum(o.found for o in outcomes)
    misses = [o for o in outcomes if not o.found]

    by_stratum = {}
    for key, group in sorted(_group(outcomes, lambda o: o.stratum).items(),
                             key=lambda kv: (ERAS.index(kv[1][0].era), RESOLUTIONS.index(kv[1][0].resolution))):
        by_stratum[key] = _row(group, {o.scene_id for o in group})
    by_era = {k: _row(v, {o.scene_id for o in v}) for k, v in
              sorted(_group(outcomes, lambda o: o.era).items(), key=lambda kv: ERAS.index(kv[0]))}
    by_resolution = {k: _row(v, {o.scene_id for o in v}) for k, v in
                     sorted(_group(outcomes, lambda o: o.resolution).items(), key=lambda kv: RESOLUTIONS.index(kv[0]))}

    stages = Counter(o.stage for o in misses)
    stage_counts = {s: stages.get(s, 0) for s in STAGES}
    ranked = sorted(VERDICT_STAGES, key=lambda s: stage_counts[s], reverse=True)
    top = ranked[0] if misses and stage_counts[ranked[0]] > 0 else None

    wrong = [o for o in misses if o.reason == "wrong_match"]
    wrong_stats = {
        "count": len(wrong),
        "median_distance_to_right_answer": median([o.best_distance for o in wrong]) if wrong else None,
        "median_winner_distance": median([o.winner["distance"] for o in wrong if o.winner]) if any(o.winner for o in wrong) else None,
        "right_person_in_top5": sum(1 for o in wrong if o.rank_on_best_face is not None),
        "no_match_at_all": sum(1 for o in wrong if o.winner is None),
    }

    sensitivity = {}
    for t in SENSITIVITY_THRESHOLDS:
        p = replace(params, attribution_max_distance=t)
        alt = Counter(o.reason for r in ok for o in classify_scene(r, p) if not o.found)
        sensitivity[str(t)] = {r: alt.get(r, 0) for r in REASONS}

    return {
        "meta": meta or {},
        "classify_params": params.to_dict(),
        "scenes": {"total": len(records), "ok": len(ok), "errors": len(errors)},
        "totals": {
            "expected": len(outcomes),
            "found": found,
            "recall": _rate(found, len(outcomes)),
            "false_positives": len(fps),
            "precision": _rate(found, found + len(fps)),
            "shown_only_as_single_frame": sum(1 for o in misses if o.shown_as_single_frame),
        },
        "reasons": {r: sum(1 for o in misses if o.reason == r) for r in REASONS},
        "stages": stage_counts,
        "verdict": {"stage": top, "next_step": NEXT_STEP.get(top) if top else None,
                    "ranking": [(s, stage_counts[s]) for s in ranked]},
        "wrong_match": wrong_stats,
        "attribution_sensitivity": sensitivity,
        "by_stratum": by_stratum,
        "by_era": by_era,
        "by_resolution": by_resolution,
        "errors": errors,
        "outcomes": [o.to_dict() for o in outcomes],
    }


def _pct(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def _num(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _table(title: str, rows: dict) -> list[str]:
    head = ["", f"### {title}", "",
            "| | Scenes | Expected | Found | Recall | " + " | ".join(REASONS) + " |",
            "|---|---:|---:|---:|---:|" + "---:|" * len(REASONS)]
    body = [f"| {k} | {v['scenes']} | {v['expected']} | {v['found']} | {_pct(v['recall'])} | "
            + " | ".join(str(v["reasons"][r]) for r in REASONS) + " |" for k, v in rows.items()]
    return head + body


def render_markdown(summary: dict) -> str:
    m, t, cp = summary["meta"], summary["totals"], summary["classify_params"]
    misses = t["expected"] - t["found"]
    lines = [
        "# Face recognition failure analysis",
        "",
        f"Run `{m.get('run_id', '?')}`, face DB `{m.get('db_version', '?')}`, "
        f"{summary['scenes']['ok']} of {summary['scenes']['total']} scenes processed.",
        "",
        "## Verdict",
        "",
        summary["verdict"]["next_step"] or "No misses to attribute.",
        "",
        "| Stage | Misses | Share of misses |",
        "|---|---:|---:|",
    ]
    for stage in STAGES:
        n = summary["stages"][stage]
        lines.append(f"| {stage} | {n} | {_pct(_rate(n, misses))} |")
    lines += [
        "",
        "`frame_choice`: missed in the sampled frames, but a keyframe elsewhere in the scene has their face large, "
        "frontal and correctly matched. `detection_or_absent` / `size_or_pose_throughout`: no such keyframe exists. "
        "`unprobed`: the probe didn't run for that scene (stream source, `--no-probe`, or probe error).",
        "",
        "## Totals",
        "",
        f"- Expected performers: {t['expected']}; found: {t['found']} (recall {_pct(t['recall'])})",
        f"- False positives (multi-frame persons matched to someone not in the scene): {t['false_positives']} "
        f"(precision {_pct(t['precision'])})",
        f"- Missed but shown as a single-frame person: {t['shown_only_as_single_frame']}",
        "",
        "## Miss reasons",
        "",
        "| Reason | Misses | Share |",
        "|---|---:|---:|",
    ]
    for r in REASONS:
        n = summary["reasons"][r]
        lines.append(f"| {r} | {n} | {_pct(_rate(n, misses))} |")

    w = summary["wrong_match"]
    lines += [
        "",
        f"Wrong matches: median distance to the right performer {_num(w['median_distance_to_right_answer'])}, "
        f"median distance of the winner {_num(w['median_winner_distance'])}. "
        f"Right person anywhere in that face's top 5: {w['right_person_in_top5']}; "
        f"face had no match under max_distance at all: {w['no_match_at_all']}.",
    ]
    lines += _table("By resolution", summary["by_resolution"])
    lines += _table("By era", summary["by_era"])
    lines += _table("By stratum (era / resolution)", summary["by_stratum"])

    lines += ["", "## Attribution sensitivity", "",
              f"Reasons depend on attributing faces to the expected performer (default threshold "
              f"{cp['attribution_max_distance']}). Counts at other thresholds:", "",
              "| Threshold | " + " | ".join(REASONS) + " |", "|---:|" + "---:|" * len(REASONS)]
    for th, counts in summary["attribution_sensitivity"].items():
        lines.append(f"| {th} | " + " | ".join(str(counts[r]) for r in REASONS) + " |")

    lines += ["", "## Settings", "", "```", f"pipeline: {m.get('run_params')}", f"classify: {cp}", "```"]
    if summary["errors"]:
        lines += ["", "## Scenes that failed", ""]
        lines += [f"- {e['scene_id']}: {e['error']}" for e in summary["errors"]]
    return "\n".join(lines) + "\n"

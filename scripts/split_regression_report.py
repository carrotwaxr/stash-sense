"""Plan every title of the #131 regression set leave-one-out and render the report against the owner's labels.

    python scripts/split_regression_report.py          # print the report
    python scripts/split_regression_report.py --write  # write docs/research/split-regression.md
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from splits import detect, profiles, signals  # noqa: E402
from splits.planner import plan_file  # noqa: E402

FIXTURE = ROOT / "api" / "tests" / "fixtures" / "split_regression"
REPORT = ROOT / "docs" / "research" / "split-regression.md"
LEVELS = ("high", "medium", "low")
EXACT_S = 2.0
NEAR_S = 30.0
BLACK_START_S = 0.5
MIN_PROFILE_BOUNDARIES = 3


def load(path: Path = FIXTURE) -> list[dict]:
    return [json.loads(line) for line in (path / "index.jsonl").read_text().splitlines() if line.strip()]


def _signal_dict(rec: dict) -> dict:
    return signals.to_dict(signals.decode((FIXTURE / f"{rec['id']}.npz").read_bytes()))


def _nearest(t: float, cuts: list[float]) -> float | None:
    return min((abs(t - c) for c in cuts), default=None)


def score(records: list[dict]) -> list[dict]:
    """One dict per record, sorted by id: label, plan, distances, black-start flags and the profile used."""
    dicts = {r["id"]: _signal_dict(r) for r in records}
    runs = {i: detect.dark_runs(d["fr"]) for i, d in dicts.items()}
    feats = {}
    for r in records:
        d = dicts[r["id"]]
        feats[r["id"]] = [detect.features_at(d, t, runs=runs[r["id"]]) for t in r["cuts"]]

    by_series = defaultdict(list)
    for r in records:
        if r["cuts"]:
            by_series[r["series"]].append(r)

    scored = []
    for rec in sorted(records, key=lambda r: r["id"]):
        others = [o for o in by_series[rec["series"]] if o["id"] != rec["id"]]
        profile = None
        if others:
            profile = profiles.build_profile(
                [
                    {
                        "key": o["id"],
                        "total": o["keep_until"] or o["duration"],
                        "cuts": o["cuts"],
                        "feat": feats[o["id"]],
                    }
                    for o in others
                ]
            )
            if profile["n_boundaries"] < MIN_PROFILE_BOUNDARIES:
                profile = None
        plan = plan_file(dicts[rec["id"]], rec["duration"], profile=profile)
        label = list(rec["cuts"])
        got = [c["t"] for c in plan.cuts]
        starts = [r[0] for r in runs[rec["id"]]]
        at_black = [any(abs(s - t) <= BLACK_START_S for s in starts) for t in label]
        dist = [_nearest(t, got) for t in label]
        exact = len(got) == len(label) and all(abs(a - b) <= EXACT_S for a, b in zip(sorted(got), sorted(label)))
        scored.append(
            {
                "id": rec["id"],
                "series": rec["series"],
                "leave_whole": rec["leave_whole"],
                "label": label,
                "label_keep_until": rec["keep_until"],
                "plan": got,
                "keep_until": plan.keep_until,
                "confidence": plan.confidence,
                "reasons": list(plan.reasons),
                "profile_used": plan.profile_used,
                "relaxed": plan.relaxed,
                "at_black": at_black,
                "dist": dist,
                "exact": exact,
            }
        )
    return scored


def _cell(text: str) -> str:
    return str(text).replace("|", "\\|")


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else "n/a"


def _ts(ts: list[float], at_black: list[bool] | None = None) -> str:
    if at_black is None:
        return ", ".join(f"{t:.1f}" for t in ts) or "none"
    return ", ".join(f"{t:.1f}{'' if b else '*'}" for t, b in zip(ts, at_black)) or "none"


def _missed(r: dict) -> list[int]:
    return [i for i, d in enumerate(r["dist"]) if d is None or d > EXACT_S]


def _miss_verdict(r: dict) -> str:
    miss = _missed(r)
    if not miss:
        return "no label missed (the plan has a different cut count)"
    off = [i for i in miss if not r["at_black"][i]]
    if len(off) == len(miss):
        return "yes: the missed label is not at a black start" if len(miss) == 1 else f"yes: all {len(miss)} missed labels are not at a black start"
    if off:
        return f"partly: {len(off)} of {len(miss)} missed labels are not at a black start"
    return "no: the missed label is at a black start" if len(miss) == 1 else f"no: all {len(miss)} missed labels are at a black start"


def _is_miss(r: dict) -> bool:
    return not r["leave_whole"] and not r["exact"]


def render(scored: list[dict]) -> str:
    labelled = [r for r in scored if not r["leave_whole"]]
    whole = [r for r in scored if r["leave_whole"]]
    n_cuts = sum(len(r["label"]) for r in labelled)
    n_black = sum(sum(r["at_black"]) for r in labelled)
    lines = [
        "# Split regression (#131)",
        "",
        "This is the split planner (`api/splits/planner.py`) run on every title of the regression set and scored "
        "against the owner's labelled cuts. The set is `api/tests/fixtures/split_regression/`: `index.jsonl` holds "
        "one record per title (anonymized id, series label, duration, the labelled cuts, whether the owner left it "
        "whole, and the labelled `keep_until`), and `<id>.npz` the extracted video and audio signals. The report "
        "holds only fixture ids, series labels and numbers.",
        "",
        "Each title is planned leave-one-out: its profile is built (`profiles.build_profile`) from the other titles "
        "of its series that have labelled cuts, using `features_at` on their labelled cuts, with each title's total "
        "taken as its labelled `keep_until` when it has one and its duration otherwise. A series with no other "
        "labelled title, or a profile with fewer than 3 boundaries, gives no profile (the planner then needs every "
        "cut to be unmistakable for `high`).",
        "",
        "Terms:",
        "",
        f"- **exact**: the plan has as many cuts as the label and every cut is within {EXACT_S:g} s of its label.",
        f"- **recall**: the share of labelled cuts that have a plan cut within {EXACT_S:g} s (or {NEAR_S:g} s). "
        "Median distance is from each labelled cut to the nearest plan cut, over the labelled cuts of the plans in "
        "that row that have one.",
        f"- **at a black start**: a dark run (`detect.dark_runs`) starts within {BLACK_START_S:g} s of the labelled "
        "cut. Some labels were never snapped to a black start: titles with static transitions (owner scrub times, "
        "which land a few seconds after the first static burst) and a few labels from before the cut-at-black-start "
        "rule (they sit mid-black). In the tables a label that is not at a black start is marked `*`. This column is "
        "for reading only; the pass/fail rule does not use it.",
        "- **relaxed**: the plan used the fallback that accepts shorter blacks that carry silence "
        "(`SplitPlan.relaxed`).",
        "",
        f"Titles: {len(scored)} ({len(labelled)} labelled, {len(whole)} left whole). Labelled cuts: {n_cuts}, of "
        f"which {_rate(n_black, n_cuts)} are at a black start.",
        "",
        "## Totals",
        "",
        "Labelled titles, by plan confidence.",
        "",
        "| Confidence | Plans | Exact | Recall 2 s | Recall 30 s | Median distance (s) | With profile | Relaxed |",
        "|---|---|---|---|---|---|---|---|",
    ]

    def totals_row(name: str, rows: list[dict]) -> str:
        dists = [d for r in rows for d in r["dist"]]
        n = len(dists)
        near = [d for d in dists if d is not None]
        med = f"{statistics.median(near):.2f}" if near else "n/a"
        return (
            f"| {name} | {len(rows)} | {sum(r['exact'] for r in rows)} | "
            f"{_rate(sum(1 for d in dists if d is not None and d <= EXACT_S), n)} | "
            f"{_rate(sum(1 for d in dists if d is not None and d <= NEAR_S), n)} | {med} | "
            f"{sum(r['profile_used'] for r in rows)} | {sum(r['relaxed'] for r in rows)} |"
        )

    for lv in LEVELS:
        lines.append(totals_row(lv, [r for r in labelled if r["confidence"] == lv]))
    lines.append(totals_row("all", labelled))

    lines += [
        "",
        "Labelled cuts by whether they sit at a black start (all confidences).",
        "",
        "| Labels | Cuts | Recall 2 s | Recall 30 s |",
        "|---|---|---|---|",
    ]
    for name, want in (("at a black start", True), ("not at a black start", False)):
        ds = [d for r in labelled for d, b in zip(r["dist"], r["at_black"]) if b == want]
        lines.append(
            f"| {name} | {len(ds)} | {_rate(sum(1 for d in ds if d is not None and d <= EXACT_S), len(ds))} | "
            f"{_rate(sum(1 for d in ds if d is not None and d <= NEAR_S), len(ds))} |"
        )

    with_ku = [r for r in labelled if r["label_keep_until"]]
    ku_ok = [r for r in with_ku if r["keep_until"] is not None and abs(r["keep_until"] - r["label_keep_until"]) <= NEAR_S]
    spurious = [r for r in labelled if not r["label_keep_until"] and r["keep_until"] is not None]
    lines += [
        "",
        f"Menu tail (`keep_until`): {_rate(len(ku_ok), len(with_ku))} labelled `keep_until` values are found within "
        f"{NEAR_S:g} s; {len(spurious)} titles without a label got one.",
        "",
        "Titles left whole, by plan confidence.",
        "",
        "| Confidence | Plans | Plans with cuts | Cuts planned |",
        "|---|---|---|---|",
    ]
    for lv in LEVELS:
        rows = [r for r in whole if r["confidence"] == lv]
        lines.append(f"| {lv} | {len(rows)} | {sum(1 for r in rows if r['plan'])} | {sum(len(r['plan']) for r in rows)} |")

    cols = "| Title | Series | Label cuts | Plan cuts | Profile | Relaxed | Misses not at a black start | Reasons |"
    sep = "|---|---|---|---|---|---|---|---|"

    def row(r: dict) -> str:
        cells = [
            r["id"],
            r["series"],
            _ts(r["label"], r["at_black"]),
            _ts(r["plan"]),
            "yes" if r["profile_used"] else "no",
            "yes" if r["relaxed"] else "no",
            _miss_verdict(r),
            "; ".join(r["reasons"]),
        ]
        return "| " + " | ".join(_cell(c) for c in cells) + " |"

    high_miss = [r for r in labelled if r["confidence"] == "high" and _is_miss(r)]
    lines += [
        "",
        f"## High plans that miss ({len(high_miss)} of {sum(1 for r in labelled if r['confidence'] == 'high')})",
        "",
        "Times are seconds; `*` marks a label that is not at a black start. \"Misses not at a black start\" says "
        "whether the labels with no plan cut within "
        f"{EXACT_S:g} s are all, partly or not at all of that kind.",
        "",
        cols,
        sep,
    ]
    lines += [row(r) for r in high_miss]

    high_whole = [r for r in whole if r["confidence"] == "high"]
    lines += [
        "",
        f"## Left-whole titles with a high plan ({len(high_whole)})",
        "",
        "| Title | Series | Plan cuts | Profile | Relaxed | Reasons |",
        "|---|---|---|---|---|---|",
    ]
    for r in high_whole:
        cells = [
            r["id"],
            r["series"],
            _ts(r["plan"]),
            "yes" if r["profile_used"] else "no",
            "yes" if r["relaxed"] else "no",
            "; ".join(r["reasons"]),
        ]
        lines.append("| " + " | ".join(_cell(c) for c in cells) + " |")

    lines += [
        "",
        "## Medium and low plans that miss",
        "",
        "Counts only.",
        "",
        "| Confidence | Plans that miss | Of those, every missed label is not at a black start |",
        "|---|---|---|",
    ]
    for lv in ("medium", "low"):
        miss = [r for r in labelled if r["confidence"] == lv and _is_miss(r)]
        off = [r for r in miss if _missed(r) and all(not r["at_black"][i] for i in _missed(r))]
        lines.append(f"| {lv} | {len(miss)} | {len(off)} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help=f"write {REPORT.relative_to(ROOT)}")
    args = ap.parse_args()
    text = render(score(load()))
    if args.write:
        REPORT.write_text(text)
        print(f"wrote {REPORT}")
    else:
        print(text, end="")


if __name__ == "__main__":
    main()

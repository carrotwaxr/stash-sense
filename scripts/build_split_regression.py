#!/usr/bin/env python
"""Build the anonymized #131 split-point regression set from the owner's labelled titles.

Reads the curator's October labels, plan files and cached per-frame signals (all private, outside this repo),
keeps single-file labelled titles whose signals cover at least 90% of the title, and writes:

  index.jsonl   one record per title: id, series, duration, cuts, leave_whole, keep_from, keep_until, ref_cuts
  <id>.npz      the quantized signals (splits.signals.encode)

The id is sha256(key)[:12] and the series is an opaque label (s001, s002, ...). No title, path or URL is
written, so the output is safe to commit. ref_cuts are the cuts the original curator planner finds on the
quantized signals, so a port can be checked against them exactly.

    python scripts/build_split_regression.py --out api/tests/fixtures/split_regression

--curator-log and --tools default to the curator checkout's splits directory in the home directory.
"""

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from splits import signals as sg  # noqa: E402

MIN_COVERAGE = 0.9


def load_labels(path: Path) -> dict:
    """Last line per key wins."""
    labels = {}
    for line in path.read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            labels[str(d["key"])] = d
    return labels


def number(x, nd=2):
    return round(float(x), nd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--curator-log",
        type=Path,
        default=Path.home() / "code/stash-curator/curator-log/splits-2026-10",
    )
    ap.add_argument("--tools", type=Path, default=None)
    args = ap.parse_args()
    S = args.curator_log
    T = args.tools or S / "tools"
    sys.path.insert(0, str(T))
    import series_key as skmod  # noqa: E402
    import series_profiles as sp  # noqa: E402

    labels = load_labels(S / "labels.jsonl")
    dropped = collections.Counter()
    kept = []  # (key, label, plan, signals)
    for key, lab in labels.items():
        if lab.get("deleted"):
            dropped["deleted"] += 1
            continue
        if lab.get("learn") is False:
            dropped["learn false"] += 1
            continue
        plan_path = S / f"{key}.json"
        if not plan_path.exists():
            dropped["no plan file"] += 1
            continue
        plan = json.loads(plan_path.read_text())
        # Leave-whole labels carry no `scenes`: the plan file's decide.
        scenes = lab.get("scenes") or plan.get("scenes") or []
        if len(scenes) > 1 or len(plan.get("scenes") or []) > 1:
            dropped["multi-part"] += 1
            continue
        sig_path = T / "signals" / f"{key}.json"
        if not sig_path.exists():
            dropped["no signals"] += 1
            continue
        raw = json.loads(sig_path.read_text())
        duration = float(plan["duration"])
        covered = sum(b - a for a, b in raw.get("covered", []))
        if covered < MIN_COVERAGE * duration:
            dropped["coverage < 90%"] += 1
            continue
        kept.append((key, lab, plan, sg.from_curator(raw, duration)))

    series_keys = sorted(
        {k for k in (skmod.series_key(p["title"]) for _, _, p, _ in kept) if k}
    )
    series_label = {k: f"s{i:03d}" for i, k in enumerate(series_keys, 1)}
    next_label = len(series_keys) + 1

    args.out.mkdir(parents=True, exist_ok=True)
    records = []
    for key, lab, plan, s in sorted(
        kept, key=lambda x: hashlib.sha256(x[0].encode()).hexdigest()
    ):
        sk = skmod.series_key(plan["title"])
        if sk:
            series = series_label[sk]
        else:  # an empty key is its own series
            series = f"s{next_label:03d}"
            next_label += 1
        tid = hashlib.sha256(key.encode()).hexdigest()[:12]
        blob = sg.encode(s)
        (args.out / f"{tid}.npz").write_bytes(blob)
        d = sg.to_dict(sg.decode(blob))
        total = float(plan["duration"])
        ref = sp.plan_cuts(
            sp.refine_candidates(d, sp.candidates(d, 0.17, min_gap=15, min_t=0)),
            None,
            total,
        )[0]
        leave_whole = bool(lab.get("leave_whole"))
        records.append(
            {
                "id": tid,
                "series": series,
                "duration": number(total, 1),
                "cuts": []
                if leave_whole
                else [number(b) for b in lab.get("boundaries", [])],
                "leave_whole": leave_whole,
                "keep_from": lab.get("keep_from"),
                "keep_until": lab.get("keep_until"),
                "ref_cuts": [number(c["t"], 1) for c in ref],
            }
        )
    with open(args.out / "index.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    total_bytes = sum(p.stat().st_size for p in args.out.iterdir())
    print(f"titles read: {len(labels)}")
    for reason, n in sorted(dropped.items()):
        print(f"dropped ({reason}): {n}")
    print(f"written: {len(records)}")
    print(f"cuts: {sum(len(r['cuts']) for r in records)}")
    print(f"leave-whole: {sum(r['leave_whole'] for r in records)}")
    print(f"total bytes: {total_bytes}")


if __name__ == "__main__":
    main()

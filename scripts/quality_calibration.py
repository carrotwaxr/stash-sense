#!/usr/bin/env python
"""Calibration report for the #133 quality flags, from the owner's reviewed `scene_quality` cards.

Reads only `scene_quality` recommendations from the sidecar's database (opened read-only) and prints, per flag:

  flagged_reviewed  flagged cards that were accepted or dismissed (pending cards are ignored)
  yes / no          the owner's pick for that flag; a dismissed card, or a pick missing from the
                    accepted option, counts as no
  precision         yes / flagged_reviewed
  control_reviewed  reviewed control cards (a random draw of unflagged scenes)
  control_yes       control cards where the owner raised this flag
  recall_estimate   p*F / (p*F + q*U): p = precision, F = flagged_reviewed, q = control_yes /
                    control_reviewed (the miss rate among unflagged scenes), U = --unflagged

The accepted option is the review service's `resolution_details["option"]`, e.g. {"broken": "yes", "rip": "no"}.

    python scripts/quality_calibration.py /path/to/stash_sense.db --unflagged 25000 [--json]
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path

FLAGS = ("broken", "rip", "below_studio", "format")
REVIEWED = ("resolved", "dismissed")


def _load(db_path: Path) -> list[dict]:
    uri = f"file:{Path(db_path).resolve()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            "SELECT status, details, resolution_details FROM recommendations "
            "WHERE type = 'scene_quality' AND status IN (?, ?) "
            "AND COALESCE(resolution_action, '') != 'stale'",  # the job retired it; the owner never answered
            REVIEWED,
        ).fetchall()
    finally:
        conn.close()
    recs = []
    for status, details, resolution in rows:
        details = json.loads(details) if details else {}
        resolution = json.loads(resolution) if resolution else {}
        option = resolution.get("option") if isinstance(resolution, dict) else None
        recs.append({
            "status": status,
            "flags": details.get("flags") or {},
            "control": bool(details.get("control")),
            "option": option if isinstance(option, dict) else {},
        })
    return recs


def _says_yes(rec: dict, flag: str) -> bool:
    return rec["status"] == "resolved" and rec["option"].get(flag) == "yes"


def report(db_path, unflagged_total: int) -> dict:
    recs = _load(Path(db_path))
    out = {}
    for flag in FLAGS:
        flagged = [r for r in recs if not r["control"] and flag in r["flags"]]
        controls = [r for r in recs if r["control"]]
        yes = sum(_says_yes(r, flag) for r in flagged)
        control_yes = sum(_says_yes(r, flag) for r in controls)
        precision = yes / len(flagged) if flagged else None
        recall = None
        if precision is not None and controls:
            caught = precision * len(flagged)
            missed = (control_yes / len(controls)) * unflagged_total
            if caught + missed > 0:
                recall = caught / (caught + missed)
        out[flag] = {
            "flagged_reviewed": len(flagged),
            "yes": yes,
            "no": len(flagged) - yes,
            "precision": precision,
            "control_reviewed": len(controls),
            "control_yes": control_yes,
            "recall_estimate": recall,
        }
    return out


def _fmt(v) -> str:
    return "-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


def print_table(result: dict) -> None:
    cols = ["flagged_reviewed", "yes", "no", "precision", "control_reviewed", "control_yes", "recall_estimate"]
    print(f"{'flag':<14}" + "".join(f"{c:>18}" for c in cols))
    for flag, row in result.items():
        print(f"{flag:<14}" + "".join(f"{_fmt(row[c]):>18}" for c in cols))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("db", type=Path, help="the sidecar's stash_sense.db")
    ap.add_argument("--unflagged", type=int, required=True, help="number of scenes the job did not flag (U)")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    args = ap.parse_args(argv)
    if not args.db.is_file():
        print(f"no such database: {args.db}", file=sys.stderr)
        return 2
    result = report(args.db, args.unflagged)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print_table(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Score the #130 keeper suggestion against the owner's recorded picks and render the mismatch report.

    python scripts/dupes_keeper_report.py          # print the report
    python scripts/dupes_keeper_report.py --write  # write docs/research/dupes-keeper-regression.md
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from duplicates.keeper import FileFeatures, suggest  # noqa: E402

FIXTURE = ROOT / "api" / "tests" / "fixtures" / "dupes_keeper_regression.jsonl"
REPORT = ROOT / "docs" / "research" / "dupes-keeper-regression.md"
OUTCOMES = ("agree", "superset", "tie", "disagree")
KINDS = ("duplicate", "contained")
SHORT_SIDE_NOTE = "short side by duration differs from cover"


def load(path: Path = FIXTURE) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _features(key: str, side: dict) -> FileFeatures:
    return FileFeatures(
        key=key,
        codec=side["codec"],
        width=side["width"],
        height=side["height"],
        bit_rate=side["bit_rate"],
        fps=side["fps"],
        duration=side["duration"],
        size=side["size"],
        ext=side["ext"],
        upscale_name=side["upscale_name"],
        upscale_path=side["upscale_path"],
    )


def describe(side: dict) -> str:
    rate = f"{side['bit_rate'] / 1e6:.1f} Mbps" if side["bit_rate"] else "unknown bitrate"
    return (
        f"{side['codec'] or '?'} {side['width']}x{side['height']} {rate} "
        f"{side['fps']:g} fps {side['duration'] / 60:.1f} min {side['ext']}"
    )


def score(records: list[dict]) -> list[dict]:
    """One dict per record: id, kind, a, b (one-line features), human, suggestion, reason, outcome."""
    scored = []
    for rec in sorted(records, key=lambda r: (r["kind"], r["id"])):
        files = [_features("a", rec["a"]), _features("b", rec["b"])]
        s = suggest(files, kind=rec["kind"])
        reason = s.reason
        if rec["kind"] == "duplicate":
            pick = rec["pick"]
            other = "b" if pick == "a" else "a"
            if s.keep == [pick]:
                outcome = "agree"
            elif pick in s.keep and len(s.keep) == 2:
                outcome = "superset"
            elif "tie" in s.flags and s.keep == [other]:
                outcome = "tie"
            else:
                outcome = "disagree"
            human = f"keep {pick.upper()}"
            suggestion = "keep " + " + ".join(k.upper() for k in s.keep)
        else:
            short = "a" if rec["cover_a"] >= rec["cover_b"] else "b"
            by_duration = "a" if rec["a"]["duration"] < rec["b"]["duration"] else "b"
            if rec["a"]["duration"] == rec["b"]["duration"]:
                by_duration = "a"  # suggest() sorts by (duration, key)
            if by_duration != short:
                reason = f"{reason} ({SHORT_SIDE_NOTE})"
            keeps_short = short in s.keep
            human_keeps_short = rec["pick_raw"] == "keep_short"
            outcome = "agree" if keeps_short == human_keeps_short else "disagree"
            human = f"keep short ({short.upper()}) too" if human_keeps_short else f"delete short ({short.upper()})"
            suggestion = f"keep short ({short.upper()}) too" if keeps_short else f"delete short ({short.upper()})"
        scored.append(
            {
                "id": rec["id"],
                "kind": rec["kind"],
                "a": describe(rec["a"]),
                "b": describe(rec["b"]),
                "human": human,
                "suggestion": suggestion,
                "reason": reason,
                "outcome": outcome,
                "keep": list(s.keep),
                "flags": list(s.flags),
            }
        )
    return scored


def _cell(text: str) -> str:
    return text.replace("|", "\\|")


def _rate(num: int, den: int) -> str:
    return f"{num}/{den} ({100 * num / den:.1f}%)" if den else "n/a"


def render(scored: list[dict]) -> str:
    counts = {k: Counter(r["outcome"] for r in scored if r["kind"] == k) for k in KINDS}
    lines = [
        "# Duplicate keeper regression (#130)",
        "",
        "This is the keeper suggestion (`api/duplicates/keeper.py`) scored against the owner's recorded picks. The set is "
        "`api/tests/fixtures/dupes_keeper_regression.jsonl`: one record per decided pair, anonymized to codec, "
        "resolution, bit rate, frame rate, duration, size, container and the two upscale flags (no names, paths or "
        "hashes). The two files of each record are keyed `a` and `b`.",
        "",
        "Outcomes:",
        "",
        "- **agree**: duplicate pairs: the suggestion keeps exactly the file the owner kept. Contained pairs: the "
        "suggestion keeps the short file only when the owner did (`keep_short`), and drops it when the owner "
        "chose `long_better`.",
        "- **superset**: duplicate pairs only: the suggestion keeps the owner's file plus the other one (an upscale "
        "pair or a longer cut). Nothing the owner wanted is deleted, so this counts as agreement.",
        "- **tie**: the suggestion flagged the pair as practically equal and kept the other file. Either is right, so "
        "ties are left out of the agreement rate.",
        "- **disagree**: everything else. Every one is listed below with the suggestion's reason.",
        "",
        "For contained pairs the short side is the one with the larger cover. The keeper decides by duration; where "
        f"the two differ the reason is marked \"{SHORT_SIDE_NOTE}\".",
        "",
        "Agreement is (agree + superset) / (records - tie).",
        "",
        "## Totals",
        "",
        "| Kind | Records | agree | superset | tie | disagree | Agreement |",
        "|---|---|---|---|---|---|---|",
    ]
    total = Counter()
    for k in KINDS:
        c = counts[k]
        n = sum(c.values())
        total.update(c)
        lines.append(
            f"| {k} | {n} | {c['agree']} | {c['superset']} | {c['tie']} | {c['disagree']} | "
            f"{_rate(c['agree'] + c['superset'], n - c['tie'])} |"
        )
    n = sum(total.values())
    lines.append(
        f"| all | {n} | {total['agree']} | {total['superset']} | {total['tie']} | {total['disagree']} | "
        f"{_rate(total['agree'] + total['superset'], n - total['tie'])} |"
    )

    cols = "| Record | Kind | Side A | Side B | Human | Suggestion | Reason |"
    sep = "|---|---|---|---|---|---|---|"

    def row(r: dict) -> str:
        cells = [r["id"], r["kind"], r["a"], r["b"], r["human"], r["suggestion"], r["reason"]]
        return "| " + " | ".join(_cell(c) for c in cells) + " |"

    disagree = [r for r in scored if r["outcome"] == "disagree"]
    lines += ["", f"## Disagreements ({len(disagree)})", "", cols, sep]
    lines += [row(r) for r in disagree]

    superset = [r for r in scored if r["outcome"] == "superset"]
    lines += [
        "",
        f"## Superset ({len(superset)})",
        "",
        "The suggestion keeps the owner's file and one more.",
        "",
        cols,
        sep,
    ]
    lines += [row(r) for r in superset]

    lines += ["", "## Ties", "", f"{total['tie']} records were practically equal files.", ""]
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

#!/usr/bin/env python
"""Build `split_seed.json`, the #131 series-profile seed, from the owner's labelled titles (local use only).

Reads the curator's October applied splits, owner labels, per-scene info and cached per-frame signals (all
private, outside this repo) and writes the shape `SplitStore.import_seed` reads:

  {"aliases": [[pattern, name], ...],
   "labels": [{"series", "key", "total", "cuts", "feats", "source"}, ...]}

Only single-scene titles are used. `source` is "owner" or "applied" (an owner label wins over an applied
split of the same title). `feats` is `detect.features_at` on the signals at each cut, computed on the same
quantized values production sees, or null where the +-6 s window around the cut is not covered.

The output holds real titles' series keys, so it is never written into the repo: the script refuses an
`--out` inside it. Put it under /tmp and copy it into the sidecar's data dir by hand.

    python scripts/build_split_seed.py --out /tmp/split-seed/split_seed.json

--tools and --splits default to the curator checkout's splits directories under the home directory.
"""

import argparse
import collections
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "api"))

from splits import detect, profiles, signals as sg  # noqa: E402

_CURATOR_SPLITS = Path("code") / "stash-curator" / "curator-log" / "splits-2026-10"
WINDOW_S = 6.0
MIN_BOUNDARIES = 3


def default_splits() -> Path:
    return Path.home() / _CURATOR_SPLITS


def default_tools() -> Path:
    return default_splits() / "tools"


def load_aliases(tools: Path) -> list[list[str]]:
    """`ALIASES` from the curator's series_key.py (regex over series keys, name), as [pattern, name] pairs."""
    spec = importlib.util.spec_from_file_location(
        "curator_series_key", tools / "series_key.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [[pat, name] for pat, name in mod.ALIASES]


def _jsonl(path: Path):
    for line in path.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def labelled(splits: Path) -> dict:
    """{key: {"scenes", "title", "cuts", "source"}}: applied results, then owner labels over them.

    Same selection as the curator's `build_series_profiles.labelled`: a result whose status starts with
    "applied" and whose plan line carries the scenes and cut points; owner labels with boundaries, except
    `learn: false`.
    """
    last, plan = {}, {}
    for r in _jsonl(splits / "applied.jsonl"):
        if r.get("state") == "result":
            last[r["key"]] = r
        elif "scenes" in r:
            plan[r["key"]] = r
    out = {}
    for k, r in last.items():
        if r["status"].startswith("applied") and k in plan:
            title = json.loads((splits / f"{k}.json").read_text())["title"]
            out[k] = {
                "scenes": plan[k]["scenes"],
                "title": title,
                "cuts": plan[k]["at"],
                "source": "applied",
            }
    for lb in _jsonl(splits / "labels.jsonl"):
        if lb.get("learn") is False or not lb.get("boundaries") or not lb.get("scenes"):
            continue
        title = json.loads((splits / f"{lb['key']}.json").read_text())["title"]
        out[lb["key"]] = {
            "scenes": lb["scenes"],
            "title": title,
            "cuts": lb["boundaries"],
            "source": "owner",
        }
    return out


def quantized(tools: Path, scene_id, duration: float) -> dict:
    """The scene's cached signals as production will see them: encode, decode, back to the curator dict."""
    f = tools / "signals" / f"{scene_id}.json"
    raw = (
        json.loads(f.read_text())
        if f.exists()
        else {"fr": [], "au": [], "sil": [], "covered": []}
    )
    return sg.to_dict(sg.decode(sg.encode(sg.from_curator(raw, duration))))


def feats_for(d: dict, cuts) -> list:
    out = []
    for t in cuts:
        covered = any(a <= t - WINDOW_S and t + WINDOW_S <= b for a, b in d["covered"])
        out.append(detect.features_at(d, t) if covered else None)
    return out


def build(tools: Path, splits: Path) -> dict:
    aliases = load_aliases(tools)
    info = json.loads((tools / "scene_info.json").read_text())
    labels = []
    for key, lb in sorted(labelled(splits).items()):
        if len(lb["scenes"]) != 1:
            continue
        series = profiles.series_key(lb["title"], aliases)
        if not series:
            continue  # untitled files are not a series
        sid = lb["scenes"][0]
        total = float(info[str(sid)]["files"][0]["duration"])
        d = quantized(tools, sid, total)
        labels.append(
            {
                "series": series,
                "key": str(sid),
                "total": total,
                "cuts": [float(c) for c in lb["cuts"]],
                "feats": feats_for(d, lb["cuts"]),
                "source": lb["source"],
            }
        )
    return {"aliases": aliases, "labels": labels}


def inside(path: Path, root: Path) -> bool:
    p = path.resolve()
    return p == root or root in p.parents


def repo_roots() -> list[Path]:
    """This checkout and, for a worktree, the main checkout that holds it."""
    roots = [ROOT]
    r = subprocess.run(
        [
            "git",
            "-C",
            str(ROOT),
            "rev-parse",
            "--path-format=absolute",
            "--git-common-dir",
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode == 0 and r.stdout.strip():
        roots.append(Path(r.stdout.strip()).parent)
    return roots


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="split_seed.json to write (outside the repo)",
    )
    ap.add_argument("--tools", type=Path, default=default_tools())
    ap.add_argument("--splits", type=Path, default=default_splits())
    args = ap.parse_args(argv)

    if any(inside(args.out, r) for r in repo_roots()):
        ap.error(
            "--out is inside the repo; the seed names real titles, write it under /tmp"
        )
    seed = build(args.tools, args.splits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(seed, separators=(",", ":")))

    by_source = collections.Counter(lb["source"] for lb in seed["labels"])
    per_series = collections.Counter()
    for lb in seed["labels"]:
        per_series[lb["series"]] += len(lb["cuts"])
    n3 = sum(1 for n in per_series.values() if n >= MIN_BOUNDARIES)
    with_feats = sum(f is not None for lb in seed["labels"] for f in lb["feats"])
    n_cuts = sum(len(lb["cuts"]) for lb in seed["labels"])
    print(
        f"labels: {len(seed['labels'])} ({', '.join(f'{k} {v}' for k, v in sorted(by_source.items()))})"
    )
    print(f"aliases: {len(seed['aliases'])}")
    print(f"series: {len(per_series)}; with at least {MIN_BOUNDARIES} boundaries: {n3}")
    print(f"cuts: {n_cuts}; with features: {with_feats}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python
"""Build the anonymized #130 keeper regression set from the owner's recorded duplicate/contained picks.

Reads the curator's decisions.jsonl (append-only, last record per id wins) and the matcher pair files, orients each
card by the decision's a_path (the matcher's A file, never the id's sorted hash order), and looks up each side's file
attributes by oshash in live Stash, then in two Stash database backups. Only attributes survive: no path, title or
name is written, the output is safe to commit.

    python scripts/build_dupes_regression.py --out api/tests/fixtures/dupes_keeper_regression.jsonl

Live Stash comes from STASH_URL and STASH_API_KEY in the environment (or --env-file).
"""

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import urllib.request
from pathlib import Path

# Same pattern as the curator's build_review.py: an AI upscale shows in the file name.
UPNAME_RE = re.compile(
    r"upscal|_\d{3,4}p_\d{8}|tvai|topaz|starlight|proteus|\bai[ -]?enhanced",
    re.IGNORECASE,
)
PICKS = ("keep_a", "keep_b", "keep_short", "long_better")

CURATOR = Path.home() / "code/stash-curator"
DEFAULT_REVIEW = CURATOR / "curator-log/review"
DEFAULT_PAIRS = ("pairs.jsonl", "pairs-library-1028.jsonl", "pairs-library-1030.jsonl")
DEFAULT_BACKUPS = (
    "/tmp/dupes-regression/stash-go.sqlite.85.pre-v0320.bak",  # 11-16
    "/tmp/dupes-regression/stash-go.sqlite.85.20260928_193215",  # 09-28
)
DEFAULT_ENV = Path.home() / "code/stash-sense/api/.env"

FILE_FIELDS = "id size basename path width height video_codec bit_rate frame_rate duration fingerprints { type value }"


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_env(path):
    env = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            m = re.match(r"\s*([A-Z_]+)\s*=\s*(.*?)\s*$", line)
            if m and not line.lstrip().startswith("#"):
                env[m.group(1)] = m.group(2).strip("'\"")
    return env


def last_decisions(path):
    last = {}
    for rec in read_jsonl(path):
        last[rec["id"]] = rec
    return last


def pair_key(h1, h2):
    return "|".join(sorted((h1, h2)))


def load_pairs(paths):
    """{sorted oshash key: [pair records]}"""
    idx = {}
    for p in paths:
        if not Path(p).exists():
            continue
        for rec in read_jsonl(p):
            idx.setdefault(pair_key(rec["a"]["oshash"], rec["b"]["oshash"]), []).append(
                rec
            )
    return idx


def to_data_path(path):
    return (
        "/data/" + path[len("/mnt/stash/") :]
        if path.startswith("/mnt/stash/")
        else path
    )


def find_card(decision, pair_idx):
    """The matcher pair whose A file is the decision's a_path, or None."""
    _, h1, h2 = decision["id"].split("|", 2)
    for rec in pair_idx.get(pair_key(h1, h2), []):
        if to_data_path(rec["a"]["last_path"]) == decision.get("a_path"):
            return rec
    return None


def side_attrs(codec, width, height, bit_rate, fps, duration, size, path):
    base = os.path.basename(path)
    return {
        "codec": re.sub(r"\s+", "", (codec or "").lower()),
        "width": int(width or 0),
        "height": int(height or 0),
        "bit_rate": int(bit_rate or 0),
        "fps": round(float(fps or 0), 3),
        "duration": round(float(duration or 0), 2),
        "size": int(size or 0),
        "ext": os.path.splitext(base)[1].lower().lstrip("."),
        "upscale_name": bool(UPNAME_RE.search(base)),
        "upscale_path": "/upscaled/" in path and "/originals" not in path,
    }


def live_lookup(url, api_key, hashes):
    """{oshash: attrs} for files still in Stash, via findSceneByHash, 50 aliases per request."""
    out, hashes = {}, sorted(set(hashes))
    for i in range(0, len(hashes), 50):
        chunk = hashes[i : i + 50]
        doc = (
            "{ "
            + " ".join(
                f'h{j}: findSceneByHash(input: {{ oshash: "{h}" }}) {{ files {{ {FILE_FIELDS} }} }}'
                for j, h in enumerate(chunk)
            )
            + " }"
        )
        req = urllib.request.Request(
            url.rstrip("/") + "/graphql",
            json.dumps({"query": doc}).encode(),
            {"Content-Type": "application/json", "ApiKey": api_key},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.load(resp)
        if body.get("errors"):
            raise RuntimeError(body["errors"][0].get("message"))
        for j, h in enumerate(chunk):
            scene = body["data"].get(f"h{j}")
            for f in (scene or {}).get("files", []):
                if any(
                    fp["type"] == "oshash" and fp["value"] == h
                    for fp in f["fingerprints"]
                ):
                    out[h] = side_attrs(
                        f["video_codec"],
                        f["width"],
                        f["height"],
                        f["bit_rate"],
                        f["frame_rate"],
                        f["duration"],
                        f["size"],
                        f["path"],
                    )
                    break
    return out


def backup_lookup(db_path, hashes):
    """{oshash: attrs} from a Stash sqlite backup, opened read-only."""
    out = {}
    if not Path(db_path).exists():
        return out
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        sql = (
            "SELECT v.video_codec, v.width, v.height, v.bit_rate, v.frame_rate, v.duration, f.size, f.basename "
            "FROM files_fingerprints fp JOIN files f ON f.id = fp.file_id JOIN video_files v ON v.file_id = f.id "
            "WHERE fp.type = 'oshash' AND fp.fingerprint = ? LIMIT 1"
        )
        for h in set(hashes):
            row = con.execute(sql, (h,)).fetchone()
            if row:
                out[h] = side_attrs(*row[:7], row[7])
    finally:
        con.close()
    return out


def build(decisions, pair_idx, lookups):
    """Returns (records, stats). lookups is an ordered list of {oshash: attrs}, first hit wins."""
    stats = {"decisions": 0, "no_card": 0, "missing_files": 0, "written": 0}
    out = []
    for dec in decisions:
        stats["decisions"] += 1
        card = find_card(dec, pair_idx)
        if not card:
            stats["no_card"] += 1
            continue
        sides = []
        for s in ("a", "b"):
            h = card[s]["oshash"]
            sides.append(next((lk[h] for lk in lookups if h in lk), None))
        if not all(sides):
            stats["missing_files"] += 1
            continue
        kind, raw = card["kind"], dec["pick"]
        if kind == "duplicate" and raw in ("keep_a", "keep_b"):
            pick = raw[-1]
        elif kind == "contained" and raw in ("keep_short", "long_better"):
            short = "a" if card["cover_a"] >= card["cover_b"] else "b"
            pick = short if raw == "keep_short" else ("b" if short == "a" else "a")
        else:
            stats["no_card"] += 1  # pick and kind disagree: no usable card
            continue
        out.append(
            {
                "id": hashlib.sha256(dec["id"].encode()).hexdigest()[:12],
                "kind": kind,
                "pick": pick,
                "pick_raw": raw,
                "cover_a": card["cover_a"],
                "cover_b": card["cover_b"],
                "a": sides[0],
                "b": sides[1],
            }
        )
        stats["written"] += 1
    return out, stats


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--review-dir",
        type=Path,
        default=DEFAULT_REVIEW,
        help="curator-log/review directory",
    )
    ap.add_argument(
        "--pairs",
        nargs="*",
        default=None,
        help="matcher pair files (default: the three in --review-dir)",
    )
    ap.add_argument(
        "--backup",
        nargs="*",
        default=list(DEFAULT_BACKUPS),
        help="Stash sqlite backups, newest first",
    )
    ap.add_argument(
        "--env-file",
        default=str(DEFAULT_ENV),
        help="file holding STASH_URL and STASH_API_KEY",
    )
    ap.add_argument("--no-live", action="store_true", help="skip live Stash")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args(argv)

    last = last_decisions(args.review_dir / "decisions.jsonl")
    decisions = [
        r for i, r in last.items() if i.startswith("dupe|") and r.get("pick") in PICKS
    ]
    pair_files = (
        args.pairs
        if args.pairs is not None
        else [args.review_dir / n for n in DEFAULT_PAIRS]
    )
    pair_idx = load_pairs(pair_files)

    hashes = []
    for dec in decisions:
        card = find_card(dec, pair_idx)
        if card:
            hashes += [card["a"]["oshash"], card["b"]["oshash"]]

    lookups = []
    if not args.no_live:
        env = {**load_env(args.env_file), **os.environ}
        lookups.append(live_lookup(env["STASH_URL"], env["STASH_API_KEY"], hashes))
    for db in args.backup:
        lookups.append(backup_lookup(db, hashes))

    records, stats = build(decisions, pair_idx, lookups)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        f.writelines(json.dumps(rec, sort_keys=True) + "\n" for rec in records)
    print(f"decisions read: {stats['decisions']}")
    print(f"skipped, no card: {stats['no_card']}")
    print(f"skipped, missing files: {stats['missing_files']}")
    print(f"written: {stats['written']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Where else does a file's audio appear? Builds the index from the store, queries files against it, verifies, and
records the overlaps.

  PYTHONPATH=api python -m content_fingerprint.match --db /data/fp/content_fp.db --index /data/fp/index [--rebuild]
      [--query-list files.jsonl | --all] [--min-overlap 20] [--out overlaps.jsonl]

An overlap says: seconds a_start..a_end of file a are the same audio as b_start..b_end of file b. One covering most of
both files is a duplicate; one covering a short file's whole length inside a long file is a scene inside a DVD (or a
part of a multi-part scene); several in a row in two files are re-cuts of the same material.
"""

import argparse
import json
import logging
import os
import sys

from . import AUDIO_ALGO, AUDIO_ALGO_VERSION, index as fpindex
from .store import ContentFpStore

log = logging.getLogger("content_fingerprint")


def overlaps_for(fid, fp, idx, fps, min_overlap_s=20.0, max_ber=0.25, min_votes=10, exclude=None):
    """[(fid, other, a_start, a_end, b_start, b_end, ber)] in seconds, for every other file sharing audio with ``fid``
    (``fid`` itself and the files in ``exclude`` skipped)."""
    out = []
    min_windows = max(3, int(round(min_overlap_s / fpindex.ITEM_SECONDS / 32)))
    for other, off, _votes in fpindex.candidates(idx, fp, min_votes=min_votes, exclude={fid} | set(exclude or ())):
        ofp = fps.get(other)
        if ofp is None:
            continue
        off, _ = fpindex.best_offset(fp, ofp, off)
        for a0, a1, b0, b1, ber in fpindex.matched_runs(fp, ofp, off, max_ber=max_ber, min_windows=min_windows):
            s = fpindex.ITEM_SECONDS
            out.append((fid, other, round(a0 * s, 2), round(a1 * s, 2), round(b0 * s, 2), round(b1 * s, 2), round(ber, 4)))
    return out


def _union(spans):
    """Total length of the union of ``spans`` [(start, end)]."""
    total, end = 0.0, None
    for s, e in sorted(spans):
        if end is None or s > end:
            total += e - s
            end = e
        elif e > end:
            total += e - end
            end = e
    return total


def classify(rows, dur_a, dur_b, min_shared_s=60.0):
    """What one pair's overlaps mean. ``rows`` are this pair's (a_start, a_end, b_start, b_end, ber), a and b as
    stored; durations in seconds. Returns {kind, shared_s, cover_a, cover_b}, kind one of:

      duplicate   most of both files is the same audio (a re-encode, a re-release, an upscale)
      contained   most of the shorter file is inside the longer one (a scene in a DVD, a part in a full movie)
      partial     a sizeable shared stretch (a compilation reusing footage, a re-cut)
      incidental  a short shared bit, or many short scraps (an intro sting, a music bed); not worth a look
    """
    shared_a = _union([(r[0], r[1]) for r in rows])
    shared_b = _union([(r[2], r[3]) for r in rows])
    cover_a = min(1.0, shared_a / dur_a) if dur_a else 0.0
    cover_b = min(1.0, shared_b / dur_b) if dur_b else 0.0
    shared = min(shared_a, shared_b)
    short_cover = cover_a if (dur_a or 0) <= (dur_b or 0) else cover_b
    long_cover = cover_b if (dur_a or 0) <= (dur_b or 0) else cover_a
    if short_cover >= 0.8 and long_cover >= 0.8:
        kind = "duplicate"
    elif short_cover >= 0.8:
        kind = "contained"
    elif shared >= min_shared_s and (short_cover >= 0.2 or shared >= 180) and max(r[1] - r[0] for r in rows) >= min_shared_s:
        # real shared footage runs on unbroken; a music bed reused under different footage matches in short scraps
        # (owner 10-22: Pitch A Tent vs a Shane Diesel scene, 222 s in 12 pieces of ~18 s, "not the same")
        kind = "partial"
    else:
        kind = "incidental"
    return {"kind": kind, "shared_s": round(shared, 1), "cover_a": round(cover_a, 3), "cover_b": round(cover_b, 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(prog="content_fingerprint.match", description="Find shared audio between files.")
    ap.add_argument("--db", required=True)
    ap.add_argument("--index", required=True, help="directory of the index arrays")
    ap.add_argument("--rebuild", action="store_true", help="rebuild the index from the store first")
    q = ap.add_mutually_exclusive_group(required=True)
    q.add_argument("--all", action="store_true", help="query every fingerprinted file")
    q.add_argument("--query-list", help="JSON-lines files ({oshash, size}) to query")
    ap.add_argument("--min-overlap", type=float, default=20.0, help="seconds")
    ap.add_argument("--max-ber", type=float, default=0.25)
    ap.add_argument("--out", help="write one JSON line per pair worth a look (kind, coverage, the matching pieces)")
    ap.add_argument("--include-incidental", action="store_true", help="also write the short incidental matches")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    store = ContentFpStore(args.db)
    entries = store.all_audio(AUDIO_ALGO, AUDIO_ALGO_VERSION)
    fps = dict(entries)
    if args.rebuild or not os.path.exists(os.path.join(args.index, "values.npy")):
        idx = fpindex.build(entries)
        idx.save(args.index)
        log.info("index built: %d files, %d entries", len(entries), len(idx))
    else:
        idx = fpindex.Index.load(args.index)
    files = store.files()
    if args.all:
        queries = list(fps)
    else:
        by_key = {(f["oshash"], f["size_bytes"]): fid for fid, f in files.items()}
        queries = []
        with open(args.query_list) as fh:
            for line in fh:
                if line.strip():
                    d = json.loads(line)
                    fid = by_key.get((d["oshash"], int(d["size"])))
                    if fid in fps:
                        queries.append(fid)
    log.info("%d query files against %d fingerprinted files", len(queries), len(fps))
    found = []
    done = set()
    for n, fid in enumerate(queries, 1):
        done.add(fid)   # a pair is found once, from whichever of its files is queried first
        rows = overlaps_for(fid, fps[fid], idx, fps, args.min_overlap, args.max_ber, exclude=done)
        if rows:
            store.put_overlaps(AUDIO_ALGO_VERSION, rows)
            found += rows
        if n % 100 == 0:
            log.info("%d/%d queried, %d overlaps so far", n, len(queries), len(found))
    by_pair = {}
    for a, b, a0, a1, b0, b1, ber in found:
        by_pair.setdefault((a, b), []).append((a0, a1, b0, b1, ber))
    def dur(f, fid):
        return f["duration_s"] or len(fps[fid]) * fpindex.ITEM_SECONDS

    kinds = {}
    with open(args.out, "w") if args.out else open(os.devnull, "w") as fh:
        for (a, b), rows in by_pair.items():
            fa, fb = files[a], files[b]
            c = classify(rows, dur(fa, a), dur(fb, b))
            kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
            if c["kind"] != "incidental" or args.include_incidental:
                fh.write(json.dumps({**c, "a": fa, "b": fb, "pieces": [
                    {"a_start": r[0], "a_end": r[1], "b_start": r[2], "b_end": r[3], "ber": r[4]} for r in rows]}) + "\n")
    print(json.dumps({"queries": len(queries), "overlaps": len(found), "pairs": len(by_pair), "kinds": kinds}))
    store.close()


if __name__ == "__main__":
    sys.exit(main())

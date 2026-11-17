"""Standalone batch worker.

  PYTHONPATH=api python -m content_fingerprint.cli --db /data/content_fp.db \\
      --stash-url http://10.0.0.4:6969 --mount-map /data=/mnt/stash [--group 206] [--scene-id 123 ...] [--workers 3]
  PYTHONPATH=api python -m content_fingerprint.cli --db ... --list files.jsonl --mount-map /data=/mnt/stash

The API key comes from --api-key or STASH_API_KEY. SIGINT/SIGTERM stop after the files in flight; rerun to resume.
"""

import argparse
import functools
import json
import logging
import os
import signal
import sys
import threading

from . import AUDIO_ALGO, AUDIO_ALGO_VERSION, audio, stash_source, worker
from .store import ContentFpStore


def main(argv=None):
    ap = argparse.ArgumentParser(prog="content_fingerprint", description="Fingerprint the audio of Stash files.")
    ap.add_argument("--db", required=True, help="content_fp.db path (created if missing)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stash-url", help="Stash base URL, e.g. http://10.0.0.4:6969")
    src.add_argument("--list", help="JSON-lines file list instead of asking Stash")
    ap.add_argument("--api-key", default=os.environ.get("STASH_API_KEY"))
    ap.add_argument("--group", action="append", default=[], help="only scenes in this group (repeatable)")
    ap.add_argument("--scene-id", action="append", default=[], help="only these scenes (repeatable)")
    ap.add_argument("--mount-map", action="append", default=[], help="container prefix = local prefix (repeatable)")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, help="fingerprint at most this many files this run")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per file")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    mapping = stash_source.parse_mount_map(args.mount_map)
    if args.list:
        records = stash_source.from_list(args.list, mapping)
    else:
        records = stash_source.from_stash(args.stash_url, args.api_key, mapping, args.group or None, args.scene_id or None)
    missing = [r.path for r in records[:50] if not os.path.exists(r.path)]
    if records and len(missing) == min(50, len(records)):
        sys.exit(f"none of the first {len(missing)} paths exist here (e.g. {missing[0]}): check --mount-map")

    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: (logging.info("stopping after the files in flight"), stop.set()))
    store = ContentFpStore(args.db)
    fp = functools.partial(audio.fingerprint, timeout=args.timeout)
    stats = worker.run(records, store, workers=args.workers, retry_failed=args.retry_failed, stop=stop,
                       fingerprint_fn=fp, limit=args.limit)
    stats["store"] = store.counts(AUDIO_ALGO, AUDIO_ALGO_VERSION)
    store.close()
    print(json.dumps(stats))


if __name__ == "__main__":
    main()

"""The batch loop: fingerprint every file not yet in the store, a few at a time, resumable at any point."""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import AUDIO_ALGO, AUDIO_ALGO_VERSION, audio

log = logging.getLogger("content_fingerprint")


def run(records, store, workers=3, retry_failed=False, stop=None, fingerprint_fn=None, params=None,
        progress_every=25, limit=None):
    """Fingerprint ``records`` into ``store``; returns {"done", "skipped", "ok", "no_audio", "failed", "seconds"}.

    Files already fingerprinted at this algorithm version are skipped, so an interrupted run picks up where it left
    off. ``stop`` (a threading.Event) ends the run after the files in flight.
    """
    fingerprint_fn = fingerprint_fn or audio.fingerprint
    params = params if params is not None else audio.params()
    stop = stop or threading.Event()
    todo = []
    for rec in records:
        fid = store.upsert_file(rec.oshash, rec.size, rec.md5, rec.duration, rec.path)
        if not store.has_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, retry_failed=retry_failed):
            todo.append((fid, rec))
    stats = {"done": 0, "skipped": len(records) - len(todo), "ok": 0, "no_audio": 0, "failed": 0}
    if limit is not None:
        todo = todo[:limit]
    started = time.monotonic()
    log.info("%d files selected, %d already fingerprinted, %d to do this run", len(records), stats["skipped"], len(todo))

    def one(fid, rec):
        if stop.is_set():
            return None
        t0 = time.monotonic()
        status, fp, err = fingerprint_fn(rec.path)
        store.put_audio(fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, params, status, fp, err, round(time.monotonic() - t0, 2))
        return status, rec, err

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, fid, rec) for fid, rec in todo]
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            status, rec, err = res
            stats["done"] += 1
            stats[status] += 1
            if status == "failed":
                log.warning("failed: %s: %s", rec.path, (err or "").splitlines()[-1:] or "")
            if stats["done"] % progress_every == 0 or stats["done"] == len(todo):
                el = time.monotonic() - started
                rate = stats["done"] / el if el else 0
                left = (len(todo) - stats["done"]) / rate if rate else 0
                log.info("%d/%d (ok %d, no audio %d, failed %d), %.2f files/s, ~%.1f h left",
                         stats["done"], len(todo), stats["ok"], stats["no_audio"], stats["failed"], rate, left / 3600)
    stats["seconds"] = round(time.monotonic() - started, 1)
    return stats

"""The content index (#129) as a queue job: fingerprint every file's audio and keyframes, then find shared audio.

Phase 1, fingerprinting. Every file of every scene, one row per content (oshash + size), files without a fingerprint
first and the most recently added scenes first. ffmpeg runs through JobContext.run_media_command, so the
media_job_nice / media_job_threads settings (#124) apply. Progress lives in the store itself: a finished file has a row,
so a stopped or restarted job repeats no finished file.

Phase 2, matching. Rebuild the inverted index when the set of fingerprints changed, then look up every file not yet
queried and store the verified overlaps (``audio_overlaps``).

A store built elsewhere is imported first when it is dropped at DATA_DIR/content_fp_import.db (see
docs/content-index.md). It is renamed to .imported or .refused afterwards so it is tried once.

Excluded scenes (#125) are skipped: a file is skipped when any of its scenes is excluded, and excluded files are left
out of the matching. The exclusion list is read fresh at the start and the job fails closed when Stash can't say.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from base_job import BaseJob, JobContext
from content_fingerprint import (
    AUDIO_ALGO,
    AUDIO_ALGO_VERSION,
    VIDEO_ALGO,
    VIDEO_ALGO_VERSION,
    audio,
    importer,
    match,
    video,
)
from content_fingerprint.store import ContentFpStore

logger = logging.getLogger(__name__)

STORE_NAME = "content_fp.db"
INDEX_NAME = "content_index"
IMPORT_NAME = "content_fp_import.db"
FILE_TIMEOUT_S = 3600
MISSING_ABORT = 50       # this many files in a row missing from the mount: the mount or the path map is wrong
ERROR_ABORT = 20         # this many unexpected errors in a row (ffmpeg missing, database broken): stop the run
QUERY_BATCH = 25


@dataclass
class LibraryFile:
    oshash: str
    size: int
    path: str                       # as Stash reports it
    md5: str | None = None
    duration: float | None = None
    scene_ids: list = field(default_factory=list)
    added: str = ""                 # newest created_at of its scenes


async def library_files(stash, per_page: int = 500) -> list[LibraryFile]:
    """Every file of every scene, one per content; files without an oshash are skipped."""
    seen: dict[tuple[str, int], LibraryFile] = {}
    page = 1
    while True:
        scenes, count = await stash.get_scene_files_page(page, per_page)
        for scene in scenes:
            for f in scene.get("files") or []:
                fps = {x["type"]: x["value"] for x in f.get("fingerprints") or []}
                if "oshash" not in fps:
                    continue
                key = (fps["oshash"], int(f["size"]))
                rec = seen.get(key)
                if rec is None:
                    rec = seen[key] = LibraryFile(key[0], key[1], f["path"], fps.get("md5"), f.get("duration"))
                rec.scene_ids.append(str(scene["id"]))
                rec.added = max(rec.added, scene.get("created_at") or "")
        if page * per_page >= count or not scenes:
            break
        page += 1
    return list(seen.values())


def _data_dir() -> str:
    return os.environ.get("DATA_DIR", "./data")


def _workers() -> int:
    from settings import SETTING_DEFS, get_setting
    try:
        return int(get_setting("content_index_workers"))
    except RuntimeError:
        return int(SETTING_DEFS["content_index_workers"].fallback)


class ContentFingerprintJob(BaseJob):
    def __init__(self, stash=None, media=None, data_dir: str | None = None, workers: int | None = None,
                 exclusion=None):
        self._stash = stash
        self._media = media
        self._data_dir = data_dir
        self._workers = workers
        self._exclusion = exclusion

    # -- dependencies, overridable for tests ------------------------------------------------------------------

    def _deps(self):
        stash = self._stash
        if stash is None:
            from recommendations_router import get_stash_client
            stash = get_stash_client()
        media = self._media
        if media is None:
            from media_library import get_media_library
            media = get_media_library()
        return stash, media, self._data_dir or _data_dir(), self._workers or _workers()

    async def _load_exclusion(self, stash):
        if self._exclusion is not None:
            return self._exclusion
        import curation_settings
        return await curation_settings.current(stash, max_age=0)

    # -- run --------------------------------------------------------------------------------------------------

    async def run(self, context: JobContext, cursor: str | None = None) -> str | None:
        stash, media, data_dir, workers = self._deps()
        exclusion = await self._load_exclusion(stash)
        files = await library_files(stash)
        kept = [f for f in files if not exclusion.excluded(f.scene_ids)]
        result = {"files": len(files), "excluded": len(files) - len(kept)}

        store = await asyncio.to_thread(ContentFpStore, os.path.join(data_dir, STORE_NAME))
        try:
            ids = await asyncio.to_thread(store.upsert_files, kept)
            local = {}
            for f in kept:
                try:
                    local[(f.oshash, f.size)] = media.to_local(f.path)
                except ValueError:      # MediaPathError: outside every MEDIA_PATH_MAP entry
                    pass
            result["unmapped"] = len(kept) - len(local)

            imp = os.path.join(data_dir, IMPORT_NAME)
            if os.path.exists(imp):
                result["import"] = await self._import(context, store, imp, local)

            stopped = await self._fingerprint(context, store, kept, ids, local, workers, result)
            if stopped is not None:
                context.set_result(result)
                return stopped
            stopped = await self._match(context, store, {ids[k] for k in local if k in ids},
                                        os.path.join(data_dir, INDEX_NAME), result)
            result["audio"] = store.counts(AUDIO_ALGO, AUDIO_ALGO_VERSION)
            result["video"] = store.counts(VIDEO_ALGO, VIDEO_ALGO_VERSION, table="video_signatures")
            result["overlaps"] = store.overlap_count(AUDIO_ALGO_VERSION)
            context.set_result(result)
            return stopped
        finally:
            store.close()

    async def _import(self, context, store, path, local):
        async def fp(p):
            r = await context.run_media_command(audio.command(p), timeout=FILE_TIMEOUT_S)
            return audio.parse(r.returncode, r.stdout, r.stderr)
        try:
            out = await importer.check_and_import(path, store, local, fp, audio.ffmpeg_version())
        except Exception as e:   # refused, or not a store at all: either way, tried once and set aside
            os.replace(path, path + ".refused")
            logger.warning("content index: import refused: %s", e)
            return {"status": "refused", "reason": str(e)}
        os.replace(path, path + ".imported")
        logger.warning("content index: imported %s", out)
        return {"status": "imported", **out}

    async def _fingerprint(self, context, store, kept, ids, local, workers, result) -> str | None:
        """Phase 1. Returns a cursor when the job yields to a higher-priority job, else None."""
        audio_done = store.done_ids("audio_fingerprints", AUDIO_ALGO, AUDIO_ALGO_VERSION)
        video_done = store.done_ids("video_signatures", VIDEO_ALGO, VIDEO_ALGO_VERSION)
        todo = []
        for f in kept:
            key = (f.oshash, f.size)
            fid = ids.get(key)
            if fid is None or key not in local:
                continue
            need_a, need_v = fid not in audio_done, fid not in video_done
            if need_a or need_v:
                todo.append((f, fid, need_a, need_v))
        todo.sort(key=lambda t: t[0].added, reverse=True)
        stats = {"to_do": len(todo), "done": 0, "missing": 0, "timed_out": 0, "errors": 0}
        result["fingerprint"] = stats
        await context.report_progress(0, len(todo))

        a_params, v_params = audio.params(), {**video.params(), "ffmpeg": audio.ffmpeg_version()}
        sem = asyncio.Semaphore(max(1, workers))
        missing_run = error_run = 0
        abort: list[str] = []

        async def one(f, fid, need_a, need_v):
            nonlocal missing_run
            path = local[(f.oshash, f.size)]
            if not await asyncio.to_thread(os.path.exists, path):
                stats["missing"] += 1
                missing_run += 1
                if missing_run >= MISSING_ABORT and not abort:
                    abort.append(f"{MISSING_ABORT} files in a row are missing under the media mount (e.g. {path}); "
                                 "check MEDIA_PATH_MAP and the /media mount")
                return
            missing_run = 0
            if need_a:
                t0 = time.monotonic()
                r = await context.run_media_command(audio.command(path), timeout=FILE_TIMEOUT_S)
                status, fp, err = audio.parse(r.returncode, r.stdout, r.stderr)
                await asyncio.to_thread(store.put_audio, fid, AUDIO_ALGO, AUDIO_ALGO_VERSION, a_params, status, fp,
                                        err, round(time.monotonic() - t0, 2))
            if need_v:
                t0 = time.monotonic()
                r = await context.run_media_command(video.command(path), timeout=FILE_TIMEOUT_S)
                status, times, hashes, err = video.parse(r.returncode, r.stdout, r.stderr)
                await asyncio.to_thread(store.put_video, fid, VIDEO_ALGO, VIDEO_ALGO_VERSION, v_params, status,
                                        times, hashes, err, round(time.monotonic() - t0, 2))
            stats["done"] += 1

        async def guarded(item):
            # Nothing is stored for a timeout or an unexpected error, so the next run tries the file again; a file
            # ffmpeg can't read is stored as failed and not retried.
            nonlocal error_run
            async with sem:
                if context.is_stop_requested() or abort:
                    return
                try:
                    await one(*item)
                    error_run = 0
                except TimeoutError:
                    stats["timed_out"] += 1
                    logger.warning("content index: %s timed out after %d s", item[0].path, FILE_TIMEOUT_S)
                except Exception as e:
                    stats["errors"] += 1
                    error_run += 1
                    logger.warning("content index: %s: %s", item[0].path, e)
                    if error_run >= ERROR_ABORT and not abort:
                        abort.append(f"{ERROR_ABORT} files in a row failed unexpectedly; last error: {e}")
                await context.report_progress(sum(stats[k] for k in ("done", "missing", "timed_out", "errors")),
                                              len(todo))

        yielded = False
        i = 0
        while i < len(todo):
            if context.is_stop_requested() or abort:
                break
            if await context.should_yield():
                yielded = True
                break
            batch = todo[i:i + max(1, workers) * 4]
            i += len(batch)
            await asyncio.gather(*(guarded(item) for item in batch))
        if abort:
            raise RuntimeError(abort[0])
        return "fingerprint" if yielded else None

    async def _match(self, context, store, allowed: set[int], index_path: str, result) -> str | None:
        """Phase 2: index the fingerprints and look up the files not yet queried."""
        if context.is_stop_requested():
            return None
        ok = store.ok_audio_ids(AUDIO_ALGO, AUDIO_ALGO_VERSION)
        todo = [f for f in store.unqueried(AUDIO_ALGO, AUDIO_ALGO_VERSION) if f in allowed]
        stats = {"to_query": len(todo), "queried": 0, "overlaps_found": 0, "index_rebuilt": False}
        result["match"] = stats
        if not todo:
            return None
        meta = match.index_meta(index_path) or {}
        fresh = (meta.get("n_files"), meta.get("id_sum"), meta.get("algo_version")) == \
            (len(ok), sum(ok), AUDIO_ALGO_VERSION)
        if not fresh:
            idx = await asyncio.to_thread(match.build_index, store, index_path)
            stats["index_rebuilt"] = True
        else:
            idx = match.fpindex.Index.load(index_path)
        fps = match.StoreFingerprints(store)
        # Skipped as candidates: files no longer in the library or excluded, and files queried earlier in this run
        # (their pairs with the file at hand were found from their side). One set, grown in place.
        skip = ok - allowed
        await context.report_progress(0, len(todo))

        def query(batch):
            found = 0
            for fid in batch:
                fp = fps.get(fid)
                skip.add(fid)
                if fp is not None:
                    rows = match.overlaps_for(fid, fp, idx, fps, exclude=skip)
                    if rows:
                        store.put_overlaps(AUDIO_ALGO_VERSION, rows)
                        found += len(rows)
            store.mark_queried(batch, AUDIO_ALGO_VERSION)
            return found

        for i in range(0, len(todo), QUERY_BATCH):
            if context.is_stop_requested():
                return None
            if await context.should_yield():
                return "match"
            batch = todo[i:i + QUERY_BATCH]
            stats["overlaps_found"] += await asyncio.to_thread(query, batch)
            stats["queried"] += len(batch)
            await context.report_progress(stats["queried"], len(todo))
        return None

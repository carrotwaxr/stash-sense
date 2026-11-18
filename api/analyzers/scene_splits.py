"""Split plans for long and multi-scene files (#131).

One recommendation per candidate scene (``target_type="scene"``, ``target_id`` the scene id), reviewed by
``review.splits``. Per file the job extracts the signals once (two ffmpeg passes, cached in ``DATA_DIR/splits.db`` by
(oshash, size)), then plans every candidate from the cache on every run, so a planner change re-scores all open plans;
the job result reports how many changed confidence or moved a cut by more than MOVED_S.

Like ``scene_duplicates`` it upserts in place and never deletes (the driver feed cannot report a deletion): a plan that
changed is rewritten, a scene that is no longer a candidate is resolved ``stale`` (only when the run finished), and
dismissed rows are never touched. A plan the owner accepted is not proposed again for the same file (oshash): accepting
writes nothing to Stash, so the scene is still a candidate until the curator has split it.

Accepted plans teach the series profiles: each ``applied`` plan with cuts becomes a label (``rec:<id>``, so a re-run
overwrites it) with the features of the signals at each accepted cut.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from collections import Counter
from typing import Optional

import media_exec
import media_library
import curation_settings
from analyzers.base import AnalysisResult, BaseAnalyzer
from content_fingerprint.store import ContentFpStore
from splits import candidates as cand_mod
from splits import detect, planner
from splits import signals as sg
from splits.profiles import series_key
from splits.store import SplitStore

logger = logging.getLogger(__name__)

TYPE_ID = "scene_splits"
TARGET_TYPE = "scene"
STORE_NAME = "splits.db"
CONFIDENCE = {"high": 0.9, "medium": 0.6, "low": 0.3}
SIGNAL_WORKERS = 2
EXTRACT_TIMEOUT_S = 3 * 3600
MOVED_S = 2.0               # a cut that moved more than this counts as moved when re-scoring
LABEL_WINDOW_S = 6.0        # a label's features need the signals to cover its cut +- this
REDUNDANT_REASON = ("every stretch outside the first and last %d s is held by other scenes' files at equal or better "
                    "quality" % cand_mod.GUARD_S)


def _oshash(f: dict) -> Optional[str]:
    return next((fp["value"] for fp in f.get("fingerprints") or [] if fp.get("type") == "oshash"), None)


def _plain(o):
    """json.dumps default: numpy scalars and arrays, tuples handled by json itself."""
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _clean(details: dict) -> dict:
    return json.loads(json.dumps(details, default=_plain))


def _covered(covered, a: float, b: float) -> bool:
    return any(s <= a and b <= e for s, e in covered)


class ExtractFailed(Exception):
    pass


class SceneSplitsAnalyzer(BaseAnalyzer):
    """Plans the split points of every candidate scene and proposes them for review."""

    type = TYPE_ID

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats: Counter = Counter()
        self.rescored = {"confidence_changed": 0, "cuts_moved": 0}
        self._lib = None
        self._sem = asyncio.Semaphore(SIGNAL_WORKERS)

    # ---------------- helpers ----------------

    def _local(self, stash_path: str) -> Optional[str]:
        if self._lib is None:
            self._lib = media_library.get_media_library()
        try:
            return self._lib.to_local(stash_path)
        except ValueError:     # outside every MEDIA_PATH_MAP entry
            return None

    async def _extract(self, local: str, duration: float) -> bytes:
        """Run both ffmpeg passes and return the encoded signals; raises ExtractFailed."""
        try:
            video = await media_exec.run_command(media_exec.media_job_command(sg.video_command(local)),
                                                 timeout=EXTRACT_TIMEOUT_S)
            if video.returncode != 0:
                raise ExtractFailed(f"video pass exited {video.returncode}: {video.stderr[-300:].decode(errors='replace')}")
            audio = await media_exec.run_command(media_exec.media_job_command(sg.audio_command(local)),
                                                 timeout=EXTRACT_TIMEOUT_S)
            if audio.returncode != 0:
                raise ExtractFailed(f"audio pass exited {audio.returncode}: {audio.stderr[-300:].decode(errors='replace')}")
        except ExtractFailed:
            raise
        except Exception as e:
            raise ExtractFailed(f"{type(e).__name__}: {e}") from e

        def parse() -> bytes:
            fr = sg.parse_video(video.stdout.decode(errors="replace"))
            if not fr:
                raise ExtractFailed("the video pass printed no frames")
            au, sil = sg.parse_audio(audio.stdout.decode(errors="replace"), audio.stderr.decode(errors="replace"),
                                     duration)
            end = min(float(duration), fr[-1][0] + 1.0) if duration else fr[-1][0]
            return sg.encode(sg.from_curator({"fr": fr, "au": au, "sil": sil, "covered": [[0.0, end]]}, duration))

        return await asyncio.to_thread(parse)

    async def _signals(self, store: SplitStore, cand) -> Optional[sg.Signals]:
        """The file's signals from the cache, else extracted and cached; None (counted) when that is not possible."""
        f = cand.file
        osh, size = _oshash(f), int(f.get("size") or 0)
        if not osh:
            self.stats["no_oshash"] += 1
            return None
        cached = await asyncio.to_thread(store.get_signals, osh, size)
        if cached is not None:
            self.stats["cached"] += 1
            return cached
        local = self._local(f.get("path") or "") if f.get("path") else None
        if not local or not await asyncio.to_thread(os.path.exists, local):
            self.stats["missing"] += 1
            logger.warning("scene_splits: scene %s: file %s is not visible on the media mount", cand.scene_id,
                           f.get("path"))
            return None
        async with self._sem:
            if self.is_stop_requested():
                return None
            try:
                blob = await self._extract(local, float(f.get("duration") or 0))
            except ExtractFailed as e:
                self.stats["failed"] += 1
                logger.warning("scene_splits: scene %s: ffmpeg failed on %s: %s", cand.scene_id, local, e)
                return None
        await asyncio.to_thread(store.put_signals, osh, size, sg.SIGNALS_VERSION, blob)
        self.stats["extracted"] += 1
        # plan from what the cache would give, so a re-run plans exactly the same
        return await asyncio.to_thread(sg.decode, blob)

    # ---------------- labels from accepted plans ----------------

    def _accepted(self):
        return self.rec_db.get_recommendations(status="resolved", type=TYPE_ID, target_type=TARGET_TYPE,
                                               limit=1_000_000_000)

    def _label_accepted(self, store: SplitStore, accepted) -> int:
        added = 0
        for rec in accepted:
            if rec.resolution_action != "applied":
                continue
            split = (rec.resolution_details or {}).get("split")
            series = (rec.details or {}).get("series")
            if not split or not split.get("cuts") or not series:
                continue
            cuts = [float(c) for c in split["cuts"]]
            feats: list = [None] * len(cuts)
            sigs = store.get_signals(split.get("oshash"), int(split.get("size") or 0))
            if sigs is not None:
                d = sg.to_dict(sigs)
                for i, c in enumerate(cuts):
                    if _covered(sigs.covered, c - LABEL_WINDOW_S, c + LABEL_WINDOW_S):
                        feats[i] = detect.features_at(d, c)
            store.add_label(series, f"rec:{rec.id}", float(split.get("duration") or 0), cuts,
                            feats if any(feats) else None, "accepted")
            added += 1
        return added

    # ---------------- planning ----------------

    def _plan(self, sigs: sg.Signals, cand, profile) -> dict:
        total = float(cand.file.get("duration") or sigs.duration)
        plan = planner.plan_file(sg.to_dict(sigs), total, profile=profile, coverage=cand.coverage or None)
        kind = "leave_whole" if plan.leave_whole_suggested else "split"
        return {"cuts": plan.cuts, "keep_until": plan.keep_until, "confidence": plan.confidence,
                "reasons": plan.reasons, "leftovers": plan.leftovers, "profile_used": plan.profile_used,
                "kind": kind}

    def _details(self, cand, series: str, plan: dict) -> dict:
        f = cand.file
        return _clean({
            "file": {"id": f["id"], "oshash": _oshash(f), "size": f.get("size"), "duration": f.get("duration")},
            "cuts": plan["cuts"],
            "keep_until": plan["keep_until"],
            "confidence": plan["confidence"],
            "reasons": plan["reasons"],
            "leftovers": plan["leftovers"],
            "series": series,
            "profile_used": plan["profile_used"],
            "planner_version": planner.PLANNER_VERSION,
            "signals_version": sg.SIGNALS_VERSION,
            "coverage": [list(c) for c in cand.coverage],
            "covering": cand.covering,
            "kind": plan["kind"],
        })

    def _note_rescore(self, old: dict, new: dict) -> None:
        if old.get("confidence") != new["confidence"]:
            self.rescored["confidence_changed"] += 1
        a = [c["t"] for c in old.get("cuts") or []]
        b = [c["t"] for c in new["cuts"]]
        if len(a) != len(b) or any(abs(x - y) > MOVED_S for x, y in zip(a, b)):
            self.rescored["cuts_moved"] += 1

    # ---------------- run ----------------

    async def run(self, incremental: bool = True) -> AnalysisResult:
        excl = await curation_settings.current(self.stash)     # fails closed: ExclusionUnavailable propagates
        self.exclusion = excl
        from jobs.content_fingerprint_job import STORE_NAME as CONTENT_STORE, _data_dir
        data_dir = _data_dir()
        os.makedirs(data_dir, exist_ok=True)
        content_path = os.path.join(data_dir, CONTENT_STORE)
        content = await asyncio.to_thread(ContentFpStore, content_path) if os.path.exists(content_path) else None
        store = await asyncio.to_thread(SplitStore, os.path.join(data_dir, STORE_NAME))
        try:
            return await self._run(excl, store, content, data_dir)
        finally:
            store.close()
            if content is not None:
                content.close()

    async def _run(self, excl, store: SplitStore, content, data_dir: str) -> AnalysisResult:
        stats = self.stats
        stats["seed_labels"] = await asyncio.to_thread(store.import_seed, data_dir)
        accepted = await asyncio.to_thread(self._accepted)
        stats["accepted_labels"] = await asyncio.to_thread(self._label_accepted, store, accepted)
        profiles = await asyncio.to_thread(store.profiles)
        aliases = await asyncio.to_thread(store.aliases)

        candidates = await cand_mod.generate(self.stash, content, set(excl.scene_ids))
        self.set_items_total(len(candidates))

        pending = {r.target_id: r for r in self.rec_db.get_recommendations(
            status="pending", type=TYPE_ID, target_type=TARGET_TYPE, limit=1_000_000_000)}
        accepted_files = {r.target_id: (r.details or {}).get("file", {}).get("oshash")
                          for r in accepted if r.resolution_action == "applied"}
        produced: set[str] = set()
        done = 0

        async def one(cand) -> None:
            nonlocal done
            try:
                if self.is_stop_requested():
                    return
                sid = cand.scene_id
                produced.add(sid)     # a candidate we could not plan keeps its open plan
                if self.is_dismissed(TARGET_TYPE, sid):
                    stats["skipped"] += 1
                    return
                osh = _oshash(cand.file)
                if osh and accepted_files.get(sid) == osh:
                    stats["accepted_skipped"] += 1
                    return
                series = series_key(cand.title, aliases)
                if cand.redundant:
                    plan = {"cuts": [], "keep_until": None, "confidence": "high", "reasons": [REDUNDANT_REASON],
                            "leftovers": [], "profile_used": False, "kind": "redundant"}
                else:
                    sigs = await self._signals(store, cand)
                    if sigs is None:
                        return
                    plan = await asyncio.to_thread(self._plan, sigs, cand, profiles.get(series))
                details = self._details(cand, series, plan)
                old = pending.get(sid)
                _, how = self.rec_db.upsert_recommendation(
                    type=TYPE_ID, target_type=TARGET_TYPE, target_id=sid, details=details,
                    confidence=CONFIDENCE[plan["confidence"]], source_analysis_id=self.run_id)
                stats[how] += 1
                if old is not None and how == "updated":
                    self._note_rescore(old.details or {}, details)
            finally:
                done += 1
                self.update_progress(done, stats["created"])

        await asyncio.gather(*(one(c) for c in candidates))

        if not self.is_stop_requested():
            for sid, rec in pending.items():
                if sid not in produced:
                    self.rec_db.resolve_recommendation(rec.id, "stale", {"reason": "no longer a split candidate"})
                    stats["stale"] += 1

        logger.warning(
            "scene_splits: %d candidates; signals cached %d, extracted %d, file missing %d, ffmpeg failed %d, no oshash %d; "
            "created %d, updated %d, unchanged %d, reopened %d, accepted and kept %d, dismissed %d, stale %d; "
            "%d labels from the seed, %d from accepted plans; re-scored: %d confidence changed, %d cuts moved%s",
            len(candidates), stats["cached"], stats["extracted"], stats["missing"], stats["failed"], stats["no_oshash"],
            stats["created"], stats["updated"], stats["unchanged"], stats["reopened"], stats["accepted_skipped"],
            stats["skipped"], stats["stale"], stats["seed_labels"], stats["accepted_labels"],
            self.rescored["confidence_changed"], self.rescored["cuts_moved"],
            " (stopped)" if self.is_stop_requested() else "")
        return AnalysisResult(
            items_processed=len(candidates), recommendations_created=stats["created"],
            extra={"rescored": dict(self.rescored),
                   "signals": {k: stats[k] for k in ("cached", "extracted", "missing", "failed", "no_oshash")}})

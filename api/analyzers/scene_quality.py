"""Quality flags for each scene's primary file (#133).

Per file the job runs the cheap pass (probe, container index, tail decode; cached in ``DATA_DIR/quality.db`` by
(oshash, size)) and, for a cheap-pass suspect only, the 24-window health check of #130, cached in the content store's
health table like ``scene_duplicates`` does. From that evidence it raises up to four flags per scene:

- ``broken``: the tiered check says the file is missing or suspect
- ``rip``: file-name and tag clues (plus low detail when below the studio) reach the rip threshold
- ``below_studio``: the file's effective detail sits well under its studio+year peers'; carries blur/block sharpness
- ``format``: a disfavored codec or container, raised only when no other flag is

Details never carry the file's path: the review card reads it from Stash. Pre-roll detection is not in stage 1.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from collections import Counter
from typing import Optional

import curation_settings
import media_library
from analyzers.base import AnalysisResult, BaseAnalyzer
from analyzers.scene_duplicates import _content_id
from content_fingerprint.store import ContentFpStore
from duplicates import keeper
from quality import peers, probe, rip, sharpness
from quality.store import QualityStore

logger = logging.getLogger(__name__)

TYPE_ID = "scene_quality"
TARGET_TYPE = "scene"
STORE_NAME = "quality.db"
PROBE_WORKERS = 2
FLAGS = ("broken", "rip", "below_studio", "format")
QUALITY_VERSION = 1
CONTROL_SAMPLE = 20      # unflagged scenes kept open for review in report-only mode, to estimate recall
FORMAT_CONFIDENCE = 0.5
STALE_REASON = "no quality flag on the current file"
PER_PAGE = 500
LOW_DETAIL_TEXT = "effective detail well below the studio's release"


def _oshash(f: dict) -> Optional[str]:
    return next((fp["value"] for fp in f.get("fingerprints") or [] if fp.get("type") == "oshash"), None)


def _plain(o):
    """json.dumps default: numpy scalars and arrays."""
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def _clean(details: dict) -> dict:
    return json.loads(json.dumps(details, default=_plain))


def _year(date: Optional[str]) -> Optional[int]:
    try:
        return int(date[:4]) if date else None
    except ValueError:
        return None


def _file_details(f: dict) -> dict:
    return {"id": f["id"], "oshash": _oshash(f), "size": f.get("size"), "basename": f.get("basename"),
            "codec": f.get("video_codec"), "width": f.get("width"), "height": f.get("height"),
            "bit_rate": f.get("bit_rate"), "duration": f.get("duration"), "format": f.get("format")}


class SceneQualityAnalyzer(BaseAnalyzer):
    """Checks every scene's primary file and raises its quality flags."""

    type = TYPE_ID

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats: Counter = Counter()
        self.raised: Counter = Counter()
        self._lib = None

    # ---------------- helpers ----------------

    def _local(self, stash_path: str) -> Optional[str]:
        if self._lib is None:
            self._lib = media_library.get_media_library()
        try:
            return self._lib.to_local(stash_path)
        except ValueError:     # outside every MEDIA_PATH_MAP entry
            return None

    async def _files(self, excl) -> list[tuple[dict, dict]]:
        """(scene, primary file) for every scene that is not excluded and has a file with an oshash."""
        out = []
        page = 1
        while True:
            scenes, total = await self.stash.get_scenes_with_files(page, PER_PAGE)
            for scene in scenes:
                if excl.is_excluded(scene["id"]):
                    self.stats["excluded"] += 1
                    continue
                files = scene.get("files") or []
                if not files:
                    self.stats["no_file"] += 1
                    continue
                if not _oshash(files[0]):
                    self.stats["no_oshash"] += 1
                    continue
                out.append((scene, files[0]))
            if not scenes or page * PER_PAGE >= total:
                return out
            page += 1

    async def _known_deep(self, content: Optional[ContentFpStore], key: tuple[str, int]) -> Optional[dict]:
        if content is None:
            return None

        def read():
            cid = _content_id(content, key)
            return None if cid is None else content.get_health(cid)
        return await asyncio.to_thread(read)

    async def _store_deep(self, content: Optional[ContentFpStore], key: tuple[str, int], deep: dict) -> None:
        if content is None:
            return

        def write():
            cid = _content_id(content, key, create=True)
            content.put_health(cid, deep["status"], deep["reasons"], deep.get("params") or {})
        await asyncio.to_thread(write)

    async def _check(self, f: dict, local: str, qstore: QualityStore, content) -> dict:
        """The tiered result for one file, from the caches where possible."""
        key = (_oshash(f), int(f.get("size") or 0))
        duration = float(f.get("duration") or 0)
        exists = await asyncio.to_thread(os.path.exists, local)
        cheap = await asyncio.to_thread(qstore.get_cheap, *key, probe.CHEAP_VERSION) if exists else None
        if cheap is not None:
            self.stats["cached"] += 1
        else:
            cheap = await probe.cheap_check(local, duration)
            if cheap["status"] == "missing":    # not cached: the mount may come back
                self.stats["missing"] += 1
            else:
                self.stats["probed"] += 1
                await asyncio.to_thread(qstore.put_cheap, *key, probe.CHEAP_VERSION, cheap)
        deep = await self._known_deep(content, key) if cheap["status"] == "suspect" else None
        res = await probe.tiered(local, duration, cheap=cheap, deep=deep)
        if res["tier"] == "deep" and deep is None:
            self.stats["deep_checked"] += 1
            await self._store_deep(content, key, res["deep"])
        return res

    # ---------------- evidence ----------------

    async def collect(self, excl, qstore: QualityStore, content: Optional[ContentFpStore]) -> list[dict]:
        """One row per checked scene: {scene_id, file, studio_id, year, broken (tiered result), tags}."""
        pairs = await self._files(excl)
        pairs.sort(key=lambda p: _oshash(p[1]))     # oshash order spreads the workers over the disks (#159)
        self.set_items_total(len(pairs))
        sem = asyncio.Semaphore(PROBE_WORKERS)
        rows: list[dict] = []
        done = 0

        async def one(scene: dict, f: dict) -> None:
            nonlocal done
            try:
                local = self._local(f.get("path") or "") if f.get("path") else None
                if local is None:
                    self.stats["unmapped"] += 1
                    return
                async with sem:
                    if self.is_stop_requested():
                        return
                    res = await self._check(f, local, qstore, content)
                rows.append({"scene_id": scene["id"], "file": f,
                             "studio_id": (scene.get("studio") or {}).get("id"), "year": _year(scene.get("date")),
                             "broken": res, "tags": res["cheap"].get("tags") or {}})
            finally:
                done += 1
                self.update_progress(done, 0)

        await asyncio.gather(*(one(s, f) for s, f in pairs))
        rows.sort(key=lambda r: int(r["scene_id"]) if str(r["scene_id"]).isdigit() else 0)
        return rows

    async def _sharpness(self, row: dict, qstore: QualityStore) -> Optional[dict]:
        f = row["file"]
        if row["broken"]["status"] == "missing":
            return None
        key = (_oshash(f), int(f.get("size") or 0))
        cached = await asyncio.to_thread(qstore.get_sharpness, *key, sharpness.SHARPNESS_VERSION)
        if cached is not None:
            return cached
        local = self._local(f.get("path") or "")
        if local is None or self.is_stop_requested():
            return None
        res = await sharpness.measure(local, float(f.get("duration") or 0))
        self.stats["sharpness_measured"] += 1
        if res is not None:
            await asyncio.to_thread(qstore.put_sharpness, *key, sharpness.SHARPNESS_VERSION, res)
        return res

    async def assemble(self, rows: list[dict], settings, qstore: QualityStore) -> dict[str, dict]:
        """scene_id -> details: the file, the raised flags with their evidence, and every score."""
        effs = [peers.PeerFile(r["scene_id"], r["studio_id"], r["year"],
                               keeper.effective_detail(keeper.FileFeatures.from_stash_file(r["file"]))[0])
                for r in rows]
        below = peers.below_studio(effs, percentile=settings.quality_peer_percentile)
        sem = asyncio.Semaphore(PROBE_WORKERS)

        async def sharp(row):
            async with sem:
                return await self._sharpness(row, qstore)
        flagged = [r for r in rows if r["scene_id"] in below]
        sharp_of = dict(zip((r["scene_id"] for r in flagged), await asyncio.gather(*(sharp(r) for r in flagged))))

        out: dict[str, dict] = {}
        for r in rows:
            sid, f, broken = r["scene_id"], r["file"], r["broken"]
            flags: dict[str, dict] = {}
            if probe.is_broken(broken):
                flags["broken"] = {"status": broken["status"], "reasons": broken["reasons"], "tier": broken["tier"]}
            clues = rip.file_clues(f.get("basename") or os.path.basename(f.get("path") or ""), r["tags"])
            if sid in below:
                clues.append({"signal": "low_detail", "text": LOW_DETAIL_TEXT, "weight": rip.WEIGHTS["low_detail"]})
            score = round(rip.rip_score([c["weight"] for c in clues]), 3)
            if score >= rip.RIP_THRESHOLD:
                flags["rip"] = {"score": score, "signals": clues}
            if sid in below:
                flags["below_studio"] = {**below[sid], "sharpness": sharp_of.get(sid)}
            if not flags:
                fmt = rip.format_flag(f, settings)
                if fmt is not None:
                    flags["format"] = fmt
            for flag in flags:
                self.raised[flag] += 1
            out[sid] = _clean({
                "file": _file_details(f),
                "flags": flags,
                "scores": {"rip": score, "below_studio": below[sid]["score"] if sid in below else None,
                           "broken": broken["status"]},
                "control": False,
                "quality_version": QUALITY_VERSION,
            })
        return out

    def _confidence(self, details: dict) -> float:
        flags = details["flags"]
        scores = [1.0 if f == "broken" else FORMAT_CONFIDENCE if f == "format" else float(ev.get("score") or 0.0)
                  for f, ev in flags.items()]
        return round(max(scores), 3)

    def _persist(self, rows: list[dict], details: dict[str, dict], settings) -> None:
        """Upsert one rec per flagged scene, retire the open ones that no longer have a flag, and top up the
        control sample."""
        stats = self.stats
        scene_ids = {r["scene_id"] for r in rows}
        pending = {r.target_id: r for r in self.rec_db.get_recommendations(
            status="pending", type=TYPE_ID, target_type=TARGET_TYPE, limit=1_000_000_000)}
        for sid, d in details.items():
            if not d["flags"]:
                continue
            if self.is_dismissed(TARGET_TYPE, sid):
                stats["dismissed"] += 1
                continue
            _, how = self.rec_db.upsert_recommendation(
                type=TYPE_ID, target_type=TARGET_TYPE, target_id=sid, details=d,
                confidence=self._confidence(d), source_analysis_id=self.run_id)
            stats[how] += 1
        if self.is_stop_requested():
            return
        for sid, rec in pending.items():
            if sid in scene_ids and not details[sid]["flags"] and not (rec.details or {}).get("control"):
                self.rec_db.resolve_recommendation(rec.id, "stale", {"reason": STALE_REASON})
                stats["stale"] += 1
        if settings.quality_report_only:
            self._control_sample(details, pending)

    def _control_sample(self, details: dict[str, dict], pending: dict) -> None:
        """Keep CONTROL_SAMPLE unflagged scenes open as control cards. Only scenes that never had a quality rec
        qualify, so the sample is not biased toward scenes the job once flagged."""
        open_controls = sum(1 for r in pending.values() if (r.details or {}).get("control"))
        want = CONTROL_SAMPLE - open_controls
        if want <= 0:
            return
        known = {r.target_id for r in self.rec_db.get_recommendations(
            type=TYPE_ID, target_type=TARGET_TYPE, limit=1_000_000_000)}
        eligible = sorted((sid for sid, d in details.items() if not d["flags"] and sid not in known), key=str)
        for sid in random.Random(self.run_id).sample(eligible, min(want, len(eligible))):
            d = dict(details[sid], control=True)
            _, how = self.rec_db.upsert_recommendation(
                type=TYPE_ID, target_type=TARGET_TYPE, target_id=sid, details=d, confidence=0.0,
                source_analysis_id=self.run_id)
            if how == "created":
                self.stats["controls"] += 1

    # ---------------- run ----------------

    async def run(self, incremental: bool = True) -> AnalysisResult:
        excl = await curation_settings.current(self.stash)     # fails closed: ExclusionUnavailable propagates
        self.exclusion = excl
        from jobs.content_fingerprint_job import STORE_NAME as CONTENT_STORE, _data_dir
        data_dir = _data_dir()
        os.makedirs(data_dir, exist_ok=True)
        content_path = os.path.join(data_dir, CONTENT_STORE)
        content = await asyncio.to_thread(ContentFpStore, content_path) if os.path.exists(content_path) else None
        qstore = await asyncio.to_thread(QualityStore, os.path.join(data_dir, STORE_NAME))
        try:
            rows = await self.collect(excl, qstore, content)
            details = await self.assemble(rows, excl.settings, qstore)
            self._persist(rows, details, excl.settings)
        finally:
            qstore.close()
            if content is not None:
                content.close()

        stats = self.stats
        checks = {k: stats[k] for k in ("cached", "probed", "deep_checked", "missing", "unmapped", "excluded",
                                        "no_file", "no_oshash", "sharpness_measured")}
        flags = {f: self.raised[f] for f in FLAGS}
        written = {k: stats[k] for k in ("created", "updated", "unchanged", "reopened", "skipped", "dismissed",
                                         "stale", "controls")}
        logger.warning(
            "scene_quality: %d files checked; cheap cached %d, probed %d, deep-checked %d, missing %d, unmapped %d, "
            "excluded %d, no file %d, no oshash %d, sharpness measured %d; flags: broken %d, rip %d, below studio %d, "
            "format %d; created %d, updated %d, unchanged %d, reopened %d, kept %d, dismissed %d, stale %d, "
            "controls %d%s",
            len(rows), *checks.values(), *flags.values(), *written.values(),
            " (stopped)" if self.is_stop_requested() else "")
        return AnalysisResult(items_processed=len(rows), recommendations_created=stats["created"],
                              extra={"checks": checks, "flags": flags, "written": written,
                                     "controls": stats["controls"],
                                     "flagged_scenes": sum(1 for d in details.values() if d["flags"])})

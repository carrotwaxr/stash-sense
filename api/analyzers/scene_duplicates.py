"""Duplicate scene sets: candidates verified with audio, sprite and keyframe evidence, grouped, upserted in place (#130).

One recommendation per set (``target_type="scene_set"``, ``target_id`` the set's key from ``duplicates.sets``), reviewed
by ``review.duplicates``. Recommendations are never deleted: external tools follow them through the driver API's change
feed (ordered by ``updated_at, id``), which cannot report a deletion. A set whose evidence or keeper suggestion changed
is rewritten in place; a set that grew a scene below its anchor is resolved ``superseded`` (by the new key); a set that
is no longer found is resolved ``stale`` (only when the run finished). A resolved row whose key returns with
different details is reopened; dismissed rows are never touched.

Per candidate pair the evidence is, in the order it is gathered: the audio bit error rate (the content index's stored
print when it has the file, else a 5 minute ffmpeg run on the local file), the keyframe match of the index's stored
signatures, and only when those cannot settle the pair, the sprite sheet score. Sprite sheets are fetched lazily and
cached per scene; tiles that recur across scenes of one studio (logos, intro cards) are ignored, judged against at most
STOP_SCENES other candidate scenes of that studio.
The keeper of each set that deletes anything is health checked once (the result is cached in the content store).
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import curation_settings
import media_library
from analyzers.base import AnalysisResult, BaseAnalyzer
from content_fingerprint import VIDEO_ALGO, VIDEO_ALGO_VERSION, video
from content_fingerprint.store import ContentFpStore
from duplicates import audio, health, keeper, sprites
from duplicates.candidates import Candidate, generate
from duplicates.sets import Item, build_items
from duplicates.verdict import AUDIO_STRONG, KEYFRAME_STRONG, PairEvidence, pair_class

logger = logging.getLogger(__name__)

TYPE_ID = "scene_duplicates"
TARGET_TYPE = "scene_set"
CONFIDENCE = {"duplicate": 0.9, "contained": 0.8, "mislabel": 0.6, "language_variant": 0.6, "partial": 0.4,
              "review": 0.3}
STOP_SCENES = 40     # other scenes of a studio a pair's sprite tiles are compared against to find recurring tiles
STOP_MIN_SCENES = 4  # a tile is a stop tile when it matches tiles in STOP_MIN_SCENES - 1 of them
DELETING = frozenset({"duplicate", "contained", "mislabel"})   # classes whose accept can delete files


def _oshash(f: dict) -> Optional[str]:
    return next((fp["value"] for fp in f.get("fingerprints") or [] if fp.get("type") == "oshash"), None)


@dataclass
class _File:
    scene_id: str
    stash: dict

    @property
    def oshash(self) -> Optional[str]:
        return _oshash(self.stash)

    @property
    def content(self) -> Optional[tuple[str, int]]:
        """(oshash, size): the content index's identity of a file."""
        osh = self.oshash
        return (osh, int(self.stash.get("size") or 0)) if osh else None

    @property
    def path(self) -> str:
        return self.stash.get("path") or ""

    @property
    def cache_key(self) -> tuple:
        return self.content or ("id", str(self.stash["id"]))


def _content_id(store: ContentFpStore, content: tuple[str, int], create: bool = False) -> Optional[int]:
    row = store._one("SELECT id FROM content_files WHERE oshash = ? AND size_bytes = ?", content)
    if row is not None:
        return row[0]
    return store.upsert_file(content[0], content[1]) if create else None


class SceneDuplicatesAnalyzer(BaseAnalyzer):
    """Finds scenes holding the same content, verifies them with audio and picture evidence, and proposes a keeper."""

    type = TYPE_ID

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats: Counter = Counter()
        self._store: Optional[ContentFpStore] = None
        self._lib = None
        self._scenes: dict[str, Optional[dict]] = {}
        self._prints: dict[tuple, object] = {}
        self._health: dict[tuple, str] = {}
        self._sigs: dict[str, Optional[dict]] = {}
        self._studio_scenes: dict[str, list[str]] = {}

    # ---------------- helpers ----------------

    def _local(self, stash_path: str) -> Optional[str]:
        if self._lib is None:
            self._lib = media_library.get_media_library()
        try:
            return self._lib.to_local(stash_path)
        except ValueError:     # outside every MEDIA_PATH_MAP entry
            return None

    async def _scene(self, scene_id: str) -> Optional[dict]:
        if scene_id not in self._scenes:
            self._scenes[scene_id] = await self.stash.get_scene_for_review(scene_id)
        return self._scenes[scene_id]

    @staticmethod
    def _pick(scene: dict, oshash: Optional[str], avoid: Optional[str] = None) -> Optional[dict]:
        files = [f for f in scene.get("files") or [] if str(f["id"]) != avoid]
        if oshash:
            for f in files:
                if _oshash(f) == oshash:
                    return f
        return files[0] if files else None

    def _pair_files(self, cand: Candidate, sa: dict, sb: dict) -> tuple[Optional[_File], Optional[_File]]:
        """The two files a candidate compares: the ones it names, else each scene's primary file."""
        fa = self._pick(sa, cand.file_a)
        fb = self._pick(sb, cand.file_b, avoid=str(fa["id"]) if fa and cand.scene_a == cand.scene_b else None)
        if fa is None or fb is None:
            return None, None
        return _File(cand.scene_a, fa), _File(cand.scene_b, fb)

    # ---------------- evidence ----------------

    async def _print(self, f: _File):
        """The audio print of a file's head: the store's, else a fast ffmpeg run; None when neither is possible."""
        if f.cache_key in self._prints:
            return self._prints[f.cache_key]
        fp = None
        if self._store is not None and f.content:
            stored = await asyncio.to_thread(audio.stored_fp, self._store, *f.content)
            if stored is not None:
                fp = audio.head_items(stored, audio.AUDIO_SECONDS)
        if fp is None:
            local = self._local(f.path) if f.path else None
            if local and await asyncio.to_thread(os.path.exists, local):
                fp = await audio.fingerprint_head(local)
        self._prints[f.cache_key] = fp
        return fp

    async def _audio(self, fa: _File, fb: _File):
        a, b = await self._print(fa), await self._print(fb)
        if a is None or b is None:
            return None
        return await asyncio.to_thread(audio.audio_ber, a, b)

    async def _keyframes(self, fa: _File, fb: _File):
        if self._store is None or not fa.content or not fb.content:
            return None
        store = self._store

        def work():
            ida, idb = _content_id(store, fa.content), _content_id(store, fb.content)
            if ida is None or idb is None:
                return None
            sig_a = store.get_video(ida, VIDEO_ALGO, VIDEO_ALGO_VERSION)
            sig_b = store.get_video(idb, VIDEO_ALGO, VIDEO_ALGO_VERSION)
            if sig_a is None or sig_b is None:
                return None
            return video.match(sig_a, sig_b)

        return await asyncio.to_thread(work)

    async def _signature(self, scene_id: str) -> Optional[dict]:
        """The scene's sprite signature, fetched on first use; None when it has no usable sprite sheet."""
        if scene_id not in self._sigs:
            sig = None
            self.stats["sprite_fetches"] += 1
            try:
                got = await sprites.fetch_sprite(getattr(self.stash, "base_url", "") or "",
                                                 getattr(self.stash, "api_key", "") or "", scene_id)
                if got is not None:
                    sig = await asyncio.to_thread(sprites.sprite_signature, *got)
            except Exception as e:     # no sprite for this scene is only missing evidence
                logger.warning("scene_duplicates: no sprite evidence for scene %s: %s", scene_id, e)
            self._sigs[scene_id] = sig if sig is not None and "full" in sig else None
        return self._sigs[scene_id]

    async def _sprite(self, cand: Candidate, sa: dict, sb: dict) -> Optional[tuple[int, int]]:
        """Sprite score of a pair of scenes; tiles recurring in the studio's other scenes are left out."""
        sig_a, sig_b = await self._signature(cand.scene_a), await self._signature(cand.scene_b)
        if sig_a is None or sig_b is None:
            return None
        skip_a = skip_b = None
        studio_a, studio_b = (sa.get("studio") or {}).get("id"), (sb.get("studio") or {}).get("id")
        if studio_a and studio_a == studio_b:
            pool = [s for s in self._studio_scenes.get(str(studio_a), [])
                    if s not in (cand.scene_a, cand.scene_b)][:STOP_SCENES]
            others = [o for o in [await self._signature(s) for s in pool] if o is not None]
            skip_a = await asyncio.to_thread(sprites.stop_tiles_for, sig_a, others, STOP_MIN_SCENES)
            skip_b = await asyncio.to_thread(sprites.stop_tiles_for, sig_b, others, STOP_MIN_SCENES)
        score = sprites.sprite_score(sig_a, sig_b, skip_a, skip_b)
        return (score["d85"], score["d80"]) if score else None

    async def _verify(self, cand: Candidate) -> Optional[tuple[PairEvidence, str]]:
        sa, sb = await self._scene(cand.scene_a), await self._scene(cand.scene_b)
        if not sa or not sb:
            return None
        ev = PairEvidence(overlap=cand.overlap)
        fa, fb = self._pair_files(cand, sa, sb)
        if fa and fb:
            res = await self._audio(fa, fb)
            if res:
                ev.ber, ev.ber_offset_s = res
            kf = await self._keyframes(fa, fb)
            if kf:
                ev.keyframe_fraction, ev.keyframe_offset_s = kf["fraction"], kf["offset_s"]
        # the picture can only change the class when the audio is missing or not strong, and the keyframes aren't
        audio_settles = ev.ber is not None and ev.ber < AUDIO_STRONG
        keyframes_settle = ev.keyframe_fraction is not None and ev.keyframe_fraction >= KEYFRAME_STRONG
        if cand.scene_a != cand.scene_b and not audio_settles and not keyframes_settle:
            ev.sprite = await self._sprite(cand, sa, sb)
        return ev, pair_class(ev, sa, sb)

    # ---------------- keeper ----------------

    async def _known_health(self, f: _File) -> Optional[str]:
        if f.cache_key in self._health:
            return self._health[f.cache_key]
        if self._store is not None and f.content:
            def read():
                cid = _content_id(self._store, f.content)
                return None if cid is None else self._store.get_health(cid)
            row = await asyncio.to_thread(read)
            if row:
                self._health[f.cache_key] = row["status"]
                return row["status"]
        return None

    async def _check_health(self, f: _File) -> Optional[str]:
        """Run the health check on the file's local path and cache the result; None when it has no local path."""
        local = self._local(f.path) if f.path else None
        if local is None:
            return None
        res = await health.check_health(local)
        self._health[f.cache_key] = res["status"]
        if self._store is not None and f.content:
            def write():
                cid = _content_id(self._store, f.content, create=True)
                self._store.put_health(cid, res["status"], res["reasons"], res["params"])
            await asyncio.to_thread(write)
        return res["status"]

    async def _suggestion(self, item: Item, excl) -> Optional[dict]:
        """The keeper suggestion for a set from its scenes as they are now, or None when fewer than 2 files are left."""
        scenes = []
        for sid in item.scene_ids:
            scene = await self.stash.get_scene_for_review(sid)
            if not scene:
                return None
            scenes.append(scene)
        # the same keys as review.duplicates, so the suggestion names the files the card does
        files: list[tuple[_File, keeper.FileFeatures]] = []
        seen_ids: set[str] = set()
        used: set[str] = set()
        for s in scenes:
            for f in s.get("files") or []:
                if str(f["id"]) in seen_ids:
                    continue
                seen_ids.add(str(f["id"]))
                feat = keeper.FileFeatures.from_stash_file(f)
                key = feat.key or f"id:{f['id']}"
                if key in used:
                    key = f"{key}@{f['id']}"
                used.add(key)
                files.append((_File(str(s["id"]), f), dataclasses.replace(feat, key=key)))
        if len(files) < 2:
            return None

        kind = "contained" if item.cls == "contained" else "duplicate"
        codecs = set(excl.settings.disfavored_video_codecs)
        containers = set(excl.settings.disfavored_containers)
        status: dict[str, str] = {}
        for f, feat in files:
            st = await self._known_health(f)
            if st:
                status[feat.key] = st
        attempted: set[str] = set()
        while True:
            sug = keeper.suggest([feat for _, feat in files], kind=kind, health=status,
                                 disfavored_codecs=codecs, disfavored_containers=containers)
            todo = [(f, feat) for f, feat in files
                    if feat.key in sug.keep and feat.key not in status and feat.key not in attempted]
            if not todo or self.is_stop_requested():
                break
            for f, feat in todo:      # each pass attempts at least one more file, so this ends
                attempted.add(feat.key)
                st = await self._check_health(f)
                if st:
                    status[feat.key] = st
        return {"keep": sug.keep, "delete": sug.delete, "reason": sug.reason, "flags": sug.flags, "held": sug.held}

    # ---------------- run ----------------

    async def run(self, incremental: bool = True) -> AnalysisResult:
        excl = await curation_settings.current(self.stash)     # fails closed: ExclusionUnavailable propagates
        self.exclusion = excl
        from jobs.content_fingerprint_job import STORE_NAME, _data_dir
        store_path = os.path.join(_data_dir(), STORE_NAME)
        self._store = await asyncio.to_thread(ContentFpStore, store_path) if os.path.exists(store_path) else None
        try:
            return await self._run(excl)
        finally:
            if self._store is not None:
                self._store.close()

    async def _run(self, excl) -> AnalysisResult:
        stats = self.stats
        candidates = await generate(self.stash, self._store, set(excl.scene_ids))
        self.set_items_total(len(candidates))

        scene_ids = sorted({s for c in candidates for s in (c.scene_a, c.scene_b)}, key=int)
        for sid in scene_ids:
            if self.is_stop_requested():
                break
            await self._scene(sid)
        for sid in scene_ids:
            studio = ((self._scenes.get(sid) or {}).get("studio") or {}).get("id")
            if studio:
                self._studio_scenes.setdefault(str(studio), []).append(sid)

        verified: list[tuple[Candidate, PairEvidence, str]] = []
        for i, cand in enumerate(candidates, 1):
            if self.is_stop_requested():
                break
            out = await self._verify(cand)
            if out and out[1] != "different":
                verified.append((cand, out[0], out[1]))
            self.update_progress(i, stats["created"])

        pending = {r.target_id: r for r in self.rec_db.get_recommendations(
            status="pending", type=TYPE_ID, target_type=TARGET_TYPE, limit=1_000_000_000)}
        previous = {key: {str(s) for s in (r.details or {}).get("scene_ids") or []} for key, r in pending.items()}
        items, superseded = build_items(verified, previous)
        stats.update({f"cls_{i.cls}": 1 for i in items})

        produced: set[str] = set()
        for item in items:
            if self.is_stop_requested():
                break
            details = item.to_details()
            if item.cls in DELETING:
                suggestion = await self._suggestion(item, excl)
                if suggestion is None:
                    continue
                details["suggestion"] = suggestion
            produced.add(item.key)
            if self.is_dismissed(TARGET_TYPE, item.key):
                stats["skipped"] += 1
                continue
            _, how = self.rec_db.upsert_recommendation(
                type=TYPE_ID, target_type=TARGET_TYPE, target_id=item.key, details=details,
                confidence=CONFIDENCE[item.cls], source_analysis_id=self.run_id)
            stats[how] += 1

        retired: set[str] = set()
        for old in superseded:
            new = next((i.key for i in items if i.cls == "duplicate" and i.key in produced
                        and set(i.scene_ids) & previous.get(old, set())), None)
            if new and old in pending:
                self.rec_db.resolve_recommendation(pending[old].id, "superseded", {"superseded_by": new})
                retired.add(old)
                stats["superseded"] += 1
        if not self.is_stop_requested():
            for key, rec in pending.items():
                if key not in produced and key not in retired:
                    self.rec_db.resolve_recommendation(rec.id, "stale", {"reason": "no longer found as a duplicate"})
                    stats["stale"] += 1

        classes = {k[4:]: v for k, v in sorted(stats.items()) if k.startswith("cls_")}
        logger.warning(
            "scene_duplicates: %d candidates, %d verified, sets %s; created %d, updated %d, unchanged %d, skipped %d, "
            "superseded %d, stale %d%s",
            len(candidates), len(verified), classes or "none", stats["created"], stats["updated"],
            stats["unchanged"], stats["skipped"], stats["superseded"], stats["stale"],
            " (stopped)" if self.is_stop_requested() else "")
        return AnalysisResult(items_processed=len(candidates), recommendations_created=stats["created"])

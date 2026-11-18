"""scene_duplicates: review a duplicate set from #130 as one card.

A recommendation names a set of scenes (details.scene_ids) and the verified pairs between them. The card
re-reads every scene, suggests which files to keep (duplicates.keeper), and plans the accept: merge the scenes
into the one with the most history, then delete every file not kept.

Files are picked by oshash, never by position, so a pick means the same file however the card orders them.
When two files of the set share an oshash (byte-identical copies), the later ones are keyed
``<oshash>@<file id>`` so a pick still names one file; a file without an oshash is keyed ``id:<file id>``.

File health comes from the content index store (DATA_DIR/content_fp.db), opened read-only with plain sqlite
(ContentFpStore would create its schema); content files match by (oshash, size). No store means no health.

Per class:
- duplicate: merge, keep the keeper (and a second file when the keeper suggests one), delete the rest
- contained: the long file is never deleted. When the short file is deleted, its scene merges into the long file's
  scene; when nothing is deleted (split the longer file first, or both kept) the scenes stay separate
- mislabel: merge into the scene the user says has the right metadata; nothing is unioned
- language_variant: merge, keep every file
- partial, review: nothing changes; accept only resolves
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging
import os
import sqlite3

import curation_settings
from duplicates import keeper
from duplicates.sets import MOMENT_START_S

from .plan import (
    FileRef,
    Plan,
    Preview,
    PrimaryFileChange,
    SceneMerge,
    scene_field_values,
    values_equal,
)
from .types import InvalidOption, ReviewType, TargetMissing, register_review_type

logger = logging.getLogger(__name__)

TYPE_ID = "scene_duplicates"
KEEP_ALL = frozenset({"language_variant", "partial", "review"})
NO_MERGE = frozenset({"partial", "review"})
SCALAR_FIELDS = ("title", "details", "date", "code", "director", "studio_id", "rating100")
LIST_FIELDS = ("urls", "performer_ids", "tag_ids", "gallery_ids")
SAME_ENCODE_DURATION_S = 1.0


def _num(scene_id) -> tuple:
    s = str(scene_id)
    return (0, int(s), s) if s.isdigit() else (1, 0, s)


def _fps(f: dict) -> dict[str, str]:
    return {fp["type"]: fp["value"] for fp in f.get("fingerprints") or []}


def _size_text(n: int) -> str:
    if n >= 1024 ** 3:
        return f"{n / 1024 ** 3:.1f} GB"
    return f"{n / 1024 ** 2:.0f} MB"


def _empty(v) -> bool:
    return v is None or v == ""


@dataclasses.dataclass
class _File:
    scene_id: str
    stash: dict
    feat: keeper.FileFeatures

    @property
    def key(self) -> str:
        return self.feat.key

    @property
    def file_id(self) -> str:
        return str(self.stash["id"])


def _data_dir() -> str:
    from jobs.content_fingerprint_job import _data_dir as job_data_dir
    return job_data_dir()


def _read_health(files: list[tuple[str, int]]) -> dict[tuple[str, int], str]:
    """Health status by (oshash, size) from the content store, read-only; {} without a store."""
    from jobs.content_fingerprint_job import STORE_NAME
    path = os.path.join(_data_dir(), STORE_NAME)
    if not files or not os.path.exists(path):
        return {}
    out: dict[tuple[str, int], str] = {}
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)
        try:
            for oshash, size in files:
                row = conn.execute(
                    "SELECT h.status FROM content_files f JOIN file_health h ON h.file_id = f.id "
                    "WHERE f.oshash = ? AND f.size_bytes = ?", (oshash, size)).fetchone()
                if row:
                    out[(oshash, size)] = row[0]
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.warning("Could not read file health from %s: %s", path, e)
        return {}
    return out


def _history(scene: dict) -> tuple:
    return (
        bool(scene.get("play_count") or scene.get("o_counter") or scene.get("rating100") is not None),
        bool(scene.get("organized")),
        len(scene.get("groups") or []),
    )


def _rank(scene: dict) -> tuple:
    """Sort key: most history first."""
    return tuple(-int(x) for x in _history(scene))


def _merge_values(dest: dict, sources: list[dict]) -> dict:
    """Destination's values, empty scalars filled and lists unioned from sources (in history order)."""
    current = scene_field_values(dest)
    merged = {k: (list(v) if isinstance(v, list) else v) for k, v in current.items()}
    src_values = [scene_field_values(s) for s in sources]
    for k in SCALAR_FIELDS:
        if _empty(merged[k]):
            merged[k] = next((sv[k] for sv in src_values if not _empty(sv[k])), merged[k])
    for k in LIST_FIELDS:
        for sv in src_values:
            merged[k].extend(v for v in sv[k] if v not in merged[k])
    seen = {g["group_id"] for g in merged["groups"]}
    for sv in src_values:
        for g in sv["groups"]:
            if g["group_id"] not in seen:
                seen.add(g["group_id"])
                merged["groups"].append(g)
    merged["organized"] = current["organized"] or any(sv["organized"] for sv in src_values)
    return {k: v for k, v in merged.items() if not values_equal(k, v, current[k])}


def _identical(f: _File, keepers: list[_File]) -> bool:
    """f shares an md5 or oshash with a keeper."""
    fp = _fps(f.stash)
    return any(any(fp.get(t) and fp.get(t) == _fps(k.stash).get(t) for t in ("md5", "oshash")) for k in keepers)


def _same_encode(f: _File, keepers: list[_File]) -> bool:
    """f has a keeper's codec and frame size, and its duration within 1 s."""
    a = f.feat
    return any(a.codec and a.codec == b.codec and a.width == b.width and a.height == b.height
               and abs(a.duration - b.duration) <= SAME_ENCODE_DURATION_S for b in (k.feat for k in keepers))


def _batch_eligible(cls: str, chosen: dict, default: dict, sug: keeper.KeeperSuggestion,
                    kept: list[_File], doomed: list[_File]) -> bool:
    """Batch accept only the plain case: the default for a duplicate set, one file kept, at least one deleted,
    no flag from the keeper (a tie only when every deleted file is identical to a keeper), and every deleted
    file identical to a keeper or the same encode. Anything else is reviewed one by one."""
    if chosen != default or cls != "duplicate" or sug.held or not doomed or chosen["keep_extra"]:
        return False
    identical = all(_identical(f, kept) for f in doomed)
    flags = set(sug.flags)
    if flags - {"tie"} or ("tie" in flags and not identical):
        return False
    return identical or all(_identical(f, kept) or _same_encode(f, kept) for f in doomed)


@register_review_type
class SceneDuplicatesReviewType(ReviewType):
    type_id = TYPE_ID
    display_name = "Duplicate Scenes"
    description = "Scenes holding the same content: merge them and keep the best file"

    def scene_ids(self, rec) -> set[str]:
        return {str(s) for s in (rec.details or {}).get("scene_ids") or []}

    async def preview(self, rec, option: dict | None, stash) -> Preview:
        details = rec.details or {}
        cls = details.get("cls") or "review"
        pairs = details.get("pairs") or []

        # 1. current state
        scenes: list[dict] = []
        for sid in details.get("scene_ids") or []:
            scene = await stash.get_scene_for_review(str(sid))
            if not scene:
                raise TargetMissing(f"Scene {sid} no longer exists")
            scenes.append(scene)
        by_scene = {str(s["id"]): s for s in scenes}

        # 2. candidates: every file of every scene, keyed by oshash (unique within the set)
        files: list[_File] = []
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
                files.append(_File(str(s["id"]), f, dataclasses.replace(feat, key=key)))
        if len(files) < 2:
            raise TargetMissing(f"Duplicate set {rec.target_id} has fewer than 2 files left")
        by_key = {f.key: f for f in files}

        content = {f.key: (_fps(f.stash).get("oshash"), int(f.stash.get("size") or 0)) for f in files}
        status = await asyncio.to_thread(_read_health, [c for c in content.values() if c[0]])
        health = {k: status[c] for k, c in content.items() if c in status}
        try:
            settings = (await curation_settings.current(stash)).settings
            codecs, containers = set(settings.disfavored_video_codecs), set(settings.disfavored_containers)
        except Exception as e:
            logger.warning("Could not read disfavored formats, using none: %s", e)
            codecs, containers = set(), set()

        # 3. the suggestion
        kind = "contained" if cls == "contained" else "duplicate"
        sug = keeper.suggest([f.feat for f in files], kind=kind, health=health,
                             disfavored_codecs=codecs, disfavored_containers=containers)
        keep_all = cls in KEEP_ALL or sug.held
        long_file = max(files, key=lambda f: (f.feat.duration, f.key)) if cls == "contained" else None

        def destination_for(main: _File) -> str:
            """Most history first, then the scene holding the main keeper, then the lowest id."""
            return str(min(scenes, key=lambda s: (_rank(s), str(s["id"]) != main.scene_id, _num(s["id"])))["id"])

        default = {"keep": sug.keep[0], "keep_extra": sug.keep[1] if len(sug.keep) > 1 else ""}
        if cls == "mislabel":
            default["metadata_from"] = destination_for(by_key[sug.keep[0]])

        # 4. the chosen option, resolved to current files
        option = option or {}
        chosen = {k: str(option[k]) if option.get(k) is not None else default[k] for k in default}
        if chosen["keep"] not in by_key:
            raise InvalidOption(f"File {chosen['keep']!r} is not in this duplicate set")
        if chosen["keep_extra"] and chosen["keep_extra"] not in by_key:
            raise InvalidOption(f"File {chosen['keep_extra']!r} is not in this duplicate set")
        if chosen["keep_extra"] == chosen["keep"]:
            raise InvalidOption("The second file to keep must differ from the first")
        if cls == "mislabel" and chosen["metadata_from"] not in by_scene:
            raise InvalidOption(f"Scene {chosen['metadata_from']} is not in this duplicate set")

        main = by_key[chosen["keep"]]
        kept_keys = {chosen["keep"]} | ({chosen["keep_extra"]} if chosen["keep_extra"] else set())
        if keep_all:
            kept_keys = set(by_key)
        if long_file is not None:
            kept_keys.add(long_file.key)
        kept = [f for f in files if f.key in kept_keys]
        doomed = [f for f in files if f.key not in kept_keys]

        plan = Plan(rec_id=rec.id, rec_type=self.type_id, option=chosen)
        plan.keep_files = [FileRef.from_stash(f.scene_id, f.stash) for f in kept]
        plan.delete_files = [FileRef.from_stash(f.scene_id, f.stash) for f in doomed]

        # 5-6. destination and merge
        merge = len(scenes) > 1 and cls not in NO_MERGE
        if cls == "mislabel":
            dest_id = chosen["metadata_from"]
        elif cls == "contained":
            # merge first, then delete the lower quality file: the scene of the longer cut stays; with nothing
            # deleted (split the long file first, or both kept) the scenes stay separate
            dest_id = long_file.scene_id
            merge = merge and bool(doomed)
        else:
            dest_id = destination_for(main)
        dest = by_scene[dest_id]
        if merge:
            sources = sorted((s for s in scenes if str(s["id"]) != dest_id), key=lambda s: (_rank(s), _num(s["id"])))
            values = {} if cls == "mislabel" else _merge_values(dest, sources)
            plan.merge = SceneMerge(dest_id, [str(s["id"]) for s in sources], values, sources)

        # 7. primary file
        dest_files = dest.get("files") or []
        doomed_ids = {f.file_id for f in doomed}
        if dest_files and str(dest_files[0]["id"]) in doomed_ids:
            plan.primary_file = PrimaryFileChange(dest_id, str(dest_files[0]["id"]), main.file_id)

        # 8. evidence
        def still_t(f: _File) -> float:
            oshash = _fps(f.stash).get("oshash")
            for p in pairs:
                m = p.get("moment") or {}
                if oshash and p.get("file_a") == oshash and str(p.get("scene_a")) == f.scene_id:
                    return m.get("t_a", MOMENT_START_S)
                if oshash and p.get("file_b") == oshash and str(p.get("scene_b")) == f.scene_id:
                    return m.get("t_b", MOMENT_START_S)
            for p in pairs:
                m = p.get("moment") or {}
                if str(p.get("scene_a")) == f.scene_id:
                    return m.get("t_a", MOMENT_START_S)
                if str(p.get("scene_b")) == f.scene_id:
                    return m.get("t_b", MOMENT_START_S)
            return MOMENT_START_S

        if cls == "language_variant":
            reason = "The same scene in another language: merges the scenes and keeps every file."
        elif cls == "partial":
            reason = "The scenes only partly overlap, so both stay as they are."
        elif cls == "review":
            reason = "The evidence is unclear, so nothing changes; look at the pair before acting on it."
        else:
            reason = sug.reason

        primary_ids = {str(s["id"]): str((s.get("files") or [{}])[0].get("id")) for s in scenes}
        evidence = {
            "scenes": [
                {
                    "id": str(s["id"]), "title": s.get("title") or "",
                    "studio_id": (s.get("studio") or {}).get("id"),
                    "performer_ids": [p["id"] for p in s.get("performers") or []],
                    "date": s.get("date"), "play_count": s.get("play_count") or 0,
                    "o_counter": s.get("o_counter") or 0, "rating100": s.get("rating100"),
                    "organized": bool(s.get("organized")),
                    "files": [
                        {
                            "file_id": f.file_id, "oshash": f.key, "path": f.stash.get("path"),
                            "size": int(f.stash.get("size") or 0), "duration": f.stash.get("duration"),
                            "video_codec": f.stash.get("video_codec"), "width": f.stash.get("width"),
                            "height": f.stash.get("height"), "bit_rate": f.stash.get("bit_rate"),
                            "frame_rate": f.stash.get("frame_rate"), "format": f.stash.get("format"),
                            "is_primary": primary_ids.get(f.scene_id) == f.file_id,
                            "health": health.get(f.key),
                        }
                        for f in files if f.scene_id == str(s["id"])
                    ],
                }
                for s in scenes
            ],
            "pairs": pairs,
            "cls": cls,
            "keeper_reason": reason,
            "flags": list(sug.flags),
            "stills": [{"scene_id": f.scene_id, "file_id": f.file_id, "t": still_t(f)} for f in files],
            "playback": {"scene_id": dest_id, "t": still_t(main)},
        }

        parts = []
        if plan.merge:
            parts.append(f"merge {len(plan.merge.scene_ids)} scenes")
        if plan.delete_files:
            n = len(plan.delete_files)
            parts.append(f"delete {n} file{'s' if n != 1 else ''}, reclaim {_size_text(plan.bytes_reclaimed)}")
        summary = f"{reason} Accept: {', '.join(parts)}." if parts else f"{reason} Accept only resolves."

        # 9. choices
        def file_label(f: _File) -> str:
            st = f.stash
            res = f"{st.get('width') or '?'}x{st.get('height') or '?'}"
            base = st.get("basename") or os.path.basename(st.get("path") or "")
            return (f"scene {f.scene_id} · {res} {st.get('video_codec') or '?'} · "
                    f"{_size_text(int(st.get('size') or 0))} · {base}")

        choices: list[dict] = []
        if cls not in NO_MERGE:
            values = [{"value": f.key, "label": file_label(f)} for f in files]
            choices.append({"field": "keep", "label": "File to keep", "kind": "pick", "values": values})
            choices.append({"field": "keep_extra", "label": "Also keep", "kind": "pick",
                            "values": [{"value": "", "label": "none"}, *values]})
        if cls == "mislabel":
            choices.append({"field": "metadata_from", "label": "Metadata from", "kind": "pick", "values": [
                {"value": str(s["id"]), "label": f"scene {s['id']}: {s.get('title') or '(untitled)'}"}
                for s in scenes]})

        # 10. batch eligibility
        high = _batch_eligible(cls, chosen, default, sug, kept, doomed)

        return Preview(plan=plan, evidence=evidence, choices=choices, default_option=default,
                       high_confidence=high, summary=summary)

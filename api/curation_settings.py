"""Curation settings shared through Stash (#125): the exclusion list and disfavored formats.

They live in the stash-sense plugin's settings in Stash, so any tool with a
Stash API key can read them from ``configuration { plugins }``. Each value is
a comma-separated string. Stash's configurePlugin replaces the plugin's whole
settings map, so writers must read, merge, then write.

The exclusion list resolves to a set of scene ids. Nothing in stash-sense
fingerprints, recommends or writes an excluded scene. When Stash can't be
read the guard fails closed: callers get ExclusionUnavailable, not an empty set.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import asdict, dataclass

PLUGIN_ID = "stash-sense"

KEY_TAGS = "excludeTagIds"
KEY_GROUPS = "excludeGroupIds"
KEY_STUDIOS = "excludeStudioIds"
KEY_CODECS = "disfavoredVideoCodecs"
KEY_CONTAINERS = "disfavoredContainers"

CACHE_SECONDS = 60.0

# Where a recommendation names scenes: the target when target_type is
# "scene" (some types use a composite target_id such as "12:34" or
# "12|endpoint|uuid", which never matches a scene id), plus these details keys. rec_scene_ids() and
# scene_exclusion_sql() must agree; tests check that they do.
DETAIL_SCENE_KEYS = ("scene_id", "scene_a_id", "scene_b_id", "local_scene_id")
# Details keys holding a list of scene ids (#130 duplicate sets: target_type "scene_set").
DETAIL_SCENE_LIST_KEYS = ("scene_ids",)


class ExclusionUnavailable(RuntimeError):
    """The exclusion list could not be read from Stash."""


class SceneExcluded(PermissionError):
    """A request named scenes on the exclusion list."""

    def __init__(self, scene_ids):
        self.scene_ids = sorted(scene_ids, key=lambda s: (len(s), s))
        super().__init__(
            f"Scene(s) {', '.join(self.scene_ids)} are on the Stash Sense exclusion list; "
            "remove them from it in Settings > Curation to act on them"
        )


def _split(value, *, lower: bool = False, ids: bool = False) -> tuple[str, ...]:
    if value is None:
        return ()
    items = value if isinstance(value, list) else str(value).split(",")
    out: list[str] = []
    for item in items:
        s = str(item).strip()
        if lower:
            s = s.lower()
        # Stash ids are integers. Anything else can't name an entity, and
        # Stash's hierarchical filters fail on it, so it is dropped.
        if ids and not s.isdigit():
            continue
        if s and s not in out:
            out.append(s)
    return tuple(out)


@dataclass(frozen=True)
class CurationSettings:
    exclude_tag_ids: tuple[str, ...] = ()
    exclude_group_ids: tuple[str, ...] = ()
    exclude_studio_ids: tuple[str, ...] = ()
    disfavored_video_codecs: tuple[str, ...] = ()
    disfavored_containers: tuple[str, ...] = ()

    @classmethod
    def from_plugin_config(cls, cfg: dict | None) -> CurationSettings:
        cfg = cfg or {}
        return cls(
            exclude_tag_ids=_split(cfg.get(KEY_TAGS), ids=True),
            exclude_group_ids=_split(cfg.get(KEY_GROUPS), ids=True),
            exclude_studio_ids=_split(cfg.get(KEY_STUDIOS), ids=True),
            disfavored_video_codecs=_split(cfg.get(KEY_CODECS), lower=True),
            disfavored_containers=_split(cfg.get(KEY_CONTAINERS), lower=True),
        )

    @property
    def has_exclusions(self) -> bool:
        return bool(self.exclude_tag_ids or self.exclude_group_ids or self.exclude_studio_ids)

    def is_disfavored(self, file: dict) -> bool:
        """True when a Stash file's codec or container is on a disfavored list."""
        codec = (file.get("video_codec") or "").lower()
        container = (file.get("format") or "").lower()
        return (bool(codec) and codec in self.disfavored_video_codecs) or (
            bool(container) and container in self.disfavored_containers
        )

    def to_dict(self) -> dict:
        return {k: list(v) for k, v in asdict(self).items()}


@dataclass(frozen=True)
class Exclusion:
    """The settings and the scene ids they exclude, as of resolved_at."""

    settings: CurationSettings
    scene_ids: frozenset[str]
    resolved_at: float

    def excluded(self, scene_ids) -> set[str]:
        return {str(s) for s in scene_ids if s is not None} & self.scene_ids

    def is_excluded(self, scene_id) -> bool:
        return scene_id is not None and str(scene_id) in self.scene_ids

    def require_not_excluded(self, scene_ids) -> None:
        hit = self.excluded(scene_ids)
        if hit:
            raise SceneExcluded(hit)


EMPTY = Exclusion(CurationSettings(), frozenset(), 0.0)


async def load_settings(stash) -> CurationSettings:
    return CurationSettings.from_plugin_config(await stash.get_plugin_configuration(PLUGIN_ID))


async def resolve(stash, settings: CurationSettings) -> frozenset[str]:
    """Scene ids excluded by settings; tags, groups and studios include their descendants."""
    ids: set[str] = set()
    for field, values in (
        ("tags", settings.exclude_tag_ids),
        ("groups", settings.exclude_group_ids),
        ("studios", settings.exclude_studio_ids),
    ):
        if values:
            found = await stash.find_scene_ids(
                {field: {"value": list(values), "modifier": "INCLUDES", "depth": -1}}
            )
            ids.update(str(i) for i in found)
    return frozenset(ids)


class ExclusionCache:
    """The current Exclusion, refreshed from Stash at most every CACHE_SECONDS."""

    def __init__(self, ttl: float = CACHE_SECONDS, clock=time.monotonic):
        self.ttl = ttl
        self.clock = clock
        self._value: Exclusion | None = None
        self._last_good: Exclusion | None = None
        self._generation = 0
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        # A read already in flight started before this call; the generation
        # bump stops it from caching what it read.
        self._generation += 1
        self._value = None

    async def get(self, stash, *, max_age: float | None = None, allow_stale: bool = False) -> Exclusion:
        """The current Exclusion.

        allow_stale: when Stash can't be read, return the last list read
        successfully instead of raising. For read-only listings only; jobs
        and writes must fail closed.
        """
        max_age = self.ttl if max_age is None else max_age
        async with self._lock:
            v = self._value
            if v is not None and self.clock() - v.resolved_at <= max_age:
                return v
            generation = self._generation
            try:
                settings = await load_settings(stash)
                scene_ids = await resolve(stash, settings)
            except Exception as e:
                if allow_stale and self._last_good is not None:
                    return self._last_good
                raise ExclusionUnavailable(f"Could not read the exclusion list from Stash: {e}") from e
            fresh = Exclusion(settings, scene_ids, self.clock())
            self._last_good = fresh
            if generation == self._generation:
                self._value = fresh
            return fresh


_cache = ExclusionCache()


async def current(stash, *, max_age: float | None = None, allow_stale: bool = False) -> Exclusion:
    """The exclusion list. Jobs and writes pass max_age=0 for a fresh read; see ExclusionCache.get."""
    return await _cache.get(stash, max_age=max_age, allow_stale=allow_stale)


def invalidate() -> None:
    _cache.invalidate()


# ---------------- recommendations ----------------


def rec_scene_ids(target_type: str, target_id, details: dict | None) -> set[str]:
    """Scenes a recommendation names."""
    ids: set[str] = set()
    if target_type == "scene" and target_id is not None:
        ids.add(str(target_id))
    for key in DETAIL_SCENE_KEYS:
        v = (details or {}).get(key)
        if v is not None:
            ids.add(str(v))
    for key in DETAIL_SCENE_LIST_KEYS:
        v = (details or {}).get(key)
        if isinstance(v, list):
            ids.update(str(i) for i in v if i is not None)
    return ids


def scene_exclusion_sql(scene_ids) -> tuple[str, list]:
    """An SQL condition (and its params) that is true when a recommendations row names none of scene_ids.

    Mirrors rec_scene_ids(). The ids travel as one JSON array parameter, so
    the statement doesn't grow with the size of the list.
    """
    if not scene_ids:
        return "1=1", []
    import json

    arr = json.dumps(sorted(str(s) for s in scene_ids))
    in_set = "IN (SELECT value FROM json_each(?))"
    parts = [f"(target_type = 'scene' AND target_id {in_set})"]
    parts += [f"CAST(json_extract(details, '$.{k}') AS TEXT) {in_set}" for k in DETAIL_SCENE_KEYS]
    # Each part is true when the row names one of the ids; the whole is NOT (any part).
    parts += [
        f"(json_type(details, '$.{k}') = 'array' AND EXISTS ("
        f"SELECT 1 FROM json_each(details, '$.{k}') WHERE CAST(value AS TEXT) {in_set}))"
        for k in DETAIL_SCENE_LIST_KEYS
    ]
    return "NOT (" + " OR ".join(f"COALESCE({p}, 0)" for p in parts) + ")", [arr] * len(parts)

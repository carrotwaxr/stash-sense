"""The benchmark set: StashDB-linked scenes with known performers.

A scene qualifies when it has a StashDB id, a dated file with a duration, at
least one performer, and every performer linked to StashDB (so the expected
set is complete). Scenes are sampled evenly across era x resolution strata
with a fixed seed, then frozen to JSON so later runs score the same scenes.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

STASHDB_ENDPOINT = "https://stashdb.org/graphql"
SET_FORMAT_VERSION = 1

ERAS = ["pre-2010", "2010-2014", "2015-2019", "2020+"]
RESOLUTIONS = ["SD", "720p", "1080p", "4K"]


@dataclass
class ExpectedPerformer:
    stashdb_id: str
    name: str


@dataclass
class BenchmarkScene:
    scene_id: str
    title: str
    date: str
    era: str
    resolution: str
    width: int
    height: int
    duration: float
    path: Optional[str]
    expected: list[ExpectedPerformer] = field(default_factory=list)

    @property
    def stratum(self) -> str:
        return f"{self.era} / {self.resolution}"


@dataclass
class BenchmarkSet:
    seed: int
    created_at: str
    scenes: list[BenchmarkScene]
    version: int = SET_FORMAT_VERSION

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=2))
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "BenchmarkSet":
        data = json.loads(Path(path).read_text())
        if data.get("version") != SET_FORMAT_VERSION:
            raise ValueError(f"{path}: unsupported set version {data.get('version')}")
        scenes = [
            BenchmarkScene(
                **{k: v for k, v in s.items() if k != "expected"},
                expected=[ExpectedPerformer(**p) for p in s["expected"]],
            )
            for s in data["scenes"]
        ]
        return cls(seed=data["seed"], created_at=data["created_at"], scenes=scenes)


def era_for(date: Optional[str]) -> Optional[str]:
    """Bucket a YYYY[-MM-DD] date into an era, or None when undated."""
    if not date or len(date) < 4 or not date[:4].isdigit():
        return None
    year = int(date[:4])
    if year < 2010:
        return "pre-2010"
    if year < 2015:
        return "2010-2014"
    if year < 2020:
        return "2015-2019"
    return "2020+"


def resolution_for(width: int, height: int) -> Optional[str]:
    """Bucket by the short side, so 1280x720 is 720p whatever its aspect."""
    short = min(width or 0, height or 0)
    if short <= 0:
        return None
    if short < 720:
        return "SD"
    if short < 1080:
        return "720p"
    if short < 2160:
        return "1080p"
    return "4K"


def _stashdb_id(stash_ids: list[dict]) -> Optional[str]:
    for sid in stash_ids or []:
        if sid.get("endpoint") == STASHDB_ENDPOINT:
            return sid.get("stash_id")
    return None


def scene_from_stash(data: dict) -> Optional[BenchmarkScene]:
    """Convert a findScenes row, or None when it doesn't qualify."""
    if not _stashdb_id(data.get("stash_ids")):
        return None
    files = data.get("files") or []
    if not files:
        return None
    f = files[0]
    era = era_for(data.get("date"))
    res = resolution_for(f.get("width") or 0, f.get("height") or 0)
    if not era or not res or not f.get("duration"):
        return None
    performers = data.get("performers") or []
    if not performers:
        return None
    expected = []
    for p in performers:
        pid = _stashdb_id(p.get("stash_ids"))
        if not pid:
            return None
        expected.append(ExpectedPerformer(stashdb_id=pid, name=p.get("name", "")))
    return BenchmarkScene(
        scene_id=str(data["id"]),
        title=data.get("title") or "",
        date=data["date"],
        era=era,
        resolution=res,
        width=f["width"],
        height=f["height"],
        duration=float(f["duration"]),
        path=f.get("path"),
        expected=expected,
    )


def stratified_sample(
    scenes: list[BenchmarkScene], count: int, seed: int
) -> list[BenchmarkScene]:
    """Take ``count`` scenes spread as evenly as availability allows.

    Each stratum gets an equal share; shares a small stratum can't fill are
    handed to the others in later passes. Output order is stable for a seed.
    """
    rng = random.Random(seed)
    pools: dict[str, list[BenchmarkScene]] = {}
    for s in sorted(scenes, key=lambda s: s.scene_id):
        pools.setdefault(s.stratum, []).append(s)
    for key in sorted(pools):
        rng.shuffle(pools[key])

    taken: dict[str, int] = {k: 0 for k in pools}
    remaining = min(count, len(scenes))
    while remaining > 0:
        open_keys = [k for k in sorted(pools) if taken[k] < len(pools[k])]
        if not open_keys:
            break
        share = max(1, remaining // len(open_keys))
        for k in open_keys:
            if remaining == 0:
                break
            add = min(share, len(pools[k]) - taken[k], remaining)
            taken[k] += add
            remaining -= add

    picked = [s for k in sorted(pools) for s in pools[k][: taken[k]]]
    return sorted(picked, key=lambda s: (ERAS.index(s.era), RESOLUTIONS.index(s.resolution), s.scene_id))


SCENES_QUERY = """
query BenchmarkScenes($filter: FindFilterType, $scene_filter: SceneFilterType) {
  findScenes(filter: $filter, scene_filter: $scene_filter) {
    count
    scenes {
      id
      title
      date
      files { path width height duration }
      stash_ids { endpoint stash_id }
      performers { name stash_ids { endpoint stash_id } }
    }
  }
}
"""


async def fetch_candidates(client, per_page: int = 500) -> list[BenchmarkScene]:
    """Page through every StashDB-linked scene and keep the qualifying ones."""
    scene_filter = {
        "stash_id_endpoint": {"endpoint": STASHDB_ENDPOINT, "modifier": "NOT_NULL"},
        "performer_count": {"value": 0, "modifier": "GREATER_THAN"},
    }
    out: list[BenchmarkScene] = []
    page = 1
    while True:
        data = await client._execute(
            SCENES_QUERY,
            {"filter": {"page": page, "per_page": per_page, "sort": "id", "direction": "ASC"},
             "scene_filter": scene_filter},
        )
        rows = data["findScenes"]["scenes"]
        for row in rows:
            scene = scene_from_stash(row)
            if scene:
                out.append(scene)
        if len(rows) < per_page:
            return out
        page += 1


async def build_set(client, count: int, seed: int) -> tuple[BenchmarkSet, dict[str, int]]:
    """Build a new set; also return candidate counts per stratum."""
    candidates = await fetch_candidates(client)
    available: dict[str, int] = {}
    for s in candidates:
        available[s.stratum] = available.get(s.stratum, 0) + 1
    picked = stratified_sample(candidates, count, seed)
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return BenchmarkSet(seed=seed, created_at=created, scenes=picked), available

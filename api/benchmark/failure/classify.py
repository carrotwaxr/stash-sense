"""Give every missed performer exactly one reason.

Works on the raw per-scene record the runner saves (plain dicts), so a saved
run can be re-scored with different thresholds without touching the GPU.

A scene face is *attributed* to an expected performer when its fused
distance to that performer's DB faces is within ``attribution_max_distance``
and no other expected performer in the scene is closer. Attribution uses the
answer key, which production never sees; it exists only to explain misses.

Reasons, checked in this order (first hit wins):

1. ``not_in_db``: no faces for the performer in the face DB.
2. ``no_face_detected``: no detected face is attributed to them.
3. ``dropped_by_aggregation``: they were the top-1 match (within
   max_distance) on at least one face production saw, but scene aggregation
   didn't report them as a multi-frame person.
4. ``small_or_profile``: every attributed face is under ``usable_min_size``
   px (short side) or turned past ``profile_yaw`` degrees.
5. ``wrong_match``: a usable face of theirs was detected and embedded, and
   another performer won it.
"""

from dataclasses import asdict, dataclass
from typing import Optional

import face_config

REASONS = [
    "not_in_db",
    "no_face_detected",
    "dropped_by_aggregation",
    "small_or_profile",
    "wrong_match",
]

# Which pipeline stage each outcome blames (see rollup()).
STAGES = [
    "db_coverage",
    "frame_choice",
    "detection_or_absent",
    "size_or_pose_throughout",
    "matching",
    "unprobed",
]


@dataclass(frozen=True)
class ClassifyParams:
    attribution_max_distance: float = 0.65
    usable_min_size: int = 80
    profile_yaw: float = 45.0
    max_distance: float = face_config.MAX_DISTANCE
    min_person_frames: int = 2  # the plugin's "multi-frame" group: frame_count > 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PerformerOutcome:
    scene_id: str
    stratum: str
    era: str
    resolution: str
    stashdb_id: str
    name: str
    found: bool
    reason: Optional[str] = None
    stage: Optional[str] = None
    best_distance: Optional[float] = None  # closest attributed face to their DB faces
    attributed_faces: int = 0
    usable_faces: int = 0
    correct_top1_faces: int = 0
    shown_as_single_frame: bool = False
    winner: Optional[dict] = None  # wrong_match: who took their best face
    rank_on_best_face: Optional[int] = None  # their rank in that face's top list
    frame_choice_recoverable: Optional[bool] = None

    def to_dict(self) -> dict:
        return asdict(self)


def _is_usable(face: dict, p: ClassifyParams) -> bool:
    if not face.get("eligible"):
        return False
    if min(face["w"], face["h"]) < p.usable_min_size:
        return False
    yaw = face.get("yaw")
    return yaw is None or abs(yaw) <= p.profile_yaw


def is_usable_face(face: dict, p: ClassifyParams = ClassifyParams()) -> bool:
    """Exposed for the probe: a face recognition can be expected to handle."""
    return _is_usable(face, p)


def attributed_faces(scene: dict, stashdb_id: str, p: ClassifyParams) -> list[dict]:
    out = []
    for face in scene["faces"]:
        dists = face.get("dist") or {}
        mine = dists.get(stashdb_id)
        if mine is None or mine > p.attribution_max_distance:
            continue
        others = [d for sid, d in dists.items() if sid != stashdb_id and d is not None]
        if others and min(others) < mine:
            continue
        out.append(face)
    return out


def _top1_is(face: dict, ids: set[str], max_distance: float) -> bool:
    top = face.get("top") or []
    return bool(top) and top[0]["stashdb_id"] in ids and top[0]["distance"] <= max_distance


def _rank(face: dict, ids: set[str]) -> Optional[int]:
    for i, m in enumerate(face.get("top") or []):
        if m["stashdb_id"] in ids:
            return i + 1
    return None


def stage_for(reason: Optional[str], recoverable: Optional[bool]) -> Optional[str]:
    if reason is None:
        return None
    if reason == "not_in_db":
        return "db_coverage"
    if reason in ("wrong_match", "dropped_by_aggregation"):
        return "matching"
    if recoverable is None:
        return "unprobed"
    if recoverable:
        return "frame_choice"
    return "detection_or_absent" if reason == "no_face_detected" else "size_or_pose_throughout"


def classify_scene(scene: dict, p: ClassifyParams = ClassifyParams()) -> list[PerformerOutcome]:
    """One outcome per expected performer in a successfully processed scene."""
    outcomes = []
    persons = scene.get("persons") or []
    probe = scene.get("probe") or {}
    for exp in scene["expected"]:
        sid = exp["stashdb_id"]
        ids = set(exp.get("ids") or []) | {sid}
        multi = [x for x in persons if x.get("best") and x["frame_count"] >= p.min_person_frames]
        single = [x for x in persons if x.get("best") and x["frame_count"] < p.min_person_frames]
        o = PerformerOutcome(
            scene_id=scene["scene_id"], stratum=scene["stratum"], era=scene["era"],
            resolution=scene["resolution"], stashdb_id=sid, name=exp.get("name", ""),
            found=any(x["best"]["stashdb_id"] in ids for x in multi),
        )
        o.shown_as_single_frame = any(x["best"]["stashdb_id"] in ids for x in single)
        mine = attributed_faces(scene, sid, p)
        o.attributed_faces = len(mine)
        usable = [f for f in mine if _is_usable(f, p)]
        o.usable_faces = len(usable)
        correct = [f for f in mine if f.get("eligible") and _top1_is(f, ids, p.max_distance)]
        o.correct_top1_faces = len(correct)
        if mine:
            best = min(mine, key=lambda f: f["dist"][sid])
            o.best_distance = best["dist"][sid]

        if not o.found:
            if not exp.get("in_db"):
                o.reason = "not_in_db"
            elif not mine:
                o.reason = "no_face_detected"
            elif correct:
                o.reason = "dropped_by_aggregation"
            elif not usable:
                o.reason = "small_or_profile"
            else:
                o.reason = "wrong_match"
                face = min(usable, key=lambda f: f["dist"][sid])
                o.best_distance = face["dist"][sid]
                top = face.get("top") or []
                o.winner = top[0] if top else None
                o.rank_on_best_face = _rank(face, ids)
            if o.reason in ("no_face_detected", "small_or_profile") and sid in probe:
                o.frame_choice_recoverable = bool(probe[sid].get("recoverable"))
            o.stage = stage_for(o.reason, o.frame_choice_recoverable)
        outcomes.append(o)
    return outcomes


def false_positives(scene: dict, p: ClassifyParams = ClassifyParams()) -> list[dict]:
    """Multi-frame persons whose best match isn't any expected performer."""
    ids = set()
    for exp in scene["expected"]:
        ids |= set(exp.get("ids") or []) | {exp["stashdb_id"]}
    return [
        x["best"] for x in scene.get("persons") or []
        if x.get("best") and x["frame_count"] >= p.min_person_frames and x["best"]["stashdb_id"] not in ids
    ]

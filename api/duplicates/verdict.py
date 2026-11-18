"""Verdict and pair class from audio, sprite and keyframe evidence (#130).

``verdict`` is stash-curator's ``dupecheck.verdict`` (SAME / DIFFERENT / REVIEW), with the
content index's keyframe match fraction counted as picture evidence next to the sprite
score. ``pair_class`` adds what the verdict alone cannot say: a re-scored or dubbed copy,
a scene inside a longer file, a pair whose metadata contradicts the files.
"""
from dataclasses import asdict, dataclass
from urllib.parse import urlparse

from duplicates.sprites import SPRITE_STRONG, SPRITE_WEAK

# Audio bit-error rate over the best offset (measured on the owner's library, 2026-10):
# the same recording 0.04-0.2, unrelated 0.45+, and 0.3-0.4 is grey: often different scenes
# of one production sharing a music bed. Values are dupecheck's, validated 2026-10-13.
AUDIO_STRONG = 0.20      # below this audio alone decides SAME
AUDIO_SAME = 0.30        # 0.20-0.30 needs some picture support
AUDIO_DIFF = 0.40        # above this the audio is foreign

# Keyframe match fraction (content index video match): duplicates 0.25-1.0, other scenes
# from the same studio <= 0.03. Set between those, at the same tiers as the sprite score.
KEYFRAME_STRONG = 0.15   # counts like strong sprites
KEYFRAME_WEAK = 0.05     # counts like weak sprites


@dataclass
class PairEvidence:
    ber: float | None = None
    ber_offset_s: float | None = None
    sprite: tuple[int, int] | None = None        # (d85, d80)
    keyframe_fraction: float | None = None       # content index video match
    keyframe_offset_s: float | None = None
    overlap: dict | None = None                  # classify() output when the index has it

    def to_dict(self) -> dict:
        return asdict(self)


def _picture(ev: PairEvidence) -> str:
    """'strong', 'weak' or 'none' from the sprite score and the keyframe fraction."""
    if ev.sprite and (ev.sprite[0] >= SPRITE_STRONG[0] or ev.sprite[1] >= SPRITE_STRONG[1]):
        return "strong"
    if ev.keyframe_fraction is not None and ev.keyframe_fraction >= KEYFRAME_STRONG:
        return "strong"
    if ev.sprite and (ev.sprite[0] >= SPRITE_WEAK[0] or ev.sprite[1] >= SPRITE_WEAK[1]):
        return "weak"
    if ev.keyframe_fraction is not None and ev.keyframe_fraction >= KEYFRAME_WEAK:
        return "weak"
    return "none"


def _no_video_evidence(ev: PairEvidence) -> bool:
    """No sprite tile matched at all and the keyframes show nothing (absent or below the weak tier)."""
    if ev.sprite and (ev.sprite[0] > 0 or ev.sprite[1] > 0):
        return False
    return ev.keyframe_fraction is None or ev.keyframe_fraction < KEYFRAME_WEAK


def verdict(ev: PairEvidence) -> str:
    """SAME / DIFFERENT / REVIEW; audio decides when it can."""
    s = _picture(ev)
    ber = ev.ber
    if ber is not None and ber < AUDIO_STRONG:
        return "SAME"
    if ber is not None and ber < AUDIO_SAME:
        # one production's shared music bed lands here too; with no picture match it is a REVIEW
        return "SAME" if s != "none" else "REVIEW"
    if ber is not None and ber > AUDIO_DIFF:
        return "REVIEW" if s == "strong" else "DIFFERENT"   # strong pictures + foreign audio: re-scored, long intro
    if s == "strong":
        return "SAME"                                       # grey or missing audio, strong pictures
    return "REVIEW"


_ENDPOINT_NAMES = {"stashdb.org": "StashDB", "pmvstash.org": "PMVStash", "fansdb.cc": "FansDB"}


def _endpoint_name(endpoint: str) -> str:
    host = urlparse(endpoint).netloc or endpoint
    return _ENDPOINT_NAMES.get(host, host)


def metadata_conflict(scene_a: dict, scene_b: dict) -> str | None:
    """Why two scenes' metadata cannot describe the same scene, or None. Missing data is not a conflict."""
    ids_a: dict[str, set] = {}
    ids_b: dict[str, set] = {}
    for scene, ids in ((scene_a, ids_a), (scene_b, ids_b)):
        for sid in scene.get("stash_ids") or []:
            if sid.get("endpoint") and sid.get("stash_id"):
                ids.setdefault(sid["endpoint"], set()).add(sid["stash_id"])
    for endpoint in sorted(ids_a.keys() & ids_b.keys()):
        if not ids_a[endpoint] & ids_b[endpoint]:
            return f"different {_endpoint_name(endpoint)} ids"
    studio_a = (scene_a.get("studio") or {}).get("id")
    studio_b = (scene_b.get("studio") or {}).get("id")
    if studio_a and studio_b and str(studio_a) != str(studio_b):
        return "different studios"
    perf_a = {str(p["id"]) for p in scene_a.get("performers") or [] if p.get("id")}
    perf_b = {str(p["id"]) for p in scene_b.get("performers") or [] if p.get("id")}
    if perf_a and perf_b and not perf_a & perf_b:
        return "no performer in common"
    return None


def pair_class(ev: PairEvidence, scene_a: dict, scene_b: dict) -> str:
    """duplicate | contained | partial | language_variant | mislabel | review | different."""
    kind = (ev.overlap or {}).get("kind")
    if kind == "incidental" and _no_video_evidence(ev):
        return "different"
    v = verdict(ev)
    if v == "DIFFERENT":
        return "different"
    if _picture(ev) == "strong" and ev.ber is not None and ev.ber > AUDIO_DIFF:
        return "language_variant"
    if ev.ber is not None and AUDIO_SAME <= ev.ber <= AUDIO_DIFF and _no_video_evidence(ev):
        return "different"          # a shared music bed, not a copy
    if v == "SAME" and metadata_conflict(scene_a, scene_b):
        return "mislabel"
    if kind in ("duplicate", "contained", "partial"):
        return kind
    return "duplicate" if v == "SAME" else "review"

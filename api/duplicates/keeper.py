"""Pairwise keeper rules for duplicate detection v2 (#130).

`better(a, b)` says which of two files is the better one to keep and why. The rules, in order: a rip loses;
an AI upscale loses to the original; more effective detail; a disfavored codec or container loses a near tie;
bits per pixel; frame rate; duration (duplicates only); container. Reasons are short plain sentences and only
the disfavored rule may name a codec.
"""

import os
import re
from dataclasses import dataclass

# Relative compression efficiency against h264 (a kb/s of hevc carries 1.6x the detail of a kb/s of h264).
CODEC_EFF = {"hevc": 1.6, "av1": 1.7, "vp9": 1.4, "h264": 1.0, "vc1": 0.8, "wmv3": 0.8, "mpeg4": 0.6, "msmpeg4v3": 0.6,
             "msmpeg4v2": 0.55, "wmv2": 0.55, "wmv1": 0.5, "rv40": 0.7, "rv30": 0.6, "mpeg2video": 0.5, "mpeg1video": 0.4}
UNKNOWN_CODEC_EFF = 0.8
UPNAME_RE = re.compile(r"upscal|_\d{3,4}p_\d{8}|tvai|topaz|starlight|proteus|\bai[ -]?enhanced", re.I)
ENCODER_RE = re.compile(r"topaz|tvai|video ai", re.I)
GOOD_CONTAINERS = {"mp4", "m4v", "mkv"}
UPSCALE_CODECS = {"av1"}  # the size/pixel guess is only trusted for av1; hevc and vp9 releases at HD are real

DETAIL_BAND = 1.15
BPP_BAND = 1.25
FPS_RATIO = 1.2
DURATION_GAP = 60.0
BITS_PER_PIXEL_FLOOR = 0.08  # bits per pixel per frame a file needs to hold its resolution
SPLIT_FIRST_RATIO = 1.5  # a contained pair whose long file is this much longer than the short one is a multi-scene file
UPSCALE_PIXEL_RATIO = 1.2


@dataclass(frozen=True)
class FileFeatures:
    key: str  # oshash; the identity a pick is stored by
    codec: str
    width: int
    height: int
    bit_rate: int
    fps: float
    duration: float
    size: int
    ext: str
    upscale_name: bool = False
    upscale_path: bool = False
    encoder: str = ""
    rip: bool = False  # filled by #133

    @classmethod
    def from_stash_file(cls, f: dict) -> "FileFeatures":
        path = f.get("path") or ""
        base = f.get("basename") or os.path.basename(path)
        key = ""
        for fp in f.get("fingerprints") or []:
            if fp.get("type") == "oshash":
                key = fp.get("value") or ""
                break
        return cls(
            key=key,
            codec=(f.get("video_codec") or "").lower(),
            width=int(f.get("width") or 0),
            height=int(f.get("height") or 0),
            bit_rate=int(f.get("bit_rate") or 0),
            fps=float(f.get("frame_rate") or 0),
            duration=float(f.get("duration") or 0),
            size=int(f.get("size") or 0),
            ext=os.path.splitext(base)[1].lstrip(".").lower(),
            upscale_name=bool(UPNAME_RE.search(base)),
            upscale_path="/upscaled/" in path and "/originals" not in path,
            encoder=f.get("encoder") or "",
        )


def _px(f: FileFeatures) -> int:
    return (f.width or 0) * (f.height or 0)


def is_upscale(f: FileFeatures, siblings: list[FileFeatures]) -> bool:
    """Whether `f` looks like an AI upscale of one of its siblings (`f` itself in the list is ignored)."""
    if f.upscale_name or f.upscale_path or (f.encoder and ENCODER_RE.search(f.encoder)):
        return True
    if f.codec not in UPSCALE_CODECS:
        return False
    others = [s for s in siblings if s is not f and not (f.key and s.key == f.key)]
    plain = [s for s in others if s.codec not in UPSCALE_CODECS]
    if not plain:
        return False
    px = _px(f)
    return all(px >= UPSCALE_PIXEL_RATIO * _px(s) for s in plain) or all(f.size > s.size for s in others)


def _detail(f: FileFeatures) -> tuple[float, float]:
    """(effective detail in pixels, bits per pixel per frame). A file with no bitrate counts at its pixels."""
    px = _px(f)
    fps = f.fps or 30.0
    kb = (f.bit_rate or 0) / 1000 * CODEC_EFF.get(f.codec, UNKNOWN_CODEC_EFF)
    if not kb:
        return float(px), 0.0
    eff = min(px, kb * 1000 / (fps * BITS_PER_PIXEL_FLOOR))
    return eff, (kb * 1000 / (px * fps) if px else 0.0)


def effective_detail(f: FileFeatures) -> tuple[float, float]:
    """Public name for `_detail`: (effective detail in pixels, bits per pixel per frame)."""
    return _detail(f)


def _describe(f: FileFeatures) -> str:
    rate = f"{f.bit_rate / 1e6:.1f} Mbps" if f.bit_rate else "unknown bitrate"
    return f"{f.width}x{f.height} at {rate}"


def better(
    a: FileFeatures,
    b: FileFeatures,
    *,
    kind: str = "duplicate",
    disfavored_codecs: frozenset[str] | set[str] = frozenset(),
    disfavored_containers: frozenset[str] | set[str] = frozenset(),
    siblings: list[FileFeatures] | None = None,
) -> tuple[str, str]:
    """Which of two files to keep: ("a" | "b" | "tie", reason). `siblings` is the whole set for `is_upscale`."""

    def win(a_wins: bool, reason: str) -> tuple[str, str]:
        return ("a" if a_wins else "b"), reason

    # 1. a rip loses
    if a.rip != b.rip:
        return win(b.rip, "keeps the clean file: the other file is a rip")

    # 2. an upscale loses to the original
    ua = is_upscale(a, siblings if siblings is not None else [b])
    ub = is_upscale(b, siblings if siblings is not None else [a])
    if ua != ub:
        return win(ub, "keeps the original: the other file is an AI upscale")

    # 3. effective detail
    (ea, ba), (eb, bb) = _detail(a), _detail(b)
    if max(ea, eb) > DETAIL_BAND * min(ea, eb):
        w, lo = (a, b) if ea > eb else (b, a)
        return win(w is a, f"more effective detail ({_describe(w)} beats {_describe(lo)})")

    # 4. disfavored codec or container, inside the detail band only
    da, db = a.codec in disfavored_codecs, b.codec in disfavored_codecs
    if da != db:
        loser = a if da else b
        return win(db, f"avoids {loser.codec}: the other file has about the same detail")
    ca, cb = a.ext in disfavored_containers, b.ext in disfavored_containers
    if ca != cb:
        loser = a if ca else b
        return win(cb, f"avoids the .{loser.ext} container: the other file has about the same detail")

    # 5. bits per pixel
    if ba > 0 and bb > 0 and max(ba, bb) > BPP_BAND * min(ba, bb):
        return win(ba > bb, "more bits per pixel at about the same detail (cleaner, fewer blocks)")

    # 6. frame rate
    if a.fps > 0 and b.fps > 0 and max(a.fps, b.fps) >= FPS_RATIO * min(a.fps, b.fps):
        w = a if a.fps > b.fps else b
        return win(w is a, f"same picture at a higher frame rate ({w.fps:g} fps)")

    # 7. duration (a contained file inside a longer one says nothing about length)
    if kind == "duplicate" and abs(a.duration - b.duration) > DURATION_GAP:
        w = a if a.duration > b.duration else b
        return win(w is a, f"same picture, {abs(a.duration - b.duration) / 60:.1f} min longer (more complete)")

    # 8. container
    ga, gb = a.ext in GOOD_CONTAINERS, b.ext in GOOD_CONTAINERS
    if ga != gb:
        return win(ga, "same picture; the better container")

    return "tie", "practically equal: either is fine"


@dataclass
class KeeperSuggestion:
    keep: list[str]  # FileFeatures.key, best first
    delete: list[str]
    reason: str  # one sentence shown on the card
    flags: list[str]  # supercut_candidate, upscale_pair, held_unhealthy, tie, split_first
    held: bool = False


BAD_HEALTH = frozenset({"suspect", "missing"})
SUPERCUT_MIN_GAP = 60.0
SUPERCUT_FRACTION = 0.05


def _sentence(why: str) -> str:
    if why.startswith(("keeps", "avoids")):
        return why[0].upper() + why[1:]
    return f"Keeps the best copy: {why}"


def suggest(
    files: list[FileFeatures],
    *,
    kind: str = "duplicate",
    health: dict[str, str] | None = None,
    disfavored_codecs: frozenset[str] | set[str] = frozenset(),
    disfavored_containers: frozenset[str] | set[str] = frozenset(),
) -> KeeperSuggestion:
    """Which files of a duplicate set to keep. Sorted by key first, so the card order never matters."""
    fs = sorted(files, key=lambda f: f.key)
    health = health or {}
    kw = dict(kind=kind, disfavored_codecs=disfavored_codecs, disfavored_containers=disfavored_containers,
              siblings=fs)

    def failed(f: FileFeatures) -> bool:
        return health.get(f.key, "ok") in BAD_HEALTH

    def fold(pool: list[FileFeatures]) -> FileFeatures:
        champ = pool[0]
        for f in pool[1:]:
            if better(champ, f, **kw)[0] == "b":  # a tie keeps the current champion
                champ = f
        return champ

    def result(keep: list[FileFeatures], reason: str, flags: list[str]) -> KeeperSuggestion:
        kept = {f.key for f in keep}
        return KeeperSuggestion([f.key for f in keep], [f.key for f in fs if f.key not in kept], reason, flags)

    healthy = [f for f in fs if not failed(f)]
    if not healthy:
        return KeeperSuggestion([f.key for f in fs], [], "No file passed its health check, so nothing is deleted.",
                                ["held_unhealthy"], held=True)

    # contained: the long file holds extra content and is never deleted
    if kind == "contained" and len(fs) == 2:
        short, long_ = sorted(fs, key=lambda f: (f.duration, f.key))
        if failed(long_):
            return result([short, long_], "The longer file failed its health check, so both files are kept.", [])
        if long_.duration >= SPLIT_FIRST_RATIO * short.duration:
            return result([short, long_], "Split the longer file first: it probably holds several scenes. "
                                          "Both files are kept until then.", ["split_first"])
        if failed(short):
            return result([long_], "Its content is inside the longer file, and the shorter file failed its health "
                                   "check.", [])
        if better(short, long_, **kw)[0] == "a":
            return result([short, long_], "The shorter file is better quality and the longer one holds extra "
                                          "content, so both are kept.", [])
        return result([long_], "Its content is inside the longer file at equal or better quality.", [])

    flags: list[str] = []
    prefix = ""
    ups = [f for f in healthy if is_upscale(f, fs)]
    originals = [f for f in healthy if f not in ups]
    best_overall = fold(fs)

    if ups and originals:
        keep = [fold(originals), fold(ups)]
        flags.append("upscale_pair")
        reason = "Keeps the best original and the best AI upscale; the other copies are worse."
        return result(keep, reason, flags)

    champ = fold(healthy)
    if champ is not best_overall:
        prefix = "The best file failed its health check, so the next best is kept. "
    others = [f for f in healthy if f is not champ]
    if others:
        runner = fold(others)
        who, why = better(champ, runner, **kw)
        if any(better(champ, o, **kw)[0] == "tie" for o in others):
            flags.append("tie")
        reason = _sentence(why) if who != "tie" else "Practically equal files: keeps one, either is fine."
    else:
        reason = "Keeps the only file that passed its health check."
    keep = [champ]

    if kind == "duplicate":
        longest = max(healthy, key=lambda f: f.duration)  # first by key on equal durations
        if champ.duration < longest.duration - max(SUPERCUT_MIN_GAP, SUPERCUT_FRACTION * longest.duration):
            keep.append(longest)
            flags.append("supercut_candidate")
            reason += " The longest file is kept too because it may hold extra content."
    return result(keep, prefix + reason, flags)

"""Rip evidence (#133): file-name and tag clues, a combined rip score and the disfavored-format flag."""

import math
import re

from curation_settings import CurationSettings

TUBE_DOMAINS = (
    "spankbang", "pornhub", "xvideos", "xnxx", "xhamster", "eporner", "goodporn", "youporn",
    "redtube", "tnaflix", "txxx", "hqporner", "porntrex", "tube8", "beeg", "motherless",
    "keep2share", "k2s", "sxyprn", "daftsex", "porndig", "fullporner", "noodlemagazine",
)  # fmt: skip
WEIGHTS = {
    "tag_tube": 0.9,
    "tag_url": 0.3,
    "tube_name": 0.6,
    "preroll": 0.7,
    "low_detail": 0.2,
}
RIP_THRESHOLD = 0.5
TEXT_MAX = 120

_TUBE_RE = re.compile(
    r"(?<![a-z0-9])(?:" + "|".join(TUBE_DOMAINS) + r")(?![a-z0-9])", re.I
)
_URL_RE = re.compile(
    r"https?://\S+|www\.\S+|(?<![a-z0-9.-])[a-z0-9-]+\.(?:com|net|org|xxx|tv|co|info|biz|me|cc|to|porn|sex)(?![a-z0-9])",
    re.I,
)


def _clue(signal: str, text: str) -> dict:
    return {
        "signal": signal,
        "text": text.strip()[:TEXT_MAX],
        "weight": WEIGHTS[signal],
    }


def file_clues(basename: str, tags: dict) -> list[dict]:
    """One clue per kind found: a tube site or other URL in a tag value, a tube site in the file name."""
    values = [str(v) for v in (tags or {}).values() if v]
    clues = []
    tube = next((v for v in values if _TUBE_RE.search(v)), None)
    if tube is not None:
        clues.append(_clue("tag_tube", tube))
    else:
        url = next((v for v in values if _URL_RE.search(v)), None)
        if url is not None:
            clues.append(_clue("tag_url", url))
    if _TUBE_RE.search(basename or ""):
        clues.append(_clue("tube_name", basename))
    return clues


def rip_score(weights: list[float]) -> float:
    """Noisy-or of the clue weights: 1 - prod(1 - w)."""
    return 1 - math.prod(1 - w for w in weights)


def format_flag(file: dict, settings: CurationSettings) -> dict | None:
    """The file's codec and container with a plain reason when either is on a disfavored list, else None."""
    if not (settings.disfavored_video_codecs or settings.disfavored_containers):
        return None
    if not settings.is_disfavored(file):
        return None
    codec = (file.get("video_codec") or "").lower()
    container = (file.get("format") or "").lower()
    if codec and codec in settings.disfavored_video_codecs:
        reason = f"{codec} video is on the disfavored list"
    else:
        reason = f"{container} container is on the disfavored list"
    return {"codec": codec, "container": container, "reason": reason}

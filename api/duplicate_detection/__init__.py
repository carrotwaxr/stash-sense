"""Duplicate scene detection module."""

from .models import (
    FaceAppearance,
    SceneFingerprint,
    SceneMetadata,
    SignalBreakdown,
    DuplicateMatch,
    StashID,
)
from .scoring import (
    check_stashbox_match,
    metadata_score,
    face_signature_similarity,
    calculate_duplicate_confidence,
    StashboxMatchResult,
)

__all__ = [
    "FaceAppearance",
    "SceneFingerprint",
    "SceneMetadata",
    "SignalBreakdown",
    "DuplicateMatch",
    "StashID",
    "check_stashbox_match",
    "metadata_score",
    "face_signature_similarity",
    "calculate_duplicate_confidence",
    "StashboxMatchResult",
]

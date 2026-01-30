"""Scoring functions for duplicate scene detection."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .models import (
    SceneFingerprint,
    SceneMetadata,
    FaceAppearance,
    SignalBreakdown,
    DuplicateMatch,
)


@dataclass
class StashboxMatchResult:
    """Result of checking for stash-box ID match."""

    matched: bool
    endpoint: Optional[str] = None
    stash_id: Optional[str] = None


def check_stashbox_match(scene_a: SceneMetadata, scene_b: SceneMetadata) -> StashboxMatchResult:
    """Check if two scenes share the same stash-box ID on the same endpoint."""
    for sid_a in scene_a.stash_ids:
        for sid_b in scene_b.stash_ids:
            if sid_a.endpoint == sid_b.endpoint and sid_a.stash_id == sid_b.stash_id:
                return StashboxMatchResult(
                    matched=True,
                    endpoint=sid_a.endpoint,
                    stash_id=sid_a.stash_id,
                )
    return StashboxMatchResult(matched=False)


def _has_useful_metadata(scene: SceneMetadata) -> bool:
    """Check if scene has enough metadata for comparison."""
    return bool(scene.studio_id or scene.performer_ids)


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _days_between(date_a: Optional[str], date_b: Optional[str]) -> Optional[int]:
    """Calculate days between two date strings (YYYY-MM-DD format)."""
    if not date_a or not date_b:
        return None
    try:
        d1 = datetime.strptime(date_a, "%Y-%m-%d")
        d2 = datetime.strptime(date_b, "%Y-%m-%d")
        return abs((d2 - d1).days)
    except ValueError:
        return None


def metadata_score(scene_a: SceneMetadata, scene_b: SceneMetadata) -> tuple[float, str]:
    """
    Calculate metadata similarity score (0-60).

    Requires a base signal (studio or performers) to return non-zero.
    Duration and date are confirmation multipliers.
    """
    if not _has_useful_metadata(scene_a) or not _has_useful_metadata(scene_b):
        return 0.0, "Insufficient metadata"

    base = 0.0
    reasons = []

    # Base signals
    if scene_a.studio_id and scene_b.studio_id and scene_a.studio_id == scene_b.studio_id:
        base += 20.0
        reasons.append("Same studio")

    performer_overlap = _jaccard_similarity(scene_a.performer_ids, scene_b.performer_ids)
    if performer_overlap == 1.0 and scene_a.performer_ids:
        base += 20.0
        reasons.append("Exact performer match")
    elif performer_overlap >= 0.5:
        base += 12.0
        reasons.append(f"Performers overlap ({performer_overlap:.0%})")

    if base == 0:
        return 0.0, "No studio or performer match"

    # Confirmation multipliers
    multiplier = 1.0

    days = _days_between(scene_a.date, scene_b.date)
    if days is not None:
        if days == 0:
            multiplier += 0.5
            reasons.append("Same release date")
        elif days <= 7:
            multiplier += 0.3
            reasons.append("Release dates within 7 days")

    if scene_a.duration_seconds and scene_b.duration_seconds:
        diff = abs(scene_a.duration_seconds - scene_b.duration_seconds)
        if diff <= 5:
            multiplier += 0.5
            reasons.append("Duration within 5s")
        elif diff <= 30:
            multiplier += 0.3
            reasons.append("Duration within 30s")

    score = min(base * multiplier, 60.0)
    return score, " + ".join(reasons)


def face_signature_similarity(
    fp_a: SceneFingerprint, fp_b: SceneFingerprint
) -> tuple[float, str]:
    """
    Calculate face signature similarity score (0-85).

    Compares which performers appear and in what proportions.
    """
    all_performers = set(fp_a.faces.keys()) | set(fp_b.faces.keys())
    all_performers.discard("unknown")

    if not all_performers:
        return 0.0, "No identified performers in either scene"

    proportion_diffs = []
    matches = []

    for performer_id in all_performers:
        prop_a = fp_a.faces.get(performer_id, FaceAppearance(performer_id, 0, 0, 0)).proportion
        prop_b = fp_b.faces.get(performer_id, FaceAppearance(performer_id, 0, 0, 0)).proportion

        diff = abs(prop_a - prop_b)
        proportion_diffs.append(diff)

        # Both have meaningful presence (>10%)
        if prop_a > 0.1 and prop_b > 0.1:
            matches.append(performer_id)

    avg_diff = sum(proportion_diffs) / len(proportion_diffs)

    # Convert to similarity score (0-85 range)
    # Perfect match (avg_diff=0) -> 85%
    # Completely different (avg_diff>=0.5) -> 0%
    similarity = max(0.0, 85.0 * (1.0 - avg_diff * 2.0))

    reason = f"{len(matches)} shared performers, {avg_diff:.1%} avg proportion difference"
    return similarity, reason


def calculate_duplicate_confidence(
    scene_a: SceneMetadata,
    scene_b: SceneMetadata,
    fp_a: Optional[SceneFingerprint],
    fp_b: Optional[SceneFingerprint],
) -> Optional[DuplicateMatch]:
    """
    Calculate overall duplicate confidence combining all signals.

    Returns None if scenes are the same or confidence is too low to consider.
    """
    # Guard: same scene
    if scene_a.scene_id == scene_b.scene_id:
        return None

    # Tier 1: Authoritative stash-box match
    stashbox = check_stashbox_match(scene_a, scene_b)
    if stashbox.matched:
        return DuplicateMatch(
            scene_a_id=int(scene_a.scene_id),
            scene_b_id=int(scene_b.scene_id),
            confidence=100.0,
            reasoning=[f"Identical stash-box ID: {stashbox.stash_id}"],
            signal_breakdown=SignalBreakdown(
                stashbox_match=True,
                stashbox_endpoint=stashbox.endpoint,
                face_score=0.0,
                face_reasoning="",
                metadata_score=0.0,
                metadata_reasoning="",
            ),
        )

    # Tier 2: Face + Metadata signals
    face_score = 0.0
    face_reasoning = "No fingerprint available"

    if fp_a and fp_b:
        if fp_a.total_faces_detected == 0 and fp_b.total_faces_detected == 0:
            face_reasoning = "No faces detected in either scene"
        elif fp_a.total_faces_detected == 0 or fp_b.total_faces_detected == 0:
            face_reasoning = "Asymmetric face detection"
        else:
            face_score, face_reasoning = face_signature_similarity(fp_a, fp_b)

    meta_score, meta_reasoning = metadata_score(scene_a, scene_b)

    # No signals = no match
    if face_score == 0 and meta_score == 0:
        return None

    # Combine with diminishing returns
    primary = max(face_score, meta_score)
    secondary = min(face_score, meta_score)
    combined = primary + (secondary * 0.3)

    # Cap at 95% without stash-box confirmation
    confidence = min(combined, 95.0)

    # Build reasoning
    reasoning = []
    if face_score > 0:
        reasoning.append(f"Face analysis: {face_reasoning}")
    if meta_score > 0:
        reasoning.append(f"Metadata: {meta_reasoning}")

    # Add confidence qualifier
    if confidence >= 80:
        reasoning.insert(0, "High confidence duplicate")
    elif confidence >= 50:
        reasoning.insert(0, "Likely duplicate")
    elif confidence >= 30:
        reasoning.insert(0, "Possible duplicate")
    else:
        reasoning.insert(0, "Low confidence match")

    return DuplicateMatch(
        scene_a_id=int(scene_a.scene_id),
        scene_b_id=int(scene_b.scene_id),
        confidence=round(confidence, 1),
        reasoning=reasoning,
        signal_breakdown=SignalBreakdown(
            stashbox_match=False,
            stashbox_endpoint=None,
            face_score=face_score,
            face_reasoning=face_reasoning,
            metadata_score=meta_score,
            metadata_reasoning=meta_reasoning,
        ),
    )

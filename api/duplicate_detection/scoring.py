"""Scoring functions for duplicate scene detection."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .models import (
    SceneFingerprint,
    SceneMetadata,
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

    Date + performers is the strongest signal. Studio is a mild boost.
    Duration only penalizes at extremes, never adds confidence.
    """
    score = 0.0
    reasons = []

    # Date comparison
    days = _days_between(scene_a.date, scene_b.date)
    same_date = days is not None and days == 0
    close_date = days is not None and days <= 7

    # Performer comparison
    performer_overlap = _jaccard_similarity(scene_a.performer_ids, scene_b.performer_ids)
    exact_performers = performer_overlap == 1.0 and bool(scene_a.performer_ids)
    partial_performers = performer_overlap >= 0.5

    # Primary scoring: date + performers is strongest
    if same_date and exact_performers:
        score += 45.0
        reasons.append("Same date + exact performer match")
    elif same_date and partial_performers:
        score += 35.0
        reasons.append(f"Same date + performers overlap ({performer_overlap:.0%})")
    elif close_date and exact_performers:
        score += 35.0
        reasons.append("Release dates within 7 days + exact performer match")
    elif exact_performers:
        score += 25.0
        reasons.append("Exact performer match")
    elif same_date and (scene_a.performer_ids or scene_b.performer_ids):
        score += 20.0
        reasons.append("Same date")
    elif partial_performers:
        score += 15.0
        reasons.append(f"Performers overlap ({performer_overlap:.0%})")

    # Studio boost (mild, not decisive)
    if scene_a.studio_id and scene_b.studio_id and scene_a.studio_id == scene_b.studio_id:
        score += 10.0
        reasons.append("Same studio")

    # Duration penalty (only subtracts, never adds)
    if scene_a.duration_seconds and scene_b.duration_seconds:
        shorter = min(scene_a.duration_seconds, scene_b.duration_seconds)
        longer = max(scene_a.duration_seconds, scene_b.duration_seconds)
        ratio = shorter / longer if longer > 0 else 1.0
        if ratio < 0.15:
            score -= 20.0
            reasons.append("Duration ratio < 15% (penalty -20)")
        elif ratio < 0.30:
            score -= 10.0
            reasons.append("Duration ratio < 30% (penalty -10)")

    if not reasons:
        return 0.0, "No metadata signals"

    return min(max(score, 0.0), 60.0), " + ".join(reasons)


def face_signature_similarity(
    fp_a: SceneFingerprint, fp_b: SceneFingerprint
) -> tuple[float, str]:
    """
    Calculate face signature similarity score (0-75).

    Requires at least one shared performer. Score based on Jaccard
    similarity of performer sets, with proportion bonus for shared performers.
    """
    all_a = set(fp_a.faces.keys()) - {"unknown"}
    all_b = set(fp_b.faces.keys()) - {"unknown"}
    shared = all_a & all_b

    if not all_a and not all_b:
        return 0.0, "No identified performers in either scene"

    if not shared:
        return 0.0, "No shared performers"

    # Jaccard similarity of performer sets
    union = all_a | all_b
    jaccard = len(shared) / len(union) if union else 0.0

    # Proportion similarity for shared performers only
    proportion_diffs = []
    for pid in shared:
        diff = abs(fp_a.faces[pid].proportion - fp_b.faces[pid].proportion)
        proportion_diffs.append(diff)
    avg_diff = sum(proportion_diffs) / len(proportion_diffs)

    # Score: Jaccard drives the base, proportion similarity is a bonus
    base = jaccard * 60.0
    proportion_bonus = max(0.0, 15.0 * (1.0 - avg_diff * 4.0))
    score = min(base + proportion_bonus, 75.0)

    reason = f"{len(shared)} shared performers (Jaccard {jaccard:.0%}), {avg_diff:.1%} avg proportion diff"
    return score, reason


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

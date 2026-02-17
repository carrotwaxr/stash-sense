"""
Signal scoring functions for multi-signal performer identification.

Provides adjustment multipliers based on body proportion and tattoo signals
to re-rank face recognition candidates.
"""

from typing import Optional

from body_proportions import BodyProportions
from tattoo_detector import TattooResult


def body_ratio_penalty(
    query_ratios: Optional[BodyProportions],
    candidate_ratios: Optional[BodyProportions],
) -> float:
    """
    Compute adjustment multiplier based on body proportion comparison.

    Compares the shoulder_hip_ratio (most discriminating metric) between
    query and candidate proportions.

    Args:
        query_ratios: Body proportions from the query image (may be None)
        candidate_ratios: Body proportions from the candidate (may be None)

    Returns:
        Adjustment multiplier:
        - 1.0 if either input is None (no penalty)
        - 1.0 if diff <= 0.12 (compatible)
        - 0.85 if diff > 0.12 (slight mismatch)
        - 0.6 if diff > 0.2 (moderate mismatch)
        - 0.3 if diff > 0.35 (severe mismatch)
    """
    if query_ratios is None or candidate_ratios is None:
        return 1.0

    diff = abs(query_ratios.shoulder_hip_ratio - candidate_ratios.shoulder_hip_ratio)

    if diff > 0.35:
        return 0.3
    elif diff > 0.2:
        return 0.6
    elif diff > 0.12:
        return 0.85
    else:
        return 1.0


def tattoo_adjustment(
    query_result: Optional[TattooResult],
    candidate_id: str,
    tattoo_scores: Optional[dict[str, float]] = None,
    has_tattoo_embeddings: bool = False,
) -> float:
    """
    Compute adjustment multiplier based on tattoo embedding similarity.

    Uses visual similarity scores from TattooMatcher (Voyager kNN on
    EfficientNet-B0 embeddings) instead of binary has/doesn't-have presence.

    Args:
        query_result: Tattoo detection result from the query image (may be None)
        candidate_id: Universal ID of the candidate performer
        tattoo_scores: Dict of universal_id -> best similarity score from TattooMatcher
        has_tattoo_embeddings: Whether the candidate has any tattoo embeddings in the index

    Returns:
        Adjustment multiplier:
        - 1.0 if query_result is None or no tattoos detected (neutral)
        - 1.3-1.5 if high tattoo similarity (>0.7) between query and candidate
        - 1.1 if moderate tattoo similarity (>0.5)
        - 0.7 if query has tattoos but candidate has no tattoo embeddings (penalty)
        - 0.95 if query has no tattoos but candidate has many tattoo embeddings
        - 1.0 otherwise (neutral)
    """
    if query_result is None:
        return 1.0

    query_has_tattoos = query_result.has_tattoos

    # No tattoos in query image
    if not query_has_tattoos:
        if has_tattoo_embeddings:
            return 0.95  # Slight penalty: candidate has tattoos, query doesn't
        return 1.0

    # Query has tattoos — check embedding similarity scores
    if tattoo_scores:
        score = tattoo_scores.get(candidate_id, 0.0)
        if score > 0.7:
            # High similarity — strong boost (scale linearly 1.3-1.5)
            return 1.3 + (score - 0.7) * (0.2 / 0.3)
        elif score > 0.5:
            return 1.1  # Moderate similarity — modest boost

    # Query has tattoos but candidate has no tattoo embeddings at all
    if not has_tattoo_embeddings:
        return 0.7

    # Query has tattoos, candidate has embeddings, but low/no similarity
    return 1.0

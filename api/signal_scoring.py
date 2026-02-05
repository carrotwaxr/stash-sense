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
    has_tattoos: bool,
    locations: set[str],
) -> float:
    """
    Compute adjustment multiplier based on tattoo comparison.

    Args:
        query_result: Tattoo detection result from the query image (may be None)
        has_tattoos: Whether the candidate has tattoos
        locations: Set of tattoo location hints for the candidate

    Returns:
        Adjustment multiplier:
        - 1.0 if query_result is None (neutral)
        - 0.7 if query shows tattoos but candidate has none (penalty)
        - 0.95 if query shows no tattoos but candidate has tattoos (slight penalty)
        - 1.15 if both have tattoos and locations overlap (boost)
        - 0.9 if both have tattoos but different locations
        - 1.0 otherwise (neutral)
    """
    if query_result is None:
        return 1.0

    query_has_tattoos = query_result.has_tattoos
    query_locations = query_result.locations

    # Query has tattoos, candidate does not
    if query_has_tattoos and not has_tattoos:
        return 0.7

    # Query has no tattoos, candidate does
    if not query_has_tattoos and has_tattoos:
        return 0.95

    # Both have tattoos - check location overlap
    if query_has_tattoos and has_tattoos:
        # Check if any locations overlap
        if query_locations & locations:
            return 1.15
        else:
            return 0.9

    # Neither has tattoos (or other edge cases)
    return 1.0

"""Robust face matching with multi-model fusion.

Implements intelligent matching that:
1. Queries both FaceNet and ArcFace indices
2. Detects when a model produces degenerate output
3. Uses adaptive weighted fusion based on model reliability
4. Returns matches with confidence scores
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np

from voyager import Index

import face_config


# =============================================================================
# CONFIGURATION - Tune these values in face_config.py
# =============================================================================

@dataclass
class MatchingConfig:
    """Configuration for face matching. All values are tunable."""

    # Query parameters
    query_k: int = 100  # Number of candidates to fetch from each index

    # Fusion weights (when both models are healthy) - from face_config.py
    facenet_weight: float = face_config.FACENET_WEIGHT
    arcface_weight: float = face_config.ARCFACE_WEIGHT

    # Model health detection thresholds
    # Degenerate outputs have very low variance (all distances nearly identical)
    # Normal FaceNet: variance ~0.002-0.01, Degenerate ArcFace: variance ~0.0001
    min_distance_variance: float = 0.001  # Min variance in top-k distances
    max_suspicious_min_distance: float = 0.10  # If min distance < this AND low variance, suspicious
    health_check_k: int = 20  # How many results to check for health
    min_distance_range: float = 0.05  # Min range (max-min) in top-k distances

    # Output filtering
    max_results: int = 10
    max_distance: float = 0.8  # Maximum combined distance to return


DEFAULT_CONFIG = MatchingConfig()


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class ModelHealth(Enum):
    """Health status of a model's output for a given query."""
    HEALTHY = "healthy"
    DEGENERATE = "degenerate"  # Low variance, suspicious distances
    NO_RESULTS = "no_results"


@dataclass
class ModelQueryResult:
    """Result from querying a single model's index."""
    neighbors: np.ndarray  # Face indices
    distances: np.ndarray  # Distances to neighbors
    health: ModelHealth
    health_reason: str = ""

    # Statistics for debugging
    min_distance: float = 0.0
    max_distance: float = 0.0
    distance_variance: float = 0.0


@dataclass
class CandidateMatch:
    """A candidate match from the fusion process."""
    face_index: int
    universal_id: str
    name: str

    # Per-model scores (None if not in that model's results)
    facenet_distance: Optional[float] = None
    arcface_distance: Optional[float] = None

    # Combined score
    combined_distance: float = 0.0
    confidence: float = 0.0  # 0-1, higher is better

    # Which models contributed
    in_facenet: bool = False
    in_arcface: bool = False

    # For debugging
    facenet_rank: Optional[int] = None
    arcface_rank: Optional[int] = None


@dataclass
class MatchingResult:
    """Complete result from the matching process."""
    matches: list[CandidateMatch]

    # Model health info
    facenet_health: ModelHealth = ModelHealth.HEALTHY
    arcface_health: ModelHealth = ModelHealth.HEALTHY
    facenet_health_reason: str = ""
    arcface_health_reason: str = ""
    fusion_strategy: str = "weighted"  # "weighted", "facenet_only", "arcface_only"

    # Statistics
    facenet_candidates: int = 0
    arcface_candidates: int = 0
    candidates_in_both: int = 0


# =============================================================================
# HEALTH DETECTION
# =============================================================================

def check_model_health(
    distances: np.ndarray,
    config: MatchingConfig = DEFAULT_CONFIG,
) -> tuple[ModelHealth, str]:
    """
    Check if a model's output appears healthy or degenerate.

    The key insight from testing:
    - FaceNet healthy: min_dist typically 0.25-0.40, even with low variance
    - ArcFace degenerate: min_dist typically < 0.15, with suspiciously uniform distances

    Degenerate outputs are detected by:
    1. Suspiciously low minimum distance (< 0.15) - the primary indicator
    2. Combined with low variance or narrow range

    Returns:
        (health_status, reason_string)
    """
    if len(distances) == 0:
        return ModelHealth.NO_RESULTS, "No results returned"

    check_distances = distances[:config.health_check_k]

    min_dist = float(np.min(check_distances))
    max_dist = float(np.max(check_distances))
    variance = float(np.var(check_distances))
    dist_range = max_dist - min_dist

    # Primary check: minimum distance
    # If min_dist >= 0.20, it's almost certainly healthy (normal face recognition range)
    if min_dist >= 0.20:
        return ModelHealth.HEALTHY, f"min={min_dist:.3f}, range={dist_range:.3f}, var={variance:.5f}"

    # If min_dist < 0.15 AND distances are tightly clustered, it's degenerate
    # This is the pathological case where ArcFace outputs a "generic" embedding
    if min_dist < config.max_suspicious_min_distance:
        if variance < 0.002 or dist_range < 0.05:
            return ModelHealth.DEGENERATE, f"Suspicious min_dist={min_dist:.3f} with tight clustering (var={variance:.5f}, range={dist_range:.3f})"

    # Edge case: moderate min_dist (0.15-0.20) but very tight clustering
    if dist_range < 0.03 and variance < 0.001:
        return ModelHealth.DEGENERATE, f"Very tight clustering (range={dist_range:.4f}, var={variance:.6f})"

    return ModelHealth.HEALTHY, f"min={min_dist:.3f}, range={dist_range:.3f}, var={variance:.5f}"


# =============================================================================
# MATCHING LOGIC
# =============================================================================

def query_model(
    embedding: np.ndarray,
    index: Index,
    config: MatchingConfig = DEFAULT_CONFIG,
) -> ModelQueryResult:
    """Query a single model's index and assess health."""
    neighbors, distances = index.query(embedding, k=config.query_k)

    health, reason = check_model_health(distances, config)

    return ModelQueryResult(
        neighbors=neighbors,
        distances=distances,
        health=health,
        health_reason=reason,
        min_distance=float(np.min(distances)) if len(distances) > 0 else 0.0,
        max_distance=float(np.max(distances[:20])) if len(distances) >= 20 else float(np.max(distances)),
        distance_variance=float(np.var(distances[:20])) if len(distances) >= 20 else float(np.var(distances)),
    )


def fuse_results(
    facenet_result: ModelQueryResult,
    arcface_result: ModelQueryResult,
    faces_mapping: list[str],  # index -> universal_id
    performers: dict[str, dict],  # universal_id -> performer info
    config: MatchingConfig = DEFAULT_CONFIG,
) -> MatchingResult:
    """
    Fuse results from both models using adaptive weighting.

    Strategy:
    - If both healthy: weighted combination
    - If only FaceNet healthy: use FaceNet only
    - If only ArcFace healthy: use ArcFace only
    - If neither healthy: use FaceNet with lower confidence
    """
    # Determine fusion strategy based on health
    fn_healthy = facenet_result.health == ModelHealth.HEALTHY
    af_healthy = arcface_result.health == ModelHealth.HEALTHY

    if fn_healthy and af_healthy:
        strategy = "weighted"
        fn_weight = config.facenet_weight
        af_weight = config.arcface_weight
    elif fn_healthy:
        strategy = "facenet_only"
        fn_weight = 1.0
        af_weight = 0.0
    elif af_healthy:
        strategy = "arcface_only"
        fn_weight = 0.0
        af_weight = 1.0
    else:
        # Neither healthy - fall back to FaceNet (typically more stable)
        strategy = "facenet_fallback"
        fn_weight = 1.0
        af_weight = 0.0

    # Build candidate map
    candidates: dict[int, CandidateMatch] = {}
    faces_count = len(faces_mapping)

    # Process FaceNet results
    for rank, (idx, dist) in enumerate(zip(facenet_result.neighbors, facenet_result.distances)):
        idx = int(idx)
        # Skip if index is out of bounds (can happen with index/metadata mismatch)
        if idx < 0 or idx >= faces_count:
            continue
        uid = faces_mapping[idx]
        # Skip null entries (gaps from deleted faces or missing stashbox IDs)
        if uid is None:
            continue
        if idx not in candidates:
            info = performers.get(uid, {})
            candidates[idx] = CandidateMatch(
                face_index=idx,
                universal_id=uid,
                name=info.get("name", "Unknown"),
            )
        candidates[idx].facenet_distance = float(dist)
        candidates[idx].facenet_rank = rank + 1
        candidates[idx].in_facenet = True

    # Process ArcFace results (only if healthy or we're using it)
    if af_weight > 0:
        for rank, (idx, dist) in enumerate(zip(arcface_result.neighbors, arcface_result.distances)):
            idx = int(idx)
            # Skip if index is out of bounds (can happen with index/metadata mismatch)
            if idx < 0 or idx >= faces_count:
                continue
            uid = faces_mapping[idx]
            # Skip null entries (gaps from deleted faces or missing stashbox IDs)
            if uid is None:
                continue
            if idx not in candidates:
                info = performers.get(uid, {})
                candidates[idx] = CandidateMatch(
                    face_index=idx,
                    universal_id=uid,
                    name=info.get("name", "Unknown"),
                )
            candidates[idx].arcface_distance = float(dist)
            candidates[idx].arcface_rank = rank + 1
            candidates[idx].in_arcface = True

    # Calculate combined scores
    for candidate in candidates.values():
        fn_dist = candidate.facenet_distance
        af_dist = candidate.arcface_distance

        if fn_dist is not None and af_dist is not None:
            # Both models have this candidate
            candidate.combined_distance = fn_dist * fn_weight + af_dist * af_weight
        elif fn_dist is not None:
            # Only FaceNet
            if strategy == "weighted":
                # Penalize for missing in ArcFace (use median-ish distance)
                candidate.combined_distance = fn_dist * fn_weight + 0.5 * af_weight
            else:
                candidate.combined_distance = fn_dist
        elif af_dist is not None:
            # Only ArcFace
            if strategy == "weighted":
                candidate.combined_distance = 0.5 * fn_weight + af_dist * af_weight
            else:
                candidate.combined_distance = af_dist

        # Convert distance to confidence (0-1, higher is better)
        candidate.confidence = max(0.0, min(1.0, 1.0 - candidate.combined_distance))

    # Sort by combined distance and filter
    sorted_candidates = sorted(candidates.values(), key=lambda c: c.combined_distance)
    filtered = [c for c in sorted_candidates if c.combined_distance <= config.max_distance]
    final = filtered[:config.max_results]

    # Count statistics
    in_both = sum(1 for c in candidates.values() if c.in_facenet and c.in_arcface)

    return MatchingResult(
        matches=final,
        facenet_health=facenet_result.health,
        arcface_health=arcface_result.health,
        facenet_health_reason=facenet_result.health_reason,
        arcface_health_reason=arcface_result.health_reason,
        fusion_strategy=strategy,
        facenet_candidates=len([c for c in candidates.values() if c.in_facenet]),
        arcface_candidates=len([c for c in candidates.values() if c.in_arcface]),
        candidates_in_both=in_both,
    )


def match_face(
    facenet_embedding: np.ndarray,
    arcface_embedding: np.ndarray,
    facenet_index: Index,
    arcface_index: Index,
    faces_mapping: list[str],
    performers: dict[str, dict],
    config: MatchingConfig = DEFAULT_CONFIG,
) -> MatchingResult:
    """
    Match a face against the database using both models.

    This is the main entry point for face matching.

    Args:
        facenet_embedding: FaceNet512 embedding vector
        arcface_embedding: ArcFace embedding vector
        facenet_index: Voyager index for FaceNet
        arcface_index: Voyager index for ArcFace
        faces_mapping: List mapping face index to universal_id
        performers: Dict mapping universal_id to performer info
        config: Matching configuration

    Returns:
        MatchingResult with candidates and diagnostics
    """
    # Query both indices
    fn_result = query_model(facenet_embedding, facenet_index, config)
    af_result = query_model(arcface_embedding, arcface_index, config)

    # Fuse results
    return fuse_results(fn_result, af_result, faces_mapping, performers, config)


# =============================================================================
# DEBUGGING UTILITIES
# =============================================================================

def format_matching_result(result: MatchingResult, expected_names: list[str] = None) -> str:
    """Format a matching result for debugging output."""
    lines = []

    lines.append(f"Strategy: {result.fusion_strategy}")
    lines.append(f"FaceNet: {result.facenet_health.value} [{result.facenet_health_reason}] ({result.facenet_candidates} candidates)")
    lines.append(f"ArcFace: {result.arcface_health.value} [{result.arcface_health_reason}] ({result.arcface_candidates} candidates)")
    lines.append(f"In both: {result.candidates_in_both}")
    lines.append("")

    expected_names = expected_names or []

    for i, match in enumerate(result.matches[:10]):
        marker = "★" if match.name in expected_names else " "
        fn_str = f"fn={match.facenet_distance:.3f}@{match.facenet_rank}" if match.facenet_distance else "fn=-----"
        af_str = f"af={match.arcface_distance:.3f}@{match.arcface_rank}" if match.arcface_distance else "af=-----"

        lines.append(f"{marker} {i+1}. {match.name[:30]:<30} combined={match.combined_distance:.3f} ({fn_str}, {af_str})")

    return "\n".join(lines)

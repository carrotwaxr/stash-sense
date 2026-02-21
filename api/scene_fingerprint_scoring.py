"""Quality scoring for scene fingerprint matches.

Scores a candidate match between a local scene and a stash-box scene
based on fingerprint overlap, type, duration agreement, and community votes.
"""

DURATION_TOLERANCE_SECONDS = 5.0
EXACT_HASH_ALGORITHMS = {"MD5", "OSHASH"}
DEFAULT_MIN_COUNT = 2
DEFAULT_MIN_PERCENTAGE = 66


def score_match(
    matching_fingerprints: list[dict],
    total_local_fingerprints: int,
    local_duration: float,
) -> dict:
    match_count = len(matching_fingerprints)

    if total_local_fingerprints > 0:
        match_percentage = (match_count / total_local_fingerprints) * 100
    else:
        match_percentage = 0.0

    has_exact_hash = any(
        fp["algorithm"] in EXACT_HASH_ALGORITHMS for fp in matching_fingerprints
    )

    duration_diffs = []
    for fp in matching_fingerprints:
        remote_dur = fp.get("duration")
        if remote_dur is not None and local_duration is not None:
            duration_diffs.append(abs(local_duration - remote_dur))

    if duration_diffs:
        avg_diff = sum(duration_diffs) / len(duration_diffs)
        duration_agreement = avg_diff <= DURATION_TOLERANCE_SECONDS
        duration_diff = round(avg_diff, 2)
    else:
        duration_agreement = True
        duration_diff = None

    total_submissions = sum(fp.get("submissions", 0) for fp in matching_fingerprints)

    return {
        "match_count": match_count,
        "match_percentage": round(match_percentage, 1),
        "has_exact_hash": has_exact_hash,
        "duration_agreement": duration_agreement,
        "duration_diff": duration_diff,
        "total_submissions": total_submissions,
    }


def is_high_confidence(
    match_count: int,
    match_percentage: float,
    min_count: int = DEFAULT_MIN_COUNT,
    min_percentage: int = DEFAULT_MIN_PERCENTAGE,
) -> bool:
    return match_count >= min_count and match_percentage >= min_percentage

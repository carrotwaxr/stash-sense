"""Scene-level face matching: clustering, frequency, and hybrid strategies.

Provides algorithms for identifying performers across multiple video frames:
- Embedding-based face clustering (cluster_faces_by_person)
- Frequency-based matching (frequency_based_matching)
- Clustered frequency matching (clustered_frequency_matching)
- Hybrid matching combining cluster and frequency (hybrid_matching)
- Multi-signal re-ranking with body/tattoo signals (_rerank_scene_persons)
"""

import logging
from collections import defaultdict

import numpy as np

from recognizer import FaceRecognizer, PerformerMatch, RecognitionResult

logger = logging.getLogger(__name__)


def _extract_scene_signals(
    frames: list,
    detected_faces: list[tuple[int, "DetectedFace"]],
    matcher: "MultiSignalMatcher",
    use_body: bool,
    use_tattoo: bool,
) -> tuple:
    """Extract body and tattoo signals from representative scene frames.

    Args:
        frames: List of ExtractedFrame objects with .image attribute
        detected_faces: List of (frame_index, DetectedFace) tuples
        matcher: MultiSignalMatcher instance
        use_body: Whether to extract body proportions
        use_tattoo: Whether to run tattoo detection

    Returns:
        (body_ratios, tattoo_result, tattoo_scores, signals_used, tattoos_detected)
    """
    from tattoo_detector import TattooResult

    body_ratios = None
    tattoo_result = None
    tattoo_scores = None
    signals_used = ["face"]
    tattoos_detected = 0

    # Build frame index -> image lookup
    frame_map = {f.frame_index: f.image for f in frames}

    # Body: extract from one representative frame (the one with the largest face)
    if use_body and matcher.body_extractor is not None and detected_faces:
        # Pick frame with largest face
        best_face_area = 0
        best_frame_img = None
        for frame_idx, face in detected_faces:
            area = face.bbox["w"] * face.bbox["h"]
            if area > best_face_area and frame_idx in frame_map:
                best_face_area = area
                best_frame_img = frame_map[frame_idx]

        if best_frame_img is not None:
            body_ratios = matcher.body_extractor.extract(best_frame_img)
            if body_ratios is not None:
                signals_used.append("body")

    # Tattoo: detect on up to 3 frames with the most/largest faces
    if use_tattoo and matcher.tattoo_detector is not None and detected_faces:
        # Score each frame by number of faces * total face area
        frame_scores: dict[int, float] = defaultdict(float)
        for frame_idx, face in detected_faces:
            frame_scores[frame_idx] += face.bbox["w"] * face.bbox["h"]

        # Pick top 3 frames
        top_frame_idxs = sorted(frame_scores, key=frame_scores.get, reverse=True)[:3]

        # Merge detections across frames
        all_detections = []
        best_frame_for_matching = None
        best_frame_detection_count = 0

        for frame_idx in top_frame_idxs:
            if frame_idx not in frame_map:
                continue
            frame_img = frame_map[frame_idx]
            result = matcher.tattoo_detector.detect(frame_img)
            if result.has_tattoos:
                all_detections.extend(result.detections)
                if len(result.detections) > best_frame_detection_count:
                    best_frame_detection_count = len(result.detections)
                    best_frame_for_matching = (frame_img, result)

        tattoos_detected = len(all_detections)

        if all_detections:
            # Build a merged TattooResult for signal_scoring
            tattoo_result = TattooResult(
                detections=all_detections,
                has_tattoos=True,
                confidence=max(d.confidence for d in all_detections),
            )
            signals_used.append("tattoo")

            # Run embedding matching on the frame with the most detections
            if matcher.tattoo_matcher is not None and best_frame_for_matching:
                frame_img, frame_result = best_frame_for_matching
                tattoo_scores = matcher.tattoo_matcher.match(
                    frame_img, frame_result.detections
                )
        else:
            # No tattoos detected -- still useful signal (absence)
            tattoo_result = TattooResult(
                detections=[], has_tattoos=False, confidence=0.0,
            )

    return body_ratios, tattoo_result, tattoo_scores, signals_used, tattoos_detected


def _rerank_scene_persons(
    persons: list,
    matcher: "MultiSignalMatcher",
    body_ratios,
    tattoo_result,
    tattoo_scores: dict | None,
    signals_used: list[str],
    tattoos_detected: int,
) -> list:
    """Re-rank PersonResult matches using multi-signal scoring.

    Adjusts the distance-based scores in each PersonResult's all_matches list
    using body + tattoo multipliers, then re-selects best_match. Also attaches
    signals_used and tattoos_detected to each PersonResult.

    Args:
        persons: List of PersonResult from any matching mode
        matcher: MultiSignalMatcher instance (for body data lookup and tattoo embedding set)
        body_ratios: Body proportions from the scene (may be None)
        tattoo_result: Tattoo detection result (may be None)
        tattoo_scores: Tattoo embedding similarity scores (may be None)
        signals_used: List of signal names used (e.g. ["face", "body", "tattoo"])
        tattoos_detected: Number of YOLO tattoo detections in scene frames

    Returns:
        Persons list with re-ranked matches and signal metadata attached
    """
    from signal_scoring import body_ratio_penalty, tattoo_adjustment

    for person in persons:
        person.signals_used = signals_used.copy()
        person.tattoos_detected = tattoos_detected

        if not person.all_matches:
            continue

        # Re-rank all_matches using multi-signal adjustments
        scored = []
        for m in person.all_matches:
            # Derive universal_id from stashdb_id (matches recognizer convention)
            uid = f"stashdb.org:{m.stashdb_id}"

            # Base score from face distance (higher = better)
            base_score = 1.0 / (1.0 + m.distance)

            body_mult = body_ratio_penalty(body_ratios, matcher._get_candidate_body_ratios(uid))
            has_embeddings = uid in matcher.performers_with_tattoo_embeddings
            tattoo_mult = tattoo_adjustment(tattoo_result, uid, tattoo_scores, has_embeddings)

            final_score = base_score * body_mult * tattoo_mult
            scored.append((m, final_score))

        # Sort by final score descending
        scored.sort(key=lambda x: x[1], reverse=True)
        person.all_matches = [m for m, _ in scored]
        person.best_match = person.all_matches[0] if person.all_matches else None

    return persons


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


def cluster_faces_by_person(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: FaceRecognizer,
    distance_threshold: float = 0.6,
) -> list[list[tuple[int, RecognitionResult]]]:
    """
    Cluster detected faces by person using embedding cosine similarity.

    Uses greedy clustering: assign each face to the nearest existing
    cluster or create a new cluster if too far from all existing ones.

    Uses cosine distance (consistent with Voyager indices) instead of L2,
    and reuses pre-computed embeddings from recognition when available.

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        recognizer: FaceRecognizer instance (fallback for embedding generation)
        distance_threshold: Max cosine distance to consider same person (default 0.6)

    Returns:
        List of clusters, each containing faces of the same person
    """
    if not all_results:
        return []

    clusters: list[list[tuple[int, RecognitionResult, np.ndarray]]] = []

    for frame_idx, result in all_results:
        # Use stored embedding if available, otherwise recompute
        if result.embedding is not None:
            face_vector = np.concatenate([result.embedding.facenet, result.embedding.arcface])
        else:
            embedding = recognizer.generator.get_embedding(result.face.image)
            face_vector = np.concatenate([embedding.facenet, embedding.arcface])

        # Find nearest cluster by cosine distance
        best_cluster_idx = -1
        best_distance = float("inf")

        for cluster_idx, cluster in enumerate(clusters):
            # Compare to cluster centroid (average of all faces in cluster)
            cluster_vectors = [c[2] for c in cluster]
            centroid = np.mean(cluster_vectors, axis=0)
            distance = _cosine_distance(face_vector, centroid)

            if distance < best_distance:
                best_distance = distance
                best_cluster_idx = cluster_idx

        # Add to existing cluster or create new one
        if best_distance < distance_threshold and best_cluster_idx >= 0:
            clusters[best_cluster_idx].append((frame_idx, result, face_vector))
        else:
            clusters.append([(frame_idx, result, face_vector)])

    # Remove embedding vectors from output
    return [[(f, r) for f, r, _ in cluster] for cluster in clusters]


def merge_clusters_by_match(
    clusters: list[list[tuple[int, RecognitionResult]]],
) -> list[list[tuple[int, RecognitionResult]]]:
    """
    Merge clusters that have the same best performer match.

    If multiple clusters all have "Xander Corvus" as their top match,
    they're probably the same person and should be merged.
    """
    if len(clusters) <= 1:
        return clusters

    # Get best match for each cluster
    cluster_best_match: list[tuple[str, float]] = []
    for cluster in clusters:
        # Find best match across all faces in cluster
        best_match_id = None
        best_score = float("inf")

        for _, result in cluster:
            if result.matches:
                top_match = result.matches[0]
                if top_match.combined_score < best_score:
                    best_score = top_match.combined_score
                    best_match_id = top_match.stashdb_id

        cluster_best_match.append((best_match_id, best_score))

    # Group clusters by their best match
    match_to_clusters: dict[str, list[int]] = defaultdict(list)
    for i, (match_id, _) in enumerate(cluster_best_match):
        if match_id:
            match_to_clusters[match_id].append(i)

    # Merge clusters with same best match
    merged_indices: set[int] = set()
    merged_clusters: list[list[tuple[int, RecognitionResult]]] = []

    for match_id, cluster_indices in match_to_clusters.items():
        if len(cluster_indices) > 1:
            # Merge all these clusters
            merged = []
            for idx in cluster_indices:
                merged.extend(clusters[idx])
                merged_indices.add(idx)
            merged_clusters.append(merged)
        elif cluster_indices[0] not in merged_indices:
            merged_clusters.append(clusters[cluster_indices[0]])
            merged_indices.add(cluster_indices[0])

    # Add clusters that had no matches
    for i, cluster in enumerate(clusters):
        if i not in merged_indices:
            merged_clusters.append(cluster)

    return merged_clusters


def aggregate_matches(
    cluster: list[tuple[int, RecognitionResult]],
    top_k: int = 3,
    _match_to_response=None,
    _distance_to_confidence=None,
) -> list:
    """
    Aggregate matches across multiple frames for a person.

    Combines match scores across frames, preferring performers that appear
    consistently with low distances.

    Args:
        cluster: List of (frame_index, RecognitionResult) tuples
        top_k: Maximum number of matches to return
        _match_to_response: Callback to convert PerformerMatch to response model
        _distance_to_confidence: Callback to convert distance to confidence score
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from main import _match_to_response
    if _distance_to_confidence is None:
        from main import distance_to_confidence as _distance_to_confidence

    # Collect all matches across frames
    match_scores: dict[str, list[float]] = defaultdict(list)
    match_info: dict[str, PerformerMatch] = {}

    for _, result in cluster:
        for match in result.matches:
            match_scores[match.stashdb_id].append(match.combined_score)
            match_info[match.stashdb_id] = match

    # Score by: average distance * (1 / appearance_ratio)
    # This prefers performers who appear consistently with low scores
    aggregated = []
    for stashdb_id, scores in match_scores.items():
        avg_score = np.mean(scores)
        appearance_ratio = len(scores) / len(cluster)
        # Boost score for performers that appear in more frames
        adjusted_score = avg_score / (appearance_ratio ** 0.5)

        match = match_info[stashdb_id]
        aggregated.append(_match_to_response(
            match,
            confidence=_distance_to_confidence(adjusted_score),
            distance=adjusted_score,
        ))

    # Sort by adjusted score (lower is better)
    aggregated.sort(key=lambda m: m.distance)
    return aggregated[:top_k]


def frequency_based_matching(
    all_results: list[tuple[int, RecognitionResult]],
    top_k: int = 5,
    min_appearances: int = 2,
    min_unique_frames: int = 2,
    max_distance: float = 0.5,
    min_confidence: float = 0.35,
    _match_to_response=None,
    _distance_to_confidence=None,
) -> list:
    """
    Identify performers by counting appearances across all face matches.

    Instead of clustering faces first (which can fail when embeddings vary too much),
    this approach:
    1. Collects all matches from all detected faces
    2. Counts how many times each performer appears in the top matches
    3. Weights by match distance (closer matches count more)
    4. Returns performers sorted by weighted frequency

    This is more robust when clustering fails but may include false positives
    if the same wrong performer happens to match multiple faces.

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        top_k: Number of performers to return
        min_appearances: Minimum number of face matches to include a performer
        min_unique_frames: Minimum unique frames a performer must appear in
        max_distance: Only count matches below this distance
        min_confidence: Minimum confidence threshold (filters low-quality matches)
        _match_to_response: Callback to convert PerformerMatch to response model
        _distance_to_confidence: Callback to convert distance to confidence score

    Returns:
        List of PersonResult objects, one per identified performer
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from main import _match_to_response
    if _distance_to_confidence is None:
        from main import distance_to_confidence as _distance_to_confidence
    from main import PersonResult

    # Collect all matches across all faces
    performer_matches: dict[str, list[tuple[float, PerformerMatch, int]]] = defaultdict(list)

    for frame_idx, result in all_results:
        for match in result.matches:
            if match.combined_score <= max_distance:
                performer_matches[match.stashdb_id].append((match.combined_score, match, frame_idx))

    # Calculate weighted frequency score for each performer
    # Score = appearances * (1 / avg_distance) - rewards frequent, close matches
    performer_scores = []
    for stashdb_id, matches in performer_matches.items():
        if len(matches) < min_appearances:
            continue

        distances = [m[0] for m in matches]
        avg_distance = np.mean(distances)
        min_distance = min(distances)
        appearances = len(matches)
        unique_frames = len(set(m[2] for m in matches))

        # Require appearance in multiple unique frames to reduce false positives
        if unique_frames < min_unique_frames:
            continue

        # Filter by minimum confidence (1 - distance)
        if (1 - min_distance) < min_confidence:
            continue

        # Weighted score: primarily based on match quality, with modest bonus for frame count
        # Formula: confidence * (1 + small_frame_bonus)
        # This ensures a single excellent match beats multiple mediocre matches
        confidence = 1 - min_distance  # Use best match, not average
        frame_bonus = 0.1 * (unique_frames - 1)  # Small bonus: +10% per additional frame
        weighted_score = confidence * (1 + frame_bonus)

        # For display, use the match with the best (lowest) distance
        best_match = min(matches, key=lambda m: m[0])[1]

        performer_scores.append({
            "stashdb_id": stashdb_id,
            "appearances": appearances,
            "unique_frames": unique_frames,
            "avg_distance": avg_distance,
            "min_distance": min_distance,
            "weighted_score": weighted_score,
            "best_match": best_match,
        })

    # Sort by weighted score (higher is better)
    performer_scores.sort(key=lambda p: p["weighted_score"], reverse=True)

    # Convert to PersonResult format
    persons = []
    for i, p in enumerate(performer_scores[:top_k]):
        match = p["best_match"]
        resp = _match_to_response(
            match,
            confidence=_distance_to_confidence(p["avg_distance"]),
            distance=p["avg_distance"],
        )
        persons.append(PersonResult(
            person_id=i,
            frame_count=p["unique_frames"],
            best_match=resp,
            all_matches=[resp],
        ))

    return persons


def clustered_frequency_matching(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: "FaceRecognizer",
    cluster_threshold: float = 0.6,
    top_k: int = 5,
    max_distance: float = 0.5,
    min_confidence: float = 0.35,
    scene_performer_stashdb_ids: list[str] | None = None,
    tagged_boost: float = 0.03,
    _match_to_response=None,
    _distance_to_confidence=None,
) -> list:
    """
    Cluster faces first to determine person count, then use frequency matching
    within each cluster to identify who each person is.

    This combines the strengths of both approaches:
    - Clustering answers "how many distinct people are there?"
    - Frequency matching answers "who is each person?"

    The result is one PersonResult per face cluster, with alternative matches
    for each cluster shown as all_matches.
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from main import _match_to_response
    if _distance_to_confidence is None:
        from main import distance_to_confidence as _distance_to_confidence
    from main import PersonResult

    if not all_results:
        return []

    tagged_ids = set(scene_performer_stashdb_ids or [])

    # Step 1: Cluster faces by embedding similarity
    clusters = cluster_faces_by_person(
        all_results, recognizer, distance_threshold=cluster_threshold
    )
    # Merge clusters that have the same best match
    clusters = merge_clusters_by_match(clusters)

    print(f"[clustered_freq] {len(all_results)} face detections -> {len(clusters)} person clusters")

    # Step 2: For each cluster, run frequency matching to find the best performer
    persons = []
    used_performers: set[str] = set()

    # Sort clusters by size (most prominent person first)
    sorted_clusters = sorted(enumerate(clusters), key=lambda x: len(x[1]), reverse=True)

    for cluster_idx, cluster in sorted_clusters:
        cluster_size = len(cluster)
        unique_frames = len(set(frame_idx for frame_idx, _ in cluster))

        # Collect all matches from faces in this cluster
        performer_matches: dict[str, list[tuple[float, PerformerMatch, int]]] = defaultdict(list)

        for frame_idx, result in cluster:
            for match in result.matches:
                if match.combined_score <= max_distance:
                    performer_matches[match.stashdb_id].append(
                        (match.combined_score, match, frame_idx)
                    )

        if not performer_matches:
            # No matches in this cluster - unknown person
            persons.append(PersonResult(
                person_id=len(persons),
                frame_count=unique_frames,
                best_match=None,
                all_matches=[],
            ))
            continue

        # Score each performer within this cluster
        candidates = []
        for stashdb_id, matches in performer_matches.items():
            distances = [m[0] for m in matches]
            min_distance = min(distances)
            match_unique_frames = len(set(m[2] for m in matches))

            if (1 - min_distance) < min_confidence:
                continue

            confidence = 1 - min_distance
            frame_bonus = 0.1 * (match_unique_frames - 1)
            weighted_score = confidence * (1 + frame_bonus)

            # Apply small boost for already-tagged performers
            if stashdb_id in tagged_ids:
                weighted_score += tagged_boost

            best_match = min(matches, key=lambda m: m[0])[1]

            candidates.append({
                "stashdb_id": stashdb_id,
                "appearances": len(matches),
                "unique_frames": match_unique_frames,
                "min_distance": min_distance,
                "avg_distance": float(np.mean(distances)),
                "weighted_score": weighted_score,
                "best_match": best_match,
                "is_tagged": stashdb_id in tagged_ids,
            })

        # Sort by weighted score (higher is better)
        candidates.sort(key=lambda c: c["weighted_score"], reverse=True)

        # Pick best performer not yet used by a higher-priority cluster
        best_candidate = None
        alt_candidates = []
        for c in candidates:
            if c["stashdb_id"] not in used_performers:
                if best_candidate is None:
                    best_candidate = c
                    used_performers.add(c["stashdb_id"])
                else:
                    alt_candidates.append(c)
            else:
                alt_candidates.append(c)

        if best_candidate is None:
            # All candidates already used
            persons.append(PersonResult(
                person_id=len(persons),
                frame_count=unique_frames,
                best_match=None,
                all_matches=[],
            ))
            continue

        # Build all_matches list: best first, then alternatives (up to top_k)
        def _to_match_response(c: dict):
            m = c["best_match"]
            return _match_to_response(
                m,
                confidence=_distance_to_confidence(c["avg_distance"]),
                distance=c["avg_distance"],
                already_tagged=c["is_tagged"],
            )

        best_response = _to_match_response(best_candidate)
        all_matches = [best_response]
        for alt in alt_candidates[:top_k - 1]:
            all_matches.append(_to_match_response(alt))

        persons.append(PersonResult(
            person_id=len(persons),
            frame_count=unique_frames,
            best_match=best_response,
            all_matches=all_matches,
        ))

    # Sort: persons with matches first (by frame count desc), then unknowns
    persons.sort(key=lambda p: (p.best_match is not None, p.frame_count), reverse=True)

    # Re-assign person IDs after sorting
    for i, p in enumerate(persons):
        p.person_id = i

    return persons


def hybrid_matching(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: "FaceRecognizer",
    cluster_threshold: float = 0.6,
    top_k: int = 5,
    max_distance: float = 0.5,
    min_appearances: int = 2,
    min_unique_frames: int = 2,
    min_confidence: float = 0.35,
    _match_to_response=None,
    _distance_to_confidence=None,
) -> list:
    """
    Hybrid matching combining cluster and frequency approaches.

    Runs both methods and combines results:
    - Performers found by BOTH methods get a significant boost
    - Uses the best (lowest) distance from either method
    - Sorts by combined score

    This helps when:
    - Clustering works well (cluster mode catches it)
    - Clustering fails but frequency catches appearances (frequency mode catches it)

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        recognizer: FaceRecognizer instance for clustering
        cluster_threshold: Distance threshold for face clustering
        top_k: Maximum number of performers to return
        max_distance: Maximum distance threshold for matches
        min_appearances: Minimum face matches required per performer
        min_unique_frames: Minimum unique frames a performer must appear in
        min_confidence: Minimum confidence threshold (1 - distance)
    """
    # Lazy import to avoid circular dependency
    if _match_to_response is None:
        from main import _match_to_response
    if _distance_to_confidence is None:
        from main import distance_to_confidence as _distance_to_confidence
    from main import PersonResult

    # Get frequency results (as a dict for lookup)
    freq_persons = frequency_based_matching(
        all_results,
        top_k=top_k * 3,
        min_appearances=min_appearances,
        min_unique_frames=min_unique_frames,
        max_distance=max_distance,
        min_confidence=min_confidence,
        _match_to_response=_match_to_response,
        _distance_to_confidence=_distance_to_confidence,
    )
    freq_by_id = {p.best_match.stashdb_id: p for p in freq_persons if p.best_match}

    # Get cluster results
    clusters = cluster_faces_by_person(all_results, recognizer, cluster_threshold)
    clusters = merge_clusters_by_match(clusters)

    cluster_persons = []
    for cluster in clusters:
        aggregated = aggregate_matches(
            cluster, top_k=3,
            _match_to_response=_match_to_response,
            _distance_to_confidence=_distance_to_confidence,
        )
        if aggregated:
            cluster_persons.append({
                "stashdb_id": aggregated[0].stashdb_id,
                "name": aggregated[0].name,
                "frame_count": len(cluster),
                "distance": aggregated[0].distance,
                "match": aggregated[0],
            })

    cluster_by_id = {p["stashdb_id"]: p for p in cluster_persons}

    # Combine results
    all_performers = set(freq_by_id.keys()) | set(cluster_by_id.keys())

    combined_scores = []
    for stashdb_id in all_performers:
        freq_result = freq_by_id.get(stashdb_id)
        cluster_result = cluster_by_id.get(stashdb_id)

        # Determine best distance and frame count
        if freq_result and cluster_result:
            # Found by both - use best distance, combine frame counts
            best_distance = min(freq_result.best_match.distance, cluster_result["distance"])
            frame_count = max(freq_result.frame_count, cluster_result["frame_count"])
            found_by = "both"
            # Strong boost for being found by both methods (high confidence signal)
            confidence_boost = 0.25
        elif freq_result:
            best_distance = freq_result.best_match.distance
            frame_count = freq_result.frame_count
            found_by = "frequency"
            confidence_boost = 0.0
        else:
            best_distance = cluster_result["distance"]
            frame_count = cluster_result["frame_count"]
            found_by = "cluster"
            confidence_boost = 0.0

        # Calculate hybrid score: confidence with boost, plus small frame bonus
        base_confidence = 1 - best_distance
        boosted_confidence = min(1.0, base_confidence + confidence_boost)
        frame_bonus = 0.05 * (frame_count - 1)  # Small bonus per additional frame
        hybrid_score = boosted_confidence * (1 + frame_bonus)

        # Get the match object
        if freq_result:
            match_obj = freq_result.best_match
        else:
            match_obj = cluster_result["match"]

        combined_scores.append({
            "stashdb_id": stashdb_id,
            "name": match_obj.name,
            "frame_count": frame_count,
            "distance": best_distance,
            "hybrid_score": hybrid_score,
            "found_by": found_by,
            "match": match_obj,
        })

    # Sort by hybrid score (higher is better)
    combined_scores.sort(key=lambda p: p["hybrid_score"], reverse=True)

    # Filter by minimum confidence and frame requirements
    filtered_scores = [
        p for p in combined_scores
        if (1 - p["distance"]) >= min_confidence and p["frame_count"] >= min_unique_frames
    ]

    # Heuristic: limit max performers based on number of clusters
    # A scene with 3 face clusters probably has 2-3 performers, not 10
    max_performers = max(2, min(top_k, len(clusters)))
    filtered_scores = filtered_scores[:max_performers]

    # Convert to PersonResult format
    persons = []
    for i, p in enumerate(filtered_scores):
        match = p["match"]
        resp = _match_to_response(
            match,
            confidence=_distance_to_confidence(p["distance"]),
            distance=p["distance"],
        )
        persons.append(PersonResult(
            person_id=i,
            frame_count=p["frame_count"],
            best_match=resp,
            all_matches=[resp],
        ))

    return persons

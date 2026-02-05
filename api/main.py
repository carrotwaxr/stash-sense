"""FastAPI sidecar for Stash Sense.

Provides REST API endpoints for:
- Face recognition (identify performers in images)
- Recommendations engine (library analysis and curation)
"""
import base64
import json
import os
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

# Environment variables
STASH_URL = os.environ.get("STASH_URL", "").rstrip("/")
STASH_API_KEY = os.environ.get("STASH_API_KEY", "")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Optional

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from config import DatabaseConfig, MultiSignalConfig
from recognizer import FaceRecognizer, PerformerMatch, RecognitionResult
from body_proportions import BodyProportionExtractor
from tattoo_detector import TattooDetector
from multi_signal_matcher import MultiSignalMatcher, MultiSignalMatch
from embeddings import load_image
from sprite_parser import parse_vtt_file, extract_frames_from_sprite
from frame_extractor import (
    FrameExtractionConfig,
    extract_frames_from_stash_scene,
    check_ffmpeg_available,
)
from matching import MatchingConfig
from recommendations_router import router as recommendations_router, init_recommendations, save_scene_fingerprint, set_db_version


# Pydantic models for API
class FaceBox(BaseModel):
    """Bounding box for a detected face."""
    x: int
    y: int
    width: int
    height: int
    confidence: float


class PerformerMatchResponse(BaseModel):
    """A potential performer match."""
    stashdb_id: str = Field(description="StashDB performer UUID")
    name: str = Field(description="Performer name")
    confidence: float = Field(description="Match confidence (0-1, higher is better)")
    distance: float = Field(description="Combined distance score (lower is better)")
    facenet_distance: float
    arcface_distance: float
    country: Optional[str] = None
    image_url: Optional[str] = Field(None, description="StashDB profile image URL")


class FaceResult(BaseModel):
    """Recognition result for a single detected face."""
    box: FaceBox
    matches: list[PerformerMatchResponse]


class IdentifyRequest(BaseModel):
    """Request to identify performers in an image."""
    image_url: Optional[str] = Field(None, description="URL to fetch image from")
    image_base64: Optional[str] = Field(None, description="Base64-encoded image data")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")
    use_multi_signal: bool = True
    use_body: bool = True
    use_tattoo: bool = True


class IdentifyResponse(BaseModel):
    """Response with identification results."""
    faces: list[FaceResult]
    face_count: int


class DatabaseInfo(BaseModel):
    """Information about the loaded database."""
    version: str
    performer_count: int
    face_count: int
    sources: list[str]
    created_at: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_loaded: bool
    performer_count: int = 0
    face_count: int = 0


# Global recognizer instance
recognizer: Optional[FaceRecognizer] = None
db_manifest: dict = {}

# Multi-signal components
multi_signal_matcher: Optional[MultiSignalMatcher] = None
body_extractor: Optional[BodyProportionExtractor] = None
tattoo_detector: Optional[TattooDetector] = None
multi_signal_config: Optional[MultiSignalConfig] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the recognizer and initialize recommendations on startup."""
    global recognizer, db_manifest
    global multi_signal_matcher, body_extractor, tattoo_detector, multi_signal_config

    data_dir = Path(DATA_DIR)
    print(f"Loading face database from {data_dir}...")

    # Load face recognition database
    try:
        db_config = DatabaseConfig(data_dir=data_dir)

        # Load manifest for database info
        if db_config.manifest_json_path.exists():
            with open(db_config.manifest_json_path) as f:
                db_manifest = json.load(f)

        recognizer = FaceRecognizer(db_config)
        print("Face database loaded successfully!")

        # Initialize multi-signal components
        multi_signal_config = MultiSignalConfig.from_env()

        if multi_signal_config.enable_body:
            print("Initializing body proportion extractor...")
            body_extractor = BodyProportionExtractor()

        if multi_signal_config.enable_tattoo:
            print("Initializing tattoo detector...")
            tattoo_detector = TattooDetector()

        if recognizer.db_reader and (body_extractor or tattoo_detector):
            print("Initializing multi-signal matcher...")
            multi_signal_matcher = MultiSignalMatcher(
                face_recognizer=recognizer,
                db_reader=recognizer.db_reader,
                body_extractor=body_extractor,
                tattoo_detector=tattoo_detector,
            )
            print(f"Multi-signal ready: {len(multi_signal_matcher.body_data)} body, "
                  f"{len(multi_signal_matcher.tattoo_data)} tattoo records")

    except Exception as e:
        print(f"Warning: Failed to load face database: {e}")
        print("API will start but /identify will not work until database is available")
        recognizer = None

    # Initialize recommendations database
    rec_db_path = data_dir / "stash_sense.db"
    print(f"Initializing recommendations database at {rec_db_path}...")
    init_recommendations(
        db_path=str(rec_db_path),
        stash_url=STASH_URL,
        stash_api_key=STASH_API_KEY,
    )
    print("Recommendations database initialized!")

    # Set DB version for fingerprint tracking
    if db_manifest.get("version"):
        set_db_version(db_manifest["version"])
        print(f"Face recognition DB version: {db_manifest['version']}")

    if STASH_URL:
        print(f"Stash connection configured: {STASH_URL}")
    else:
        print("Warning: STASH_URL not set - recommendations analysis will not work")

    yield

    # Cleanup
    recognizer = None


app = FastAPI(
    title="Stash Sense API",
    description="Face recognition and recommendations engine for Stash",
    version="0.2.0",
    lifespan=lifespan,
)

# Enable CORS for Stash plugin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Stash runs on various ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include recommendations router
app.include_router(recommendations_router)


def distance_to_confidence(distance: float) -> float:
    """Convert distance score to confidence (0-1, higher is better)."""
    # Cosine distance ranges from 0 (identical) to 2 (opposite)
    # Map to confidence: 0 distance -> 1.0 confidence, 1.0 distance -> 0.0 confidence
    return max(0.0, min(1.0, 1.0 - distance))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and database status."""
    if recognizer is None:
        return HealthResponse(
            status="degraded",
            database_loaded=False,
        )

    return HealthResponse(
        status="healthy",
        database_loaded=True,
        performer_count=len(recognizer.performers),
        face_count=len(recognizer.faces),
    )


@app.get("/health/rate-limiter")
async def rate_limiter_status():
    """Get rate limiter metrics."""
    from rate_limiter import RateLimiter
    limiter = await RateLimiter.get_instance()
    return limiter.get_metrics()


@app.get("/database/info", response_model=DatabaseInfo)
async def database_info():
    """Get information about the loaded database."""
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")

    return DatabaseInfo(
        version=db_manifest.get("version", "unknown"),
        performer_count=len(recognizer.performers),
        face_count=len(recognizer.faces),
        sources=db_manifest.get("sources", ["stashdb.org"]),
        created_at=db_manifest.get("created_at"),
    )


@app.post("/identify", response_model=IdentifyResponse)
async def identify_performers(request: IdentifyRequest):
    """
    Identify performers in an image.

    Provide either `image_url` or `image_base64`. Returns detected faces
    with potential performer matches sorted by confidence.
    """
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")

    # Validate input
    if not request.image_url and not request.image_base64:
        raise HTTPException(
            status_code=400,
            detail="Must provide either image_url or image_base64"
        )

    # Fetch/decode image
    try:
        if request.image_url:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(request.image_url)
                response.raise_for_status()
                image_bytes = response.content
        else:
            image_bytes = base64.b64decode(request.image_base64)

        image = load_image(image_bytes)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode image: {e}")

    # Use multi-signal matching if available and requested
    if request.use_multi_signal and multi_signal_matcher is not None:
        multi_results = multi_signal_matcher.identify(
            image,
            top_k=request.top_k,
            use_body=request.use_body,
            use_tattoo=request.use_tattoo,
        )
        # Convert to response format
        results = []
        for mr in multi_results:
            results.append({
                "face": {
                    "bbox": mr.face.bbox,
                    "confidence": mr.face.confidence,
                },
                "matches": [
                    {
                        "universal_id": m.universal_id,
                        "stashdb_id": m.stashdb_id,
                        "name": m.name,
                        "country": m.country,
                        "image_url": m.image_url,
                        "score": m.combined_score,
                    }
                    for m in mr.matches
                ],
                "signals_used": mr.signals_used,
                "body_detected": mr.body_ratios is not None,
                "tattoos_detected": mr.tattoo_result.has_tattoos if mr.tattoo_result else False,
            })
        return {"results": results, "multi_signal": True}

    # Run recognition
    try:
        results = recognizer.recognize_image(
            image,
            top_k=request.top_k,
            max_distance=request.max_distance,
            min_face_confidence=request.min_face_confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {e}")

    # Convert to response format
    faces = []
    for result in results:
        face_box = FaceBox(
            x=int(result.face.box[0]),
            y=int(result.face.box[1]),
            width=int(result.face.box[2] - result.face.box[0]),
            height=int(result.face.box[3] - result.face.box[1]),
            confidence=result.face.confidence,
        )

        matches = [
            PerformerMatchResponse(
                stashdb_id=m.stashdb_id,
                name=m.name,
                confidence=distance_to_confidence(m.combined_score),
                distance=m.combined_score,
                facenet_distance=m.facenet_distance,
                arcface_distance=m.arcface_distance,
                country=m.country,
                image_url=m.image_url,
            )
            for m in result.matches
        ]

        faces.append(FaceResult(box=face_box, matches=matches))

    return IdentifyResponse(faces=faces, face_count=len(faces))


@app.post("/identify/url")
async def identify_from_url(
    url: str = Query(..., description="Image URL to analyze"),
    top_k: int = Query(5, ge=1, le=20),
    max_distance: float = Query(0.6, ge=0.0, le=2.0),
):
    """Convenience endpoint to identify from URL via query params."""
    return await identify_performers(
        IdentifyRequest(image_url=url, top_k=top_k, max_distance=max_distance)
    )


# Scene identification models
class SceneIdentifyRequest(BaseModel):
    """Request to identify performers in a scene using ffmpeg frame extraction."""
    stash_url: Optional[str] = Field(None, description="Base URL of Stash instance (or use STASH_URL env var)")
    scene_id: str = Field(description="Scene ID in Stash")
    api_key: Optional[str] = Field(None, description="Stash API key (or use STASH_API_KEY env var)")

    # Frame extraction settings
    num_frames: int = Field(40, ge=5, le=120, description="Number of frames to extract")
    start_offset_pct: float = Field(0.05, ge=0.0, le=0.5, description="Skip first N% of video")
    end_offset_pct: float = Field(0.95, ge=0.5, le=1.0, description="Stop at N% of video")

    # Face detection settings
    min_face_size: int = Field(40, ge=20, le=200, description="Minimum face size in pixels")
    min_face_confidence: float = Field(0.5, ge=0.1, le=1.0, description="Minimum face detection confidence")

    # Matching settings
    top_k: int = Field(3, ge=1, le=10, description="Matches per person")
    max_distance: float = Field(0.7, ge=0.0, le=2.0, description="Maximum match distance")

    # Clustering settings
    cluster_threshold: float = Field(0.6, ge=0.2, le=3.0, description="Distance threshold for face clustering")

    # Matching mode: "cluster", "frequency", or "hybrid"
    matching_mode: str = Field("frequency", description="Matching mode: 'cluster' (cluster faces then match), 'frequency' (count performer appearances), or 'hybrid' (combine both)")


class PersonResult(BaseModel):
    """A unique person detected across multiple frames."""
    person_id: int = Field(description="Unique ID for this person in the scene")
    frame_count: int = Field(description="Number of frames this person appeared in")
    best_match: Optional[PerformerMatchResponse] = Field(description="Best performer match")
    all_matches: list[PerformerMatchResponse] = Field(description="All potential matches")


class SceneIdentifyResponse(BaseModel):
    """Response with scene identification results."""
    scene_id: str
    frames_analyzed: int
    frames_requested: int = 0
    faces_detected: int
    faces_after_filter: int = 0
    persons: list[PersonResult]
    errors: list[str] = []
    fingerprint_saved: bool = False
    fingerprint_error: Optional[str] = None


def cluster_faces_by_person(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: FaceRecognizer,
    distance_threshold: float = 0.9,
) -> list[list[tuple[int, RecognitionResult]]]:
    """
    Cluster detected faces by person using embedding similarity.

    Uses a simple greedy clustering: assign each face to the nearest existing
    cluster or create a new cluster if too far from all existing ones.

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        recognizer: FaceRecognizer instance for generating embeddings
        distance_threshold: Max distance to consider same person (default 0.9 for L2 on concatenated embeddings)

    Returns:
        List of clusters, each containing faces of the same person
    """
    if not all_results:
        return []

    clusters: list[list[tuple[int, RecognitionResult, np.ndarray]]] = []

    for frame_idx, result in all_results:
        # Get embedding for this face
        embedding = recognizer.generator.get_embedding(result.face.image)
        face_vector = np.concatenate([embedding.facenet, embedding.arcface])

        # Find nearest cluster
        best_cluster_idx = -1
        best_distance = float("inf")

        for cluster_idx, cluster in enumerate(clusters):
            # Compare to cluster centroid (average of all faces in cluster)
            cluster_vectors = [c[2] for c in cluster]
            centroid = np.mean(cluster_vectors, axis=0)
            distance = np.linalg.norm(face_vector - centroid)

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
) -> list[PerformerMatchResponse]:
    """
    Aggregate matches across multiple frames for a person.

    Combines match scores across frames, preferring performers that appear
    consistently with low distances.
    """
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
        aggregated.append(PerformerMatchResponse(
            stashdb_id=match.stashdb_id,
            name=match.name,
            confidence=distance_to_confidence(adjusted_score),
            distance=adjusted_score,
            facenet_distance=np.mean([m.facenet_distance for m in [match_info[stashdb_id]] * len(scores)]),
            arcface_distance=np.mean([m.arcface_distance for m in [match_info[stashdb_id]] * len(scores)]),
            country=match.country,
            image_url=match.image_url,
        ))

    # Sort by adjusted score (lower is better)
    aggregated.sort(key=lambda m: m.distance)
    return aggregated[:top_k]


def frequency_based_matching(
    all_results: list[tuple[int, RecognitionResult]],
    top_k: int = 5,
    min_appearances: int = 1,
    max_distance: float = 0.7,
) -> list[PersonResult]:
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
        max_distance: Only count matches below this distance

    Returns:
        List of PersonResult objects, one per identified performer
    """
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
        persons.append(PersonResult(
            person_id=i,
            frame_count=p["unique_frames"],
            best_match=PerformerMatchResponse(
                stashdb_id=match.stashdb_id,
                name=match.name,
                confidence=distance_to_confidence(p["avg_distance"]),
                distance=p["avg_distance"],
                facenet_distance=match.facenet_distance,
                arcface_distance=match.arcface_distance,
                country=match.country,
                image_url=match.image_url,
            ),
            all_matches=[PerformerMatchResponse(
                stashdb_id=match.stashdb_id,
                name=match.name,
                confidence=distance_to_confidence(p["avg_distance"]),
                distance=p["avg_distance"],
                facenet_distance=match.facenet_distance,
                arcface_distance=match.arcface_distance,
                country=match.country,
                image_url=match.image_url,
            )],
        ))

    return persons


def hybrid_matching(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: "FaceRecognizer",
    cluster_threshold: float = 0.6,
    top_k: int = 5,
    max_distance: float = 0.7,
) -> list[PersonResult]:
    """
    Hybrid matching combining cluster and frequency approaches.

    Runs both methods and combines results:
    - Performers found by BOTH methods get a significant boost
    - Uses the best (lowest) distance from either method
    - Sorts by combined score

    This helps when:
    - Clustering works well (cluster mode catches it)
    - Clustering fails but frequency catches appearances (frequency mode catches it)
    """
    # Get frequency results (as a dict for lookup)
    freq_persons = frequency_based_matching(
        all_results, top_k=top_k * 3, min_appearances=1, max_distance=max_distance
    )
    freq_by_id = {p.best_match.stashdb_id: p for p in freq_persons if p.best_match}

    # Get cluster results
    clusters = cluster_faces_by_person(all_results, recognizer, cluster_threshold)
    clusters = merge_clusters_by_match(clusters)

    cluster_persons = []
    for cluster in clusters:
        aggregated = aggregate_matches(cluster, top_k=3)
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
            # Significant boost for being found by both methods
            confidence_boost = 0.15
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

    # Convert to PersonResult format
    persons = []
    for i, p in enumerate(combined_scores[:top_k]):
        match = p["match"]
        persons.append(PersonResult(
            person_id=i,
            frame_count=p["frame_count"],
            best_match=PerformerMatchResponse(
                stashdb_id=match.stashdb_id,
                name=match.name,
                confidence=distance_to_confidence(p["distance"]),
                distance=p["distance"],
                facenet_distance=match.facenet_distance,
                arcface_distance=match.arcface_distance,
                country=match.country,
                image_url=match.image_url,
            ),
            all_matches=[PerformerMatchResponse(
                stashdb_id=match.stashdb_id,
                name=match.name,
                confidence=distance_to_confidence(p["distance"]),
                distance=p["distance"],
                facenet_distance=match.facenet_distance,
                arcface_distance=match.arcface_distance,
                country=match.country,
                image_url=match.image_url,
            )],
        ))

    return persons


@app.post("/identify/scene", response_model=SceneIdentifyResponse)
async def identify_scene(request: SceneIdentifyRequest):
    """
    Identify all performers in a scene using ffmpeg frame extraction.

    Extracts full-resolution frames from the video stream using ffmpeg,
    detects faces, clusters them by person, and returns matches.
    """
    import time
    t_start = time.time()

    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")

    if not check_ffmpeg_available():
        raise HTTPException(status_code=503, detail="ffmpeg not available")

    base_url = STASH_URL.rstrip("/")
    api_key = STASH_API_KEY

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    print(f"[identify_scene] === START scene_id={request.scene_id} ===")
    print(f"[identify_scene] Settings: num_frames={request.num_frames}, min_face_size={request.min_face_size}, max_distance={request.max_distance}, mode={request.matching_mode}")

    # Get scene info from Stash
    screenshot_url = None
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            gql_query = {
                "query": f'''{{
                    findScene(id: "{request.scene_id}") {{
                        files {{
                            duration
                            width
                            height
                        }}
                        paths {{
                            screenshot
                        }}
                    }}
                }}'''
            }
            headers = {"ApiKey": api_key, "Content-Type": "application/json"}
            response = await client.post(f"{base_url}/graphql", json=gql_query, headers=headers)
            response.raise_for_status()
            data = response.json()

            scene_data = data.get("data", {}).get("findScene", {})
            if not scene_data or not scene_data.get("files"):
                raise HTTPException(status_code=404, detail="Scene not found or has no files")

            file_info = scene_data["files"][0]
            duration_sec = file_info.get("duration", 0)
            if not duration_sec:
                raise HTTPException(status_code=400, detail="Scene has no duration")

            # Get screenshot URL if available
            paths = scene_data.get("paths", {})
            screenshot_url = paths.get("screenshot") if paths else None

            print(f"[identify_scene] [{time.time()-t_start:.1f}s] Scene info: duration={duration_sec:.1f}s, resolution={file_info.get('width')}x{file_info.get('height')}, screenshot={'yes' if screenshot_url else 'no'}")

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to query scene: {e}")

    # Configure frame extraction
    config = FrameExtractionConfig(
        num_frames=request.num_frames,
        start_offset_pct=request.start_offset_pct,
        end_offset_pct=request.end_offset_pct,
        min_face_size=request.min_face_size,
        min_face_confidence=request.min_face_confidence,
    )

    # Extract frames using ffmpeg
    t_extract = time.time()
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Extracting {request.num_frames} frames...")
    extraction_result = await extract_frames_from_stash_scene(
        stash_url=base_url,
        scene_id=request.scene_id,
        duration_sec=duration_sec,
        api_key=api_key,
        config=config,
    )

    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Extracted {len(extraction_result.frames)} frames in {time.time()-t_extract:.1f}s")
    if extraction_result.errors:
        print(f"[identify_scene] Errors: {extraction_result.errors[:3]}")

    # Configure matching
    match_config = MatchingConfig(
        query_k=100,  # Get more candidates for better fusion
        facenet_weight=0.6,
        arcface_weight=0.4,
        max_results=request.top_k * 2,
        max_distance=request.max_distance,
    )

    # Detect and recognize faces in each frame
    all_results: list[tuple[int, RecognitionResult]] = []
    total_faces = 0
    filtered_faces = 0

    for frame in extraction_result.frames:
        # Detect faces
        faces = recognizer.generator.detect_faces(
            frame.image,
            min_confidence=request.min_face_confidence,
        )

        for face in faces:
            total_faces += 1

            # Apply minimum face size filter
            if face.bbox["w"] < request.min_face_size or face.bbox["h"] < request.min_face_size:
                continue

            filtered_faces += 1

            # Recognize this face using V2 matching (with health detection)
            matches, _match_result = recognizer.recognize_face_v2(face, match_config)

            result = RecognitionResult(face=face, matches=matches)
            all_results.append((frame.frame_index, result))

    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Face detection: {total_faces} detected, {filtered_faces} after size filter")

    # Process screenshot if available (high-quality cover image often has clear faces)
    # Stash serves thumbnails via paths.screenshot, so we scale up if needed
    screenshot_faces = 0
    if screenshot_url:
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"ApiKey": api_key}
                screenshot_resp = await client.get(screenshot_url, headers=headers)
                if screenshot_resp.status_code == 200:
                    screenshot_image = load_image(screenshot_resp.content)
                    img_h, img_w = screenshot_image.shape[:2]

                    # Scale up thumbnail if significantly smaller than video resolution
                    video_width = file_info.get("width", 1920)
                    if img_w < video_width * 0.8:
                        import cv2
                        scale_factor = video_width / img_w
                        new_w = int(img_w * scale_factor)
                        new_h = int(img_h * scale_factor)
                        screenshot_image = cv2.resize(screenshot_image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot upscaled: {img_w}x{img_h} -> {new_w}x{new_h}")
                        img_w, img_h = new_w, new_h

                    screenshot_detected = recognizer.generator.detect_faces(
                        screenshot_image,
                        min_confidence=request.min_face_confidence,
                    )
                    for face in screenshot_detected:
                        face_w, face_h = face.bbox["w"], face.bbox["h"]
                        if face_w >= request.min_face_size and face_h >= request.min_face_size:
                            matches, _ = recognizer.recognize_face_v2(face, match_config)
                            result = RecognitionResult(face=face, matches=matches)
                            # Use frame_index -1 to indicate screenshot
                            all_results.append((-1, result))
                            screenshot_faces += 1
                            top_match = matches[0].name if matches else "no match"
                            print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot face: {face_w}x{face_h}px -> {top_match}")
                        else:
                            print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot face too small: {face_w}x{face_h}px < {request.min_face_size}")
                    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot ({img_w}x{img_h}): {len(screenshot_detected)} faces, {screenshot_faces} usable")
        except Exception as e:
            print(f"[identify_scene] Screenshot processing failed: {e}")

    # Choose matching mode
    t_match = time.time()
    if request.matching_mode == "hybrid":
        # Hybrid matching: combine cluster and frequency approaches
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Using hybrid matching...")
        persons = hybrid_matching(
            all_results,
            recognizer,
            cluster_threshold=request.cluster_threshold,
            top_k=request.top_k * 2,
            max_distance=request.max_distance,
        )
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Hybrid matching: {len(persons)} performers in {time.time()-t_match:.1f}s")
    elif request.matching_mode == "frequency":
        # Frequency-based matching: count performer appearances across all faces
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Using frequency matching...")
        persons = frequency_based_matching(
            all_results,
            top_k=request.top_k * 2,  # Get more candidates, we'll filter later
            min_appearances=1,
            max_distance=request.max_distance,
        )
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Frequency matching: {len(persons)} performers in {time.time()-t_match:.1f}s")
    else:
        # Cluster-based matching (original approach)
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Using cluster matching...")

        # Cluster faces by person
        clusters = cluster_faces_by_person(
            all_results,
            recognizer,
            distance_threshold=request.cluster_threshold,
        )
        print(f"[identify_scene] Initial clusters: {len(clusters)}")

        # Merge clusters that have the same best match
        clusters = merge_clusters_by_match(clusters)
        print(f"[identify_scene] After merge: {len(clusters)} clusters")

        # Build response with deduplication
        persons = []
        used_performers: set[str] = set()  # Track which performers we've assigned

        # First pass: build all persons sorted by frame count
        all_persons = []
        for person_id, cluster in enumerate(clusters):
            aggregated_matches = aggregate_matches(cluster, top_k=request.top_k)
            all_persons.append((len(cluster), PersonResult(
                person_id=person_id,
                frame_count=len(cluster),
                best_match=aggregated_matches[0] if aggregated_matches else None,
                all_matches=aggregated_matches,
            )))

        # Sort by frame count (most prominent people first)
        all_persons.sort(key=lambda x: x[0], reverse=True)

        # Second pass: deduplicate - each performer can only be the best match once
        for _, person in all_persons:
            if person.best_match:
                if person.best_match.stashdb_id in used_performers:
                    # This performer already assigned to a more prominent person
                    # Find next best match that isn't used
                    for alt_match in person.all_matches[1:]:
                        if alt_match.stashdb_id not in used_performers:
                            person.best_match = alt_match
                            used_performers.add(alt_match.stashdb_id)
                            break
                    else:
                        # No unused matches, set best_match to None
                        person.best_match = None
                else:
                    used_performers.add(person.best_match.stashdb_id)

            # Also filter all_matches to not include already-used performers
            person.all_matches = [m for m in person.all_matches if m.stashdb_id not in used_performers or m.stashdb_id == (person.best_match.stashdb_id if person.best_match else None)]
            persons.append(person)

        # Re-assign person IDs after sorting
        for i, person in enumerate(persons):
            person.person_id = i

    top_names = [p.best_match.name for p in persons[:3] if p.best_match]
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] === DONE === Top matches: {', '.join(top_names)}")

    # Persist fingerprint to stash_sense.db for duplicate detection
    fingerprint_saved = False
    fingerprint_error = None

    if persons:
        performer_data = []
        for person in persons:
            if person.best_match:
                # Convert distance to confidence (0-1 scale, lower distance = higher confidence)
                avg_distance = person.best_match.distance
                avg_confidence = max(0, 1 - avg_distance) if avg_distance is not None else None
                performer_data.append({
                    "performer_id": person.best_match.stashdb_id,
                    "face_count": person.frame_count,
                    "avg_confidence": avg_confidence,
                })

        if performer_data:
            current_db_version = db_manifest.get("version")
            fp_id, fp_error = save_scene_fingerprint(
                scene_id=int(request.scene_id),
                frames_analyzed=len(extraction_result.frames),
                performer_data=performer_data,
                db_version=current_db_version,
            )
            if fp_id:
                fingerprint_saved = True
                print(f"[identify_scene] [{time.time()-t_start:.1f}s] Saved fingerprint #{fp_id} with {len(performer_data)} performers")
            else:
                fingerprint_error = fp_error
                print(f"[identify_scene] [{time.time()-t_start:.1f}s] Failed to save fingerprint: {fp_error}")

    return SceneIdentifyResponse(
        scene_id=request.scene_id,
        frames_analyzed=len(extraction_result.frames),
        frames_requested=request.num_frames,
        faces_detected=total_faces,
        faces_after_filter=filtered_faces,
        persons=persons,
        errors=extraction_result.errors[:5] if extraction_result.errors else [],
        fingerprint_saved=fingerprint_saved,
        fingerprint_error=fingerprint_error,
    )


# Health check for ffmpeg
@app.get("/health/ffmpeg")
async def ffmpeg_health():
    """Check if ffmpeg is available for V2 scene identification."""
    available = check_ffmpeg_available()
    return {
        "ffmpeg_available": available,
        "v2_endpoint_ready": available,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

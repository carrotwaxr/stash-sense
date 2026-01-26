"""FastAPI sidecar for face recognition.

Provides REST API endpoints for identifying performers in images.
"""
import base64
import json
import os
from collections import defaultdict

# Environment variables for Stash connection
STASH_URL = os.environ.get("STASH_URL", "").rstrip("/")
STASH_API_KEY = os.environ.get("STASH_API_KEY", "")
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

from config import DatabaseConfig
from recognizer import FaceRecognizer, PerformerMatch, RecognitionResult
from embeddings import load_image
from sprite_parser import parse_vtt_file, extract_frames_from_sprite


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the recognizer on startup."""
    global recognizer, db_manifest

    data_dir = Path(os.environ.get("DATA_DIR", "./data"))
    print(f"Loading database from {data_dir}...")

    try:
        db_config = DatabaseConfig(data_dir=data_dir)

        # Load manifest for database info
        if db_config.manifest_json_path.exists():
            with open(db_config.manifest_json_path) as f:
                db_manifest = json.load(f)

        recognizer = FaceRecognizer(db_config)
        print("Database loaded successfully!")
    except Exception as e:
        print(f"Warning: Failed to load database: {e}")
        print("API will start but /identify will not work until database is available")
        recognizer = None

    yield

    # Cleanup
    recognizer = None


app = FastAPI(
    title="Stash Face Recognition API",
    description="Identify performers in images using face recognition",
    version="0.1.0",
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
    """Request to identify performers in a scene using sprite sheet."""
    stash_url: Optional[str] = Field(None, description="Base URL of Stash instance (or use STASH_URL env var)")
    scene_id: str = Field(description="Scene ID in Stash")
    api_key: Optional[str] = Field(None, description="Stash API key (or use STASH_API_KEY env var)")
    max_frames: int = Field(20, ge=1, le=100, description="Max frames to analyze")
    top_k: int = Field(3, ge=1, le=10, description="Matches per person")
    max_distance: float = Field(0.6, ge=0.0, le=2.0)


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
    faces_detected: int
    persons: list[PersonResult]


def cluster_faces_by_person(
    all_results: list[tuple[int, RecognitionResult]],
    recognizer: FaceRecognizer,
    distance_threshold: float = 0.5,
) -> list[list[tuple[int, RecognitionResult]]]:
    """
    Cluster detected faces by person using embedding similarity.

    Uses a simple greedy clustering: assign each face to the nearest existing
    cluster or create a new cluster if too far from all existing ones.

    Args:
        all_results: List of (frame_index, RecognitionResult) tuples
        recognizer: FaceRecognizer instance for generating embeddings
        distance_threshold: Max distance to consider same person

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


@app.post("/identify/scene", response_model=SceneIdentifyResponse)
async def identify_scene(request: SceneIdentifyRequest):
    """
    Identify all performers in a scene using its sprite sheet.

    Fetches the sprite sheet and VTT from Stash, extracts frames,
    detects faces, clusters them by person, and returns matches.
    """
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")

    # ALWAYS use environment variables (ignore request values to avoid stale plugin issues)
    base_url = STASH_URL.rstrip("/")
    api_key = STASH_API_KEY

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    headers = {"ApiKey": api_key} if api_key else {}

    print(f"[identify_scene] stash_url={base_url}, scene_id={request.scene_id}")
    print(f"[identify_scene] api_key={'<set>' if api_key else '<empty>'}")

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            # Query GraphQL for sprite path (Stash uses hash-based paths)
            gql_query = {
                "query": f'{{ findScene(id: "{request.scene_id}") {{ paths {{ sprite }} }} }}'
            }
            gql_headers = {**headers, "Content-Type": "application/json"}
            gql_response = await client.post(f"{base_url}/graphql", json=gql_query, headers=gql_headers)
            gql_response.raise_for_status()
            gql_data = gql_response.json()

            sprite_url = gql_data.get("data", {}).get("findScene", {}).get("paths", {}).get("sprite")
            if not sprite_url:
                raise HTTPException(status_code=400, detail="Scene has no sprite sheet generated")

            print(f"[identify_scene] Fetching sprite from: {sprite_url}")
            sprite_response = await client.get(sprite_url, headers=headers)
            sprite_response.raise_for_status()
            sprite_image = Image.open(BytesIO(sprite_response.content))

            # Fetch VTT file
            vtt_url = f"{base_url}/scene/{request.scene_id}/vtt/thumbs"
            vtt_response = await client.get(vtt_url, headers=headers)
            vtt_response.raise_for_status()
            vtt_content = vtt_response.text

    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch sprite sheet: {e}")

    # Parse VTT and extract frames
    cues = parse_vtt_file(vtt_content)
    if not cues:
        raise HTTPException(status_code=400, detail="No frames found in VTT file")

    # Sample frames evenly across the scene
    sample_interval = max(1, len(cues) // request.max_frames)
    frames = extract_frames_from_sprite(
        sprite_image,
        cues,
        max_frames=request.max_frames,
        sample_interval=sample_interval,
    )

    # Detect and recognize faces in each frame
    all_results: list[tuple[int, RecognitionResult]] = []
    total_faces = 0

    for frame in frames:
        results = recognizer.recognize_image(
            frame.image,
            top_k=request.top_k * 2,  # Get more for aggregation
            max_distance=request.max_distance,
            min_face_confidence=0.5,
        )
        total_faces += len(results)
        for result in results:
            all_results.append((frame.index, result))

    # Cluster faces by person
    clusters = cluster_faces_by_person(all_results, recognizer)

    # Build response
    persons = []
    for person_id, cluster in enumerate(clusters):
        aggregated_matches = aggregate_matches(cluster, top_k=request.top_k)
        persons.append(PersonResult(
            person_id=person_id,
            frame_count=len(cluster),
            best_match=aggregated_matches[0] if aggregated_matches else None,
            all_matches=aggregated_matches,
        ))

    # Sort by frame count (most prominent people first)
    persons.sort(key=lambda p: p.frame_count, reverse=True)

    return SceneIdentifyResponse(
        scene_id=request.scene_id,
        frames_analyzed=len(frames),
        faces_detected=total_faces,
        persons=persons,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

"""FastAPI sidecar for Stash Sense.

Provides REST API endpoints for:
- Face recognition (identify performers in images)
- Recommendations engine (library analysis and curation)
"""
import asyncio
import base64
import json
import logging
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
from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

import face_config
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
from recommendations_router import router as recommendations_router, init_recommendations, save_scene_fingerprint, save_image_fingerprint, set_db_version
from stashbox_client import StashBoxClient
from database_updater import DatabaseUpdater, UpdateStatus
from hardware import init_hardware
from settings import init_settings, get_setting, migrate_env_vars
from settings_router import router as settings_router, init_settings_router
from queue_router import router as queue_router

logger = logging.getLogger(__name__)

# StashBox endpoint configuration
ENDPOINT_URLS = {
    "stashdb.org": "https://stashdb.org/graphql",
    "fansdb.cc": "https://fansdb.cc/graphql",
    "theporndb.net": "https://theporndb.net/graphql",
    "pmvstash.org": "https://pmvstash.org/graphql",
    "javstash.org": "https://javstash.org/graphql",
}

ENDPOINT_API_KEY_ENVS = {
    "stashdb.org": "STASHDB_API_KEY",
    "fansdb.cc": "FANSDB_API_KEY",
    "theporndb.net": "THEPORNDB_API_KEY",
    "pmvstash.org": "PMVSTASH_API_KEY",
    "javstash.org": "JAVSTASH_API_KEY",
}

# Lazily initialized StashBox clients
_stashbox_clients: dict[str, StashBoxClient] = {}

# Image URL cache keyed by universal_id
_image_cache: dict[str, Optional[str]] = {}


def _get_stashbox_client(endpoint_domain: str) -> Optional[StashBoxClient]:
    """Get or create a StashBox client for the given endpoint domain."""
    if endpoint_domain in _stashbox_clients:
        return _stashbox_clients[endpoint_domain]

    url = ENDPOINT_URLS.get(endpoint_domain)
    env_key = ENDPOINT_API_KEY_ENVS.get(endpoint_domain)
    if not url or not env_key:
        return None

    api_key = os.environ.get(env_key, "")
    if not api_key:
        return None

    client = StashBoxClient(url, api_key)
    _stashbox_clients[endpoint_domain] = client
    return client


def _extract_endpoint(universal_id: str | None) -> str | None:
    """Extract endpoint domain from universal_id (e.g. 'stashdb.org:uuid' -> 'stashdb.org')."""
    if universal_id and ":" in universal_id:
        return universal_id.split(":")[0]
    return None


async def _fetch_image_for_match(match: PerformerMatch) -> None:
    """Fetch and cache image URL for a single match from StashBox."""
    uid = match.universal_id
    if uid in _image_cache:
        match.image_url = _image_cache[uid]
        return

    endpoint = _extract_endpoint(uid)
    if not endpoint:
        return

    client = _get_stashbox_client(endpoint)
    if not client:
        return

    try:
        performer = await client.get_performer(match.stashdb_id)
        if performer:
            images = performer.get("images") or []
            image_url = images[0].get("url") if images else None
            _image_cache[uid] = image_url
            match.image_url = image_url
        else:
            _image_cache[uid] = None
    except Exception as e:
        logger.debug(f"Failed to fetch image for {uid}: {e}")
        _image_cache[uid] = None


async def _fetch_missing_images(all_matches: list[PerformerMatch]) -> None:
    """Fetch missing image URLs from StashBox for matches that have None."""
    needs_fetch = [
        m for m in all_matches
        if m.image_url is None and m.universal_id not in _image_cache
    ]

    # Apply cache hits for already-cached entries
    for m in all_matches:
        if m.image_url is None and m.universal_id in _image_cache:
            m.image_url = _image_cache[m.universal_id]

    if not needs_fetch:
        return

    tasks = [_fetch_image_for_match(m) for m in needs_fetch]
    await asyncio.gather(*tasks, return_exceptions=True)


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
    endpoint: Optional[str] = Field(None, description="StashBox endpoint domain (e.g. 'stashdb.org')")
    already_tagged: bool = Field(False, description="Whether this performer is already tagged on the scene")


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


class ImageIdentifyRequest(BaseModel):
    """Request to identify performers in a Stash image by ID."""
    image_id: str = Field(description="Stash image ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")
    use_multi_signal: bool = True
    use_body: bool = True
    use_tattoo: bool = True


class GalleryPerformerResult(BaseModel):
    """A performer identified across a gallery."""
    performer_id: str = Field(description="StashDB performer UUID")
    name: str
    best_distance: float
    avg_distance: float
    confidence: float = Field(description="Best match confidence (0-1)")
    image_count: int = Field(description="Number of images this performer appeared in")
    image_ids: list[str] = Field(description="Stash image IDs where performer was found")
    country: Optional[str] = None
    image_url: Optional[str] = Field(None, description="StashDB profile image URL")
    endpoint: Optional[str] = Field(None, description="StashBox endpoint domain")


class GalleryIdentifyRequest(BaseModel):
    """Request to identify performers in a Stash gallery."""
    gallery_id: str = Field(description="Stash gallery ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")


class GalleryIdentifyResponse(BaseModel):
    """Response with gallery identification results."""
    gallery_id: str
    total_images: int
    images_processed: int
    faces_detected: int
    performers: list[GalleryPerformerResult]
    errors: list[str] = []


class DatabaseInfo(BaseModel):
    """Information about the loaded database."""
    version: str
    performer_count: int
    face_count: int
    sources: list[str]
    created_at: Optional[str] = None
    tattoo_embedding_count: Optional[int] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    database_loaded: bool
    performer_count: int = 0
    face_count: int = 0


class CheckUpdateResponse(BaseModel):
    current_version: str
    latest_version: Optional[str] = None
    update_available: bool
    release_name: Optional[str] = None
    download_url: Optional[str] = None
    download_size_mb: Optional[int] = None
    published_at: Optional[str] = None
    error: Optional[str] = None


class StartUpdateResponse(BaseModel):
    job_id: str
    status: str


class UpdateStatusResponse(BaseModel):
    status: str
    progress_pct: int = 0
    current_version: Optional[str] = None
    target_version: Optional[str] = None
    error: Optional[str] = None


# Global recognizer instance
recognizer: Optional[FaceRecognizer] = None
db_manifest: dict = {}

# Multi-signal components
multi_signal_matcher: Optional[MultiSignalMatcher] = None
body_extractor: Optional[BodyProportionExtractor] = None
tattoo_detector: Optional[TattooDetector] = None
tattoo_matcher = None  # Optional[TattooMatcher]
multi_signal_config: Optional[MultiSignalConfig] = None

# Database self-updater
db_updater: Optional[DatabaseUpdater] = None


def reload_database(data_dir: Path) -> bool:
    """Load (or reload) the face recognition database and multi-signal components.

    Sets the module-level globals: recognizer, db_manifest, multi_signal_matcher,
    body_extractor, tattoo_detector, tattoo_matcher, multi_signal_config.

    When called for a hot-swap (not first startup), stale references for
    body_extractor, tattoo_detector, tattoo_matcher, and multi_signal_matcher
    are explicitly cleared before rebuilding.

    Args:
        data_dir: Path to the data directory containing DB files.

    Returns:
        True on success.

    Raises:
        Exception: on any failure during loading.
    """
    global recognizer, db_manifest
    global multi_signal_matcher, body_extractor, tattoo_detector, tattoo_matcher, multi_signal_config

    # Clear stale references so a hot-swap doesn't leave dangling pointers
    body_extractor = None
    tattoo_detector = None
    tattoo_matcher = None
    multi_signal_matcher = None

    print(f"Loading face database from {data_dir}...")

    db_config = DatabaseConfig(data_dir=data_dir)

    # Load manifest for database info
    if db_config.manifest_json_path.exists():
        with open(db_config.manifest_json_path) as f:
            db_manifest = json.load(f)

    recognizer = FaceRecognizer(db_config)
    print("Face database loaded successfully!")

    # Initialize multi-signal components
    multi_signal_config = MultiSignalConfig.from_settings()

    if multi_signal_config.enable_body:
        print("Initializing body proportion extractor...")
        body_extractor = BodyProportionExtractor()

    # Tattoo signal: "auto" enables if index files exist, "true" always enables
    enable_tattoo = multi_signal_config.enable_tattoo
    tattoo_enabled = (
        enable_tattoo == "true"
        or (enable_tattoo == "auto"
            and db_config.tattoo_index_path.exists()
            and db_config.tattoo_json_path.exists())
    )

    if tattoo_enabled:
        print("Initializing tattoo detector...")
        tattoo_detector = TattooDetector()

        # Initialize embedding-based matcher if index is available
        if recognizer.tattoo_index is not None and recognizer.tattoo_mapping is not None:
            from tattoo_matcher import TattooMatcher
            tattoo_matcher = TattooMatcher(
                tattoo_index=recognizer.tattoo_index,
                tattoo_mapping=recognizer.tattoo_mapping,
            )
            print(f"Tattoo embedding matching ready: {len(recognizer.tattoo_index)} embeddings loaded")

    if recognizer.db_reader and (body_extractor or tattoo_detector):
        print("Initializing multi-signal matcher...")
        multi_signal_matcher = MultiSignalMatcher(
            face_recognizer=recognizer,
            db_reader=recognizer.db_reader,
            body_extractor=body_extractor,
            tattoo_detector=tattoo_detector,
            tattoo_matcher=tattoo_matcher,
        )
        tattoo_count = len(multi_signal_matcher.performers_with_tattoo_embeddings)
        print(f"Multi-signal ready: {len(multi_signal_matcher.body_data)} body, "
              f"{tattoo_count} performers with tattoo embeddings")

    # Set DB version for fingerprint tracking
    if db_manifest.get("version"):
        set_db_version(db_manifest["version"])
        print(f"Face recognition DB version: {db_manifest['version']}")

    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the recognizer and initialize recommendations on startup."""
    global recognizer

    data_dir = Path(DATA_DIR)

    global db_updater

    # Detect hardware and classify tier
    hw_profile = init_hardware(str(data_dir))

    # Load face recognition database
    try:
        reload_database(data_dir)
    except Exception as e:
        print(f"Warning: Failed to load face database: {e}")
        print("API will start but /identify will not work until database is available")
        recognizer = None

    # Initialize database self-updater
    db_updater = DatabaseUpdater(
        data_dir=data_dir,
        reload_fn=reload_database,
    )

    # Initialize recommendations database
    rec_db_path = data_dir / "stash_sense.db"
    print(f"Initializing recommendations database at {rec_db_path}...")
    init_recommendations(
        db_path=str(rec_db_path),
        stash_url=STASH_URL,
        stash_api_key=STASH_API_KEY,
    )
    print("Recommendations database initialized!")

    # Initialize settings system (needs rec_db for persistence)
    from recommendations_router import get_rec_db
    settings_mgr = init_settings(get_rec_db(), hw_profile.tier)
    init_settings_router()

    # Migrate deprecated env vars to settings on first run
    migrated = migrate_env_vars(settings_mgr)
    if migrated:
        logger.warning(f"Migrated {migrated} env var(s) to settings system")

    override_count = sum(
        1 for v in settings_mgr.get_all_with_metadata()['categories'].values()
        for s in v['settings'].values() if s['is_override']
    )
    logger.warning(f"Settings initialized: tier={hw_profile.tier}, overrides={override_count}")

    # Initialize queue manager
    from queue_manager import QueueManager
    from queue_router import init_queue_router
    queue_mgr = QueueManager(get_rec_db())
    queue_mgr.recover_on_startup()
    queue_mgr.seed_default_schedules()
    init_queue_router(queue_mgr)
    await queue_mgr.start()
    logger.warning("Queue manager started")

    import signal
    def handle_sigterm(signum, frame):
        logger.warning("SIGTERM received, initiating graceful shutdown...")
        queue_mgr.request_shutdown()
    signal.signal(signal.SIGTERM, handle_sigterm)

    if STASH_URL:
        print(f"Stash connection configured: {STASH_URL}")
    else:
        print("Warning: STASH_URL not set - recommendations analysis will not work")

    yield

    # Graceful shutdown — wait for jobs to checkpoint
    await queue_mgr.stop(timeout=30.0)
    logger.warning("Queue manager stopped")

    # Cleanup
    recognizer = None


app = FastAPI(
    title="Stash Sense API",
    description="Face recognition and recommendations engine for Stash",
    version="0.3.0",
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

# Include routers
app.include_router(recommendations_router)
app.include_router(settings_router)
app.include_router(queue_router)


def _match_to_response(m, **overrides) -> "PerformerMatchResponse":
    """Convert a PerformerMatch (or MultiSignalMatch or PerformerMatchResponse) to PerformerMatchResponse."""
    uid = getattr(m, "universal_id", None)
    score = getattr(m, "combined_score", getattr(m, "distance", 0))
    return PerformerMatchResponse(
        stashdb_id=m.stashdb_id,
        name=m.name,
        confidence=distance_to_confidence(score),
        distance=score,
        facenet_distance=m.facenet_distance,
        arcface_distance=m.arcface_distance,
        country=m.country,
        image_url=m.image_url,
        endpoint=_extract_endpoint(uid) or getattr(m, "endpoint", None),
        **overrides,
    )


def distance_to_confidence(distance: float) -> float:
    """Convert distance score to confidence (0-1, higher is better)."""
    # Cosine distance ranges from 0 (identical) to 2 (opposite)
    # Map to confidence: 0 distance -> 1.0 confidence, 1.0 distance -> 0.0 confidence
    return max(0.0, min(1.0, 1.0 - distance))


async def require_db_available():
    """Return 503 if a database update is currently swapping files."""
    if db_updater and db_updater._state.status in (
        UpdateStatus.SWAPPING, UpdateStatus.RELOADING,
    ):
        raise HTTPException(
            status_code=503,
            detail="Database update in progress",
            headers={"Retry-After": "10"},
        )
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")


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

    tattoo_count = None
    if multi_signal_matcher is not None:
        tattoo_count = len(multi_signal_matcher.performers_with_tattoo_embeddings) or None

    return DatabaseInfo(
        version=db_manifest.get("version", "unknown"),
        performer_count=len(recognizer.performers),
        face_count=len(recognizer.faces),
        sources=db_manifest.get("sources", ["stashdb.org"]),
        created_at=db_manifest.get("created_at"),
        tattoo_embedding_count=tattoo_count,
    )


@app.get("/database/check-update", response_model=CheckUpdateResponse)
async def check_database_update():
    """Check GitHub for a newer database release."""
    if db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    result = await db_updater.check_update()
    return CheckUpdateResponse(**result)


@app.post("/database/update", response_model=StartUpdateResponse)
async def start_database_update():
    """Trigger a database update."""
    if db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    if db_updater._update_task and not db_updater._update_task.done():
        raise HTTPException(status_code=409, detail="Update already in progress")
    check = await db_updater.check_update()
    if not check.get("update_available"):
        raise HTTPException(status_code=400, detail="Already on latest version")
    job_id = await db_updater.start_update(
        download_url=check["download_url"],
        target_version=check["latest_version"],
    )
    return StartUpdateResponse(job_id=job_id, status="started")


@app.get("/database/update/status", response_model=UpdateStatusResponse)
async def get_update_status():
    """Get current status of a database update."""
    if db_updater is None:
        raise HTTPException(status_code=503, detail="Updater not initialized")
    return UpdateStatusResponse(**db_updater.get_status())


@app.post("/identify", response_model=IdentifyResponse)
async def identify_performers(request: IdentifyRequest, _=Depends(require_db_available)):
    """
    Identify performers in an image.

    Provide either `image_url` or `image_base64`. Returns detected faces
    with potential performer matches sorted by confidence.
    """

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
        # Fetch missing images from StashBox
        all_matches = [m for mr in multi_results for m in mr.matches]
        await _fetch_missing_images(all_matches)

        # Convert to response format
        faces = []
        for mr in multi_results:
            bbox = mr.face.bbox
            face_box = FaceBox(
                x=int(bbox["x"]),
                y=int(bbox["y"]),
                width=int(bbox["w"]),
                height=int(bbox["h"]),
                confidence=mr.face.confidence,
            )

            matches = [_match_to_response(m) for m in mr.matches]

            faces.append(FaceResult(box=face_box, matches=matches))

        return IdentifyResponse(faces=faces, face_count=len(faces))

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

    # Fetch missing images from StashBox
    all_matches = [m for r in results for m in r.matches]
    await _fetch_missing_images(all_matches)

    # Convert to response format
    faces = []
    for result in results:
        bbox = result.face.bbox  # dict with x, y, w, h in pixels
        face_box = FaceBox(
            x=int(bbox["x"]),
            y=int(bbox["y"]),
            width=int(bbox["w"]),
            height=int(bbox["h"]),
            confidence=result.face.confidence,
        )

        matches = [_match_to_response(m) for m in result.matches]

        faces.append(FaceResult(box=face_box, matches=matches))

    return IdentifyResponse(faces=faces, face_count=len(faces))


@app.post("/identify/url")
async def identify_from_url(
    url: str = Query(..., description="Image URL to analyze"),
    top_k: int = Query(5, ge=1, le=20),
    max_distance: float = Query(0.6, ge=0.0, le=2.0),
    _=Depends(require_db_available),
):
    """Convenience endpoint to identify from URL via query params."""
    return await identify_performers(
        IdentifyRequest(image_url=url, top_k=top_k, max_distance=max_distance)
    )


@app.post("/identify/image", response_model=IdentifyResponse)
async def identify_image(request: ImageIdentifyRequest, _=Depends(require_db_available)):
    """
    Identify performers in a Stash image by image ID.
    Fetches the image from Stash, runs face recognition, and stores fingerprint.
    """

    base_url = STASH_URL.rstrip("/")
    api_key = STASH_API_KEY

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    # Fetch image info from Stash
    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(base_url, api_key)
    image_data = await stash_client.get_image_by_id(request.image_id)

    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = image_data.get("paths", {}).get("image")
    if not image_url:
        raise HTTPException(status_code=400, detail="Image has no image path")

    # Fetch the image
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"ApiKey": api_key} if api_key else {}
            response = await client.get(image_url, headers=headers)
            response.raise_for_status()
            image = load_image(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    # Use multi-signal matching if available and requested
    if request.use_multi_signal and multi_signal_matcher is not None:
        try:
            multi_results = multi_signal_matcher.identify(
                image,
                top_k=request.top_k,
                use_body=request.use_body,
                use_tattoo=request.use_tattoo,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Recognition failed: {e}")

        # Fetch missing images from StashBox
        all_ms_matches = [m for mr in multi_results for m in mr.matches]
        await _fetch_missing_images(all_ms_matches)

        faces = []
        img_h, img_w = image.shape[:2]
        # Also build results list for fingerprint saving
        results = []
        for mr in multi_results:
            bbox = mr.face.bbox
            face_box = FaceBox(
                x=int(bbox["x"]),
                y=int(bbox["y"]),
                width=int(bbox["w"]),
                height=int(bbox["h"]),
                confidence=mr.face.confidence,
            )

            matches = [_match_to_response(m) for m in mr.matches]

            faces.append(FaceResult(box=face_box, matches=matches))
            # Build RecognitionResult for fingerprint
            results.append(RecognitionResult(face=mr.face, matches=mr.matches))
    else:
        # Run face-only recognition
        try:
            results = recognizer.recognize_image(
                image,
                top_k=request.top_k,
                max_distance=request.max_distance,
                min_face_confidence=request.min_face_confidence,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Recognition failed: {e}")

        # Fetch missing images from StashBox
        all_fo_matches = [m for r in results for m in r.matches]
        await _fetch_missing_images(all_fo_matches)

        faces = []
        img_h, img_w = image.shape[:2]

        for result in results:
            bbox = result.face.bbox
            face_box = FaceBox(
                x=int(bbox["x"]),
                y=int(bbox["y"]),
                width=int(bbox["w"]),
                height=int(bbox["h"]),
                confidence=result.face.confidence,
            )

            matches = [_match_to_response(m) for m in result.matches]

            faces.append(FaceResult(box=face_box, matches=matches))

    # Save fingerprint
    try:
        save_image_fingerprint(
            image_id=request.image_id,
            gallery_id=None,
            faces=results,
            image_shape=(img_h, img_w),
            db_version=db_manifest.get("version"),
        )
    except Exception as e:
        print(f"[identify_image] Failed to save fingerprint: {e}")

    return IdentifyResponse(faces=faces, face_count=len(faces))


@app.post("/identify/gallery", response_model=GalleryIdentifyResponse)
async def identify_gallery(request: GalleryIdentifyRequest, _=Depends(require_db_available)):
    """
    Identify all performers across a gallery.
    Processes each image, aggregates results per-performer, and stores fingerprints.
    """
    import time
    t_start = time.time()

    base_url = STASH_URL.rstrip("/")
    api_key = STASH_API_KEY

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    # Fetch gallery info from Stash
    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(base_url, api_key)
    gallery_data = await stash_client.get_gallery_by_id(request.gallery_id)

    if not gallery_data:
        raise HTTPException(status_code=404, detail="Gallery not found")

    images = gallery_data.get("images", [])
    if not images:
        return GalleryIdentifyResponse(
            gallery_id=request.gallery_id,
            total_images=0,
            images_processed=0,
            faces_detected=0,
            performers=[],
        )

    total_images = len(images)
    print(f"[identify_gallery] === START gallery_id={request.gallery_id}, {total_images} images ===")

    # Process each image
    performer_appearances: dict[str, list[dict]] = defaultdict(list)
    performer_info: dict[str, dict] = {}
    total_faces = 0
    images_processed = 0
    errors = []

    headers = {"ApiKey": api_key} if api_key else {}
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i, img in enumerate(images):
            img_id = img["id"]
            img_url = img.get("paths", {}).get("image")

            if not img_url:
                errors.append(f"Image {img_id} has no URL")
                continue

            try:
                resp = await client.get(img_url, headers=headers)
                resp.raise_for_status()
                image = load_image(resp.content)

                results = recognizer.recognize_image(
                    image,
                    top_k=request.top_k,
                    max_distance=request.max_distance,
                    min_face_confidence=request.min_face_confidence,
                )

                img_h, img_w = image.shape[:2]
                total_faces += len(results)
                images_processed += 1

                # Save per-image fingerprint
                try:
                    save_image_fingerprint(
                        image_id=img_id,
                        gallery_id=request.gallery_id,
                        faces=results,
                        image_shape=(img_h, img_w),
                        db_version=db_manifest.get("version"),
                    )
                except Exception as e:
                    print(f"[identify_gallery] Failed to save fingerprint for image {img_id}: {e}")

                # Collect per-performer data
                for result in results:
                    if result.matches:
                        best = result.matches[0]
                        pid = best.stashdb_id

                        performer_appearances[pid].append({
                            "image_id": img_id,
                            "distance": best.combined_score,
                        })

                        # Keep best info
                        if pid not in performer_info or best.combined_score < performer_info[pid]["distance"]:
                            performer_info[pid] = {
                                "name": best.name,
                                "distance": best.combined_score,
                                "country": best.country,
                                "image_url": best.image_url,
                                "endpoint": _extract_endpoint(best.universal_id),
                            }

                if (i + 1) % 10 == 0:
                    print(f"[identify_gallery] [{time.time()-t_start:.1f}s] Processed {i+1}/{total_images} images")

            except Exception as e:
                errors.append(f"Image {img_id}: {str(e)[:100]}")
                print(f"[identify_gallery] Error processing image {img_id}: {e}")

    # Aggregate results
    performers = []
    for pid, appearances in performer_appearances.items():
        distances = [a["distance"] for a in appearances]
        image_ids = list(set(a["image_id"] for a in appearances))
        image_count = len(image_ids)
        best_distance = min(distances)
        avg_distance = sum(distances) / len(distances)

        # Filter: 2+ appearances OR single match with distance < 0.4
        if image_count < 2 and best_distance >= 0.4:
            continue

        info = performer_info[pid]
        performers.append(GalleryPerformerResult(
            performer_id=pid,
            name=info["name"],
            best_distance=best_distance,
            avg_distance=avg_distance,
            confidence=max(0.0, min(1.0, 1.0 - best_distance)),
            image_count=image_count,
            image_ids=image_ids,
            country=info.get("country"),
            image_url=info.get("image_url"),
            endpoint=info.get("endpoint"),
        ))

    # Sort by image count desc, then best distance asc
    performers.sort(key=lambda p: (-p.image_count, p.best_distance))

    top_names = [p.name for p in performers[:3]]
    print(f"[identify_gallery] [{time.time()-t_start:.1f}s] === DONE === "
          f"{images_processed}/{total_images} images, {total_faces} faces, "
          f"{len(performers)} performers: {', '.join(top_names)}")

    return GalleryIdentifyResponse(
        gallery_id=request.gallery_id,
        total_images=total_images,
        images_processed=images_processed,
        faces_detected=total_faces,
        performers=performers,
        errors=errors[:10],
    )


# Scene identification models
class SceneIdentifyRequest(BaseModel):
    """Request to identify performers in a scene using ffmpeg frame extraction."""
    stash_url: Optional[str] = Field(None, description="Base URL of Stash instance (or use STASH_URL env var)")
    scene_id: str = Field(description="Scene ID in Stash")
    api_key: Optional[str] = Field(None, description="Stash API key (or use STASH_API_KEY env var)")

    # Frame extraction settings (defaults from face_config.py)
    num_frames: int = Field(face_config.NUM_FRAMES, ge=5, le=120, description="Number of frames to extract")
    start_offset_pct: float = Field(face_config.START_OFFSET_PCT, ge=0.0, le=0.5, description="Skip first N% of video")
    end_offset_pct: float = Field(face_config.END_OFFSET_PCT, ge=0.5, le=1.0, description="Stop at N% of video")

    # Face detection settings
    min_face_size: int = Field(face_config.MIN_FACE_SIZE, ge=20, le=200, description="Minimum face size in pixels")
    min_face_confidence: float = Field(face_config.MIN_FACE_CONFIDENCE, ge=0.1, le=1.0, description="Minimum face detection confidence")

    # Matching settings
    top_k: int = Field(face_config.TOP_K, ge=1, le=10, description="Matches per person")
    max_distance: float = Field(face_config.MAX_DISTANCE, ge=0.0, le=2.0, description="Maximum match distance")

    # Clustering settings
    cluster_threshold: float = Field(face_config.CLUSTER_THRESHOLD, ge=0.2, le=3.0, description="Distance threshold for face clustering")

    # Matching mode: "cluster", "frequency", or "hybrid"
    matching_mode: str = Field("frequency", description="Matching mode: 'cluster' (cluster faces then match), 'frequency' (count performer appearances), or 'hybrid' (combine both)")

    # Already-tagged performers (StashDB IDs) for boosting
    scene_performer_stashdb_ids: list[str] = Field(default_factory=list, description="StashDB IDs of performers already tagged on this scene")

    # Multi-signal settings
    use_multi_signal: bool = True
    use_body: bool = True
    use_tattoo: bool = True


class PersonResult(BaseModel):
    """A unique person detected across multiple frames."""
    person_id: int = Field(description="Unique ID for this person in the scene")
    frame_count: int = Field(description="Number of frames this person appeared in")
    best_match: Optional[PerformerMatchResponse] = Field(description="Best performer match")
    all_matches: list[PerformerMatchResponse] = Field(description="All potential matches")
    signals_used: list[str] = Field(default_factory=list, description="Signals used for matching, e.g. ['face', 'body', 'tattoo']")
    tattoos_detected: int = Field(0, description="Number of YOLO tattoo detections in this person's frames")


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
    timing: Optional[dict] = None
    multi_signal_used: bool = False


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
    from body_proportions import BodyProportions
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
            # No tattoos detected — still useful signal (absence)
            tattoo_result = TattooResult(
                detections=[], has_tattoos=False, confidence=0.0,
            )

    return body_ratios, tattoo_result, tattoo_scores, signals_used, tattoos_detected


def _rerank_scene_persons(
    persons: list[PersonResult],
    matcher: "MultiSignalMatcher",
    body_ratios,
    tattoo_result,
    tattoo_scores: dict | None,
    signals_used: list[str],
    tattoos_detected: int,
) -> list[PersonResult]:
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
        aggregated.append(_match_to_response(
            match,
            confidence=distance_to_confidence(adjusted_score),
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
        min_unique_frames: Minimum unique frames a performer must appear in
        max_distance: Only count matches below this distance
        min_confidence: Minimum confidence threshold (filters low-quality matches)

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
            confidence=distance_to_confidence(p["avg_distance"]),
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
) -> list[PersonResult]:
    """
    Cluster faces first to determine person count, then use frequency matching
    within each cluster to identify who each person is.

    This combines the strengths of both approaches:
    - Clustering answers "how many distinct people are there?"
    - Frequency matching answers "who is each person?"

    The result is one PersonResult per face cluster, with alternative matches
    for each cluster shown as all_matches.
    """
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
        def _to_match_response(c: dict) -> PerformerMatchResponse:
            m = c["best_match"]
            return _match_to_response(
                m,
                confidence=distance_to_confidence(c["avg_distance"]),
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
    # Get frequency results (as a dict for lookup)
    freq_persons = frequency_based_matching(
        all_results,
        top_k=top_k * 3,
        min_appearances=min_appearances,
        min_unique_frames=min_unique_frames,
        max_distance=max_distance,
        min_confidence=min_confidence,
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
            confidence=distance_to_confidence(p["distance"]),
            distance=p["distance"],
        )
        persons.append(PersonResult(
            person_id=i,
            frame_count=p["frame_count"],
            best_match=resp,
            all_matches=[resp],
        ))

    return persons


@app.post("/identify/scene", response_model=SceneIdentifyResponse)
async def identify_scene(request: SceneIdentifyRequest, _=Depends(require_db_available)):
    """
    Identify all performers in a scene using ffmpeg frame extraction.

    Extracts full-resolution frames from the video stream using ffmpeg,
    detects faces, clusters them by person, and returns matches.
    """
    import time
    t_start = time.time()

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

    # Resolve num_frames: use settings value when caller didn't override
    num_frames = request.num_frames
    try:
        from settings import get_setting
        settings_num_frames = int(get_setting("num_frames"))
        # If request used the Pydantic default, prefer settings value
        if num_frames == face_config.NUM_FRAMES:
            num_frames = settings_num_frames
    except (RuntimeError, KeyError):
        pass

    # Resolve frame extraction concurrency from settings
    max_concurrent = 8
    try:
        from settings import get_setting
        max_concurrent = int(get_setting("frame_extraction_concurrency"))
    except (RuntimeError, KeyError):
        pass

    # Configure frame extraction
    config = FrameExtractionConfig(
        num_frames=num_frames,
        start_offset_pct=request.start_offset_pct,
        end_offset_pct=request.end_offset_pct,
        min_face_size=request.min_face_size,
        min_face_confidence=request.min_face_confidence,
        max_concurrent_extractions=max_concurrent,
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
        facenet_weight=face_config.FACENET_WEIGHT,
        arcface_weight=face_config.ARCFACE_WEIGHT,
        max_results=request.top_k * 2,
        max_distance=request.max_distance,
    )

    # Phase 1: Detect all faces from all frames
    detected_faces: list[tuple[int, "DetectedFace"]] = []  # (frame_index, face)
    total_faces = 0
    t_detect_total = 0.0

    t_face_loop = time.time()
    for frame in extraction_result.frames:
        t_det = time.time()
        faces = recognizer.generator.detect_faces(
            frame.image,
            min_confidence=request.min_face_confidence,
        )
        t_detect_total += time.time() - t_det

        for face in faces:
            total_faces += 1
            if face.bbox["w"] >= request.min_face_size and face.bbox["h"] >= request.min_face_size:
                detected_faces.append((frame.frame_index, face))

    filtered_faces = len(detected_faces)
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Detection: {t_detect_total:.1f}s | {total_faces} detected, {filtered_faces} after filter")

    # Phase 2: Batch generate embeddings for ALL faces (2 model calls total)
    t_embed = time.time()
    if detected_faces:
        face_images = [face.image for _, face in detected_faces]
        embeddings = recognizer.generator.get_embeddings_batch(face_images)
    else:
        embeddings = []
    t_embed_total = time.time() - t_embed
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Batch embedding: {t_embed_total:.1f}s for {len(embeddings)} faces ({t_embed_total*1000/max(1,len(embeddings)):.1f}ms/face)")

    # Phase 3: Match each face against the index using pre-computed embeddings
    all_results: list[tuple[int, RecognitionResult]] = []
    t_recognize_total = 0.0

    for (frame_idx, face), embedding in zip(detected_faces, embeddings):
        t_rec = time.time()
        matches, _match_result, _ = recognizer.recognize_face_v2(face, match_config, embedding=embedding)
        t_recognize_total += time.time() - t_rec

        result = RecognitionResult(face=face, matches=matches, embedding=embedding)
        all_results.append((frame_idx, result))

    t_face_loop_total = time.time() - t_face_loop
    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Matching: {t_recognize_total:.1f}s | Total face pipeline: {t_face_loop_total:.1f}s")

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
                    # Filter and batch-embed screenshot faces
                    ss_faces = [f for f in screenshot_detected
                                if f.bbox["w"] >= request.min_face_size and f.bbox["h"] >= request.min_face_size]
                    if ss_faces:
                        ss_embeddings = recognizer.generator.get_embeddings_batch([f.image for f in ss_faces])
                        for face, emb in zip(ss_faces, ss_embeddings):
                            matches, _, _ = recognizer.recognize_face_v2(face, match_config, embedding=emb)
                            result = RecognitionResult(face=face, matches=matches, embedding=emb)
                            all_results.append((-1, result))
                            screenshot_faces += 1
                            top_match = matches[0].name if matches else "no match"
                            print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot face: {face.bbox['w']}x{face.bbox['h']}px -> {top_match}")
                    print(f"[identify_scene] [{time.time()-t_start:.1f}s] Screenshot ({img_w}x{img_h}): {len(screenshot_detected)} faces, {screenshot_faces} usable")
        except Exception as e:
            print(f"[identify_scene] Screenshot processing failed: {e}")

    # Fetch missing images from StashBox for all detected matches
    scene_all_matches = [m for _, r in all_results for m in r.matches]
    await _fetch_missing_images(scene_all_matches)

    # Extract multi-signal data (body + tattoo) from representative frames
    scene_body_ratios = None
    scene_tattoo_result = None
    scene_tattoo_scores = None
    scene_signals_used = ["face"]
    scene_tattoos_detected = 0
    ms_used = False

    if (request.use_multi_signal and multi_signal_matcher is not None
            and (request.use_body or request.use_tattoo) and detected_faces):
        t_ms = time.time()
        scene_body_ratios, scene_tattoo_result, scene_tattoo_scores, scene_signals_used, scene_tattoos_detected = _extract_scene_signals(
            frames=extraction_result.frames,
            detected_faces=detected_faces,
            matcher=multi_signal_matcher,
            use_body=request.use_body,
            use_tattoo=request.use_tattoo,
        )
        ms_used = len(scene_signals_used) > 1
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Multi-signal: signals={scene_signals_used}, tattoos_detected={scene_tattoos_detected} in {time.time()-t_ms:.1f}s")

    # Choose matching mode
    t_match_end = 0.0
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
        # Clustered frequency matching: cluster faces first, then identify within clusters
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Using clustered frequency matching...")
        persons = clustered_frequency_matching(
            all_results,
            recognizer,
            cluster_threshold=request.cluster_threshold,
            top_k=request.top_k,
            max_distance=request.max_distance,
            scene_performer_stashdb_ids=request.scene_performer_stashdb_ids,
        )
        print(f"[identify_scene] [{time.time()-t_start:.1f}s] Clustered frequency matching: {len(persons)} persons in {time.time()-t_match:.1f}s")
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

    # Apply multi-signal re-ranking if signals were extracted
    if ms_used and multi_signal_matcher is not None:
        persons = _rerank_scene_persons(
            persons=persons,
            matcher=multi_signal_matcher,
            body_ratios=scene_body_ratios,
            tattoo_result=scene_tattoo_result,
            tattoo_scores=scene_tattoo_scores,
            signals_used=scene_signals_used,
            tattoos_detected=scene_tattoos_detected,
        )

    t_match_end = time.time()
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

    timing_data = {
        "total_ms": round((time.time() - t_start) * 1000),
        "extraction_ms": round((t_face_loop - t_extract) * 1000),
        "face_loop_ms": round(t_face_loop_total * 1000),
        "detection_ms": round(t_detect_total * 1000),
        "embedding_ms": round(t_embed_total * 1000),
        "recognition_ms": round(t_recognize_total * 1000),
        "matching_ms": round((t_match_end - t_match) * 1000),
    }
    print(f"[identify_scene] Timing: {timing_data}")

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
        timing=timing_data,
        multi_signal_used=ms_used,
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


# ==================== StashBox / Stash Performer Endpoints ====================


class StashBoxPerformerResponse(BaseModel):
    """Performer data from StashBox, mapped to Stash's PerformerCreateInput field names."""
    name: str
    disambiguation: Optional[str] = None
    alias_list: list[str] = []
    gender: Optional[str] = None
    birthdate: Optional[str] = None
    death_date: Optional[str] = None
    ethnicity: Optional[str] = None
    country: Optional[str] = None
    eye_color: Optional[str] = None
    hair_color: Optional[str] = None
    height_cm: Optional[int] = None
    measurements: Optional[str] = None
    fake_tits: Optional[str] = None
    career_length: Optional[str] = None
    tattoos: Optional[str] = None
    piercings: Optional[str] = None
    details: Optional[str] = None
    urls: list[str] = []
    image: Optional[str] = None
    weight: Optional[int] = None
    stash_ids: list[dict] = []


def _map_stashbox_to_stash(performer: dict, endpoint_url: str, stashdb_id: str) -> StashBoxPerformerResponse:
    """Map StashBox performer fields to Stash's PerformerCreateInput field names."""
    # breast_type mapping
    breast_type = performer.get("breast_type")
    fake_tits = None
    if breast_type:
        fake_tits = {"NATURAL": "Natural", "FAKE": "Augmented"}.get(breast_type.upper())

    # measurements: "{band}{cup}-{waist}-{hip}"
    cup = performer.get("cup_size")
    band = performer.get("band_size")
    waist = performer.get("waist_size")
    hip = performer.get("hip_size")
    measurements = None
    if any([cup, band, waist, hip]):
        bust = f"{band or ''}{cup or ''}"
        parts = [bust, str(waist) if waist else "", str(hip) if hip else ""]
        measurements = "-".join(parts).strip("-") or None

    # career_length: "YYYY-YYYY" or "YYYY-"
    start = performer.get("career_start_year")
    end = performer.get("career_end_year")
    career_length = None
    if start:
        career_length = f"{start}-{end}" if end else f"{start}-"

    # tattoos/piercings: "location: description" semicolon-separated
    def _format_body_mods(mods):
        if not mods:
            return None
        parts = []
        for mod in mods:
            loc = mod.get("location", "")
            desc = mod.get("description", "")
            if loc and desc:
                parts.append(f"{loc}: {desc}")
            elif loc:
                parts.append(loc)
            elif desc:
                parts.append(desc)
        return "; ".join(parts) if parts else None

    # urls: extract URL strings
    url_list = [u.get("url") for u in (performer.get("urls") or []) if u.get("url")]

    # image: first image URL
    images = performer.get("images") or []
    image_url = images[0].get("url") if images else None

    return StashBoxPerformerResponse(
        name=performer.get("name", ""),
        disambiguation=performer.get("disambiguation"),
        alias_list=performer.get("aliases") or [],
        gender=performer.get("gender"),
        birthdate=performer.get("birth_date"),
        death_date=performer.get("death_date"),
        ethnicity=performer.get("ethnicity"),
        country=performer.get("country"),
        eye_color=performer.get("eye_color"),
        hair_color=performer.get("hair_color"),
        height_cm=performer.get("height"),
        measurements=measurements,
        fake_tits=fake_tits,
        career_length=career_length,
        tattoos=_format_body_mods(performer.get("tattoos")),
        piercings=_format_body_mods(performer.get("piercings")),
        urls=url_list,
        image=image_url,
        stash_ids=[{"endpoint": endpoint_url, "stash_id": stashdb_id}],
    )


@app.get("/stashbox/performer/{endpoint}/{stashdb_id}", response_model=StashBoxPerformerResponse)
async def get_stashbox_performer(endpoint: str, stashdb_id: str):
    """Get performer from StashBox, mapped to Stash field names."""
    client = _get_stashbox_client(endpoint)
    if not client:
        raise HTTPException(status_code=400, detail=f"Unknown or unconfigured endpoint: {endpoint}")

    performer = await client.get_performer(stashdb_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found on StashBox")

    endpoint_url = ENDPOINT_URLS[endpoint]
    return _map_stashbox_to_stash(performer, endpoint_url, stashdb_id)


class SearchPerformersRequest(BaseModel):
    query: str


class SearchPerformerResult(BaseModel):
    id: str
    name: str
    disambiguation: Optional[str] = None
    alias_list: list[str] = []
    image_path: Optional[str] = None


@app.post("/stash/search-performers", response_model=list[SearchPerformerResult])
async def search_stash_performers(request: SearchPerformersRequest):
    """Search performers in local Stash library."""
    if not STASH_URL:
        raise HTTPException(status_code=400, detail="STASH_URL not configured")

    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(STASH_URL, STASH_API_KEY)

    query = """
    query FindPerformers($filter: FindFilterType) {
        findPerformers(filter: $filter) {
            performers {
                id name disambiguation alias_list image_path
            }
        }
    }
    """
    data = await stash_client._execute(query, {
        "filter": {"q": request.query, "per_page": 10, "sort": "name", "direction": "ASC"}
    })
    performers = data.get("findPerformers", {}).get("performers", [])
    return [SearchPerformerResult(**p) for p in performers]


class CreatePerformerRequest(BaseModel):
    scene_id: str
    endpoint: str
    stashdb_id: str


class CreatePerformerResponse(BaseModel):
    performer_id: str
    name: str
    success: bool


@app.post("/stash/create-performer", response_model=CreatePerformerResponse)
async def create_performer_from_stashbox(request: CreatePerformerRequest):
    """Create a performer in Stash from StashBox data, then add to scene."""
    if not STASH_URL:
        raise HTTPException(status_code=400, detail="STASH_URL not configured")

    # 1. Fetch performer data from StashBox
    client = _get_stashbox_client(request.endpoint)
    if not client:
        raise HTTPException(status_code=400, detail=f"Unknown or unconfigured endpoint: {request.endpoint}")

    performer = await client.get_performer(request.stashdb_id)
    if not performer:
        raise HTTPException(status_code=404, detail="Performer not found on StashBox")

    endpoint_url = ENDPOINT_URLS[request.endpoint]
    mapped = _map_stashbox_to_stash(performer, endpoint_url, request.stashdb_id)

    # 2. Create performer in Stash
    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(STASH_URL, STASH_API_KEY)

    create_input = {
        "name": mapped.name,
        "stash_ids": mapped.stash_ids,
    }
    # Add optional fields
    if mapped.disambiguation:
        create_input["disambiguation"] = mapped.disambiguation
    if mapped.alias_list:
        create_input["alias_list"] = mapped.alias_list
    if mapped.gender:
        create_input["gender"] = mapped.gender
    if mapped.birthdate:
        create_input["birthdate"] = mapped.birthdate
    if mapped.death_date:
        create_input["death_date"] = mapped.death_date
    if mapped.ethnicity:
        create_input["ethnicity"] = mapped.ethnicity
    if mapped.country:
        create_input["country"] = mapped.country
    if mapped.eye_color:
        create_input["eye_color"] = mapped.eye_color
    if mapped.hair_color:
        create_input["hair_color"] = mapped.hair_color
    if mapped.height_cm is not None:
        create_input["height_cm"] = mapped.height_cm
    if mapped.measurements:
        create_input["measurements"] = mapped.measurements
    if mapped.fake_tits:
        create_input["fake_tits"] = mapped.fake_tits
    if mapped.career_length:
        create_input["career_length"] = mapped.career_length
    if mapped.tattoos:
        create_input["tattoos"] = mapped.tattoos
    if mapped.piercings:
        create_input["piercings"] = mapped.piercings
    if mapped.image:
        create_input["image"] = mapped.image
    if mapped.urls:
        create_input["url"] = mapped.urls[0]  # Stash only accepts single URL on create

    create_query = """
    mutation PerformerCreate($input: PerformerCreateInput!) {
        performerCreate(input: $input) {
            id name
        }
    }
    """
    from rate_limiter import Priority
    data = await stash_client._execute(create_query, {"input": create_input}, priority=Priority.CRITICAL)
    new_performer = data["performerCreate"]

    # 3. Add performer to scene
    get_query = """
    query GetScene($id: ID!) {
        findScene(id: $id) { performers { id } }
    }
    """
    scene_data = await stash_client._execute(get_query, {"id": request.scene_id})
    current_ids = [p["id"] for p in scene_data["findScene"]["performers"]]
    if new_performer["id"] not in current_ids:
        current_ids.append(new_performer["id"])
        await stash_client.update_scene_performers(request.scene_id, current_ids)

    return CreatePerformerResponse(
        performer_id=new_performer["id"],
        name=new_performer["name"],
        success=True,
    )


class LinkPerformerRequest(BaseModel):
    scene_id: str
    performer_id: str
    stash_ids: list[dict] = []
    update_metadata: bool = False


class LinkPerformerResponse(BaseModel):
    success: bool


@app.post("/stash/link-performer", response_model=LinkPerformerResponse)
async def link_performer_to_scene(request: LinkPerformerRequest):
    """Add an existing Stash performer to a scene. Optionally update stash_ids."""
    if not STASH_URL:
        raise HTTPException(status_code=400, detail="STASH_URL not configured")

    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(STASH_URL, STASH_API_KEY)

    # Add performer to scene
    get_query = """
    query GetScene($id: ID!) {
        findScene(id: $id) { performers { id } }
    }
    """
    scene_data = await stash_client._execute(get_query, {"id": request.scene_id})
    current_ids = [p["id"] for p in scene_data["findScene"]["performers"]]
    if request.performer_id not in current_ids:
        current_ids.append(request.performer_id)
        await stash_client.update_scene_performers(request.scene_id, current_ids)

    # Optionally update performer's stash_ids
    if request.update_metadata and request.stash_ids:
        # Get current stash_ids
        perf_query = """
        query GetPerformer($id: ID!) {
            findPerformer(id: $id) { stash_ids { endpoint stash_id } }
        }
        """
        perf_data = await stash_client._execute(perf_query, {"id": request.performer_id})
        current_stash_ids = perf_data["findPerformer"]["stash_ids"]

        # Merge new stash_ids (avoid duplicates)
        existing = {(s["endpoint"], s["stash_id"]) for s in current_stash_ids}
        merged = list(current_stash_ids)
        for new_sid in request.stash_ids:
            key = (new_sid["endpoint"], new_sid["stash_id"])
            if key not in existing:
                merged.append(new_sid)

        if len(merged) > len(current_stash_ids):
            from rate_limiter import Priority
            await stash_client.update_performer(request.performer_id, stash_ids=merged)

    return LinkPerformerResponse(success=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

"""FastAPI sidecar for Stash Sense.

Provides REST API endpoints for:
- Face recognition (identify performers in images)
- Recommendations engine (library analysis and curation)
"""
import json
import logging
import os

from dotenv import load_dotenv
load_dotenv()

# Environment variables
STASH_URL = os.environ.get("STASH_URL", "").rstrip("/")
STASH_API_KEY = os.environ.get("STASH_API_KEY", "")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import DatabaseConfig, MultiSignalConfig
from recognizer import FaceRecognizer
from body_proportions import BodyProportionExtractor
from tattoo_detector import TattooDetector
from multi_signal_matcher import MultiSignalMatcher
from frame_extractor import check_ffmpeg_available
from recommendations_router import router as recommendations_router, init_recommendations, set_db_version
from identification_router import router as identification_router, init_identification_router, update_identification_globals
from stashbox_router import router as stashbox_router, init_stashbox_router
from database_updater import DatabaseUpdater, UpdateStatus
from hardware import init_hardware
from settings import init_settings, migrate_env_vars
from settings_router import router as settings_router, init_settings_router
from queue_router import router as queue_router

logger = logging.getLogger(__name__)


# Pydantic models for database/health endpoints
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

    # Update identification router globals after hot-swap
    update_identification_globals(
        recognizer=recognizer,
        multi_signal_matcher=multi_signal_matcher,
        db_manifest=db_manifest,
    )

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

    # Initialize identification router with runtime dependencies
    init_identification_router(
        recognizer=recognizer,
        multi_signal_matcher=multi_signal_matcher,
        db_manifest=db_manifest,
        db_updater=db_updater,
        stash_url=STASH_URL,
        stash_api_key=STASH_API_KEY,
    )

    # Initialize stashbox router
    init_stashbox_router(
        stash_url=STASH_URL,
        stash_api_key=STASH_API_KEY,
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
app.include_router(identification_router)
app.include_router(stashbox_router)
app.include_router(recommendations_router)
app.include_router(settings_router)
app.include_router(queue_router)


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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

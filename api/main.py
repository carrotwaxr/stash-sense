"""FastAPI sidecar for Stash Sense.

Provides REST API endpoints for:
- Face recognition (identify performers in images)
- Recommendations engine (library analysis and curation)
"""
import asyncio
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import DatabaseConfig, MultiSignalConfig
from recognizer import FaceRecognizer
from body_proportions import BodyProportionExtractor
from tattoo_detector import TattooDetector
from multi_signal_matcher import MultiSignalMatcher
from recommendations_router import router as recommendations_router, init_recommendations, set_db_version
from identification_router import router as identification_router, init_identification_router, update_identification_globals
from stashbox_router import router as stashbox_router, init_stashbox_router
from database_health_router import router as database_health_router, init_database_health_router, update_database_health_globals
from database_updater import DatabaseUpdater
from hardware import init_hardware
from settings import init_settings, migrate_env_vars
from settings_router import router as settings_router, init_settings_router
from stashbox_connection_manager import init_connection_manager
from queue_router import router as queue_router
from model_manager import init_model_manager
from model_router import router as model_router, init_model_router
from resource_manager import ResourceManager, init_resource_manager, get_resource_manager

logger = logging.getLogger(__name__)


# Database self-updater
db_updater: Optional[DatabaseUpdater] = None

# Resource group name for face recognition resources
FACE_RECOGNITION_RESOURCE = "face_recognition"


def _load_face_recognition(data_dir: Path) -> dict:
    """Loader for face recognition resource group.

    Loads the face recognition database, multi-signal components (body,
    tattoo), and returns them as a dict. This function is called by
    ResourceManager.require() on first access.

    Uses the model manager to resolve ONNX model paths, checking the
    data volume (DATA_DIR/models/) first and falling back to local
    ./models/ for development.

    Args:
        data_dir: Path to the data directory containing DB files.

    Returns:
        Dict with keys: recognizer, multi_signal_matcher, db_manifest.

    Raises:
        Exception: on any failure during loading.
    """
    from model_manager import get_model_manager

    print(f"Loading face database from {data_dir}...")

    db_config = DatabaseConfig(data_dir=data_dir)

    # Load manifest for database info
    db_manifest = {}
    if db_config.manifest_json_path.exists():
        with open(db_config.manifest_json_path) as f:
            db_manifest = json.load(f)

    # Determine models directory from model manager
    mgr = get_model_manager()
    facenet_path = mgr.get_model_path("facenet512")
    arcface_path = mgr.get_model_path("arcface")

    if facenet_path and arcface_path:
        models_dir = facenet_path.parent
    else:
        models_dir = None  # let embeddings.py auto-detect

    recognizer = FaceRecognizer(db_config, models_dir=models_dir)
    print("Face database loaded successfully!")

    # Initialize multi-signal components
    multi_signal_config = MultiSignalConfig.from_settings()

    body_extractor = None
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

    tattoo_detector = None
    tattoo_matcher = None
    if tattoo_enabled:
        # Use model manager paths for tattoo models
        tattoo_det_path = mgr.get_model_path("tattoo_yolov5s")
        tattoo_emb_path = mgr.get_model_path("tattoo_efficientnet_b0")

        print("Initializing tattoo detector...")
        tattoo_detector = TattooDetector(
            model_path=str(tattoo_det_path) if tattoo_det_path else None,
        )

        # Initialize embedding-based matcher if index is available
        if recognizer.tattoo_index is not None and recognizer.tattoo_mapping is not None:
            from tattoo_matcher import TattooMatcher
            tattoo_matcher = TattooMatcher(
                tattoo_index=recognizer.tattoo_index,
                tattoo_mapping=recognizer.tattoo_mapping,
                embedder_model_path=str(tattoo_emb_path) if tattoo_emb_path else None,
            )
            print(f"Tattoo embedding matching ready: {len(recognizer.tattoo_index)} embeddings loaded")

    multi_signal_matcher = None
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

    resources = {
        "recognizer": recognizer,
        "multi_signal_matcher": multi_signal_matcher,
        "db_manifest": db_manifest,
    }

    # Update router globals so endpoints can use the loaded data
    update_identification_globals(
        recognizer=recognizer,
        multi_signal_matcher=multi_signal_matcher,
        db_manifest=db_manifest,
    )
    update_database_health_globals(
        recognizer=recognizer,
        multi_signal_matcher=multi_signal_matcher,
        db_manifest=db_manifest,
    )

    return resources


def _unload_face_recognition(data_dir: Path) -> None:
    """Unloader for face recognition resource group.

    Clears recognizer/matcher globals but preserves manifest so the
    database stats endpoint can still report version and counts while
    the heavy resources are unloaded.
    """
    db_config = DatabaseConfig(data_dir=data_dir)
    manifest = {}
    if db_config.manifest_json_path.exists():
        with open(db_config.manifest_json_path) as f:
            manifest = json.load(f)

    update_identification_globals(
        recognizer=None,
        multi_signal_matcher=None,
        db_manifest=manifest,
    )
    update_database_health_globals(
        recognizer=None,
        multi_signal_matcher=None,
        db_manifest=manifest,
    )


def reload_database(data_dir: Path) -> bool:
    """Reload the face recognition database via ResourceManager.

    Called by DatabaseUpdater after hot-swapping data files. Unloads the
    current face_recognition resource group so the next require() call
    reloads everything from disk.

    Args:
        data_dir: Path to the data directory containing DB files.

    Returns:
        True on success.
    """
    try:
        mgr = get_resource_manager()
        # Unload current resources (clears router globals)
        mgr.unload(FACE_RECOGNITION_RESOURCE)
        # Immediately reload so the hot-swap takes effect
        mgr.require(FACE_RECOGNITION_RESOURCE)
        return True
    except RuntimeError:
        # ResourceManager not initialized yet (shouldn't happen during
        # normal operation, but guard against it)
        logger.warning("ResourceManager not initialized, cannot reload database")
        return False


async def _idle_checker(resource_mgr: ResourceManager, interval: float = 60.0) -> None:
    """Background task that periodically checks for idle resource groups.

    Runs forever until cancelled. Calls resource_mgr.check_idle() every
    *interval* seconds to unload resources that have been idle beyond their
    configured timeout.
    """
    while True:
        await asyncio.sleep(interval)
        resource_mgr.check_idle()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup, clean up on shutdown."""

    data_dir = Path(DATA_DIR)

    global db_updater

    # Detect hardware and classify tier
    hw_profile = init_hardware(str(data_dir))

    # Initialize model manager
    manifest_path = Path(__file__).parent / "models.json"
    models_dir = data_dir / "models"
    model_mgr = init_model_manager(manifest_path, models_dir)
    init_model_router(model_mgr)

    # Load manifest early so database stats are available before lazy load
    db_config = DatabaseConfig(data_dir=data_dir)
    startup_manifest = {}
    if db_config.manifest_json_path.exists():
        with open(db_config.manifest_json_path) as f:
            startup_manifest = json.load(f)
        print(f"Database manifest loaded: v{startup_manifest.get('version')}, "
              f"{startup_manifest.get('performer_count', 0):,} performers, "
              f"{startup_manifest.get('face_count', 0):,} faces")

    # Initialize resource manager for lazy loading of heavy resources
    resource_mgr = init_resource_manager(idle_timeout_seconds=1800.0)
    resource_mgr.register(
        FACE_RECOGNITION_RESOURCE,
        loader=lambda: _load_face_recognition(data_dir),
        unloader=lambda: _unload_face_recognition(data_dir),
    )
    # Face recognition is NOT loaded eagerly — it loads on first /identify request
    print("Face recognition registered for lazy loading (loads on first use)")

    # Initialize database self-updater
    db_updater = DatabaseUpdater(
        data_dir=data_dir,
        reload_fn=reload_database,
    )

    # Initialize identification router with runtime dependencies
    # recognizer and multi_signal_matcher start as None (lazy loaded)
    init_identification_router(
        recognizer=None,
        multi_signal_matcher=None,
        db_manifest=startup_manifest,
        db_updater=db_updater,
        stash_url=STASH_URL,
        stash_api_key=STASH_API_KEY,
    )

    # Initialize stashbox router
    init_stashbox_router(
        stash_url=STASH_URL,
        stash_api_key=STASH_API_KEY,
    )

    # Initialize database health router
    init_database_health_router(
        recognizer=None,
        multi_signal_matcher=None,
        db_manifest=startup_manifest,
        db_updater=db_updater,
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

    # Set DB version from manifest so fingerprint stats are accurate before lazy load
    if startup_manifest.get("version"):
        set_db_version(startup_manifest["version"])

    # Initialize settings system (needs rec_db for persistence)
    from recommendations_router import get_rec_db
    settings_mgr = init_settings(get_rec_db(), hw_profile.tier)
    init_settings_router()

    # Load stash-box endpoint config from Stash
    if STASH_URL:
        try:
            conn_mgr = await init_connection_manager(STASH_URL, STASH_API_KEY)
            logger.warning(
                f"StashBox connections loaded: "
                f"{len(conn_mgr.get_connections())} endpoint(s)"
            )
        except Exception as e:
            logger.warning(f"Failed to load stash-box config from Stash: {e}")
            logger.warning("StashBox features will not work until config is available")

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

    # Start background idle checker for resource manager
    idle_task = asyncio.create_task(_idle_checker(resource_mgr))

    if STASH_URL:
        print(f"Stash connection configured: {STASH_URL}")
    else:
        print("Warning: STASH_URL not set - recommendations analysis will not work")

    yield

    # Graceful shutdown — cancel idle checker
    idle_task.cancel()
    try:
        await idle_task
    except asyncio.CancelledError:
        pass

    # Graceful shutdown — wait for jobs to checkpoint
    await queue_mgr.stop(timeout=30.0)
    logger.warning("Queue manager stopped")

    # Unload all managed resources
    resource_mgr.unload_all()


app = FastAPI(
    title="Stash Sense API",
    description="Face recognition and recommendations engine for Stash",
    version="0.1.0-beta.8",
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
app.include_router(database_health_router)
app.include_router(recommendations_router)
app.include_router(settings_router)
app.include_router(queue_router)
app.include_router(model_router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

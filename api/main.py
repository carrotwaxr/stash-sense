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
from stashbox_utils import _get_stashbox_client, ENDPOINT_URLS
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
            await stash_client.update_performer(request.performer_id, stash_ids=merged)

    return LinkPerformerResponse(success=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

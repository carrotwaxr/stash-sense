"""Configuration for the face recognition database builder."""
import os
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class StashConfig:
    """Local Stash instance configuration."""
    url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "StashConfig":
        return cls(
            url=os.environ.get("STASH_URL", "http://localhost:9999"),
            api_key=os.environ.get("STASH_API_KEY", ""),
        )


@dataclass
class StashDBConfig:
    """StashDB API configuration."""
    url: str
    api_key: str
    rate_limit_delay: float = 0.5  # Seconds between requests - EASILY CONFIGURABLE

    @classmethod
    def from_env(cls) -> "StashDBConfig":
        return cls(
            url=os.environ.get("STASHDB_URL", "https://stashdb.org/graphql"),
            api_key=os.environ.get("STASHDB_API_KEY", ""),
            rate_limit_delay=float(os.environ.get("STASHDB_RATE_LIMIT", "0.5")),
        )


@dataclass
class BuilderConfig:
    """Configuration for database building."""
    # Processing limits
    max_images_per_performer: int = 10
    max_performers: int = None  # None = no limit
    batch_size: int = 100
    completeness_threshold: int = 5

    # Quality filters
    min_face_confidence: float = 0.8  # RetinaFace detection confidence threshold
    min_face_size: int = 50  # Minimum face width/height in pixels

    # Output
    version: str = None  # Auto-generated if not specified

    def __post_init__(self):
        if self.version is None:
            self.version = datetime.now().strftime("%Y.%m.%d")


@dataclass
class DatabaseConfig:
    """Configuration for the face recognition database files."""
    data_dir: Path

    # Index files
    facenet_index_path: Path = None
    arcface_index_path: Path = None
    adaface_index_path: Path = None

    # Tattoo embedding index
    tattoo_index_path: Path = None
    tattoo_json_path: Path = None

    # Metadata files (SQLite is primary, JSON kept for compatibility)
    sqlite_db_path: Path = None
    faces_json_path: Path = None
    performers_json_path: Path = None
    manifest_json_path: Path = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.facenet_index_path = self.facenet_index_path or self.data_dir / "face_facenet.voy"
        self.arcface_index_path = self.arcface_index_path or self.data_dir / "face_arcface.voy"
        self.adaface_index_path = self.adaface_index_path or self.data_dir / "face_adaface.voy"
        self.tattoo_index_path = self.tattoo_index_path or self.data_dir / "tattoo_embeddings.voy"
        self.tattoo_json_path = self.tattoo_json_path or self.data_dir / "tattoo_embeddings.json"
        self.sqlite_db_path = self.sqlite_db_path or self.data_dir / "performers.db"
        self.faces_json_path = self.faces_json_path or self.data_dir / "faces.json"
        self.performers_json_path = self.performers_json_path or self.data_dir / "performers.json"
        self.manifest_json_path = self.manifest_json_path or self.data_dir / "manifest.json"


@dataclass
class MultiSignalConfig:
    """Configuration for multi-signal identification."""
    enable_body: bool = True
    enable_tattoo: str = "auto"  # "true", "false", or "auto" (auto-detect based on index)
    face_candidates: int = 20

    @classmethod
    def from_env(cls) -> "MultiSignalConfig":
        return cls(
            enable_body=os.environ.get("ENABLE_BODY_SIGNAL", "true").lower() == "true",
            enable_tattoo=os.environ.get("ENABLE_TATTOO_SIGNAL", "auto").lower(),
            face_candidates=int(os.environ.get("FACE_CANDIDATES", "20")),
        )

    @classmethod
    def from_settings(cls) -> "MultiSignalConfig":
        """Create config from the settings system. Falls back to from_env() if settings not ready."""
        try:
            from settings import get_setting
            return cls(
                enable_body=get_setting("body_signal_enabled"),
                enable_tattoo="true" if get_setting("tattoo_signal_enabled") else "false",
                face_candidates=get_setting("face_candidates"),
            )
        except RuntimeError:
            return cls.from_env()


# Embedding dimensions
FACENET_DIM = 512
ARCFACE_DIM = 512

# Default thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5

# Stash-box endpoints (for universal ID generation)
STASHBOX_ENDPOINTS = {
    "https://stashdb.org/graphql": "stashdb.org",
    "https://pmvstash.org/graphql": "pmvstash.org",
    "https://fansdb.cc/graphql": "fansdb.cc",
    "https://javstash.org/graphql": "javstash.org",
    "https://theporndb.net/graphql": "theporndb.net",  # Uses REST API, not GraphQL
}


def get_stashbox_shortname(endpoint_url: str) -> str:
    """Convert a stash-box GraphQL URL to a short name for universal IDs."""
    return STASHBOX_ENDPOINTS.get(endpoint_url, endpoint_url.replace("https://", "").replace("/graphql", ""))

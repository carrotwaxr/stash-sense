"""Configuration for the face recognition database builder."""
import os
from dataclasses import dataclass, field
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

    # Quality filters
    min_face_confidence: float = 0.9
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

    # Metadata files
    faces_json_path: Path = None
    performers_json_path: Path = None
    manifest_json_path: Path = None

    # Image cache
    image_cache_dir: Path = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.facenet_index_path = self.facenet_index_path or self.data_dir / "face_facenet.voy"
        self.arcface_index_path = self.arcface_index_path or self.data_dir / "face_arcface.voy"
        self.faces_json_path = self.faces_json_path or self.data_dir / "faces.json"
        self.performers_json_path = self.performers_json_path or self.data_dir / "performers.json"
        self.manifest_json_path = self.manifest_json_path or self.data_dir / "manifest.json"
        self.image_cache_dir = self.image_cache_dir or self.data_dir / "image_cache"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)


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
}


def get_stashbox_shortname(endpoint_url: str) -> str:
    """Convert a stash-box GraphQL URL to a short name for universal IDs."""
    return STASHBOX_ENDPOINTS.get(endpoint_url, endpoint_url.replace("https://", "").replace("/graphql", ""))

"""Configuration for the face recognition database builder."""
import os
from dataclasses import dataclass
from pathlib import Path

@dataclass
class StashConfig:
    url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "StashConfig":
        return cls(
            url=os.environ.get("STASH_URL", "http://10.0.0.4:6969"),
            api_key=os.environ.get("STASH_API_KEY", ""),
        )

@dataclass
class StashDBConfig:
    url: str
    api_key: str

    @classmethod
    def from_env(cls) -> "StashDBConfig":
        return cls(
            url=os.environ.get("STASHDB_URL", "https://stashdb.org/graphql"),
            api_key=os.environ.get("STASHDB_API_KEY", ""),
        )

@dataclass
class DatabaseConfig:
    """Configuration for the face recognition database."""
    data_dir: Path

    # Index files
    facenet_index_path: Path = None
    arcface_index_path: Path = None

    # Metadata files
    faces_json_path: Path = None
    performers_json_path: Path = None

    # Image cache
    image_cache_dir: Path = None

    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.facenet_index_path = self.facenet_index_path or self.data_dir / "face_facenet.voy"
        self.arcface_index_path = self.arcface_index_path or self.data_dir / "face_arc.voy"
        self.faces_json_path = self.faces_json_path or self.data_dir / "faces.json"
        self.performers_json_path = self.performers_json_path or self.data_dir / "performers.json"
        self.image_cache_dir = self.image_cache_dir or self.data_dir / "image_cache"
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)

# Embedding dimensions
FACENET_DIM = 512
ARCFACE_DIM = 512

# Default thresholds
DEFAULT_CONFIDENCE_THRESHOLD = 0.5
DEFAULT_IMAGES_PER_PERFORMER = 15

"""Build the face recognition database from StashDB.

This builder creates a SHAREABLE database keyed by StashDB performer IDs,
not local Stash IDs. This allows the database to work across all Stash installations.

Features:
- Incremental builds: skip already-processed performers
- Resume support: continue from where you left off after interruption
- Auto-save: saves progress periodically to prevent data loss
- Image caching: downloaded images are cached to disk
"""
import json
import hashlib
import signal
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, field
from datetime import datetime, timezone
import numpy as np
from tqdm import tqdm

from voyager import Index, Space, StorageDataType

from config import (
    DatabaseConfig,
    BuilderConfig,
    StashDBConfig,
    FACENET_DIM,
    ARCFACE_DIM,
    get_stashbox_shortname,
)
from stashdb_client import StashDBClient, StashDBPerformer
from embeddings import FaceEmbeddingGenerator, load_image

# Progress tracking schema version
PROGRESS_SCHEMA_VERSION = 2
COMPLETENESS_THRESHOLD = 5


@dataclass
class PerformerProgress:
    """Track sync progress for a single performer."""
    faces_indexed: int = 0
    images_processed: int = 0
    images_available: int = 0
    last_synced: str = ""  # ISO format timestamp

    def is_complete(self) -> bool:
        """Check if performer has enough faces for reliable recognition."""
        return self.faces_indexed >= COMPLETENESS_THRESHOLD

    def needs_recheck(self, current_images_available: int) -> bool:
        """Check if we should try to get more faces for this performer."""
        if self.is_complete():
            return False
        # Re-check if StashDB has more images than we processed
        return current_images_available > self.images_processed

    def to_dict(self) -> dict:
        return {
            "faces_indexed": self.faces_indexed,
            "images_processed": self.images_processed,
            "images_available": self.images_available,
            "last_synced": self.last_synced,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerformerProgress":
        return cls(
            faces_indexed=data.get("faces_indexed", 0),
            images_processed=data.get("images_processed", 0),
            images_available=data.get("images_available", 0),
            last_synced=data.get("last_synced", ""),
        )


@dataclass
class PerformerRecord:
    """Performer record for the shareable database."""
    universal_id: str  # e.g., "stashdb.org:50459d16-..."
    stashdb_id: str
    name: str
    country: Optional[str] = None
    image_url: Optional[str] = None  # Reference image on StashDB
    face_count: int = 0


class DatabaseBuilder:
    """Build face recognition database from StashDB with incremental/resume support."""

    # How often to auto-save (number of performers)
    AUTO_SAVE_INTERVAL = 100

    def __init__(
        self,
        db_config: DatabaseConfig,
        builder_config: BuilderConfig,
        stashdb_client: StashDBClient,
        resume: bool = False,
    ):
        self.db_config = db_config
        self.builder_config = builder_config
        self.stashdb = stashdb_client
        self.generator = FaceEmbeddingGenerator()
        self.resume = resume
        self._interrupted = False

        # Progress tracking file
        self.progress_file = self.db_config.data_dir / "progress.json"

        # Data storage
        self.performers: dict[str, PerformerRecord] = {}  # universal_id -> record
        self.faces: list[str] = []  # index -> universal_id mapping
        self.processed_ids: set[str] = set()  # StashDB IDs already processed
        self.current_face_index = 0

        # Stats
        self.stats = {
            "performers_processed": 0,
            "performers_skipped": 0,
            "performers_with_faces": 0,
            "images_processed": 0,
            "faces_indexed": 0,
            "images_failed": 0,
            "no_face_detected": 0,
        }

        # Initialize or load indices
        if resume and self._has_existing_data():
            self._load_existing_data()
        else:
            self._init_new_indices()

        # Set up signal handler for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C gracefully."""
        print("\n\n⚠️  Interrupt received! Saving progress before exit...")
        self._interrupted = True

    def _has_existing_data(self) -> bool:
        """Check if there's existing data to resume from."""
        return (
            self.db_config.facenet_index_path.exists() and
            self.db_config.faces_json_path.exists() and
            self.db_config.performers_json_path.exists()
        )

    def _init_new_indices(self):
        """Initialize fresh Voyager indices."""
        print("Initializing new database...")
        self.facenet_index = Index(
            Space.Cosine,
            num_dimensions=FACENET_DIM,
            storage_data_type=StorageDataType.E4M3,
        )
        self.arcface_index = Index(
            Space.Cosine,
            num_dimensions=ARCFACE_DIM,
            storage_data_type=StorageDataType.E4M3,
        )

    def _load_existing_data(self):
        """Load existing database for incremental updates."""
        print("Loading existing database for incremental update...")

        # Load Voyager indices
        print(f"  Loading FaceNet index from {self.db_config.facenet_index_path}")
        with open(self.db_config.facenet_index_path, "rb") as f:
            self.facenet_index = Index.load(f)

        print(f"  Loading ArcFace index from {self.db_config.arcface_index_path}")
        with open(self.db_config.arcface_index_path, "rb") as f:
            self.arcface_index = Index.load(f)

        # Load faces mapping
        print(f"  Loading faces mapping from {self.db_config.faces_json_path}")
        with open(self.db_config.faces_json_path) as f:
            self.faces = json.load(f)
        self.current_face_index = len(self.faces)

        # Load performers
        print(f"  Loading performers from {self.db_config.performers_json_path}")
        with open(self.db_config.performers_json_path) as f:
            performers_data = json.load(f)

        for uid, data in performers_data.items():
            # Extract stashdb_id from universal_id
            stashdb_id = uid.split(":", 1)[1] if ":" in uid else uid
            self.performers[uid] = PerformerRecord(
                universal_id=uid,
                stashdb_id=stashdb_id,
                name=data["name"],
                country=data.get("country"),
                image_url=data.get("image_url"),
                face_count=data.get("face_count", 0),
            )
            self.processed_ids.add(stashdb_id)

        # Load progress file if exists
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                progress = json.load(f)
                # Add any additional processed IDs from progress file
                self.processed_ids.update(progress.get("processed_ids", []))
                self.stats = progress.get("stats", self.stats)

        print(f"  Loaded {len(self.performers)} performers, {len(self.faces)} faces")
        print(f"  {len(self.processed_ids)} performers already processed (will be skipped)")

    def _get_image_cache_path(self, url: str) -> Path:
        """Get cache path for an image URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.db_config.image_cache_dir / f"{url_hash}.jpg"

    def _download_image(self, url: str) -> Optional[bytes]:
        """Download image with caching."""
        cache_path = self._get_image_cache_path(url)

        # Check cache
        if cache_path.exists():
            return cache_path.read_bytes()

        # Download from StashDB
        data = self.stashdb.download_image(url)

        # Cache if successful
        if data:
            cache_path.write_bytes(data)

        return data

    def _process_image(self, image_data: bytes, record: PerformerRecord) -> bool:
        """
        Process an image and add face embedding to the index.

        Returns: True if a face was successfully indexed
        """
        try:
            image = load_image(image_data)
        except Exception as e:
            print(f"  Failed to load image for {record.name}: {e}")
            self.stats["images_failed"] += 1
            return False

        # Detect faces
        faces = self.generator.detect_faces(image)
        if not faces:
            self.stats["no_face_detected"] += 1
            return False

        # Filter by confidence and size
        valid_faces = [
            f for f in faces
            if f.confidence >= self.builder_config.min_face_confidence
            and f.bbox["w"] >= self.builder_config.min_face_size
            and f.bbox["h"] >= self.builder_config.min_face_size
        ]

        if not valid_faces:
            self.stats["no_face_detected"] += 1
            return False

        # Take the largest face (most prominent, likely the performer)
        valid_faces.sort(key=lambda f: f.bbox["w"] * f.bbox["h"], reverse=True)
        face = valid_faces[0]

        # Generate embedding
        try:
            embedding = self.generator.get_embedding(face.image)
        except Exception as e:
            print(f"  Failed to generate embedding for {record.name}: {e}")
            self.stats["images_failed"] += 1
            return False

        # Add to indices
        self.facenet_index.add_item(embedding.facenet)
        self.arcface_index.add_item(embedding.arcface)

        # Update mappings
        self.faces.append(record.universal_id)
        record.face_count += 1
        self.current_face_index += 1
        self.stats["faces_indexed"] += 1

        return True

    def _save_progress(self):
        """Save progress to allow resuming."""
        progress = {
            "processed_ids": list(self.processed_ids),
            "stats": self.stats,
            "last_save": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.progress_file, "w") as f:
            json.dump(progress, f)

    def build_from_stashdb(
        self,
        performer_ids: list[str] = None,
    ) -> dict:
        """
        Build database from StashDB performers.

        Args:
            performer_ids: Optional list of specific StashDB performer IDs to process.
                          If None, queries all performers from StashDB.

        Returns: Stats dictionary
        """
        stashbox_name = get_stashbox_shortname(self.stashdb.url)
        print(f"\nBuilding database from {stashbox_name}...")
        print(f"  Rate limit delay: {self.stashdb.rate_limit_delay}s")
        print(f"  Max images per performer: {self.builder_config.max_images_per_performer}")
        print(f"  Auto-save interval: every {self.AUTO_SAVE_INTERVAL} performers")
        if self.resume:
            print(f"  Resume mode: ON (skipping {len(self.processed_ids)} already processed)")

        if performer_ids:
            # Process specific performers
            performers_to_process = []
            for pid in performer_ids:
                if pid in self.processed_ids:
                    self.stats["performers_skipped"] += 1
                    continue
                performer = self.stashdb.get_performer(pid)
                if performer:
                    performers_to_process.append(performer)
            total = len(performers_to_process)
        else:
            # Query all performers from StashDB
            total_count, _ = self.stashdb.query_performers(per_page=1)
            total = min(total_count, self.builder_config.max_performers or total_count)
            print(f"  Total performers in StashDB: {total_count}")
            print(f"  Target to process: {total}")
            performers_to_process = self.stashdb.iter_all_performers(
                max_performers=self.builder_config.max_performers
            )

        performers_since_save = 0

        for performer in tqdm(performers_to_process, total=total, desc="Processing"):
            # Check for interrupt
            if self._interrupted:
                break

            # Skip already processed
            if performer.id in self.processed_ids:
                self.stats["performers_skipped"] += 1
                continue

            self._process_performer(performer, stashbox_name)
            self.processed_ids.add(performer.id)
            performers_since_save += 1

            # Auto-save periodically
            if performers_since_save >= self.AUTO_SAVE_INTERVAL:
                tqdm.write(f"  Auto-saving progress ({len(self.processed_ids)} performers, {len(self.faces)} faces)...")
                self.save()
                self._save_progress()
                performers_since_save = 0

        # Print summary
        if self._interrupted:
            print(f"\n⚠️  Build interrupted!")
        else:
            print(f"\n✅ Build complete!")

        print(f"  Performers processed: {self.stats['performers_processed']}")
        print(f"  Performers skipped (already done): {self.stats['performers_skipped']}")
        print(f"  Performers with faces: {self.stats['performers_with_faces']}")
        print(f"  Total faces indexed: {self.stats['faces_indexed']}")
        print(f"  Images failed: {self.stats['images_failed']}")

        return self.stats

    def _process_performer(self, performer: StashDBPerformer, stashbox_name: str):
        """Process a single performer."""
        self.stats["performers_processed"] += 1

        # Skip performers with no images
        if not performer.image_urls:
            return

        # Create universal ID
        universal_id = f"{stashbox_name}:{performer.id}"

        # Create record
        record = PerformerRecord(
            universal_id=universal_id,
            stashdb_id=performer.id,
            name=performer.name,
            country=performer.country,
            image_url=performer.image_urls[0] if performer.image_urls else None,
        )

        # Process images (up to max)
        images_processed = 0
        for url in performer.image_urls:
            if images_processed >= self.builder_config.max_images_per_performer:
                break

            self.stats["images_processed"] += 1
            image_data = self._download_image(url)
            if image_data:
                if self._process_image(image_data, record):
                    images_processed += 1

        # Only store performer if we indexed at least one face
        if record.face_count > 0:
            self.performers[universal_id] = record
            self.stats["performers_with_faces"] += 1

    def save(self):
        """Save the database to files."""
        print("\nSaving database...")

        # Save Voyager indices
        print(f"  Saving FaceNet index to {self.db_config.facenet_index_path}")
        with open(self.db_config.facenet_index_path, "wb") as f:
            self.facenet_index.save(f)

        print(f"  Saving ArcFace index to {self.db_config.arcface_index_path}")
        with open(self.db_config.arcface_index_path, "wb") as f:
            self.arcface_index.save(f)

        # Save faces.json (index -> universal_id mapping)
        print(f"  Saving faces mapping to {self.db_config.faces_json_path}")
        with open(self.db_config.faces_json_path, "w") as f:
            json.dump(self.faces, f)

        # Save performers.json (universal_id -> metadata)
        # Format per design doc
        print(f"  Saving performers to {self.db_config.performers_json_path}")
        performers_data = {
            uid: {
                "name": record.name,
                "country": record.country,
                "image_url": record.image_url,
                "face_count": record.face_count,
            }
            for uid, record in self.performers.items()
        }
        with open(self.db_config.performers_json_path, "w") as f:
            json.dump(performers_data, f, indent=2)

        # Save progress
        self._save_progress()

        # Generate and save manifest
        self._save_manifest()

        print("Database saved successfully!")

    def _save_manifest(self):
        """Generate and save the manifest file."""
        print(f"  Saving manifest to {self.db_config.manifest_json_path}")

        # Calculate checksums
        checksums = {}
        for path in [
            self.db_config.facenet_index_path,
            self.db_config.arcface_index_path,
            self.db_config.performers_json_path,
            self.db_config.faces_json_path,
        ]:
            if path.exists():
                sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
                checksums[path.name] = f"sha256:{sha256}"

        manifest = {
            "version": self.builder_config.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "performer_count": len(self.performers),
            "face_count": len(self.faces),
            "sources": [get_stashbox_shortname(self.stashdb.url)],
            "models": {
                "detector": "retinaface",
                "facenet_dim": FACENET_DIM,
                "arcface_dim": ARCFACE_DIM,
            },
            "checksums": checksums,
        }

        with open(self.db_config.manifest_json_path, "w") as f:
            json.dump(manifest, f, indent=2)


def main():
    """Main entry point for building the database."""
    import argparse
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Build face recognition database from StashDB",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build full database (will take many hours)
  python database_builder.py --output ./data

  # Build limited sample for testing
  python database_builder.py --max-performers 1000 --output ./data

  # Resume interrupted build
  python database_builder.py --resume --output ./data

  # Faster rate for testing (be nice to StashDB servers!)
  python database_builder.py --rate-limit 0.3 --max-performers 100

Environment variables:
  STASHDB_API_KEY     Your StashDB API key (required)
  STASHDB_URL         StashDB GraphQL URL (default: https://stashdb.org/graphql)
  STASHDB_RATE_LIMIT  Default rate limit in seconds (default: 0.5)
        """
    )
    parser.add_argument("--max-performers", type=int, default=None,
                        help="Maximum number of performers to process (default: all)")
    parser.add_argument("--max-images", type=int, default=5,
                        help="Maximum images per performer (default: 5)")
    parser.add_argument("--rate-limit", type=float, default=None,
                        help="Delay between StashDB requests in seconds (default: 0.5)")
    parser.add_argument("--output", type=str, default="./data",
                        help="Output directory for database files (default: ./data)")
    parser.add_argument("--version", type=str, default=None,
                        help="Database version string (default: YYYY.MM.DD)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from previous build (skip already processed performers)")
    args = parser.parse_args()

    # Configure
    stashdb_config = StashDBConfig.from_env()
    if args.rate_limit:
        stashdb_config.rate_limit_delay = args.rate_limit

    db_config = DatabaseConfig(data_dir=Path(args.output))

    builder_config = BuilderConfig(
        max_performers=args.max_performers,
        max_images_per_performer=args.max_images,
        version=args.version,
    )

    # Initialize clients
    stashdb = StashDBClient(
        url=stashdb_config.url,
        api_key=stashdb_config.api_key,
        rate_limit_delay=stashdb_config.rate_limit_delay,
    )

    # Build
    builder = DatabaseBuilder(
        db_config,
        builder_config,
        stashdb,
        resume=args.resume,
    )
    builder.build_from_stashdb()
    builder.save()


if __name__ == "__main__":
    main()

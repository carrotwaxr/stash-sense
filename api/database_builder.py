"""Build the face recognition database from StashDB.

This builder creates a SHAREABLE database keyed by StashDB performer IDs,
not local Stash IDs. This allows the database to work across all Stash installations.

Features:
- Incremental builds: skip already-processed performers
- Resume support: continue from where you left off after interruption
- Auto-save: saves progress periodically to prevent data loss
- Memory-efficient: images downloaded, processed, then garbage collected (no disk caching)
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

    def is_complete(self, threshold: int = COMPLETENESS_THRESHOLD) -> bool:
        """Check if performer has enough faces for reliable recognition."""
        return self.faces_indexed >= threshold

    def needs_recheck(self, current_images_available: int, threshold: int = COMPLETENESS_THRESHOLD) -> bool:
        """Check if we should try to get more faces for this performer."""
        if self.is_complete(threshold):
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
class BatchTimings:
    """Track time spent in each phase during a batch."""
    fetch_performer_ms: list[float] = field(default_factory=list)
    fetch_images_ms: list[float] = field(default_factory=list)
    process_face_ms: list[float] = field(default_factory=list)

    def reset(self):
        self.fetch_performer_ms.clear()
        self.fetch_images_ms.clear()
        self.process_face_ms.clear()

    def summary(self) -> str:
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        def fmt(ms):
            if ms >= 1000:
                return f"{ms/1000:.2f}s"
            return f"{ms:.0f}ms"

        parts = []
        if self.fetch_performer_ms:
            parts.append(f"Fetch metadata: {fmt(avg(self.fetch_performer_ms))}")
        if self.fetch_images_ms:
            parts.append(f"Fetch images: {fmt(avg(self.fetch_images_ms))}")
        if self.process_face_ms:
            parts.append(f"Process face: {fmt(avg(self.process_face_ms))}")

        return " | ".join(parts) if parts else "No timing data"


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
        self.performer_progress: dict[str, PerformerProgress] = {}  # stashdb_id -> progress
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

        self.batch_timings = BatchTimings()

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

        # Load progress file if exists
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                progress = json.load(f)

            schema_version = progress.get("schema_version", 1)

            if schema_version == 1:
                # Migrate from old format
                self.performer_progress = self._migrate_progress_v1_to_v2(progress)
                self.stats = progress.get("stats", self.stats)
            else:
                # Load v2 format
                self.performer_progress = {
                    pid: PerformerProgress.from_dict(data)
                    for pid, data in progress.get("performers", {}).items()
                }
                self.stats = progress.get("stats", self.stats)
        else:
            # No progress file - infer from performers.json
            now = datetime.now(timezone.utc).isoformat()
            for universal_id, record in self.performers.items():
                self.performer_progress[record.stashdb_id] = PerformerProgress(
                    faces_indexed=record.face_count,
                    images_processed=self.builder_config.max_images_per_performer,
                    images_available=self.builder_config.max_images_per_performer,
                    last_synced=now,
                )

        print(f"  Loaded {len(self.performers)} performers, {len(self.faces)} faces")
        print(f"  {len(self.performer_progress)} performers already processed (will be skipped)")

    def _migrate_progress_v1_to_v2(self, old_progress: dict) -> dict[str, PerformerProgress]:
        """
        Migrate from v1 progress (processed_ids list) to v2 (per-performer tracking).

        Uses face_count from performers.json to infer progress.
        """
        print("  Migrating progress from v1 to v2 schema...")
        migrated = {}
        now = datetime.now(timezone.utc).isoformat()

        # Get face counts from performers.json (already loaded in self.performers)
        for universal_id, record in self.performers.items():
            # Extract stashdb_id from universal_id (e.g., "stashdb.org:abc123" -> "abc123")
            stashdb_id = record.stashdb_id

            # We don't know how many images were available, but we know:
            # - faces_indexed = record.face_count
            # - images_processed >= faces_indexed (we processed at least that many)
            # For migration, assume images_processed = max_images_per_performer (5)
            # This is conservative - if they're incomplete, they'll get rechecked
            migrated[stashdb_id] = PerformerProgress(
                faces_indexed=record.face_count,
                images_processed=self.builder_config.max_images_per_performer,
                images_available=self.builder_config.max_images_per_performer,
                last_synced=now,
            )

        # Also add any IDs from old processed_ids that aren't in performers
        # (these are performers we tried but got no faces from)
        for stashdb_id in old_progress.get("processed_ids", []):
            if stashdb_id not in migrated:
                migrated[stashdb_id] = PerformerProgress(
                    faces_indexed=0,
                    images_processed=self.builder_config.max_images_per_performer,
                    images_available=self.builder_config.max_images_per_performer,
                    last_synced=now,
                )

        print(f"  Migrated {len(migrated)} performer progress records")
        return migrated

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

    def _download_and_process_image(self, url: str, record: PerformerRecord) -> bool:
        """
        Download an image, process it for face embedding, and delete it.

        This is the disk-efficient version that doesn't cache images.

        Returns: True if a face was successfully indexed
        """
        import time

        # Download directly (no caching) - with timing
        t0 = time.perf_counter()
        try:
            image_data = self.stashdb.download_image(url)
            if not image_data:
                self.stats["images_failed"] += 1
                return False
        except Exception as e:
            print(f"  Failed to download image for {record.name}: {e}")
            self.stats["images_failed"] += 1
            return False
        self.batch_timings.fetch_images_ms.append((time.perf_counter() - t0) * 1000)

        # Process the image - with timing
        t0 = time.perf_counter()
        result = self._process_image(image_data, record)
        self.batch_timings.process_face_ms.append((time.perf_counter() - t0) * 1000)

        # Image data goes out of scope and is garbage collected
        # No disk storage needed

        return result

    def _save_progress(self):
        """Save progress to allow resuming."""
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "performers": {
                pid: prog.to_dict()
                for pid, prog in self.performer_progress.items()
            },
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
        print(f"  Completeness threshold: {self.builder_config.completeness_threshold} faces")
        if self.resume:
            print(f"  Resume mode: ON (skipping {len(self.performer_progress)} already processed)")

        if performer_ids:
            # Process specific performers
            performers_to_process = []
            for pid in performer_ids:
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

        for performer in tqdm(
            performers_to_process,
            total=total,
            desc="Processing",
            smoothing=0.05,  # Use longer history for stable estimates
            mininterval=1.0,  # Update at most once per second
        ):
            # Check for interrupt
            if self._interrupted:
                break

            # Get current progress for this performer
            progress = self.performer_progress.get(performer.id)
            images_available = len(performer.image_urls)

            if progress is not None:
                # Already seen this performer
                if progress.is_complete(self.builder_config.completeness_threshold):
                    # Complete - skip entirely
                    self.stats["performers_skipped"] += 1
                    continue
                elif not progress.needs_recheck(images_available, self.builder_config.completeness_threshold):
                    # Incomplete but no new images available - skip
                    self.stats["performers_skipped"] += 1
                    continue
                # else: incomplete and has new images - will reprocess below

            # Process this performer (new or incomplete with new images)
            self._process_performer(performer, stashbox_name, progress)
            performers_since_save += 1

            # Auto-save periodically
            if performers_since_save >= self.AUTO_SAVE_INTERVAL:
                # Print batch performance summary
                tqdm.write(f"\n  Batch complete: {self.batch_timings.summary()}")
                tqdm.write(f"  Saving progress ({len(self.performer_progress)} performers, {len(self.faces)} faces)...")

                self.save(quiet=True)
                self._save_progress()
                self.batch_timings.reset()
                performers_since_save = 0

        # Print summary
        if self._interrupted:
            print(f"\n⚠️  Build interrupted!")
        else:
            print(f"\n✅ Build complete!")

        complete_count = sum(1 for p in self.performer_progress.values()
                             if p.is_complete(self.builder_config.completeness_threshold))
        incomplete_count = len(self.performer_progress) - complete_count

        print(f"  Performers processed: {self.stats['performers_processed']}")
        print(f"  Performers skipped (complete): {self.stats['performers_skipped']}")
        print(f"  Performers with faces: {self.stats['performers_with_faces']}")
        print(f"  Total faces indexed: {self.stats['faces_indexed']}")
        print(f"  Complete performers (>={self.builder_config.completeness_threshold} faces): {complete_count}")
        print(f"  Incomplete performers (<{self.builder_config.completeness_threshold} faces): {incomplete_count}")
        print(f"  Images failed: {self.stats['images_failed']}")

        return self.stats

    def _process_performer(
        self,
        performer: StashDBPerformer,
        stashbox_name: str,
        existing_progress: Optional[PerformerProgress] = None,
    ):
        """
        Process a single performer.

        Args:
            performer: Performer data from StashDB
            stashbox_name: Short name for the stash-box (e.g., "stashdb.org")
            existing_progress: Previous progress if this is a re-check, None if new
        """
        self.stats["performers_processed"] += 1
        images_available = len(performer.image_urls)

        # Skip performers with no images
        if not performer.image_urls:
            # Still track that we checked them
            self.performer_progress[performer.id] = PerformerProgress(
                faces_indexed=0,
                images_processed=0,
                images_available=0,
                last_synced=datetime.now(timezone.utc).isoformat(),
            )
            return

        # Create universal ID
        universal_id = f"{stashbox_name}:{performer.id}"

        # Get or create record
        if universal_id in self.performers:
            record = self.performers[universal_id]
        else:
            record = PerformerRecord(
                universal_id=universal_id,
                stashdb_id=performer.id,
                name=performer.name,
                country=performer.country,
                image_url=performer.image_urls[0] if performer.image_urls else None,
            )

        # Determine which images to process
        if existing_progress is not None:
            # Re-check: only process images beyond what we already processed
            images_to_process = performer.image_urls[existing_progress.images_processed:]
            start_count = existing_progress.images_processed
        else:
            # New performer: process up to max
            images_to_process = performer.image_urls[:self.builder_config.max_images_per_performer]
            start_count = 0

        # Process images
        images_processed_this_run = 0
        for url in images_to_process:
            if start_count + images_processed_this_run >= self.builder_config.max_images_per_performer:
                break

            self.stats["images_processed"] += 1
            self._download_and_process_image(url, record)
            images_processed_this_run += 1

        # Update progress tracking
        total_images_processed = start_count + images_processed_this_run
        self.performer_progress[performer.id] = PerformerProgress(
            faces_indexed=record.face_count,
            images_processed=total_images_processed,
            images_available=images_available,
            last_synced=datetime.now(timezone.utc).isoformat(),
        )

        # Store performer if we have at least one face
        if record.face_count > 0:
            self.performers[universal_id] = record
            if existing_progress is None or existing_progress.faces_indexed == 0:
                self.stats["performers_with_faces"] += 1

    def save(self, quiet: bool = False):
        """Save the database to files."""
        if not quiet:
            print("\nSaving database...")

        # Save Voyager indices
        if not quiet:
            print(f"  Saving FaceNet index to {self.db_config.facenet_index_path}")
        with open(self.db_config.facenet_index_path, "wb") as f:
            self.facenet_index.save(f)

        if not quiet:
            print(f"  Saving ArcFace index to {self.db_config.arcface_index_path}")
        with open(self.db_config.arcface_index_path, "wb") as f:
            self.arcface_index.save(f)

        # Save faces.json (index -> universal_id mapping)
        if not quiet:
            print(f"  Saving faces mapping to {self.db_config.faces_json_path}")
        with open(self.db_config.faces_json_path, "w") as f:
            json.dump(self.faces, f)

        # Save performers.json (universal_id -> metadata)
        # Format per design doc
        if not quiet:
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

        if not quiet:
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
    parser.add_argument("--completeness-threshold", type=int, default=5,
                        help="Minimum faces for a performer to be 'complete' (default: 5)")
    parser.add_argument("--sync-updates-only", action="store_true",
                        help="Only process performers updated since last sync (for incremental updates)")
    args = parser.parse_args()

    if args.sync_updates_only:
        print("Warning: --sync-updates-only is not yet fully implemented. Running normal build.")

    # Configure
    stashdb_config = StashDBConfig.from_env()
    if args.rate_limit:
        stashdb_config.rate_limit_delay = args.rate_limit

    db_config = DatabaseConfig(data_dir=Path(args.output))

    builder_config = BuilderConfig(
        max_performers=args.max_performers,
        max_images_per_performer=args.max_images,
        version=args.version,
        completeness_threshold=args.completeness_threshold,
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

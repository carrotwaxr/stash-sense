"""Build the face recognition database from Stash and StashDB."""
import json
import hashlib
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
import numpy as np
from tqdm import tqdm

from voyager import Index, Space, StorageDataType

from config import DatabaseConfig, FACENET_DIM, ARCFACE_DIM
from stash_client import StashClient, Performer
from stashdb_client import StashDBClient
from embeddings import FaceEmbeddingGenerator, load_image


@dataclass
class PerformerRecord:
    """Performer record for the database."""
    stash_id: str
    name: str
    stashdb_id: Optional[str]
    image_url: Optional[str]
    country: Optional[str] = None
    face_indices: list[int] = None  # Indices in the Voyager index

    def __post_init__(self):
        if self.face_indices is None:
            self.face_indices = []


class DatabaseBuilder:
    """Build face recognition database from multiple sources."""

    def __init__(
        self,
        config: DatabaseConfig,
        stash_client: StashClient,
        stashdb_client: Optional[StashDBClient] = None,
    ):
        self.config = config
        self.stash = stash_client
        self.stashdb = stashdb_client
        self.generator = FaceEmbeddingGenerator()

        # Initialize Voyager indices
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

        # Data storage
        self.performers: dict[str, PerformerRecord] = {}  # stash_id -> record
        self.faces: list[str] = []  # index -> stash_id mapping
        self.current_face_index = 0

    def _get_image_cache_path(self, url: str) -> Path:
        """Get cache path for an image URL."""
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.config.image_cache_dir / f"{url_hash}.jpg"

    def _download_image(self, url: str, source: str = "stash") -> Optional[bytes]:
        """Download image with caching."""
        cache_path = self._get_image_cache_path(url)

        # Check cache
        if cache_path.exists():
            return cache_path.read_bytes()

        # Download
        if source == "stash":
            import requests
            try:
                response = requests.get(
                    url,
                    headers={"ApiKey": self.stash.headers["ApiKey"]},
                    timeout=30,
                )
                response.raise_for_status()
                data = response.content
            except Exception as e:
                print(f"Failed to download from Stash: {e}")
                return None
        elif source == "stashdb":
            data = self.stashdb.download_image(url)
        else:
            return None

        # Cache
        if data:
            cache_path.write_bytes(data)

        return data

    def _process_image(self, image_data: bytes, performer: PerformerRecord) -> int:
        """
        Process an image and add face embeddings to the index.

        Returns: Number of faces added
        """
        try:
            image = load_image(image_data)
        except Exception as e:
            print(f"Failed to load image for {performer.name}: {e}")
            return 0

        # Detect faces
        faces = self.generator.detect_faces(image)
        if not faces:
            return 0

        # For performer images, we assume the largest/most prominent face is the performer
        # Sort by face area (w * h) and take the largest
        faces.sort(key=lambda f: f.bbox["w"] * f.bbox["h"], reverse=True)
        face = faces[0]

        # Generate embedding
        try:
            embedding = self.generator.get_embedding(face.image)
        except Exception as e:
            print(f"Failed to generate embedding for {performer.name}: {e}")
            return 0

        # Add to indices
        self.facenet_index.add_item(embedding.facenet)
        self.arcface_index.add_item(embedding.arcface)

        # Update mappings
        self.faces.append(performer.stash_id)
        performer.face_indices.append(self.current_face_index)
        self.current_face_index += 1

        return 1

    def build_from_stash(
        self,
        max_performers: int = None,
        with_stashdb_id: bool = False,
        include_stashdb_images: bool = True,
        max_images_per_performer: int = 5,
    ) -> int:
        """
        Build database from local Stash performers.

        Args:
            max_performers: Limit number of performers (for testing)
            with_stashdb_id: Only include performers with StashDB IDs
            include_stashdb_images: Also fetch images from StashDB
            max_images_per_performer: Max images to process per performer

        Returns: Total number of faces indexed
        """
        print("Fetching performers from Stash...")
        total_count = self.stash.get_performer_count()
        print(f"Total performers in Stash: {total_count}")

        if max_performers:
            print(f"Limiting to {max_performers} performers")

        faces_added = 0
        performers_processed = 0

        # Iterate through performers
        for performer in tqdm(
            self.stash.iter_performers(with_stashdb_id=with_stashdb_id),
            total=min(total_count, max_performers) if max_performers else total_count,
            desc="Processing performers",
        ):
            if max_performers and performers_processed >= max_performers:
                break

            # Create performer record
            record = PerformerRecord(
                stash_id=performer.id,
                name=performer.name,
                stashdb_id=performer.stashdb_id,
                image_url=performer.image_url,
            )

            images_for_performer = 0

            # Process Stash profile image
            if performer.image_url:
                image_data = self._download_image(performer.image_url, "stash")
                if image_data:
                    added = self._process_image(image_data, record)
                    faces_added += added
                    images_for_performer += added

            # Process StashDB images if enabled and performer has StashDB ID
            if (
                include_stashdb_images
                and self.stashdb
                and performer.stashdb_id
                and images_for_performer < max_images_per_performer
            ):
                stashdb_performer = self.stashdb.get_performer(performer.stashdb_id)
                if stashdb_performer:
                    record.country = stashdb_performer.country

                    for url in stashdb_performer.image_urls:
                        if images_for_performer >= max_images_per_performer:
                            break

                        image_data = self._download_image(url, "stashdb")
                        if image_data:
                            added = self._process_image(image_data, record)
                            faces_added += added
                            images_for_performer += added

            # Store performer record
            self.performers[performer.id] = record
            performers_processed += 1

        print(f"\nProcessed {performers_processed} performers, indexed {faces_added} faces")
        return faces_added

    def save(self):
        """Save the database to files."""
        print("Saving database...")

        # Save Voyager indices
        print(f"  Saving FaceNet index to {self.config.facenet_index_path}")
        with open(self.config.facenet_index_path, "wb") as f:
            self.facenet_index.save(f)

        print(f"  Saving ArcFace index to {self.config.arcface_index_path}")
        with open(self.config.arcface_index_path, "wb") as f:
            self.arcface_index.save(f)

        # Save faces.json (index -> stash_id mapping)
        print(f"  Saving faces mapping to {self.config.faces_json_path}")
        with open(self.config.faces_json_path, "w") as f:
            json.dump(self.faces, f)

        # Save performers.json (stash_id -> full record)
        print(f"  Saving performers to {self.config.performers_json_path}")
        performers_data = {
            stash_id: {
                "name": record.name,
                "stashdb_id": record.stashdb_id,
                "image": record.image_url,
                "country": record.country,
            }
            for stash_id, record in self.performers.items()
        }
        with open(self.config.performers_json_path, "w") as f:
            json.dump(performers_data, f, indent=2)

        print("Database saved successfully!")

    def get_stats(self) -> dict:
        """Get database statistics."""
        return {
            "performers": len(self.performers),
            "faces_indexed": len(self.faces),
            "with_stashdb_id": sum(
                1 for p in self.performers.values() if p.stashdb_id
            ),
        }


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize clients
    stash = StashClient(
        url=os.environ["STASH_URL"],
        api_key=os.environ["STASH_API_KEY"],
    )

    stashdb = StashDBClient(
        url=os.environ["STASHDB_URL"],
        api_key=os.environ["STASHDB_API_KEY"],
    )

    # Initialize config
    config = DatabaseConfig(
        data_dir=Path(__file__).parent.parent / "data"
    )

    # Build database
    builder = DatabaseBuilder(config, stash, stashdb)

    # Start with a small test
    print("Building test database with 10 performers...")
    builder.build_from_stash(
        max_performers=10,
        with_stashdb_id=True,
        include_stashdb_images=True,
        max_images_per_performer=3,
    )

    # Save
    builder.save()

    # Print stats
    print("\nDatabase stats:")
    for key, value in builder.get_stats().items():
        print(f"  {key}: {value}")

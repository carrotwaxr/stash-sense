"""Face recognition against the database.

Matches detected faces against the pre-built performer database.
"""
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import numpy as np

from voyager import Index, Space

from config import DatabaseConfig, FACENET_DIM, ARCFACE_DIM
from embeddings import FaceEmbeddingGenerator, DetectedFace, FaceEmbedding


@dataclass
class PerformerMatch:
    """A potential performer match."""
    universal_id: str  # e.g., "stashdb.org:50459d16-..."
    stashdb_id: str  # Just the UUID part
    name: str
    country: Optional[str]
    image_url: Optional[str]
    facenet_distance: float
    arcface_distance: float
    combined_score: float  # Lower is better (average of distances)


@dataclass
class RecognitionResult:
    """Result of face recognition on an image."""
    face: DetectedFace
    matches: list[PerformerMatch]  # Sorted by combined_score (best first)


class FaceRecognizer:
    """Recognize faces against the performer database."""

    def __init__(self, db_config: DatabaseConfig):
        """
        Initialize the recognizer.

        Args:
            db_config: Database configuration with paths to index files
        """
        self.db_config = db_config
        self.generator = FaceEmbeddingGenerator()

        # Load indices
        print(f"Loading FaceNet index from {db_config.facenet_index_path}...")
        with open(db_config.facenet_index_path, "rb") as f:
            self.facenet_index = Index.load(f)

        print(f"Loading ArcFace index from {db_config.arcface_index_path}...")
        with open(db_config.arcface_index_path, "rb") as f:
            self.arcface_index = Index.load(f)

        # Load metadata
        print(f"Loading faces mapping from {db_config.faces_json_path}...")
        with open(db_config.faces_json_path) as f:
            self.faces = json.load(f)  # index -> universal_id

        print(f"Loading performers from {db_config.performers_json_path}...")
        with open(db_config.performers_json_path) as f:
            self.performers = json.load(f)  # universal_id -> metadata

        print(f"Loaded {len(self.faces)} faces, {len(self.performers)} performers")

    def _get_performer_info(self, universal_id: str) -> dict:
        """Get performer info from universal ID."""
        return self.performers.get(universal_id, {})

    def recognize_face(
        self,
        face: DetectedFace,
        top_k: int = 5,
        max_distance: float = 1.0,
    ) -> list[PerformerMatch]:
        """
        Recognize a single detected face.

        Args:
            face: DetectedFace object with cropped face image
            top_k: Number of top matches to return
            max_distance: Maximum distance threshold (cosine distance, 0-2)

        Returns:
            List of PerformerMatch objects, sorted by combined score
        """
        # Generate embedding
        embedding = self.generator.get_embedding(face.image)

        # Query both indices
        facenet_neighbors, facenet_distances = self.facenet_index.query(
            embedding.facenet, k=top_k * 2  # Get more to allow filtering
        )
        arcface_neighbors, arcface_distances = self.arcface_index.query(
            embedding.arcface, k=top_k * 2
        )

        # Combine results - use the intersection or union approach
        # For now, let's average the distances for faces that appear in both
        matches = {}

        for i, (idx, dist) in enumerate(zip(facenet_neighbors, facenet_distances)):
            universal_id = self.faces[idx]
            if universal_id not in matches:
                matches[universal_id] = {"facenet": dist, "arcface": None}
            else:
                matches[universal_id]["facenet"] = dist

        for i, (idx, dist) in enumerate(zip(arcface_neighbors, arcface_distances)):
            universal_id = self.faces[idx]
            if universal_id not in matches:
                matches[universal_id] = {"facenet": None, "arcface": dist}
            else:
                matches[universal_id]["arcface"] = dist

        # Build match objects
        result = []
        for universal_id, distances in matches.items():
            facenet_dist = distances["facenet"]
            arcface_dist = distances["arcface"]

            # Skip if missing either distance (not in top results for both)
            if facenet_dist is None or arcface_dist is None:
                continue

            # Calculate combined score
            combined = (facenet_dist + arcface_dist) / 2

            # Filter by max distance
            if combined > max_distance:
                continue

            # Get performer info
            info = self._get_performer_info(universal_id)
            stashdb_id = universal_id.split(":", 1)[1] if ":" in universal_id else universal_id

            result.append(PerformerMatch(
                universal_id=universal_id,
                stashdb_id=stashdb_id,
                name=info.get("name", "Unknown"),
                country=info.get("country"),
                image_url=info.get("image_url"),
                facenet_distance=facenet_dist,
                arcface_distance=arcface_dist,
                combined_score=combined,
            ))

        # Sort by combined score
        result.sort(key=lambda m: m.combined_score)

        return result[:top_k]

    def recognize_image(
        self,
        image: np.ndarray,
        top_k: int = 5,
        max_distance: float = 1.0,
        min_face_confidence: float = 0.5,
    ) -> list[RecognitionResult]:
        """
        Detect and recognize all faces in an image.

        Args:
            image: RGB image as numpy array
            top_k: Number of top matches per face
            max_distance: Maximum distance threshold
            min_face_confidence: Minimum face detection confidence

        Returns:
            List of RecognitionResult objects, one per detected face
        """
        # Detect faces
        faces = self.generator.detect_faces(image, min_confidence=min_face_confidence)

        # Recognize each face
        results = []
        for face in faces:
            matches = self.recognize_face(face, top_k=top_k, max_distance=max_distance)
            results.append(RecognitionResult(face=face, matches=matches))

        return results


if __name__ == "__main__":
    # Quick test
    import requests
    from embeddings import load_image

    db_config = DatabaseConfig(data_dir=Path("./data"))
    recognizer = FaceRecognizer(db_config)

    # Test with an image
    test_url = "https://stashdb.org/images/b0aef39d-a1d6-4e58-a136-293f02b84921"
    print(f"\nTesting with {test_url}...")

    response = requests.get(test_url)
    image = load_image(response.content)

    results = recognizer.recognize_image(image)
    print(f"\nFound {len(results)} face(s)")

    for i, result in enumerate(results):
        print(f"\nFace {i+1}: confidence={result.face.confidence:.2f}")
        for j, match in enumerate(result.matches[:3]):
            print(f"  {j+1}. {match.name} (score={match.combined_score:.3f})")
            print(f"     StashDB: {match.stashdb_id}")

"""
Voyager index management for face embeddings.

Handles loading, saving, and querying the dual FaceNet512/ArcFace indices.

See: docs/plans/2026-01-29-face-enrichment-integration.md
"""
import logging
from pathlib import Path
from typing import Tuple
import numpy as np
import voyager

from embeddings import FaceEmbedding

logger = logging.getLogger(__name__)


class IndexManager:
    """
    Manages Voyager indices for face embeddings.

    Loads indices at initialization, provides methods for adding
    and querying embeddings, and saves periodically.
    """

    FACENET_FILENAME = "face_facenet512.voy"
    ARCFACE_FILENAME = "face_arcface.voy"

    def __init__(self, data_dir: Path, min_index: int = None):
        """
        Load indices from data directory.

        Args:
            data_dir: Directory containing .voy index files
            min_index: Minimum starting index (e.g., from DB max index + 1).
                       Used to prevent index conflicts after unclean shutdowns
                       where DB committed but index wasn't saved.
        """
        self.data_dir = Path(data_dir)
        self.facenet_path = self.data_dir / self.FACENET_FILENAME
        self.arcface_path = self.data_dir / self.ARCFACE_FILENAME
        self._min_index = min_index

        self._load_indices()

    def _load_indices(self):
        """Load or create indices."""
        if self.facenet_path.exists():
            logger.info(f"Loading FaceNet index from {self.facenet_path}")
            try:
                self.facenet_index = voyager.Index.load(str(self.facenet_path))
            except RuntimeError as e:
                logger.warning(f"Failed to load FaceNet index (corrupted?): {e}")
                logger.info("Creating new FaceNet index")
                self.facenet_index = voyager.Index(
                    voyager.Space.Cosine,
                    num_dimensions=512,
                )
        else:
            logger.info("Creating new FaceNet index")
            self.facenet_index = voyager.Index(
                voyager.Space.Cosine,
                num_dimensions=512,
            )

        if self.arcface_path.exists():
            logger.info(f"Loading ArcFace index from {self.arcface_path}")
            try:
                self.arcface_index = voyager.Index.load(str(self.arcface_path))
            except RuntimeError as e:
                logger.warning(f"Failed to load ArcFace index (corrupted?): {e}")
                logger.info("Creating new ArcFace index")
                self.arcface_index = voyager.Index(
                    voyager.Space.Cosine,
                    num_dimensions=512,
                )
        else:
            logger.info("Creating new ArcFace index")
            self.arcface_index = voyager.Index(
                voyager.Space.Cosine,
                num_dimensions=512,
            )

        # Current index = max of index length and min_index
        # This prevents conflicts when DB committed faces but index wasn't saved
        index_len = len(self.facenet_index)
        if self._min_index is not None and self._min_index > index_len:
            logger.warning(
                f"DB has faces up to index {self._min_index - 1}, but Voyager only has {index_len}. "
                f"Starting from {self._min_index} to avoid conflicts."
            )
            self.current_index = self._min_index
        else:
            self.current_index = index_len
        logger.info(f"Loaded {index_len} existing embeddings, next index: {self.current_index}")

    def add_embedding(self, embedding: FaceEmbedding) -> int:
        """
        Add a face embedding to both indices.

        Args:
            embedding: FaceEmbedding with facenet and arcface vectors

        Returns:
            Index position of the added embedding
        """
        idx = self.current_index

        self.facenet_index.add_item(embedding.facenet)
        self.arcface_index.add_item(embedding.arcface)

        self.current_index += 1
        return idx

    def get_embedding(self, index: int) -> FaceEmbedding:
        """
        Retrieve embedding by index.

        Args:
            index: Position in the index

        Returns:
            FaceEmbedding with both vectors

        Raises:
            IndexError: If index is out of bounds
        """
        if index < 0 or index >= self.current_index:
            raise IndexError(f"Index {index} out of bounds (0-{self.current_index - 1})")

        facenet = self.facenet_index.get_vector(index)
        arcface = self.arcface_index.get_vector(index)

        return FaceEmbedding(
            facenet=np.array(facenet, dtype=np.float32),
            arcface=np.array(arcface, dtype=np.float32),
        )

    def query(
        self,
        embedding: np.ndarray,
        k: int = 10,
        use_arcface: bool = False,
    ) -> Tuple[list[int], list[float]]:
        """
        Query for nearest neighbors.

        Args:
            embedding: Query vector (512-dim)
            k: Number of neighbors to return
            use_arcface: If True, query ArcFace index; else FaceNet

        Returns:
            Tuple of (neighbor_indices, distances)
        """
        index = self.arcface_index if use_arcface else self.facenet_index

        neighbors, distances = index.query(embedding, k=min(k, len(index)))

        return list(neighbors), list(distances)

    def save(self):
        """Save indices to disk."""
        logger.info(f"Saving indices ({self.current_index} embeddings)")
        self.facenet_index.save(str(self.facenet_path))
        self.arcface_index.save(str(self.arcface_path))

"""Tests for Voyager index management."""
import pytest
import numpy as np
from pathlib import Path


class TestIndexManager:
    """Test index loading, saving, and querying."""

    @pytest.fixture
    def data_dir(self, tmp_path):
        """Create temp data directory for indices.

        Note: We don't pre-create indices because Voyager cannot
        load empty indices. The IndexManager will create fresh
        indices when the files don't exist.
        """
        return tmp_path

    @pytest.fixture
    def manager(self, data_dir):
        """Create index manager."""
        from index_manager import IndexManager

        return IndexManager(data_dir)

    def test_loads_existing_indices(self, manager):
        """Manager loads indices from disk."""
        assert manager.facenet_index is not None
        assert manager.arcface_index is not None
        assert manager.current_index == 0

    def test_add_embedding_returns_index(self, manager):
        """Adding embedding returns its index position."""
        from embeddings import FaceEmbedding

        embedding = FaceEmbedding(
            facenet=np.random.rand(512).astype(np.float32),
            arcface=np.random.rand(512).astype(np.float32),
        )

        idx = manager.add_embedding(embedding)

        assert idx == 0
        assert manager.current_index == 1

    def test_add_multiple_embeddings(self, manager):
        """Adding multiple embeddings increments index."""
        from embeddings import FaceEmbedding

        for i in range(5):
            embedding = FaceEmbedding(
                facenet=np.random.rand(512).astype(np.float32),
                arcface=np.random.rand(512).astype(np.float32),
            )
            idx = manager.add_embedding(embedding)
            assert idx == i

        assert manager.current_index == 5

    def test_get_embedding_by_index(self, manager):
        """Can retrieve embedding by index."""
        from embeddings import FaceEmbedding

        # Use pre-normalized vectors (Voyager normalizes for Cosine space)
        facenet_vec = np.random.rand(512).astype(np.float32)
        facenet_vec = facenet_vec / np.linalg.norm(facenet_vec)
        arcface_vec = np.random.rand(512).astype(np.float32)
        arcface_vec = arcface_vec / np.linalg.norm(arcface_vec)

        original = FaceEmbedding(facenet=facenet_vec, arcface=arcface_vec)

        idx = manager.add_embedding(original)
        retrieved = manager.get_embedding(idx)

        np.testing.assert_array_almost_equal(retrieved.facenet, original.facenet)
        np.testing.assert_array_almost_equal(retrieved.arcface, original.arcface)

    def test_save_and_reload(self, data_dir):
        """Indices persist after save and reload."""
        from index_manager import IndexManager
        from embeddings import FaceEmbedding

        # Add embeddings and save
        manager1 = IndexManager(data_dir)

        # Use pre-normalized vectors (Voyager normalizes for Cosine space)
        facenet_vec = np.random.rand(512).astype(np.float32)
        facenet_vec = facenet_vec / np.linalg.norm(facenet_vec)
        arcface_vec = np.random.rand(512).astype(np.float32)
        arcface_vec = arcface_vec / np.linalg.norm(arcface_vec)

        embedding = FaceEmbedding(facenet=facenet_vec, arcface=arcface_vec)
        manager1.add_embedding(embedding)
        manager1.save()

        # Reload and verify
        manager2 = IndexManager(data_dir)
        assert manager2.current_index == 1

        retrieved = manager2.get_embedding(0)
        np.testing.assert_array_almost_equal(retrieved.facenet, embedding.facenet)

    def test_query_nearest(self, manager):
        """Can query for nearest neighbors."""
        from embeddings import FaceEmbedding

        # Add some embeddings
        base = np.random.rand(512).astype(np.float32)
        base = base / np.linalg.norm(base)

        for i in range(10):
            noise = np.random.rand(512).astype(np.float32) * 0.1
            vec = base + noise
            vec = vec / np.linalg.norm(vec)

            embedding = FaceEmbedding(facenet=vec, arcface=vec.copy())
            manager.add_embedding(embedding)

        # Query with similar vector
        query = base + np.random.rand(512).astype(np.float32) * 0.05
        query = query / np.linalg.norm(query)

        neighbors, distances = manager.query(query, k=5)

        assert len(neighbors) == 5
        assert len(distances) == 5
        assert all(d < 0.5 for d in distances)  # Should be close

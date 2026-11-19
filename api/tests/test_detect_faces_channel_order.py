"""detect_faces hands RetinaFace BGR but aligns crops from the RGB input (#147).

InsightFace's detector calls blobFromImage(swapRB=True), so it expects BGR.
The aligned crop, and so the embedding, must not change: only which faces
are detected does.
"""
from types import SimpleNamespace

import numpy as np
import pytest

embeddings = pytest.importorskip("embeddings")
from insightface.utils.face_align import norm_crop  # noqa: E402

KPS = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=np.float32)


class RecordingAnalyzer:
    """Stands in for FaceAnalysis: records its input, returns one face."""

    def __init__(self):
        self.received = None

    def get(self, image):
        self.received = image.copy()
        return [SimpleNamespace(
            det_score=0.99,
            bbox=np.array([30, 35, 95, 105], dtype=np.float32),
            kps=KPS,
        )]


@pytest.fixture
def rgb_image():
    rng = np.random.default_rng(147)
    return rng.integers(0, 256, size=(160, 120, 3), dtype=np.uint8)


@pytest.fixture
def generator():
    gen = embeddings.FaceEmbeddingGenerator.__new__(embeddings.FaceEmbeddingGenerator)
    gen._face_analyzer = RecordingAnalyzer()
    return gen


def test_analyzer_receives_bgr(generator, rgb_image):
    generator.detect_faces(rgb_image)
    np.testing.assert_array_equal(generator._face_analyzer.received, rgb_image[:, :, ::-1])


def test_aligned_crop_stays_rgb(generator, rgb_image):
    faces = generator.detect_faces(rgb_image)
    assert len(faces) == 1
    np.testing.assert_array_equal(faces[0].image, norm_crop(rgb_image, KPS, image_size=112))


def test_input_image_not_mutated(generator, rgb_image):
    before = rgb_image.copy()
    generator.detect_faces(rgb_image)
    np.testing.assert_array_equal(rgb_image, before)

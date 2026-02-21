"""Tests for capability detection."""

import pytest

from capabilities import detect_capabilities


# -- File lists mirroring capabilities.py constants --

FACE_DATA_FILES = [
    "face_facenet.voy",
    "face_arcface.voy",
    "faces.json",
    "performers.json",
]

FACE_MODEL_FILES = [
    "facenet512.onnx",
    "arcface.onnx",
]

TATTOO_MODEL_FILES = [
    "tattoo_yolov5s.onnx",
    "tattoo_efficientnet_b0.onnx",
]

TATTOO_DATA_FILES = [
    "tattoo_embeddings.voy",
]


def _create_files(directory, filenames):
    """Create empty sentinel files in *directory*."""
    for name in filenames:
        (directory / name).write_bytes(b"")


@pytest.fixture
def data_dir(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    return d


@pytest.fixture
def models_dir(tmp_path):
    d = tmp_path / "models"
    d.mkdir()
    return d


class TestFreshInstall:
    """Empty directories -- only lightweight capabilities available."""

    def test_upstream_sync_always_true(self, data_dir, models_dir):
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["upstream_sync"] is True

    def test_duplicate_detection_always_true(self, data_dir, models_dir):
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["duplicate_detection_basic"] is True

    def test_identification_disabled(self, data_dir, models_dir):
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False

    def test_tattoo_signal_disabled(self, data_dir, models_dir):
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["tattoo_signal"] is False


class TestWithFaceDataAndModels:
    """Face data + face models present -- identification enabled."""

    def test_identification_enabled(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is True

    def test_tattoo_still_disabled(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["tattoo_signal"] is False


class TestAllDataAndModels:
    """All files present -- every capability enabled."""

    def test_all_capabilities_enabled(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES + TATTOO_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES + TATTOO_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["upstream_sync"] is True
        assert caps["duplicate_detection_basic"] is True
        assert caps["identification"] is True
        assert caps["tattoo_signal"] is True


class TestFaceModelsButNoData:
    """Models installed but no data files -- identification disabled."""

    def test_identification_disabled(self, data_dir, models_dir):
        _create_files(models_dir, FACE_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False

    def test_tattoo_disabled(self, data_dir, models_dir):
        _create_files(models_dir, FACE_MODEL_FILES + TATTOO_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["tattoo_signal"] is False


class TestFaceDataButNoModels:
    """Data files imported but models not downloaded -- identification disabled."""

    def test_identification_disabled(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False

    def test_tattoo_disabled(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES + TATTOO_DATA_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["tattoo_signal"] is False


class TestPartialFaceData:
    """Some face data files missing -- identification disabled."""

    def test_missing_one_data_file(self, data_dir, models_dir):
        # Create all but one face data file
        _create_files(data_dir, FACE_DATA_FILES[:-1])
        _create_files(models_dir, FACE_MODEL_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False

    def test_missing_one_model_file(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES[:1])
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["identification"] is False


class TestTattooRequiresIdentification:
    """Tattoo signal requires identification as a prerequisite."""

    def test_tattoo_models_without_face_data(self, data_dir, models_dir):
        _create_files(models_dir, FACE_MODEL_FILES + TATTOO_MODEL_FILES)
        _create_files(data_dir, TATTOO_DATA_FILES)
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["tattoo_signal"] is False

    def test_tattoo_missing_embeddings_voy(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES + TATTOO_MODEL_FILES)
        # No tattoo_embeddings.voy
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["tattoo_signal"] is False

    def test_tattoo_missing_one_model(self, data_dir, models_dir):
        _create_files(data_dir, FACE_DATA_FILES + TATTOO_DATA_FILES)
        _create_files(models_dir, FACE_MODEL_FILES + TATTOO_MODEL_FILES[:1])
        caps = detect_capabilities(data_dir, models_dir)
        assert caps["tattoo_signal"] is False

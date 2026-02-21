"""Tests for TattooMatcher module."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from tattoo_matcher import TattooMatcher, _TattooEmbeddingGenerator
from tattoo_detector import TattooDetection


class TestTattooMatcherCrop:
    """Tests for TattooMatcher.crop_tattoo static method."""

    def test_crop_valid_bbox(self):
        """Crop with valid normalized bbox returns correct region."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        image[20:40, 40:80] = 255  # White region

        bbox = {"x": 0.2, "y": 0.2, "w": 0.2, "h": 0.2}
        crop = TattooMatcher.crop_tattoo(image, bbox, padding=0.0)

        assert crop is not None
        assert crop.shape[0] == 20  # h: 100 * 0.2 = 20
        assert crop.shape[1] == 40  # w: 200 * 0.2 = 40

    def test_crop_with_padding(self):
        """Padding expands the crop region."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = {"x": 0.3, "y": 0.3, "w": 0.2, "h": 0.2}

        crop_no_pad = TattooMatcher.crop_tattoo(image, bbox, padding=0.0)
        crop_with_pad = TattooMatcher.crop_tattoo(image, bbox, padding=0.2)

        assert crop_with_pad.shape[0] > crop_no_pad.shape[0]
        assert crop_with_pad.shape[1] > crop_no_pad.shape[1]

    def test_crop_zero_width_returns_none(self):
        """Zero-width bbox returns None."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = {"x": 0.5, "y": 0.5, "w": 0.0, "h": 0.1}

        result = TattooMatcher.crop_tattoo(image, bbox)
        assert result is None

    def test_crop_negative_height_returns_none(self):
        """Negative-height bbox returns None."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = {"x": 0.5, "y": 0.5, "w": 0.1, "h": -0.1}

        result = TattooMatcher.crop_tattoo(image, bbox)
        assert result is None

    def test_crop_clamps_to_image_bounds(self):
        """Crop near image edges clamps to valid bounds."""
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        bbox = {"x": 0.9, "y": 0.9, "w": 0.2, "h": 0.2}

        crop = TattooMatcher.crop_tattoo(image, bbox, padding=0.0)
        assert crop is not None
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0


class TestTattooMatcherMatch:
    """Tests for TattooMatcher.match method."""

    def test_match_empty_detections_returns_empty(self):
        """No detections returns empty dict."""
        mock_index = Mock()
        matcher = TattooMatcher(mock_index, [{"universal_id": "stashdb.org:uuid1"}])

        result = matcher.match(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=[],
        )

        assert result == {}

    def test_match_none_detections_returns_empty(self):
        """None detections returns empty dict."""
        mock_index = Mock()
        matcher = TattooMatcher(mock_index, [{"universal_id": "stashdb.org:uuid1"}])

        result = matcher.match(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=None,
        )

        assert result == {}

    def test_match_aggregates_by_performer(self):
        """Match returns best score per performer."""
        mock_index = Mock()
        # Two neighbors for the same performer
        mock_index.query.return_value = (
            np.array([0, 1]),  # indices
            np.array([0.2, 0.4]),  # distances (cosine)
        )

        mapping = [{"universal_id": "stashdb.org:uuid1"}, {"universal_id": "stashdb.org:uuid1"}]
        matcher = TattooMatcher(mock_index, mapping)

        # Mock the generator
        mock_gen = Mock()
        mock_gen.get_embedding.return_value = np.zeros(1280, dtype=np.float32)
        matcher._generator = mock_gen

        detection = TattooDetection(
            bbox={"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2},
            confidence=0.9,
            location_hint="left arm",
        )

        result = matcher.match(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=[detection],
        )

        assert "stashdb.org:uuid1" in result
        # Best score: 1.0 - 0.2 = 0.8
        assert result["stashdb.org:uuid1"] == pytest.approx(0.8, abs=0.01)

    def test_match_skips_null_mappings(self):
        """Match skips None entries in mapping."""
        mock_index = Mock()
        mock_index.query.return_value = (
            np.array([0, 1]),
            np.array([0.3, 0.3]),
        )

        mapping = [None, {"universal_id": "stashdb.org:uuid1"}]
        matcher = TattooMatcher(mock_index, mapping)

        mock_gen = Mock()
        mock_gen.get_embedding.return_value = np.zeros(1280, dtype=np.float32)
        matcher._generator = mock_gen

        detection = TattooDetection(
            bbox={"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2},
            confidence=0.9,
            location_hint="torso",
        )

        result = matcher.match(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=[detection],
        )

        # Only uuid1 should be present (index 0 is None)
        assert len(result) == 1
        assert "stashdb.org:uuid1" in result

    def test_match_skips_out_of_range_indices(self):
        """Match skips indices beyond mapping size."""
        mock_index = Mock()
        mock_index.query.return_value = (
            np.array([0, 99]),  # 99 is out of range
            np.array([0.3, 0.3]),
        )

        mapping = [{"universal_id": "stashdb.org:uuid1"}]
        matcher = TattooMatcher(mock_index, mapping)

        mock_gen = Mock()
        mock_gen.get_embedding.return_value = np.zeros(1280, dtype=np.float32)
        matcher._generator = mock_gen

        detection = TattooDetection(
            bbox={"x": 0.1, "y": 0.2, "w": 0.2, "h": 0.2},
            confidence=0.9,
            location_hint="torso",
        )

        result = matcher.match(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=[detection],
        )

        assert len(result) == 1
        assert "stashdb.org:uuid1" in result

    def test_match_multiple_detections(self):
        """Multiple detections contribute scores for different performers."""
        mock_index = Mock()
        # First detection matches uuid1, second matches uuid2
        mock_index.query.side_effect = [
            (np.array([0]), np.array([0.2])),
            (np.array([1]), np.array([0.3])),
        ]

        mapping = [{"universal_id": "stashdb.org:uuid1"}, {"universal_id": "stashdb.org:uuid2"}]
        matcher = TattooMatcher(mock_index, mapping)

        mock_gen = Mock()
        mock_gen.get_embedding.return_value = np.zeros(1280, dtype=np.float32)
        matcher._generator = mock_gen

        detections = [
            TattooDetection(
                bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                confidence=0.9, location_hint="left arm",
            ),
            TattooDetection(
                bbox={"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
                confidence=0.8, location_hint="torso",
            ),
        ]

        result = matcher.match(
            np.zeros((100, 100, 3), dtype=np.uint8),
            detections=detections,
            k=1,
        )

        assert "stashdb.org:uuid1" in result
        assert "stashdb.org:uuid2" in result


class TestTattooEmbeddingGeneratorPreprocess:
    """Tests for _TattooEmbeddingGenerator._preprocess method."""

    def test_preprocess_output_shape(self):
        """Preprocessing should produce (1, 3, 224, 224) float32 tensor."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")
        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        result = gen._preprocess(image)

        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

    def test_preprocess_square_image(self):
        """Square image should still produce correct output shape."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")
        image = np.random.randint(0, 256, (300, 300, 3), dtype=np.uint8)

        result = gen._preprocess(image)

        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

    def test_preprocess_small_image(self):
        """Image smaller than 256 gets upscaled correctly."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")
        image = np.random.randint(0, 256, (50, 80, 3), dtype=np.uint8)

        result = gen._preprocess(image)

        assert result.shape == (1, 3, 224, 224)
        assert result.dtype == np.float32

    def test_preprocess_normalization_black_image(self):
        """Black image (all zeros) should normalize to -mean/std per channel."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")
        image = np.zeros((256, 256, 3), dtype=np.uint8)

        result = gen._preprocess(image)

        # Black image: pixel = 0 -> float = 0.0 -> normalized = (0 - mean) / std
        expected_r = -0.485 / 0.229
        expected_g = -0.456 / 0.224
        expected_b = -0.406 / 0.225

        # result shape: (1, 3, 224, 224), channels are R, G, B
        assert result[0, 0, 0, 0] == pytest.approx(expected_r, abs=1e-5)
        assert result[0, 1, 0, 0] == pytest.approx(expected_g, abs=1e-5)
        assert result[0, 2, 0, 0] == pytest.approx(expected_b, abs=1e-5)

    def test_preprocess_normalization_white_image(self):
        """White image (all 255) should normalize to (1 - mean) / std per channel."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")
        image = np.full((256, 256, 3), 255, dtype=np.uint8)

        result = gen._preprocess(image)

        expected_r = (1.0 - 0.485) / 0.229
        expected_g = (1.0 - 0.456) / 0.224
        expected_b = (1.0 - 0.406) / 0.225

        assert result[0, 0, 0, 0] == pytest.approx(expected_r, abs=1e-5)
        assert result[0, 1, 0, 0] == pytest.approx(expected_g, abs=1e-5)
        assert result[0, 2, 0, 0] == pytest.approx(expected_b, abs=1e-5)

    def test_preprocess_tall_image_aspect_ratio(self):
        """Tall image: width is shortest side, scaled to 256."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")
        # 800x200 image -> shortest side is 200 -> scale to 256
        # new_w = 256, new_h = round(800 * 256 / 200) = 1024
        image = np.random.randint(0, 256, (800, 200, 3), dtype=np.uint8)

        result = gen._preprocess(image)
        assert result.shape == (1, 3, 224, 224)

    def test_preprocess_wide_image_aspect_ratio(self):
        """Wide image: height is shortest side, scaled to 256."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")
        # 200x800 image -> shortest side is 200 -> scale to 256
        image = np.random.randint(0, 256, (200, 800, 3), dtype=np.uint8)

        result = gen._preprocess(image)
        assert result.shape == (1, 3, 224, 224)


class TestTattooEmbeddingGeneratorGetEmbedding:
    """Tests for _TattooEmbeddingGenerator.get_embedding with mock ONNX session."""

    def test_get_embedding_returns_normalized_vector(self):
        """get_embedding should return L2-normalized 1280-dim float32 vector."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")

        # Mock ONNX session
        mock_session = MagicMock()
        raw_embedding = np.random.randn(1, 1280).astype(np.float32)
        mock_session.run.return_value = [raw_embedding]
        mock_session.get_inputs.return_value = [MagicMock(name="input")]
        mock_session.get_outputs.return_value = [MagicMock(name="embedding")]

        # Inject mock session directly
        gen._session = mock_session
        gen._input_name = "input"
        gen._output_name = "embedding"

        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        result = gen.get_embedding(image)

        assert result.shape == (1280,)
        assert result.dtype == np.float32
        # L2 norm should be ~1.0
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)

    def test_get_embedding_zero_vector_handled(self):
        """Zero embedding vector should not cause division by zero."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")

        mock_session = MagicMock()
        zero_embedding = np.zeros((1, 1280), dtype=np.float32)
        mock_session.run.return_value = [zero_embedding]

        gen._session = mock_session
        gen._input_name = "input"
        gen._output_name = "embedding"

        image = np.random.randint(0, 256, (300, 400, 3), dtype=np.uint8)
        result = gen.get_embedding(image)

        assert result.shape == (1280,)
        assert result.dtype == np.float32
        # All zeros - norm is 0, so embedding stays zero
        assert np.allclose(result, 0.0)

    def test_get_embedding_calls_session_with_preprocessed_input(self):
        """get_embedding should pass preprocessed (1,3,224,224) input to ONNX session."""
        gen = _TattooEmbeddingGenerator(model_path="/dummy")

        mock_session = MagicMock()
        mock_session.run.return_value = [np.random.randn(1, 1280).astype(np.float32)]

        gen._session = mock_session
        gen._input_name = "input"
        gen._output_name = "embedding"

        image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        gen.get_embedding(image)

        # Verify session.run was called once
        mock_session.run.assert_called_once()
        call_args = mock_session.run.call_args

        # Check output names
        assert call_args[0][0] == ["embedding"]

        # Check input dict has correct key and shape
        input_dict = call_args[0][1]
        assert "input" in input_dict
        assert input_dict["input"].shape == (1, 3, 224, 224)
        assert input_dict["input"].dtype == np.float32


class TestTattooEmbeddingGeneratorModelPath:
    """Tests for _TattooEmbeddingGenerator model path resolution."""

    def test_explicit_model_path_not_found_raises(self):
        """Providing a non-existent model path should raise FileNotFoundError."""
        gen = _TattooEmbeddingGenerator(model_path="/nonexistent/model.onnx")

        with pytest.raises(FileNotFoundError, match="Tattoo embedder model not found"):
            _ = gen.session

    def test_default_path_search_data_dir_first(self, tmp_path):
        """DATA_DIR/models should take priority over local models dir."""
        # Create model file in DATA_DIR/models
        data_models = tmp_path / "data" / "models"
        data_models.mkdir(parents=True)
        model_file = data_models / "tattoo_efficientnet_b0.onnx"
        model_file.write_bytes(b"fake-onnx")

        gen = _TattooEmbeddingGenerator()

        with patch("tattoo_matcher.DATA_DIR", str(tmp_path / "data")):
            with patch("tattoo_matcher.ort.InferenceSession") as mock_ort:
                with patch("tattoo_matcher.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
                    mock_session = MagicMock()
                    mock_session.get_inputs.return_value = [MagicMock(name="input")]
                    mock_session.get_outputs.return_value = [MagicMock(name="embedding")]
                    mock_ort.return_value = mock_session

                    _ = gen.session

                    mock_ort.assert_called_once_with(
                        str(model_file), providers=["CPUExecutionProvider"],
                    )

    def test_default_path_falls_back_to_local_models(self, tmp_path):
        """Should fall back to local models dir when DATA_DIR doesn't have the model."""
        local_model = tmp_path / "tattoo_efficientnet_b0.onnx"
        local_model.write_bytes(b"fake-onnx")

        gen = _TattooEmbeddingGenerator()

        with patch("tattoo_matcher.DATA_DIR", "/nonexistent/data"):
            with patch("tattoo_matcher.LOCAL_MODELS_DIR", tmp_path):
                with patch("tattoo_matcher.ort.InferenceSession") as mock_ort:
                    with patch("tattoo_matcher.ort.get_available_providers", return_value=["CPUExecutionProvider"]):
                        mock_session = MagicMock()
                        mock_session.get_inputs.return_value = [MagicMock(name="input")]
                        mock_session.get_outputs.return_value = [MagicMock(name="embedding")]
                        mock_ort.return_value = mock_session

                        _ = gen.session

                        mock_ort.assert_called_once_with(
                            str(local_model), providers=["CPUExecutionProvider"],
                        )


class TestTattooMatcherInit:
    """Tests for TattooMatcher initialization with model path."""

    def test_init_passes_model_path_to_generator(self):
        """TattooMatcher should pass embedder_model_path to generator."""
        mock_index = Mock()
        matcher = TattooMatcher(
            mock_index,
            [{"universal_id": "stashdb.org:uuid1"}],
            embedder_model_path="/custom/model.onnx",
        )

        assert matcher._embedder_model_path == "/custom/model.onnx"

    def test_init_default_model_path_is_none(self):
        """Default embedder_model_path should be None."""
        mock_index = Mock()
        matcher = TattooMatcher(mock_index, [])

        assert matcher._embedder_model_path is None

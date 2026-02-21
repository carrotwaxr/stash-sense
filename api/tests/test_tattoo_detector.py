"""Tests for tattoo detection using YOLOv5 ONNX model."""

from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


class TestTattooResult:
    """Tests for TattooResult dataclass."""

    def test_to_dict(self):
        """Test that TattooResult converts to dict correctly."""
        from tattoo_detector import TattooDetection, TattooResult

        detections = [
            TattooDetection(
                bbox={"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                confidence=0.85,
                location_hint="left upper body",
            ),
            TattooDetection(
                bbox={"x": 0.5, "y": 0.6, "w": 0.1, "h": 0.2},
                confidence=0.75,
                location_hint="torso",
            ),
        ]
        result = TattooResult(
            detections=detections,
            has_tattoos=True,
            confidence=0.85,
        )

        d = result.to_dict()

        assert d == {
            "detections": [
                {
                    "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
                    "confidence": 0.85,
                    "location_hint": "left upper body",
                },
                {
                    "bbox": {"x": 0.5, "y": 0.6, "w": 0.1, "h": 0.2},
                    "confidence": 0.75,
                    "location_hint": "torso",
                },
            ],
            "has_tattoos": True,
            "confidence": 0.85,
        }

    def test_from_dict(self):
        """Test that TattooResult can be created from dict."""
        from tattoo_detector import TattooResult

        data = {
            "detections": [
                {
                    "bbox": {"x": 0.2, "y": 0.3, "w": 0.15, "h": 0.25},
                    "confidence": 0.9,
                    "location_hint": "right side",
                },
            ],
            "has_tattoos": True,
            "confidence": 0.9,
        }

        result = TattooResult.from_dict(data)

        assert result.has_tattoos is True
        assert result.confidence == 0.9
        assert len(result.detections) == 1
        assert result.detections[0].bbox == {"x": 0.2, "y": 0.3, "w": 0.15, "h": 0.25}
        assert result.detections[0].confidence == 0.9
        assert result.detections[0].location_hint == "right side"

    def test_empty_result(self):
        """Test TattooResult with no detections."""
        from tattoo_detector import TattooResult

        result = TattooResult(
            detections=[],
            has_tattoos=False,
            confidence=0.0,
        )

        d = result.to_dict()

        assert d == {
            "detections": [],
            "has_tattoos": False,
            "confidence": 0.0,
        }

    def test_locations_property(self):
        """Test that locations property returns set of location hints."""
        from tattoo_detector import TattooDetection, TattooResult

        detections = [
            TattooDetection(
                bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                confidence=0.8,
                location_hint="left upper body",
            ),
            TattooDetection(
                bbox={"x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1},
                confidence=0.7,
                location_hint="torso",
            ),
            TattooDetection(
                bbox={"x": 0.2, "y": 0.3, "w": 0.1, "h": 0.1},
                confidence=0.6,
                location_hint="left upper body",  # duplicate location
            ),
        ]
        result = TattooResult(
            detections=detections,
            has_tattoos=True,
            confidence=0.8,
        )

        locations = result.locations

        assert isinstance(locations, set)
        assert locations == {"left upper body", "torso"}


class TestTattooDetector:
    """Tests for TattooDetector class."""

    def test_detector_has_required_methods(self):
        """Test that TattooDetector has the required interface."""
        from tattoo_detector import TattooDetector

        # Check class has required attributes/methods without loading model
        assert hasattr(TattooDetector, 'MIN_CONFIDENCE')
        assert hasattr(TattooDetector, 'detect')
        assert hasattr(TattooDetector, 'session')
        assert hasattr(TattooDetector, 'model')
        assert hasattr(TattooDetector, '_estimate_location')
        assert hasattr(TattooDetector, '_letterbox')
        assert hasattr(TattooDetector, '_nms')

        # Check detect and _estimate_location are callable
        assert callable(getattr(TattooDetector, 'detect'))
        assert callable(getattr(TattooDetector, '_estimate_location'))

    def test_detector_constants(self):
        """Test that detector constants are set correctly."""
        from tattoo_detector import TattooDetector

        assert TattooDetector.MIN_CONFIDENCE == 0.25
        assert TattooDetector.INPUT_SIZE == 640
        assert TattooDetector.IOU_THRESHOLD == 0.45


class TestLetterbox:
    """Tests for the _letterbox preprocessing method."""

    def _make_detector(self):
        """Create a TattooDetector without loading the model."""
        from tattoo_detector import TattooDetector
        return TattooDetector(model_path="/fake/path.onnx")

    def test_square_image_no_padding(self):
        """A 640x640 image should pass through with no resize or padding."""
        detector = self._make_detector()
        image = np.full((640, 640, 3), 128, dtype=np.uint8)

        result, ratio, (pad_left, pad_top) = detector._letterbox(image, 640)

        assert result.shape == (640, 640, 3)
        assert ratio == 1.0
        assert pad_left == 0
        assert pad_top == 0
        # Content should be preserved (no gray padding)
        np.testing.assert_array_equal(result, image)

    def test_landscape_image_padded_vertically(self):
        """A wide image should be scaled and padded top/bottom."""
        detector = self._make_detector()
        # 200h x 400w -> scale by 640/400 = 1.6 -> 320h x 640w
        # Vertical padding: (640 - 320) = 320, half = 160 top and bottom
        image = np.full((200, 400, 3), 200, dtype=np.uint8)

        result, ratio, (pad_left, pad_top) = detector._letterbox(image, 640)

        assert result.shape == (640, 640, 3)
        assert ratio == pytest.approx(1.6)
        assert pad_left == 0
        assert pad_top == 160
        # Check that the padded region is gray (114)
        assert result[0, 0, 0] == 114  # Top-left should be pad
        assert result[159, 0, 0] == 114  # Just before image starts
        # Check that the image region is not gray
        assert result[160, 0, 0] == 200  # Image content starts

    def test_portrait_image_padded_horizontally(self):
        """A tall image should be scaled and padded left/right."""
        detector = self._make_detector()
        # 400h x 200w -> scale by 640/400 = 1.6 -> 640h x 320w
        # Horizontal padding: (640 - 320) = 320, half = 160 left and right
        image = np.full((400, 200, 3), 150, dtype=np.uint8)

        result, ratio, (pad_left, pad_top) = detector._letterbox(image, 640)

        assert result.shape == (640, 640, 3)
        assert ratio == pytest.approx(1.6)
        assert pad_left == 160
        assert pad_top == 0
        # Left pad region should be gray
        assert result[0, 0, 0] == 114
        assert result[0, 159, 0] == 114
        # Image content region
        assert result[0, 160, 0] == 150

    def test_already_correct_size(self):
        """An image already at target size shouldn't be resized."""
        detector = self._make_detector()
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result, ratio, (pad_left, pad_top) = detector._letterbox(image, 640)

        assert result.shape == (640, 640, 3)
        assert ratio == 1.0
        np.testing.assert_array_equal(result, image)

    def test_small_image_scales_up(self):
        """A small square image should be scaled up to fill the target."""
        detector = self._make_detector()
        image = np.full((100, 100, 3), 50, dtype=np.uint8)

        result, ratio, (pad_left, pad_top) = detector._letterbox(image, 640)

        assert result.shape == (640, 640, 3)
        assert ratio == pytest.approx(6.4)
        assert pad_left == 0
        assert pad_top == 0

    def test_preserves_aspect_ratio(self):
        """Aspect ratio should be preserved after letterboxing."""
        detector = self._make_detector()
        # 1080x1920 (portrait phone) -> scale by 640/1920=0.333 -> 360x640
        # Or scale by 640/1080=0.593 but max side is 1920, so 640/1920
        image = np.full((1080, 1920, 3), 100, dtype=np.uint8)

        result, ratio, (pad_left, pad_top) = detector._letterbox(image, 640)

        assert result.shape == (640, 640, 3)
        # ratio should be based on longest side
        expected_ratio = 640 / 1920
        assert ratio == pytest.approx(expected_ratio, rel=0.01)
        # Scaled dimensions
        scaled_w = round(1920 * ratio)
        scaled_h = round(1080 * ratio)
        assert scaled_w == 640  # Fits exactly
        assert pad_left == 0
        # Vertical padding
        expected_pad_top = (640 - scaled_h) // 2
        assert pad_top == expected_pad_top


class TestNMS:
    """Tests for the _nms (Non-Maximum Suppression) method."""

    def _make_detector(self):
        """Create a TattooDetector without loading the model."""
        from tattoo_detector import TattooDetector
        return TattooDetector(model_path="/fake/path.onnx")

    def test_empty_detections(self):
        """NMS on empty input returns empty output."""
        detector = self._make_detector()
        empty = np.zeros((0, 6))

        result = detector._nms(empty, iou_threshold=0.45)

        assert len(result) == 0

    def test_single_detection_passes_through(self):
        """A single detection should be kept."""
        detector = self._make_detector()
        dets = np.array([[100, 100, 200, 200, 0.9, 0]])

        result = detector._nms(dets, iou_threshold=0.45)

        assert len(result) == 1
        np.testing.assert_array_equal(result[0], dets[0])

    def test_non_overlapping_boxes_all_kept(self):
        """Boxes that don't overlap should all be kept."""
        detector = self._make_detector()
        dets = np.array([
            [0, 0, 50, 50, 0.9, 0],      # Top-left
            [100, 100, 150, 150, 0.8, 0],  # Center
            [200, 200, 250, 250, 0.7, 0],  # Bottom-right
        ])

        result = detector._nms(dets, iou_threshold=0.45)

        assert len(result) == 3

    def test_highly_overlapping_boxes_suppressed(self):
        """Nearly identical boxes should be suppressed, keeping highest conf."""
        detector = self._make_detector()
        dets = np.array([
            [100, 100, 200, 200, 0.9, 0],  # High confidence
            [102, 102, 202, 202, 0.7, 0],  # Almost same box, lower conf
            [101, 101, 201, 201, 0.5, 0],  # Almost same box, lowest conf
        ])

        result = detector._nms(dets, iou_threshold=0.45)

        assert len(result) == 1
        assert result[0][4] == 0.9  # Highest confidence kept

    def test_partial_overlap_threshold_behavior(self):
        """Boxes with partial overlap should be kept/suppressed based on IoU threshold."""
        detector = self._make_detector()
        # Two boxes with ~50% overlap
        # Box 1: 0,0 -> 100,100 (area=10000)
        # Box 2: 50,0 -> 150,100 (area=10000)
        # Intersection: 50,0 -> 100,100 (area=5000)
        # Union: 10000 + 10000 - 5000 = 15000
        # IoU = 5000/15000 = 0.333
        dets = np.array([
            [0, 0, 100, 100, 0.9, 0],
            [50, 0, 150, 100, 0.8, 0],
        ])

        # With threshold 0.45, IoU 0.333 < 0.45 -> both kept
        result = detector._nms(dets, iou_threshold=0.45)
        assert len(result) == 2

        # With threshold 0.3, IoU 0.333 > 0.3 -> second suppressed
        result_strict = detector._nms(dets, iou_threshold=0.3)
        assert len(result_strict) == 1

    def test_sorted_by_confidence(self):
        """NMS should process boxes in descending confidence order."""
        detector = self._make_detector()
        # Put low confidence first in input
        dets = np.array([
            [100, 100, 200, 200, 0.3, 0],  # Lower conf
            [100, 100, 200, 200, 0.9, 0],  # Higher conf (same box)
        ])

        result = detector._nms(dets, iou_threshold=0.45)

        # Should keep only the higher confidence one
        assert len(result) == 1
        assert result[0][4] == 0.9


class TestDetect:
    """Tests for the detect() method with mocked ONNX session."""

    def _make_detector_with_mock_session(self, raw_output):
        """Create a TattooDetector with a mocked ONNX session.

        Args:
            raw_output: The raw ONNX model output, shape (1, N, 6)
        """
        from tattoo_detector import TattooDetector

        detector = TattooDetector(model_path="/fake/path.onnx")

        # Create mock session
        mock_session = MagicMock()
        mock_session.run.return_value = [raw_output]
        mock_session.get_inputs.return_value = [MagicMock(name="images")]
        mock_session.get_outputs.return_value = [MagicMock(name="detections")]

        # Inject the mock session directly
        detector._session = mock_session
        detector._input_name = "images"
        detector._output_name = "detections"

        return detector

    def test_no_detections(self):
        """When model outputs low-confidence predictions, result is empty."""
        # All predictions below MIN_CONFIDENCE (0.25)
        raw = np.zeros((1, 25200, 6), dtype=np.float32)
        raw[0, :, 4] = 0.1  # obj_conf
        raw[0, :, 5] = 0.1  # class_conf -> score = 0.01

        detector = self._make_detector_with_mock_session(raw)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        assert result.has_tattoos is False
        assert result.confidence == 0.0
        assert len(result.detections) == 0

    def test_single_detection(self):
        """A single high-confidence detection should be returned."""
        raw = np.zeros((1, 25200, 6), dtype=np.float32)
        # Place one detection at center of 640x640 space
        # x_center=320, y_center=320, w=100, h=100, obj_conf=0.9, class_conf=0.8
        raw[0, 0] = [320, 320, 100, 100, 0.9, 0.8]

        detector = self._make_detector_with_mock_session(raw)
        # Square image so letterbox doesn't add padding
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        assert result.has_tattoos is True
        assert result.confidence == pytest.approx(0.72, abs=0.01)  # 0.9 * 0.8
        assert len(result.detections) == 1
        det = result.detections[0]
        # In 640x640 space: x1=270, y1=270, x2=370, y2=370
        # Normalized: x=270/640, y=270/640, w=100/640, h=100/640
        assert det.bbox["x"] == pytest.approx(270 / 640, abs=0.02)
        assert det.bbox["y"] == pytest.approx(270 / 640, abs=0.02)
        assert det.bbox["w"] == pytest.approx(100 / 640, abs=0.02)
        assert det.bbox["h"] == pytest.approx(100 / 640, abs=0.02)

    def test_multiple_detections_sorted_by_confidence(self):
        """Multiple detections should be sorted by confidence (highest first)."""
        raw = np.zeros((1, 25200, 6), dtype=np.float32)
        # Detection 1: lower confidence
        raw[0, 0] = [100, 100, 50, 50, 0.7, 0.5]  # score = 0.35
        # Detection 2: higher confidence (non-overlapping)
        raw[0, 1] = [400, 400, 60, 60, 0.9, 0.9]  # score = 0.81

        detector = self._make_detector_with_mock_session(raw)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        assert len(result.detections) == 2
        # Higher confidence should be first
        assert result.detections[0].confidence > result.detections[1].confidence
        assert result.confidence == pytest.approx(0.81, abs=0.01)

    def test_coordinate_mapping_with_letterbox(self):
        """Coordinates should be correctly mapped back from letterbox to original space."""
        raw = np.zeros((1, 25200, 6), dtype=np.float32)
        # Detection at center of 640x640 letterbox space
        raw[0, 0] = [320, 320, 100, 100, 0.9, 0.9]

        detector = self._make_detector_with_mock_session(raw)
        # Non-square image: 480h x 640w -> ratio = 1.0, pad_top = 80
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        assert len(result.detections) == 1
        det = result.detections[0]
        # In 640x640: x1=270, y1=270, x2=370, y2=370
        # Letterbox: ratio=1.0, pad_left=0, pad_top=80
        # After removing pad: x1=270, y1=190, x2=370, y2=290
        # After undo ratio (1.0): same
        # Normalized by original (640w, 480h):
        # x=270/640, y=190/480, w=100/640, h=100/480
        assert det.bbox["x"] == pytest.approx(270 / 640, abs=0.02)
        assert det.bbox["y"] == pytest.approx(190 / 480, abs=0.02)
        assert det.bbox["w"] == pytest.approx(100 / 640, abs=0.02)
        assert det.bbox["h"] == pytest.approx(100 / 480, abs=0.02)

    def test_detect_returns_tattoo_result_type(self):
        """detect() should return a TattooResult instance."""
        from tattoo_detector import TattooResult

        raw = np.zeros((1, 25200, 6), dtype=np.float32)
        detector = self._make_detector_with_mock_session(raw)
        image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        assert isinstance(result, TattooResult)

    def test_detection_location_hints(self):
        """Detections should have location hints based on bbox position."""
        raw = np.zeros((1, 25200, 6), dtype=np.float32)
        # Detection in upper-left area of image
        raw[0, 0] = [100, 100, 50, 50, 0.9, 0.9]

        detector = self._make_detector_with_mock_session(raw)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        assert len(result.detections) == 1
        assert result.detections[0].location_hint is not None

    def test_nms_suppresses_overlapping_in_detect(self):
        """Overlapping detections in model output should be suppressed by NMS."""
        raw = np.zeros((1, 25200, 6), dtype=np.float32)
        # Two nearly identical detections (will overlap heavily)
        raw[0, 0] = [320, 320, 100, 100, 0.9, 0.9]  # score = 0.81
        raw[0, 1] = [322, 322, 100, 100, 0.8, 0.8]  # score = 0.64, same region

        detector = self._make_detector_with_mock_session(raw)
        image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

        result = detector.detect(image)

        # NMS should suppress the lower confidence duplicate
        assert len(result.detections) == 1
        assert result.detections[0].confidence == pytest.approx(0.81, abs=0.01)


class TestEstimateLocation:
    """Tests for the _estimate_location method."""

    def _make_detector(self):
        from tattoo_detector import TattooDetector
        return TattooDetector(model_path="/fake/path.onnx")

    def test_upper_body_center(self):
        detector = self._make_detector()
        bbox = {"x": 0.4, "y": 0.0, "w": 0.2, "h": 0.2}
        assert detector._estimate_location(bbox) == "upper body"

    def test_torso_center(self):
        detector = self._make_detector()
        bbox = {"x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}
        assert detector._estimate_location(bbox) == "torso"

    def test_lower_body_center(self):
        detector = self._make_detector()
        bbox = {"x": 0.4, "y": 0.75, "w": 0.2, "h": 0.2}
        assert detector._estimate_location(bbox) == "lower body"

    def test_left_side(self):
        detector = self._make_detector()
        bbox = {"x": 0.0, "y": 0.45, "w": 0.2, "h": 0.2}
        assert detector._estimate_location(bbox) == "left side"

    def test_right_side(self):
        detector = self._make_detector()
        bbox = {"x": 0.7, "y": 0.45, "w": 0.2, "h": 0.2}
        assert detector._estimate_location(bbox) == "right side"

    def test_left_upper_body(self):
        detector = self._make_detector()
        bbox = {"x": 0.0, "y": 0.0, "w": 0.2, "h": 0.2}
        assert detector._estimate_location(bbox) == "left upper body"

    def test_right_lower_body(self):
        detector = self._make_detector()
        bbox = {"x": 0.7, "y": 0.75, "w": 0.2, "h": 0.2}
        assert detector._estimate_location(bbox) == "right lower body"


class TestFindModelPath:
    """Tests for _find_model_path model resolution logic."""

    def test_raises_when_no_model_found(self):
        """Should raise FileNotFoundError when model is not in any search path."""
        from tattoo_detector import _find_model_path

        with patch("tattoo_detector.DATA_DIR", "/nonexistent/data"):
            with patch("tattoo_detector.LOCAL_MODELS_DIR", Path("/nonexistent/local")):
                with pytest.raises(FileNotFoundError, match="Tattoo detection ONNX model not found"):
                    _find_model_path()

    def test_prefers_data_dir_over_local(self, tmp_path):
        """DATA_DIR/models should take priority over local models dir."""
        from tattoo_detector import _find_model_path, ONNX_MODEL_FILENAME

        # Create both paths
        data_models = tmp_path / "data" / "models"
        data_models.mkdir(parents=True)
        (data_models / ONNX_MODEL_FILENAME).touch()

        local_models = tmp_path / "local"
        local_models.mkdir(parents=True)
        (local_models / ONNX_MODEL_FILENAME).touch()

        with patch("tattoo_detector.DATA_DIR", str(tmp_path / "data")):
            with patch("tattoo_detector.LOCAL_MODELS_DIR", local_models):
                result = _find_model_path()
                assert str(result).startswith(str(data_models))

    def test_falls_back_to_local(self, tmp_path):
        """Should fall back to local models dir when DATA_DIR doesn't have the model."""
        from tattoo_detector import _find_model_path, ONNX_MODEL_FILENAME

        local_models = tmp_path / "local"
        local_models.mkdir(parents=True)
        (local_models / ONNX_MODEL_FILENAME).touch()

        with patch("tattoo_detector.DATA_DIR", "/nonexistent/data"):
            with patch("tattoo_detector.LOCAL_MODELS_DIR", local_models):
                result = _find_model_path()
                assert str(result).startswith(str(local_models))


# Need Path import at module level for patching
from pathlib import Path

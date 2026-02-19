"""Tests for TattooMatcher module."""

import pytest
from unittest.mock import Mock
import numpy as np

from tattoo_matcher import TattooMatcher
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

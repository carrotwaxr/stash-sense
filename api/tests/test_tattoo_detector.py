"""Tests for tattoo detection using YOLOv5."""



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
        assert hasattr(TattooDetector, 'model')
        assert hasattr(TattooDetector, '_estimate_location')

        # Check detect and _estimate_location are callable
        assert callable(getattr(TattooDetector, 'detect'))
        assert callable(getattr(TattooDetector, '_estimate_location'))

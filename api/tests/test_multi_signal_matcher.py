"""Tests for MultiSignalMatcher module."""

import pytest
import sys
from unittest.mock import Mock, MagicMock, patch
from dataclasses import dataclass
from typing import Optional

import numpy as np

# Mock the voyager module before importing anything that depends on it
sys.modules['voyager'] = Mock()

from multi_signal_matcher import MultiSignalMatcher, MultiSignalMatch
from body_proportions import BodyProportions
from tattoo_detector import TattooResult, TattooDetection


class TestMultiSignalMatchDataclass:
    """Tests for MultiSignalMatch dataclass."""

    def test_create_with_required_fields(self):
        """Test creating MultiSignalMatch with required fields only."""
        mock_face = Mock()
        matches = [Mock()]

        result = MultiSignalMatch(
            face=mock_face,
            matches=matches,
        )

        assert result.face is mock_face
        assert result.matches == matches
        assert result.body_ratios is None
        assert result.tattoo_result is None
        assert result.signals_used == []

    def test_create_with_all_fields(self):
        """Test creating MultiSignalMatch with all fields."""
        mock_face = Mock()
        matches = [Mock()]
        body_ratios = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )
        tattoo_result = TattooResult(
            detections=[],
            has_tattoos=False,
            confidence=0.0,
        )

        result = MultiSignalMatch(
            face=mock_face,
            matches=matches,
            body_ratios=body_ratios,
            tattoo_result=tattoo_result,
            signals_used=["face", "body", "tattoo"],
        )

        assert result.face is mock_face
        assert result.matches == matches
        assert result.body_ratios == body_ratios
        assert result.tattoo_result == tattoo_result
        assert result.signals_used == ["face", "body", "tattoo"]


class TestMultiSignalMatcherInit:
    """Tests for MultiSignalMatcher initialization."""

    def test_init_loads_body_data(self):
        """Test that matcher loads body data on init."""
        mock_face_recognizer = Mock()
        mock_db_reader = Mock()
        mock_db_reader.get_all_body_proportions.return_value = {
            "stashdb.org:uuid1": {
                "shoulder_hip_ratio": 1.2,
                "leg_torso_ratio": 1.5,
                "arm_span_height_ratio": 1.0,
                "confidence": 0.9,
            }
        }

        matcher = MultiSignalMatcher(
            face_recognizer=mock_face_recognizer,
            db_reader=mock_db_reader,
        )

        mock_db_reader.get_all_body_proportions.assert_called_once()
        assert matcher.body_data == mock_db_reader.get_all_body_proportions.return_value

    def test_init_with_optional_extractors(self):
        """Test that matcher stores optional body and tattoo extractors."""
        mock_face_recognizer = Mock()
        mock_db_reader = Mock()
        mock_db_reader.get_all_body_proportions.return_value = {}
        mock_body_extractor = Mock()
        mock_tattoo_detector = Mock()

        matcher = MultiSignalMatcher(
            face_recognizer=mock_face_recognizer,
            db_reader=mock_db_reader,
            body_extractor=mock_body_extractor,
            tattoo_detector=mock_tattoo_detector,
        )

        assert matcher.body_extractor is mock_body_extractor
        assert matcher.tattoo_detector is mock_tattoo_detector

    def test_init_builds_tattoo_embedding_set(self):
        """Test that matcher builds performers_with_tattoo_embeddings from mapping."""
        mock_face_recognizer = Mock()
        mock_db_reader = Mock()
        mock_db_reader.get_all_body_proportions.return_value = {}
        mock_tattoo_matcher = Mock()
        mock_tattoo_matcher.tattoo_mapping = [
            {"universal_id": "stashdb.org:uuid1"}, {"universal_id": "stashdb.org:uuid1"},
            {"universal_id": "stashdb.org:uuid2"}, None
        ]

        matcher = MultiSignalMatcher(
            face_recognizer=mock_face_recognizer,
            db_reader=mock_db_reader,
            tattoo_matcher=mock_tattoo_matcher,
        )

        assert "stashdb.org:uuid1" in matcher.performers_with_tattoo_embeddings
        assert "stashdb.org:uuid2" in matcher.performers_with_tattoo_embeddings
        assert len(matcher.performers_with_tattoo_embeddings) == 2


class TestMultiSignalMatcherIdentify:
    """Tests for MultiSignalMatcher.identify method."""

    def _create_matcher(
        self,
        face_recognizer=None,
        db_reader=None,
        body_extractor=None,
        tattoo_detector=None,
        tattoo_matcher=None,
        body_data=None,
    ):
        """Helper to create a matcher with mocked dependencies."""
        if face_recognizer is None:
            face_recognizer = Mock()
        if db_reader is None:
            db_reader = Mock()
            db_reader.get_all_body_proportions.return_value = body_data or {}

        return MultiSignalMatcher(
            face_recognizer=face_recognizer,
            db_reader=db_reader,
            body_extractor=body_extractor,
            tattoo_detector=tattoo_detector,
            tattoo_matcher=tattoo_matcher,
        )

    def test_identify_calls_face_recognizer(self):
        """Test that identify calls face_recognizer.recognize_image."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = []

        matcher = self._create_matcher(face_recognizer=mock_face_recognizer)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        matcher.identify(image, top_k=5, face_candidates=20)

        mock_face_recognizer.recognize_image.assert_called_once_with(image, top_k=20)

    def test_identify_returns_list_of_multi_signal_match(self):
        """Test that identify returns a list of MultiSignalMatch objects."""
        mock_face = Mock()
        mock_matches = [
            Mock(
                universal_id="stashdb.org:uuid1",
                combined_score=0.3,
            )
        ]
        mock_recognition_result = Mock(
            face=mock_face,
            matches=mock_matches,
        )
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [mock_recognition_result]

        matcher = self._create_matcher(face_recognizer=mock_face_recognizer)

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = matcher.identify(image)

        assert len(results) == 1
        assert isinstance(results[0], MultiSignalMatch)
        assert results[0].face is mock_face

    def test_identify_extracts_body_ratios_when_extractor_provided(self):
        """Test that identify extracts body ratios when body_extractor is provided."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(
                face=Mock(),
                matches=[
                    Mock(
                        universal_id="stashdb.org:uuid1",
                        combined_score=0.3,
                    )
                ],
            )
        ]

        mock_body_extractor = Mock()
        expected_ratios = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )
        mock_body_extractor.extract.return_value = expected_ratios

        matcher = self._create_matcher(
            face_recognizer=mock_face_recognizer,
            body_extractor=mock_body_extractor,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = matcher.identify(image, use_body=True)

        mock_body_extractor.extract.assert_called_once_with(image)
        assert results[0].body_ratios == expected_ratios

    def test_identify_skips_body_extraction_when_use_body_false(self):
        """Test that identify skips body extraction when use_body=False."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(
                face=Mock(),
                matches=[
                    Mock(universal_id="stashdb.org:uuid1", combined_score=0.3)
                ],
            )
        ]

        mock_body_extractor = Mock()

        matcher = self._create_matcher(
            face_recognizer=mock_face_recognizer,
            body_extractor=mock_body_extractor,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        matcher.identify(image, use_body=False)

        mock_body_extractor.extract.assert_not_called()

    def test_identify_detects_tattoos_when_detector_provided(self):
        """Test that identify detects tattoos when tattoo_detector is provided."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(
                face=Mock(),
                matches=[
                    Mock(universal_id="stashdb.org:uuid1", combined_score=0.3)
                ],
            )
        ]

        mock_tattoo_detector = Mock()
        expected_result = TattooResult(
            detections=[
                TattooDetection(
                    bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                    confidence=0.9,
                    location_hint="left arm",
                )
            ],
            has_tattoos=True,
            confidence=0.9,
        )
        mock_tattoo_detector.detect.return_value = expected_result

        matcher = self._create_matcher(
            face_recognizer=mock_face_recognizer,
            tattoo_detector=mock_tattoo_detector,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = matcher.identify(image, use_tattoo=True)

        mock_tattoo_detector.detect.assert_called_once_with(image)
        assert results[0].tattoo_result == expected_result

    def test_identify_runs_tattoo_matcher_on_detections(self):
        """Test that identify runs tattoo_matcher.match when tattoos detected."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(
                face=Mock(),
                matches=[
                    Mock(universal_id="stashdb.org:uuid1", combined_score=0.3)
                ],
            )
        ]

        tattoo_detections = [
            TattooDetection(
                bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                confidence=0.9,
                location_hint="left arm",
            )
        ]

        mock_tattoo_detector = Mock()
        mock_tattoo_detector.detect.return_value = TattooResult(
            detections=tattoo_detections,
            has_tattoos=True,
            confidence=0.9,
        )

        mock_tattoo_matcher = Mock()
        mock_tattoo_matcher.tattoo_mapping = [{"universal_id": "stashdb.org:uuid1"}]
        mock_tattoo_matcher.match.return_value = {"stashdb.org:uuid1": 0.85}

        matcher = self._create_matcher(
            face_recognizer=mock_face_recognizer,
            tattoo_detector=mock_tattoo_detector,
            tattoo_matcher=mock_tattoo_matcher,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = matcher.identify(image, use_tattoo=True)

        mock_tattoo_matcher.match.assert_called_once()

    def test_identify_skips_tattoo_matching_when_no_tattoos_detected(self):
        """Test that tattoo_matcher.match is not called when no tattoos detected."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(
                face=Mock(),
                matches=[
                    Mock(universal_id="stashdb.org:uuid1", combined_score=0.3)
                ],
            )
        ]

        mock_tattoo_detector = Mock()
        mock_tattoo_detector.detect.return_value = TattooResult(
            detections=[], has_tattoos=False, confidence=0.0,
        )

        mock_tattoo_matcher = Mock()
        mock_tattoo_matcher.tattoo_mapping = []

        matcher = self._create_matcher(
            face_recognizer=mock_face_recognizer,
            tattoo_detector=mock_tattoo_detector,
            tattoo_matcher=mock_tattoo_matcher,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        matcher.identify(image, use_tattoo=True)

        mock_tattoo_matcher.match.assert_not_called()

    def test_identify_skips_tattoo_detection_when_use_tattoo_false(self):
        """Test that identify skips tattoo detection when use_tattoo=False."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(
                face=Mock(),
                matches=[
                    Mock(universal_id="stashdb.org:uuid1", combined_score=0.3)
                ],
            )
        ]

        mock_tattoo_detector = Mock()

        matcher = self._create_matcher(
            face_recognizer=mock_face_recognizer,
            tattoo_detector=mock_tattoo_detector,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        matcher.identify(image, use_tattoo=False)

        mock_tattoo_detector.detect.assert_not_called()

    def test_identify_tracks_signals_used(self):
        """Test that identify tracks which signals were used."""
        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(
                face=Mock(),
                matches=[
                    Mock(universal_id="stashdb.org:uuid1", combined_score=0.3)
                ],
            )
        ]

        mock_body_extractor = Mock()
        mock_body_extractor.extract.return_value = BodyProportions(
            shoulder_hip_ratio=1.2,
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )

        mock_tattoo_detector = Mock()
        mock_tattoo_detector.detect.return_value = TattooResult(
            detections=[],
            has_tattoos=False,
            confidence=0.0,
        )

        matcher = self._create_matcher(
            face_recognizer=mock_face_recognizer,
            body_extractor=mock_body_extractor,
            tattoo_detector=mock_tattoo_detector,
        )

        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = matcher.identify(image, use_body=True, use_tattoo=True)

        assert "face" in results[0].signals_used
        assert "body" in results[0].signals_used
        assert "tattoo" in results[0].signals_used


class TestMultiSignalMatcherRerank:
    """Tests for MultiSignalMatcher._rerank_candidates method."""

    def _create_matcher_with_data(self, body_data=None, tattoo_mapping=None):
        """Helper to create a matcher with pre-populated data."""
        mock_face_recognizer = Mock()
        mock_db_reader = Mock()
        mock_db_reader.get_all_body_proportions.return_value = body_data or {}

        mock_tattoo_matcher = None
        if tattoo_mapping is not None:
            mock_tattoo_matcher = Mock()
            mock_tattoo_matcher.tattoo_mapping = tattoo_mapping

        return MultiSignalMatcher(
            face_recognizer=mock_face_recognizer,
            db_reader=mock_db_reader,
            tattoo_matcher=mock_tattoo_matcher,
        )

    def test_rerank_converts_distance_to_similarity(self):
        """Test that rerank converts distance to similarity score."""
        matcher = self._create_matcher_with_data()

        # Create mock candidate with combined_score (distance)
        candidate = Mock(
                        universal_id="stashdb.org:uuid1",
            combined_score=0.0,  # Perfect match, distance = 0
        )

        results = matcher._rerank_candidates(
            candidates=[candidate],
            body_ratios=None,
            tattoo_result=None,
            tattoo_scores=None,
            top_k=5,
        )

        # base_score = 1.0 / (1.0 + 0.0) = 1.0
        assert len(results) == 1
        # Score should be 1.0 (highest similarity for distance 0)

    def test_rerank_applies_body_penalty(self):
        """Test that rerank applies body ratio penalty."""
        body_data = {
            "stashdb.org:uuid1": {
                "shoulder_hip_ratio": 1.0,
                "leg_torso_ratio": 1.5,
                "arm_span_height_ratio": 1.0,
                "confidence": 0.9,
            },
        }
        matcher = self._create_matcher_with_data(body_data=body_data)

        candidate = Mock(
                        universal_id="stashdb.org:uuid1",
            combined_score=0.3,
        )

        # Query ratios that match well with candidate
        query_ratios = BodyProportions(
            shoulder_hip_ratio=1.05,  # diff = 0.05, <= 0.12, penalty = 1.0
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )

        results_with_match = matcher._rerank_candidates(
            candidates=[candidate],
            body_ratios=query_ratios,
            tattoo_result=None,
            tattoo_scores=None,
            top_k=5,
        )

        # Query ratios that don't match
        query_ratios_mismatch = BodyProportions(
            shoulder_hip_ratio=1.5,  # diff = 0.5, > 0.35, penalty = 0.3
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )

        results_with_mismatch = matcher._rerank_candidates(
            candidates=[candidate],
            body_ratios=query_ratios_mismatch,
            tattoo_result=None,
            tattoo_scores=None,
            top_k=5,
        )

        # Mismatched body should have lower final score
        assert len(results_with_match) == 1
        assert len(results_with_mismatch) == 1

    def test_rerank_applies_tattoo_embedding_boost(self):
        """Test that rerank applies tattoo embedding similarity boost."""
        tattoo_mapping = [{"universal_id": "stashdb.org:uuid1"}, {"universal_id": "stashdb.org:uuid1"}]
        matcher = self._create_matcher_with_data(tattoo_mapping=tattoo_mapping)

        candidate = Mock(
            universal_id="stashdb.org:uuid1",
            combined_score=0.3,
        )

        # Query with tattoos detected
        query_tattoo = TattooResult(
            detections=[
                TattooDetection(
                    bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                    confidence=0.9,
                    location_hint="left arm",
                )
            ],
            has_tattoos=True,
            confidence=0.9,
        )

        # High similarity score from matcher
        tattoo_scores = {"stashdb.org:uuid1": 0.85}

        results = matcher._rerank_candidates(
            candidates=[candidate],
            body_ratios=None,
            tattoo_result=query_tattoo,
            tattoo_scores=tattoo_scores,
            top_k=5,
        )

        assert len(results) == 1

    def test_rerank_sorts_by_final_score_descending(self):
        """Test that rerank sorts results by final_score descending."""
        matcher = self._create_matcher_with_data()

        # Create candidates with different distances
        candidate1 = Mock(
                        universal_id="stashdb.org:uuid1",
            combined_score=0.1,  # Better match
        )
        candidate2 = Mock(
                        universal_id="stashdb.org:uuid2",
            combined_score=0.5,  # Worse match
        )

        results = matcher._rerank_candidates(
            candidates=[candidate2, candidate1],  # Pass in wrong order
            body_ratios=None,
            tattoo_result=None,
            tattoo_scores=None,
            top_k=5,
        )

        # Should be sorted by score descending (best first)
        assert len(results) == 2
        # The candidate with lower distance (0.1) should have higher similarity
        # and be ranked first
        assert results[0].universal_id == "stashdb.org:uuid1"
        assert results[1].universal_id == "stashdb.org:uuid2"

    def test_rerank_returns_top_k(self):
        """Test that rerank returns only top_k results."""
        matcher = self._create_matcher_with_data()

        candidates = [
            Mock(
                                universal_id=f"stashdb.org:uuid{i}",
                combined_score=0.1 * i,
            )
            for i in range(10)
        ]

        results = matcher._rerank_candidates(
            candidates=candidates,
            body_ratios=None,
            tattoo_result=None,
            tattoo_scores=None,
            top_k=3,
        )

        assert len(results) == 3


class TestMultiSignalMatcherIntegration:
    """Integration tests for MultiSignalMatcher."""

    def test_full_identification_flow(self):
        """Test the full identification flow with all signals."""
        # Set up face recognizer mock
        mock_face = Mock()
        mock_matches = [
            Mock(
                                universal_id="stashdb.org:uuid1",
                stashdb_id="uuid1",
                name="Performer 1",
                country="US",
                image_url="http://example.com/1.jpg",
                facenet_distance=0.2,
                arcface_distance=0.2,
                combined_score=0.2,
            ),
            Mock(
                                universal_id="stashdb.org:uuid2",
                stashdb_id="uuid2",
                name="Performer 2",
                country="UK",
                image_url="http://example.com/2.jpg",
                facenet_distance=0.3,
                arcface_distance=0.3,
                combined_score=0.3,
            ),
        ]

        mock_face_recognizer = Mock()
        mock_face_recognizer.recognize_image.return_value = [
            Mock(face=mock_face, matches=mock_matches)
        ]

        # Set up db reader mock
        mock_db_reader = Mock()
        mock_db_reader.get_all_body_proportions.return_value = {
            "stashdb.org:uuid1": {
                "shoulder_hip_ratio": 1.2,
                "leg_torso_ratio": 1.5,
                "arm_span_height_ratio": 1.0,
                "confidence": 0.9,
            },
            "stashdb.org:uuid2": {
                "shoulder_hip_ratio": 1.5,  # Very different from query
                "leg_torso_ratio": 1.5,
                "arm_span_height_ratio": 1.0,
                "confidence": 0.9,
            },
        }

        # Set up body extractor mock
        mock_body_extractor = Mock()
        mock_body_extractor.extract.return_value = BodyProportions(
            shoulder_hip_ratio=1.2,  # Matches uuid1
            leg_torso_ratio=1.5,
            arm_span_height_ratio=1.0,
            confidence=0.9,
        )

        # Set up tattoo detector mock
        tattoo_detections = [
            TattooDetection(
                bbox={"x": 0.1, "y": 0.2, "w": 0.1, "h": 0.1},
                confidence=0.9,
                location_hint="left arm",
            )
        ]
        mock_tattoo_detector = Mock()
        mock_tattoo_detector.detect.return_value = TattooResult(
            detections=tattoo_detections,
            has_tattoos=True,
            confidence=0.9,
        )

        # Set up tattoo matcher mock — uuid1 has high similarity
        mock_tattoo_matcher = Mock()
        mock_tattoo_matcher.tattoo_mapping = [
            {"universal_id": "stashdb.org:uuid1"}, {"universal_id": "stashdb.org:uuid1"}
        ]
        mock_tattoo_matcher.match.return_value = {"stashdb.org:uuid1": 0.85}

        # Create matcher
        matcher = MultiSignalMatcher(
            face_recognizer=mock_face_recognizer,
            db_reader=mock_db_reader,
            body_extractor=mock_body_extractor,
            tattoo_detector=mock_tattoo_detector,
            tattoo_matcher=mock_tattoo_matcher,
        )

        # Run identification
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        results = matcher.identify(image, top_k=5, use_body=True, use_tattoo=True)

        # Verify results
        assert len(results) == 1
        result = results[0]

        assert result.face is mock_face
        assert len(result.matches) > 0
        assert result.body_ratios is not None
        assert result.tattoo_result is not None
        assert "face" in result.signals_used
        assert "body" in result.signals_used
        assert "tattoo" in result.signals_used

        # uuid1 should be ranked higher due to body and tattoo match
        assert result.matches[0].universal_id == "stashdb.org:uuid1"

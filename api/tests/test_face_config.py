"""Tests for face_config.py - face recognition constants."""

from face_config import (
    FACENET_WEIGHT,
    ARCFACE_WEIGHT,
    MAX_DISTANCE,
    NUM_FRAMES,
    START_OFFSET_PCT,
    END_OFFSET_PCT,
    MIN_FACE_SIZE,
    MIN_FACE_CONFIDENCE,
    CLUSTER_THRESHOLD,
    TOP_K,
)


def test_fusion_weights_sum_to_one():
    assert FACENET_WEIGHT + ARCFACE_WEIGHT == 1.0


def test_max_distance_in_valid_range():
    assert 0 < MAX_DISTANCE <= 1.0


def test_num_frames_positive_integer():
    assert isinstance(NUM_FRAMES, int)
    assert NUM_FRAMES > 0


def test_offsets_valid():
    assert 0 <= START_OFFSET_PCT < END_OFFSET_PCT <= 1.0


def test_face_detection_thresholds():
    assert MIN_FACE_SIZE > 0
    assert 0 < MIN_FACE_CONFIDENCE <= 1.0


def test_cluster_threshold_positive():
    assert 0 < CLUSTER_THRESHOLD <= 1.0


def test_top_k_positive():
    assert TOP_K > 0

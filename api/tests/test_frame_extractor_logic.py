"""Tests for frame_extractor.py pure functions - no ffmpeg required."""

import pytest
from frame_extractor import (
    calculate_extraction_timestamps,
    calculate_weighted_timestamps,
    FrameExtractionConfig,
)


class TestCalculateExtractionTimestamps:
    def test_normal_mode_even_spacing(self):
        config = FrameExtractionConfig(num_frames=10, start_offset_pct=0.0, end_offset_pct=1.0)
        timestamps = calculate_extraction_timestamps(100.0, config)
        assert len(timestamps) == 10
        # First should be 0, last should be 100, evenly spaced
        assert pytest.approx(timestamps[0]) == 0.0
        assert pytest.approx(timestamps[-1]) == 100.0

    def test_with_percentage_offsets(self):
        config = FrameExtractionConfig(num_frames=10, start_offset_pct=0.05, end_offset_pct=0.95)
        timestamps = calculate_extraction_timestamps(1000.0, config)
        assert len(timestamps) == 10
        assert pytest.approx(timestamps[0]) == 50.0   # 5% of 1000
        assert pytest.approx(timestamps[-1]) == 950.0  # 95% of 1000

    def test_with_absolute_offsets(self):
        config = FrameExtractionConfig(
            num_frames=5,
            start_offset_sec=30.0,
            end_offset_sec=30.0,
        )
        timestamps = calculate_extraction_timestamps(600.0, config)
        assert len(timestamps) == 5
        # Start at 30s, end at 570s (600-30)
        assert pytest.approx(timestamps[0]) == 30.0
        assert pytest.approx(timestamps[-1]) == 570.0

    def test_short_video_returns_midpoint(self):
        # end_sec <= start_sec triggers midpoint fallback
        config = FrameExtractionConfig(
            num_frames=10,
            start_offset_sec=50.0,
            end_offset_sec=60.0,
        )
        # 100 - 60 = 40 end, start=50, end(40) <= start(50)
        timestamps = calculate_extraction_timestamps(100.0, config)
        assert len(timestamps) == 1
        assert pytest.approx(timestamps[0]) == 50.0  # duration/2

    def test_single_frame_returns_midpoint(self):
        config = FrameExtractionConfig(num_frames=1, start_offset_pct=0.1, end_offset_pct=0.9)
        timestamps = calculate_extraction_timestamps(200.0, config)
        assert len(timestamps) == 1
        # Midpoint of [20, 180] = 100
        assert pytest.approx(timestamps[0]) == 100.0

    def test_burst_mode_correct_frame_count(self):
        config = FrameExtractionConfig(
            num_frames=40,
            burst_mode=True,
            frames_per_burst=4,
            burst_spread_sec=0.5,
            start_offset_pct=0.05,
            end_offset_pct=0.95,
        )
        timestamps = calculate_extraction_timestamps(1800.0, config)
        # 40 // 4 = 10 sample points, 10 * 4 = 40 timestamps
        assert len(timestamps) == 40

    def test_burst_mode_timestamps_within_range(self):
        config = FrameExtractionConfig(
            num_frames=20,
            burst_mode=True,
            frames_per_burst=4,
            burst_spread_sec=1.0,
            start_offset_pct=0.0,
            end_offset_pct=1.0,
        )
        timestamps = calculate_extraction_timestamps(600.0, config)
        for ts in timestamps:
            assert 0 <= ts <= 600.0

    def test_absolute_offset_overrides_percentage(self):
        config = FrameExtractionConfig(
            num_frames=5,
            start_offset_pct=0.1,
            end_offset_pct=0.9,
            start_offset_sec=10.0,
            end_offset_sec=10.0,
        )
        timestamps = calculate_extraction_timestamps(200.0, config)
        # start_offset_sec=10, end_offset_sec=10 -> range [10, 190]
        assert pytest.approx(timestamps[0]) == 10.0
        assert pytest.approx(timestamps[-1]) == 190.0

    def test_zero_duration_returns_midpoint(self):
        config = FrameExtractionConfig(num_frames=10)
        timestamps = calculate_extraction_timestamps(0.0, config)
        assert len(timestamps) == 1
        assert timestamps[0] == 0.0

    def test_timestamps_monotonically_increasing(self):
        config = FrameExtractionConfig(num_frames=60)
        timestamps = calculate_extraction_timestamps(3600.0, config)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


class TestCalculateWeightedTimestamps:
    def test_long_video_skips_intro(self):
        timestamps = calculate_weighted_timestamps(600.0, num_frames=40, skip_intro_sec=15.0)
        # For video >= 300s, start at skip_intro_sec=15
        assert timestamps[0] >= 15.0

    def test_short_video_uses_pct_offset(self):
        timestamps = calculate_weighted_timestamps(120.0, num_frames=20)
        # For video < 300s, start at 5% = 6.0
        assert pytest.approx(timestamps[0], abs=0.1) == 6.0

    def test_correct_total_frame_count(self):
        timestamps = calculate_weighted_timestamps(1800.0, num_frames=40)
        assert len(timestamps) == 40

    def test_timestamps_monotonically_increasing(self):
        timestamps = calculate_weighted_timestamps(1800.0, num_frames=40)
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    def test_all_timestamps_in_valid_range(self):
        duration = 1800.0
        timestamps = calculate_weighted_timestamps(duration, num_frames=40)
        for ts in timestamps:
            assert 0 <= ts <= duration

    def test_very_short_video_fallback(self):
        # If end <= start, should return [duration/2]
        timestamps = calculate_weighted_timestamps(1.0, num_frames=40)
        # With 5% offset on 1.0s: start=0.05, end=0.95, usable=0.9
        # Should still work (not trigger the fallback for this duration)
        assert len(timestamps) >= 1
        for ts in timestamps:
            assert 0 <= ts <= 1.0

    def test_front_loaded_distribution(self):
        # Front zone should have more frames per second than back zone
        timestamps = calculate_weighted_timestamps(
            1800.0, num_frames=40,
            front_weight=0.40, middle_weight=0.45, back_weight=0.15,
        )
        # Count frames in first 25% of usable range vs last 10%
        start = 15.0  # skip_intro_sec for long video
        end = 1800.0 * 0.95
        usable = end - start
        front_end = start + usable * 0.25

        front_frames = sum(1 for ts in timestamps if ts <= front_end)
        # Front zone should have roughly 40% of frames
        assert front_frames >= 10  # At least 25% of 40


class TestFrameExtractionConfig:
    def test_defaults(self):
        config = FrameExtractionConfig()
        assert config.num_frames == 60
        assert config.start_offset_pct == 0.05
        assert config.end_offset_pct == 0.95
        assert config.start_offset_sec is None
        assert config.end_offset_sec is None
        assert config.burst_mode is False
        assert config.frames_per_burst == 4
        assert config.burst_spread_sec == 0.5
        assert config.output_width is None
        assert config.output_height is None
        assert config.jpeg_quality == 95
        assert config.min_face_size == 40
        assert config.min_face_confidence == 0.5
        assert config.max_concurrent_extractions == 8
        assert config.extraction_timeout_sec == 30.0
        assert config.hwaccel is None
        assert config.ffmpeg_path == "ffmpeg"

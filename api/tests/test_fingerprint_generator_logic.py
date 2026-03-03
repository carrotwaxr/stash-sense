"""Tests for fingerprint_generator.py dataclasses and enums."""

from fingerprint_generator import (
    GeneratorStatus,
    GeneratorProgress,
    FingerprintResult,
)


class TestGeneratorStatus:
    def test_enum_values(self):
        assert GeneratorStatus.IDLE == "idle"
        assert GeneratorStatus.RUNNING == "running"
        assert GeneratorStatus.PAUSED == "paused"
        assert GeneratorStatus.STOPPING == "stopping"
        assert GeneratorStatus.COMPLETED == "completed"
        assert GeneratorStatus.ERROR == "error"

    def test_all_statuses_present(self):
        expected = {"idle", "running", "paused", "stopping", "completed", "error"}
        actual = {s.value for s in GeneratorStatus}
        assert actual == expected


class TestGeneratorProgress:
    def test_progress_pct_zero_when_total_zero(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.IDLE,
            total_scenes=0,
            processed_scenes=0,
            successful=0,
            failed=0,
            skipped=0,
        )
        assert progress.progress_pct == 0.0

    def test_progress_pct_correct_percentage(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=200,
            processed_scenes=50,
            successful=40,
            failed=5,
            skipped=5,
        )
        assert progress.progress_pct == 25.0

    def test_progress_pct_100_when_complete(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.COMPLETED,
            total_scenes=100,
            processed_scenes=100,
            successful=90,
            failed=5,
            skipped=5,
        )
        assert progress.progress_pct == 100.0

    def test_to_dict_all_keys_present(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=500,
            processed_scenes=125,
            successful=100,
            failed=10,
            skipped=15,
            current_scene_id=42,
            current_scene_title="Test Scene",
            error_message=None,
        )
        d = progress.to_dict()

        expected_keys = {
            "status", "total_scenes", "processed_scenes", "successful",
            "failed", "skipped", "progress_pct", "current_scene_id",
            "current_scene_title", "error_message",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_correct_types(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=500,
            processed_scenes=125,
            successful=100,
            failed=10,
            skipped=15,
        )
        d = progress.to_dict()

        assert isinstance(d["status"], str)
        assert isinstance(d["total_scenes"], int)
        assert isinstance(d["processed_scenes"], int)
        assert isinstance(d["successful"], int)
        assert isinstance(d["failed"], int)
        assert isinstance(d["skipped"], int)
        assert isinstance(d["progress_pct"], float)

    def test_to_dict_status_is_string_value(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.COMPLETED,
            total_scenes=10,
            processed_scenes=10,
            successful=10,
            failed=0,
            skipped=0,
        )
        d = progress.to_dict()
        assert d["status"] == "completed"

    def test_to_dict_progress_pct_rounded(self):
        progress = GeneratorProgress(
            status=GeneratorStatus.RUNNING,
            total_scenes=3,
            processed_scenes=1,
            successful=1,
            failed=0,
            skipped=0,
        )
        d = progress.to_dict()
        # 1/3 * 100 = 33.333... -> rounded to 33.3
        assert d["progress_pct"] == 33.3


class TestFingerprintResult:
    def test_creation_with_defaults(self):
        result = FingerprintResult(scene_id=42, success=True)
        assert result.scene_id == 42
        assert result.success is True
        assert result.fingerprint_id is None
        assert result.performers_found == 0
        assert result.frames_analyzed == 0
        assert result.error is None

    def test_creation_with_values(self):
        result = FingerprintResult(
            scene_id=99,
            success=True,
            fingerprint_id=5,
            performers_found=3,
            frames_analyzed=60,
        )
        assert result.fingerprint_id == 5
        assert result.performers_found == 3
        assert result.frames_analyzed == 60

    def test_creation_with_error(self):
        result = FingerprintResult(
            scene_id=42,
            success=False,
            error="Timeout",
        )
        assert result.success is False
        assert result.error == "Timeout"

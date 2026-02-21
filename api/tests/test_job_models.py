"""Tests for job model definitions and type registry."""

import datetime
from dataclasses import FrozenInstanceError

import pytest

from job_models import (
    INTERVALS_FREQUENT,
    INTERVALS_INFREQUENT,
    JOB_REGISTRY,
    JobDefinition,
    JobPriority,
    JobRecord,
    JobStatus,
    ResourceType,
)


# ---------------------------------------------------------------------------
# ResourceType enum
# ---------------------------------------------------------------------------


class TestResourceType:
    def test_values(self):
        assert ResourceType.GPU == "gpu"
        assert ResourceType.CPU_HEAVY == "cpu"
        assert ResourceType.NETWORK == "network"
        assert ResourceType.LIGHT == "light"

    def test_is_string(self):
        assert isinstance(ResourceType.GPU, str)

    def test_member_count(self):
        assert len(ResourceType) == 4


# ---------------------------------------------------------------------------
# JobPriority enum
# ---------------------------------------------------------------------------


class TestJobPriority:
    def test_ordering(self):
        assert JobPriority.CRITICAL < JobPriority.HIGH < JobPriority.NORMAL < JobPriority.LOW

    def test_values(self):
        assert JobPriority.CRITICAL == 0
        assert JobPriority.HIGH == 10
        assert JobPriority.NORMAL == 50
        assert JobPriority.LOW == 100

    def test_sortable(self):
        priorities = [JobPriority.LOW, JobPriority.CRITICAL, JobPriority.NORMAL, JobPriority.HIGH]
        assert sorted(priorities) == [
            JobPriority.CRITICAL,
            JobPriority.HIGH,
            JobPriority.NORMAL,
            JobPriority.LOW,
        ]


# ---------------------------------------------------------------------------
# JobStatus enum
# ---------------------------------------------------------------------------


class TestJobStatus:
    def test_values(self):
        assert JobStatus.PENDING == "pending"
        assert JobStatus.QUEUED == "queued"
        assert JobStatus.RUNNING == "running"
        assert JobStatus.STOPPING == "stopping"
        assert JobStatus.COMPLETED == "completed"
        assert JobStatus.FAILED == "failed"
        assert JobStatus.CANCELLED == "cancelled"

    def test_is_string(self):
        assert isinstance(JobStatus.RUNNING, str)

    def test_member_count(self):
        assert len(JobStatus) == 7


# ---------------------------------------------------------------------------
# JobDefinition (frozen dataclass)
# ---------------------------------------------------------------------------


class TestJobDefinition:
    def test_create(self):
        defn = JobDefinition(
            type_id="test_job",
            display_name="Test Job",
            description="A test job",
            resource=ResourceType.LIGHT,
            default_priority=JobPriority.NORMAL,
            supports_incremental=False,
            schedulable=True,
            default_interval_hours=24,
        )
        assert defn.type_id == "test_job"
        assert defn.display_name == "Test Job"
        assert defn.description == "A test job"
        assert defn.resource == ResourceType.LIGHT
        assert defn.default_priority == JobPriority.NORMAL
        assert defn.supports_incremental is False
        assert defn.schedulable is True
        assert defn.default_interval_hours == 24

    def test_frozen(self):
        defn = JobDefinition(
            type_id="test_job",
            display_name="Test Job",
            description="A test job",
            resource=ResourceType.GPU,
            default_priority=JobPriority.HIGH,
            supports_incremental=False,
            schedulable=False,
        )
        with pytest.raises(FrozenInstanceError):
            defn.type_id = "modified"

    def test_default_interval_hours_none(self):
        defn = JobDefinition(
            type_id="x",
            display_name="X",
            description="desc",
            resource=ResourceType.LIGHT,
            default_priority=JobPriority.NORMAL,
            supports_incremental=False,
            schedulable=False,
        )
        assert defn.default_interval_hours is None

    def test_allowed_intervals_default_empty(self):
        defn = JobDefinition(
            type_id="x",
            display_name="X",
            description="desc",
            resource=ResourceType.LIGHT,
            default_priority=JobPriority.NORMAL,
            supports_incremental=False,
            schedulable=False,
        )
        assert defn.allowed_intervals == ()

    def test_allowed_intervals(self):
        intervals = ((24, "Every day"), (168, "Every week"))
        defn = JobDefinition(
            type_id="x",
            display_name="X",
            description="desc",
            resource=ResourceType.LIGHT,
            default_priority=JobPriority.NORMAL,
            supports_incremental=False,
            schedulable=True,
            allowed_intervals=intervals,
        )
        assert defn.allowed_intervals == intervals
        assert defn.allowed_intervals[0] == (24, "Every day")


# ---------------------------------------------------------------------------
# JobRecord
# ---------------------------------------------------------------------------


class TestJobRecord:
    def _make_record(self, **overrides):
        defaults = dict(
            id="job-001",
            type="upstream_performer_changes",
            status=JobStatus.PENDING,
            priority=JobPriority.NORMAL,
            cursor=None,
            items_total=0,
            items_processed=0,
            error_message=None,
            triggered_by="manual",
            created_at=datetime.datetime(2026, 2, 18, 12, 0, 0),
            started_at=None,
            completed_at=None,
        )
        defaults.update(overrides)
        return JobRecord(**defaults)

    def test_create(self):
        rec = self._make_record()
        assert rec.id == "job-001"
        assert rec.type == "upstream_performer_changes"
        assert rec.status == JobStatus.PENDING
        assert rec.priority == JobPriority.NORMAL
        assert rec.cursor is None
        assert rec.items_total == 0
        assert rec.items_processed == 0
        assert rec.error_message is None
        assert rec.triggered_by == "manual"
        assert rec.started_at is None
        assert rec.completed_at is None

    def test_to_dict(self):
        now = datetime.datetime(2026, 2, 18, 12, 0, 0)
        rec = self._make_record(created_at=now, started_at=now)
        d = rec.to_dict()
        assert d["id"] == "job-001"
        assert d["type"] == "upstream_performer_changes"
        assert d["status"] == "pending"
        assert d["priority"] == 50
        assert d["cursor"] is None
        assert d["items_total"] == 0
        assert d["items_processed"] == 0
        assert d["error_message"] is None
        assert d["triggered_by"] == "manual"
        assert d["created_at"] == now.isoformat()
        assert d["started_at"] == now.isoformat()
        assert d["completed_at"] is None

    def test_to_dict_none_datetimes(self):
        rec = self._make_record(started_at=None, completed_at=None)
        d = rec.to_dict()
        assert d["started_at"] is None
        assert d["completed_at"] is None

    def test_from_row(self):
        now_str = "2026-02-18T12:00:00"
        row = {
            "id": "job-002",
            "type": "fingerprint_generation",
            "status": "running",
            "priority": 100,
            "cursor": "page:5",
            "items_total": 500,
            "items_processed": 250,
            "error_message": None,
            "triggered_by": "scheduler",
            "created_at": now_str,
            "started_at": now_str,
            "completed_at": None,
        }
        rec = JobRecord.from_row(row)
        assert rec.id == "job-002"
        assert rec.type == "fingerprint_generation"
        assert rec.status == JobStatus.RUNNING
        assert rec.priority == 100
        assert rec.cursor == "page:5"
        assert rec.items_total == 500
        assert rec.items_processed == 250
        assert rec.triggered_by == "scheduler"
        assert rec.created_at == datetime.datetime(2026, 2, 18, 12, 0, 0)
        assert rec.started_at == datetime.datetime(2026, 2, 18, 12, 0, 0)
        assert rec.completed_at is None

    def test_from_row_none_timestamps(self):
        row = {
            "id": "job-003",
            "type": "duplicate_performer",
            "status": "pending",
            "priority": 50,
            "cursor": None,
            "items_total": 0,
            "items_processed": 0,
            "error_message": None,
            "triggered_by": "manual",
            "created_at": "2026-02-18T12:00:00",
            "started_at": None,
            "completed_at": None,
        }
        rec = JobRecord.from_row(row)
        assert rec.started_at is None
        assert rec.completed_at is None

    def test_roundtrip(self):
        """to_dict -> from_row should produce an equivalent record."""
        now = datetime.datetime(2026, 2, 18, 14, 30, 0)
        original = self._make_record(
            status=JobStatus.COMPLETED,
            cursor="cursor:abc",
            items_total=100,
            items_processed=100,
            started_at=now,
            completed_at=now,
        )
        d = original.to_dict()
        restored = JobRecord.from_row(d)
        assert restored.id == original.id
        assert restored.type == original.type
        assert restored.status == original.status
        assert restored.priority == original.priority
        assert restored.cursor == original.cursor
        assert restored.items_total == original.items_total
        assert restored.items_processed == original.items_processed
        assert restored.created_at == original.created_at
        assert restored.started_at == original.started_at
        assert restored.completed_at == original.completed_at

    def test_mutable(self):
        """JobRecord should be mutable (not frozen)."""
        rec = self._make_record()
        rec.status = JobStatus.RUNNING
        assert rec.status == JobStatus.RUNNING


# ---------------------------------------------------------------------------
# JOB_REGISTRY
# ---------------------------------------------------------------------------


class TestIntervalTiers:
    def test_frequent_intervals_ascending(self):
        hours = [h for h, _ in INTERVALS_FREQUENT]
        assert hours == sorted(hours)

    def test_infrequent_intervals_ascending(self):
        hours = [h for h, _ in INTERVALS_INFREQUENT]
        assert hours == sorted(hours)

    def test_frequent_has_expected_options(self):
        hours = {h for h, _ in INTERVALS_FREQUENT}
        assert 6 in hours
        assert 24 in hours
        assert 168 in hours

    def test_infrequent_has_expected_options(self):
        hours = {h for h, _ in INTERVALS_INFREQUENT}
        assert 24 in hours
        assert 168 in hours
        assert 720 in hours

    def test_infrequent_minimum_is_daily(self):
        min_hours = min(h for h, _ in INTERVALS_INFREQUENT)
        assert min_hours >= 24

    def test_all_intervals_have_labels(self):
        for hours, label in INTERVALS_FREQUENT + INTERVALS_INFREQUENT:
            assert isinstance(hours, int)
            assert isinstance(label, str)
            assert len(label) > 0


class TestJobRegistry:
    def test_registered_types(self):
        expected = {
            "duplicate_performer",
            "duplicate_scene_files",
            "duplicate_scenes",
            "upstream_performer_changes",
            "upstream_scene_changes",
            "upstream_tag_changes",
            "upstream_studio_changes",
            "scene_fingerprint_match",
            "fingerprint_generation",
            "database_update",
        }
        assert set(JOB_REGISTRY.keys()) == expected

    def test_duplicate_performer(self):
        d = JOB_REGISTRY["duplicate_performer"]
        assert d.resource == ResourceType.LIGHT
        assert d.default_priority == JobPriority.NORMAL
        assert d.supports_incremental is False
        assert d.schedulable is True
        assert d.default_interval_hours == 168
        assert d.description != ""
        assert d.allowed_intervals == INTERVALS_INFREQUENT

    def test_duplicate_scene_files(self):
        d = JOB_REGISTRY["duplicate_scene_files"]
        assert d.resource == ResourceType.LIGHT
        assert d.default_priority == JobPriority.NORMAL
        assert d.schedulable is True
        assert d.default_interval_hours == 168
        assert d.allowed_intervals == INTERVALS_INFREQUENT

    def test_duplicate_scenes(self):
        d = JOB_REGISTRY["duplicate_scenes"]
        assert d.resource == ResourceType.LIGHT
        assert d.default_priority == JobPriority.NORMAL
        assert d.schedulable is True
        assert d.default_interval_hours == 168
        assert d.allowed_intervals == INTERVALS_INFREQUENT

    def test_upstream_performer_changes(self):
        d = JOB_REGISTRY["upstream_performer_changes"]
        assert d.resource == ResourceType.NETWORK
        assert d.default_priority == JobPriority.NORMAL
        assert d.supports_incremental is True
        assert d.schedulable is True
        assert d.default_interval_hours == 24
        assert d.allowed_intervals == INTERVALS_FREQUENT

    def test_fingerprint_generation(self):
        d = JOB_REGISTRY["fingerprint_generation"]
        assert d.resource == ResourceType.GPU
        assert d.default_priority == JobPriority.LOW
        assert d.supports_incremental is True
        assert d.schedulable is True
        assert d.default_interval_hours is None
        assert d.allowed_intervals == INTERVALS_INFREQUENT

    def test_database_update(self):
        d = JOB_REGISTRY["database_update"]
        assert d.resource == ResourceType.LIGHT
        assert d.default_priority == JobPriority.HIGH
        assert d.supports_incremental is False
        assert d.schedulable is True
        assert d.default_interval_hours == 24
        assert d.allowed_intervals == INTERVALS_FREQUENT

    def test_all_definitions_frozen(self):
        for type_id, defn in JOB_REGISTRY.items():
            with pytest.raises(FrozenInstanceError):
                defn.type_id = "hacked"

    def test_type_id_matches_key(self):
        for key, defn in JOB_REGISTRY.items():
            assert key == defn.type_id, f"Registry key '{key}' != definition type_id '{defn.type_id}'"

    def test_all_schedulable_have_description(self):
        for defn in JOB_REGISTRY.values():
            if defn.schedulable:
                assert defn.description, f"{defn.type_id} is schedulable but has no description"

    def test_all_schedulable_have_allowed_intervals(self):
        for defn in JOB_REGISTRY.values():
            if defn.schedulable:
                assert len(defn.allowed_intervals) > 0, (
                    f"{defn.type_id} is schedulable but has no allowed_intervals"
                )

    def test_default_interval_in_allowed(self):
        """Each job's default interval should be within its allowed options."""
        for defn in JOB_REGISTRY.values():
            if defn.default_interval_hours and defn.allowed_intervals:
                allowed_hours = {h for h, _ in defn.allowed_intervals}
                assert defn.default_interval_hours in allowed_hours, (
                    f"{defn.type_id} default_interval_hours={defn.default_interval_hours} "
                    f"not in allowed_intervals"
                )

"""Job model definitions and type registry for the operation queue system."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Optional


class ResourceType(str, Enum):
    """Resource category a job consumes."""

    GPU = "gpu"
    CPU_HEAVY = "cpu"
    NETWORK = "network"
    LIGHT = "light"


class JobPriority(IntEnum):
    """Job priority levels. Lower numeric value = higher priority."""

    CRITICAL = 0
    HIGH = 10
    NORMAL = 50
    LOW = 100


class JobStatus(str, Enum):
    """Lifecycle status of a job."""

    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class JobDefinition:
    """Immutable definition of a job type registered in the system."""

    type_id: str
    display_name: str
    resource: ResourceType
    default_priority: JobPriority
    supports_incremental: bool
    schedulable: bool
    default_interval_hours: Optional[int] = None


@dataclass
class JobRecord:
    """Mutable record representing a single job instance."""

    id: str
    type: str
    status: JobStatus
    priority: int
    cursor: Optional[str]
    items_total: int
    items_processed: int
    error_message: Optional[str]
    triggered_by: str
    created_at: datetime.datetime
    started_at: Optional[datetime.datetime]
    completed_at: Optional[datetime.datetime]

    def to_dict(self) -> dict:
        """Serialize the record to a plain dict (suitable for JSON / DB storage)."""
        return {
            "id": self.id,
            "type": self.type,
            "status": self.status.value if isinstance(self.status, JobStatus) else self.status,
            "priority": int(self.priority),
            "cursor": self.cursor,
            "items_total": self.items_total,
            "items_processed": self.items_processed,
            "error_message": self.error_message,
            "triggered_by": self.triggered_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_row(cls, row: dict) -> JobRecord:
        """Construct a JobRecord from a database row or dict."""

        def _parse_dt(val: Optional[str]) -> Optional[datetime.datetime]:
            if val is None:
                return None
            return datetime.datetime.fromisoformat(val)

        return cls(
            id=row["id"],
            type=row["type"],
            status=JobStatus(row["status"]),
            priority=row["priority"],
            cursor=row["cursor"],
            items_total=row["items_total"],
            items_processed=row["items_processed"],
            error_message=row["error_message"],
            triggered_by=row["triggered_by"],
            created_at=_parse_dt(row["created_at"]),
            started_at=_parse_dt(row["started_at"]),
            completed_at=_parse_dt(row["completed_at"]),
        )


# ---------------------------------------------------------------------------
# Job type registry
# ---------------------------------------------------------------------------

JOB_REGISTRY: dict[str, JobDefinition] = {}


def _register(
    type_id: str,
    display_name: str,
    resource: ResourceType,
    default_priority: JobPriority,
    supports_incremental: bool = False,
    schedulable: bool = False,
    default_interval_hours: Optional[int] = None,
) -> None:
    """Register a job definition in the global registry."""
    JOB_REGISTRY[type_id] = JobDefinition(
        type_id=type_id,
        display_name=display_name,
        resource=resource,
        default_priority=default_priority,
        supports_incremental=supports_incremental,
        schedulable=schedulable,
        default_interval_hours=default_interval_hours,
    )


_register(
    "duplicate_performer",
    "Duplicate Performer Detection",
    ResourceType.LIGHT,
    JobPriority.NORMAL,
    schedulable=True,
    default_interval_hours=168,
)

_register(
    "duplicate_scene_files",
    "Duplicate Scene File Detection",
    ResourceType.LIGHT,
    JobPriority.NORMAL,
    schedulable=True,
    default_interval_hours=168,
)

_register(
    "duplicate_scenes",
    "Duplicate Scene Detection",
    ResourceType.LIGHT,
    JobPriority.NORMAL,
    schedulable=True,
    default_interval_hours=168,
)

_register(
    "upstream_performer_changes",
    "Upstream Performer Change Detection",
    ResourceType.NETWORK,
    JobPriority.NORMAL,
    supports_incremental=True,
    schedulable=True,
    default_interval_hours=24,
)

_register(
    "fingerprint_generation",
    "Fingerprint Generation",
    ResourceType.GPU,
    JobPriority.LOW,
    supports_incremental=True,
    schedulable=True,
)

_register(
    "database_update",
    "Database Update",
    ResourceType.LIGHT,
    JobPriority.HIGH,
    schedulable=True,
    default_interval_hours=24,
)

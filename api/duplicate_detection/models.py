"""Data models for duplicate scene detection."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FaceAppearance:
    """Appearance metrics for a performer in a scene."""

    performer_id: str
    face_count: int
    avg_confidence: float
    proportion: float  # face_count / total_faces_detected


@dataclass
class SceneFingerprint:
    """Face-based fingerprint for a scene."""

    stash_scene_id: int
    faces: dict[str, FaceAppearance]  # performer_id -> metrics
    total_faces_detected: int
    frames_analyzed: int
    fingerprint_status: str = "complete"


@dataclass
class StashID:
    """A stash-box ID reference."""

    endpoint: str
    stash_id: str


@dataclass
class SceneMetadata:
    """Metadata for a scene from Stash."""

    scene_id: str
    title: Optional[str] = None
    date: Optional[str] = None
    studio_id: Optional[str] = None
    studio_name: Optional[str] = None
    performer_ids: set[str] = field(default_factory=set)
    duration_seconds: Optional[float] = None
    stash_ids: list[StashID] = field(default_factory=list)

    @classmethod
    def from_stash(cls, data: dict) -> "SceneMetadata":
        """Create from Stash GraphQL response."""
        performer_ids = set()
        if data.get("performers"):
            performer_ids = {p["id"] for p in data["performers"]}

        stash_ids = []
        if data.get("stash_ids"):
            stash_ids = [
                StashID(endpoint=s["endpoint"], stash_id=s["stash_id"])
                for s in data["stash_ids"]
            ]

        duration = None
        if data.get("files") and len(data["files"]) > 0:
            duration = data["files"][0].get("duration")

        studio_id = None
        studio_name = None
        if data.get("studio"):
            studio_id = data["studio"].get("id")
            studio_name = data["studio"].get("name")

        return cls(
            scene_id=data["id"],
            title=data.get("title"),
            date=data.get("date"),
            studio_id=studio_id,
            studio_name=studio_name,
            performer_ids=performer_ids,
            duration_seconds=duration,
            stash_ids=stash_ids,
        )


@dataclass
class SignalBreakdown:
    """Breakdown of signals contributing to duplicate confidence."""

    stashbox_match: bool
    stashbox_endpoint: Optional[str]
    face_score: float  # 0-85
    face_reasoning: str
    metadata_score: float  # 0-60
    metadata_reasoning: str


@dataclass
class DuplicateMatch:
    """A detected duplicate pair with confidence and reasoning."""

    scene_a_id: int
    scene_b_id: int
    confidence: float  # 0-100
    reasoning: list[str]
    signal_breakdown: SignalBreakdown

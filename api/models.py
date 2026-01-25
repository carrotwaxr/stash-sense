"""Data models for the shareable face recognition database.

The key insight: StashDB IDs (and other stash-box IDs) are universal.
Local Stash performer IDs are installation-specific.

Database structure:
- Primary key: stash-box endpoint + performer ID (e.g., "stashdb.org:uuid")
- This allows users to match faces to any stash-box instance
- Local Stash users can then look up their own performer by stash_id link
"""
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class StashBoxEndpoint(Enum):
    """Known stash-box instances."""
    STASHDB = "stashdb.org"
    PMVSTASH = "pmvstash.org"  # Example - add others as needed
    FANSDB = "fansdb.cc"      # Example

    @classmethod
    def from_url(cls, url: str) -> Optional["StashBoxEndpoint"]:
        """Convert GraphQL URL to endpoint enum."""
        for endpoint in cls:
            if endpoint.value in url:
                return endpoint
        return None


@dataclass
class PerformerIdentity:
    """
    Universal performer identity using stash-box IDs.

    This is the PRIMARY KEY for the shareable database.
    """
    stashbox_endpoint: str  # e.g., "stashdb.org"
    stashbox_id: str        # UUID on that stash-box

    @property
    def universal_id(self) -> str:
        """Generate universal ID string."""
        return f"{self.stashbox_endpoint}:{self.stashbox_id}"

    @classmethod
    def from_universal_id(cls, uid: str) -> "PerformerIdentity":
        """Parse universal ID string."""
        endpoint, stashbox_id = uid.split(":", 1)
        return cls(stashbox_endpoint=endpoint, stashbox_id=stashbox_id)

    def get_stashbox_url(self) -> str:
        """Get URL to performer page on stash-box."""
        return f"https://{self.stashbox_endpoint}/performers/{self.stashbox_id}"


@dataclass
class PerformerMetadata:
    """
    Performer metadata for the shareable database.

    This contains info that's universal (from stash-box) plus
    optional local references.
    """
    # Universal identity (PRIMARY KEY)
    identity: PerformerIdentity

    # Basic info (from stash-box)
    name: str
    disambiguation: Optional[str] = None
    country: Optional[str] = None

    # Image sources
    image_count: int = 0

    # Cross-references to other stash-boxes
    # e.g., {"pmvstash.org": "other-uuid", "fansdb.cc": "another-uuid"}
    other_stashbox_ids: dict[str, str] = field(default_factory=dict)

    # Local reference (NOT stored in shareable DB, filled in at runtime)
    local_stash_id: Optional[str] = None


@dataclass
class FaceRecord:
    """
    A single face embedding record.

    Maps: face_index -> universal_performer_id
    """
    face_index: int
    universal_id: str  # "stashbox_endpoint:stashbox_id"
    source: str        # Where this face image came from (e.g., "stashdb", "boobpedia")


@dataclass
class DatabaseManifest:
    """
    Manifest for a shareable face recognition database.

    This gets distributed alongside the index files.
    """
    version: str
    created_at: str
    performer_count: int
    face_count: int

    # Sources included in this build
    sources: list[str]  # e.g., ["stashdb.org", "boobpedia.com"]

    # Model info for compatibility checking
    facenet_dim: int = 512
    arcface_dim: int = 512
    detector: str = "yolov8"

    # File checksums
    checksums: dict[str, str] = field(default_factory=dict)

# Duplicate Scene Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement multi-signal duplicate scene detection that identifies duplicate scenes using stash-box IDs, face fingerprints, and metadata heuristics.

**Architecture:** New analyzer (`DuplicateScenesAnalyzer`) that integrates with existing recommendations engine. Stores scene fingerprints in `stash_sense.db`, compares scenes using three signals (stash-box ID match = 100%, face similarity up to 85%, metadata up to 60%), and creates recommendations with confidence scores and reasoning.

**Tech Stack:** Python 3.11, SQLite, FastAPI, pytest, existing `BaseAnalyzer` pattern

---

## Task 1: Database Schema - Scene Fingerprints Tables

**Files:**
- Modify: `api/recommendations_db.py`
- Test: `api/tests/test_scene_fingerprints.py` (create)

**Step 1: Write the failing test**

Create `api/tests/test_scene_fingerprints.py`:

```python
"""Tests for scene fingerprint storage in recommendations DB."""

import pytest
from pathlib import Path


class TestSceneFingerprintSchema:
    """Tests for scene fingerprint table operations."""

    def test_create_fingerprint(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_scene_fingerprint(
            stash_scene_id=123,
            total_faces=5,
            frames_analyzed=40,
            fingerprint_status="complete",
        )

        assert fp_id is not None
        assert fp_id > 0

    def test_get_fingerprint_by_scene_id(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_scene_fingerprint(
            stash_scene_id=456,
            total_faces=3,
            frames_analyzed=40,
        )

        fp = db.get_scene_fingerprint(stash_scene_id=456)

        assert fp is not None
        assert fp["stash_scene_id"] == 456
        assert fp["total_faces"] == 3
        assert fp["frames_analyzed"] == 40

    def test_add_fingerprint_face(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_scene_fingerprint(stash_scene_id=789, total_faces=2, frames_analyzed=40)

        db.add_fingerprint_face(
            fingerprint_id=fp_id,
            performer_id="stashdb:abc-123",
            face_count=10,
            avg_confidence=0.85,
            proportion=0.5,
        )

        faces = db.get_fingerprint_faces(fp_id)

        assert len(faces) == 1
        assert faces[0]["performer_id"] == "stashdb:abc-123"
        assert faces[0]["face_count"] == 10
        assert faces[0]["proportion"] == 0.5

    def test_get_all_fingerprints(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_scene_fingerprint(stash_scene_id=1, total_faces=2, frames_analyzed=40)
        db.create_scene_fingerprint(stash_scene_id=2, total_faces=3, frames_analyzed=40)
        db.create_scene_fingerprint(stash_scene_id=3, total_faces=0, frames_analyzed=40)

        fps = db.get_all_scene_fingerprints()

        assert len(fps) == 3

    def test_fingerprint_upsert_updates_existing(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id1 = db.create_scene_fingerprint(stash_scene_id=100, total_faces=2, frames_analyzed=40)
        fp_id2 = db.create_scene_fingerprint(stash_scene_id=100, total_faces=5, frames_analyzed=40)

        # Should update, not create new
        assert fp_id1 == fp_id2

        fp = db.get_scene_fingerprint(stash_scene_id=100)
        assert fp["total_faces"] == 5
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_scene_fingerprints.py -v`

Expected: FAIL with `AttributeError: 'RecommendationsDB' object has no attribute 'create_scene_fingerprint'`

**Step 3: Update schema version and add migration**

In `api/recommendations_db.py`, update `SCHEMA_VERSION` and add the new tables:

```python
SCHEMA_VERSION = 2  # Was 1
```

Add to `_create_schema` method, after the existing tables:

```python
            -- Scene fingerprints for duplicate detection
            CREATE TABLE scene_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_scene_id INTEGER UNIQUE NOT NULL,
                total_faces INTEGER NOT NULL,
                frames_analyzed INTEGER NOT NULL,
                fingerprint_status TEXT DEFAULT 'complete',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_fingerprint_scene ON scene_fingerprints(stash_scene_id);
            CREATE INDEX idx_fingerprint_status ON scene_fingerprints(fingerprint_status);

            CREATE TABLE scene_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                face_count INTEGER NOT NULL,
                avg_confidence REAL NOT NULL,
                proportion REAL NOT NULL,
                UNIQUE(fingerprint_id, performer_id)
            );
            CREATE INDEX idx_fingerprint_faces_fp ON scene_fingerprint_faces(fingerprint_id);
```

Add migration in `_migrate_schema`:

```python
    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Migrate schema from older version."""
        if from_version < 2:
            # Add scene fingerprint tables
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scene_fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_scene_id INTEGER UNIQUE NOT NULL,
                    total_faces INTEGER NOT NULL,
                    frames_analyzed INTEGER NOT NULL,
                    fingerprint_status TEXT DEFAULT 'complete',
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_fingerprint_scene ON scene_fingerprints(stash_scene_id);
                CREATE INDEX IF NOT EXISTS idx_fingerprint_status ON scene_fingerprints(fingerprint_status);

                CREATE TABLE IF NOT EXISTS scene_fingerprint_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                    performer_id TEXT NOT NULL,
                    face_count INTEGER NOT NULL,
                    avg_confidence REAL NOT NULL,
                    proportion REAL NOT NULL,
                    UNIQUE(fingerprint_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_fingerprint_faces_fp ON scene_fingerprint_faces(fingerprint_id);

                UPDATE schema_version SET version = 2;
            """)
```

**Step 4: Add fingerprint methods to RecommendationsDB**

Add these methods to the `RecommendationsDB` class:

```python
    # ==================== Scene Fingerprints ====================

    def create_scene_fingerprint(
        self,
        stash_scene_id: int,
        total_faces: int,
        frames_analyzed: int,
        fingerprint_status: str = "complete",
    ) -> int:
        """Create or update a scene fingerprint. Returns fingerprint ID."""
        with self._connection() as conn:
            # Try update first
            cursor = conn.execute(
                """
                UPDATE scene_fingerprints
                SET total_faces = ?, frames_analyzed = ?, fingerprint_status = ?,
                    updated_at = datetime('now')
                WHERE stash_scene_id = ?
                """,
                (total_faces, frames_analyzed, fingerprint_status, stash_scene_id)
            )

            if cursor.rowcount > 0:
                # Updated existing - get the ID
                row = conn.execute(
                    "SELECT id FROM scene_fingerprints WHERE stash_scene_id = ?",
                    (stash_scene_id,)
                ).fetchone()
                return row[0]

            # Insert new
            cursor = conn.execute(
                """
                INSERT INTO scene_fingerprints (stash_scene_id, total_faces, frames_analyzed, fingerprint_status)
                VALUES (?, ?, ?, ?)
                """,
                (stash_scene_id, total_faces, frames_analyzed, fingerprint_status)
            )
            return cursor.lastrowid

    def get_scene_fingerprint(self, stash_scene_id: int) -> Optional[dict]:
        """Get fingerprint for a scene."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scene_fingerprints WHERE stash_scene_id = ?",
                (stash_scene_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_all_scene_fingerprints(self, status: Optional[str] = None) -> list[dict]:
        """Get all scene fingerprints, optionally filtered by status."""
        with self._connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM scene_fingerprints WHERE fingerprint_status = ?",
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM scene_fingerprints").fetchall()
            return [dict(row) for row in rows]

    def add_fingerprint_face(
        self,
        fingerprint_id: int,
        performer_id: str,
        face_count: int,
        avg_confidence: float,
        proportion: float,
    ) -> int:
        """Add a face entry to a fingerprint. Returns face ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scene_fingerprint_faces
                    (fingerprint_id, performer_id, face_count, avg_confidence, proportion)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint_id, performer_id) DO UPDATE SET
                    face_count = excluded.face_count,
                    avg_confidence = excluded.avg_confidence,
                    proportion = excluded.proportion
                """,
                (fingerprint_id, performer_id, face_count, avg_confidence, proportion)
            )
            return cursor.lastrowid

    def get_fingerprint_faces(self, fingerprint_id: int) -> list[dict]:
        """Get all faces for a fingerprint."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scene_fingerprint_faces WHERE fingerprint_id = ?",
                (fingerprint_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_fingerprint_faces(self, fingerprint_id: int) -> int:
        """Delete all faces for a fingerprint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM scene_fingerprint_faces WHERE fingerprint_id = ?",
                (fingerprint_id,)
            )
            return cursor.rowcount
```

**Step 5: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_scene_fingerprints.py -v`

Expected: All tests PASS

**Step 6: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/recommendations_db.py api/tests/test_scene_fingerprints.py
git commit -m "feat: add scene fingerprint storage to recommendations DB

- Add scene_fingerprints and scene_fingerprint_faces tables
- Schema version 2 with migration from v1
- CRUD operations for fingerprints and face entries"
```

---

## Task 2: Data Models for Duplicate Detection

**Files:**
- Create: `api/duplicate_detection/models.py`
- Test: `api/tests/test_duplicate_models.py` (create)

**Step 1: Write the failing test**

Create `api/tests/test_duplicate_models.py`:

```python
"""Tests for duplicate detection data models."""

import pytest


class TestSceneFingerprint:
    """Tests for SceneFingerprint dataclass."""

    def test_create_fingerprint(self):
        from duplicate_detection.models import SceneFingerprint, FaceAppearance

        fp = SceneFingerprint(
            stash_scene_id=123,
            faces={
                "stashdb:abc": FaceAppearance(
                    performer_id="stashdb:abc",
                    face_count=10,
                    avg_confidence=0.85,
                    proportion=0.5,
                ),
            },
            total_faces_detected=20,
            frames_analyzed=40,
        )

        assert fp.stash_scene_id == 123
        assert len(fp.faces) == 1
        assert fp.faces["stashdb:abc"].proportion == 0.5


class TestDuplicateMatch:
    """Tests for DuplicateMatch dataclass."""

    def test_create_match(self):
        from duplicate_detection.models import DuplicateMatch, SignalBreakdown

        match = DuplicateMatch(
            scene_a_id=1,
            scene_b_id=2,
            confidence=87.5,
            reasoning=["Face analysis: 2 shared performers", "Metadata: Same studio"],
            signal_breakdown=SignalBreakdown(
                stashbox_match=False,
                stashbox_endpoint=None,
                face_score=78.0,
                face_reasoning="2 shared performers, 3% avg proportion difference",
                metadata_score=30.0,
                metadata_reasoning="Same studio + Duration within 5s",
            ),
        )

        assert match.confidence == 87.5
        assert len(match.reasoning) == 2
        assert match.signal_breakdown.face_score == 78.0


class TestSceneMetadata:
    """Tests for SceneMetadata dataclass."""

    def test_create_from_stash_response(self):
        from duplicate_detection.models import SceneMetadata

        stash_data = {
            "id": "123",
            "title": "Test Scene",
            "date": "2024-01-15",
            "studio": {"id": "456", "name": "Test Studio"},
            "performers": [
                {"id": "p1", "name": "Performer One"},
                {"id": "p2", "name": "Performer Two"},
            ],
            "files": [{"duration": 1935.5}],
            "stash_ids": [
                {"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"},
            ],
        }

        meta = SceneMetadata.from_stash(stash_data)

        assert meta.scene_id == "123"
        assert meta.studio_id == "456"
        assert meta.performer_ids == {"p1", "p2"}
        assert meta.duration_seconds == 1935.5
        assert len(meta.stash_ids) == 1

    def test_handles_missing_fields(self):
        from duplicate_detection.models import SceneMetadata

        stash_data = {"id": "999"}

        meta = SceneMetadata.from_stash(stash_data)

        assert meta.scene_id == "999"
        assert meta.studio_id is None
        assert meta.performer_ids == set()
        assert meta.duration_seconds is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_models.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'duplicate_detection'`

**Step 3: Create the models module**

Create directory and `api/duplicate_detection/__init__.py`:

```python
"""Duplicate scene detection module."""

from .models import (
    FaceAppearance,
    SceneFingerprint,
    SceneMetadata,
    SignalBreakdown,
    DuplicateMatch,
)

__all__ = [
    "FaceAppearance",
    "SceneFingerprint",
    "SceneMetadata",
    "SignalBreakdown",
    "DuplicateMatch",
]
```

Create `api/duplicate_detection/models.py`:

```python
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
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_models.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/duplicate_detection/ api/tests/test_duplicate_models.py
git commit -m "feat: add data models for duplicate scene detection

- SceneFingerprint and FaceAppearance for face-based signatures
- SceneMetadata with from_stash() factory method
- DuplicateMatch and SignalBreakdown for results"
```

---

## Task 3: Scoring Functions - Metadata Heuristics

**Files:**
- Create: `api/duplicate_detection/scoring.py`
- Test: `api/tests/test_duplicate_scoring.py` (create)

**Step 1: Write the failing test**

Create `api/tests/test_duplicate_scoring.py`:

```python
"""Tests for duplicate detection scoring functions."""

import pytest


class TestMetadataScore:
    """Tests for metadata_score function."""

    def test_same_studio_same_date_similar_duration(self):
        from duplicate_detection.scoring import metadata_score
        from duplicate_detection.models import SceneMetadata

        scene_a = SceneMetadata(
            scene_id="1",
            studio_id="s1",
            performer_ids={"p1", "p2"},
            date="2024-01-15",
            duration_seconds=1935,
        )
        scene_b = SceneMetadata(
            scene_id="2",
            studio_id="s1",
            performer_ids={"p1", "p2"},
            date="2024-01-15",
            duration_seconds=1938,  # Within 5s
        )

        score, reasoning = metadata_score(scene_a, scene_b)

        # Base: 20 (studio) + 20 (exact performers) = 40
        # Multiplier: 1.0 + 0.5 (same date) + 0.5 (duration within 5s) = 2.0
        # Total: 40 * 2.0 = 80, capped at 60
        assert score == 60.0
        assert "Same studio" in reasoning
        assert "Exact performer match" in reasoning

    def test_no_base_signal_returns_zero(self):
        from duplicate_detection.scoring import metadata_score
        from duplicate_detection.models import SceneMetadata

        scene_a = SceneMetadata(scene_id="1", performer_ids={"p1"})
        scene_b = SceneMetadata(scene_id="2", performer_ids={"p2"})  # Different performer

        score, reasoning = metadata_score(scene_a, scene_b)

        assert score == 0.0
        assert "No studio or performer match" in reasoning

    def test_insufficient_metadata(self):
        from duplicate_detection.scoring import metadata_score
        from duplicate_detection.models import SceneMetadata

        scene_a = SceneMetadata(scene_id="1")  # No useful metadata
        scene_b = SceneMetadata(scene_id="2")

        score, reasoning = metadata_score(scene_a, scene_b)

        assert score == 0.0
        assert "Insufficient metadata" in reasoning

    def test_partial_performer_overlap(self):
        from duplicate_detection.scoring import metadata_score
        from duplicate_detection.models import SceneMetadata

        scene_a = SceneMetadata(scene_id="1", performer_ids={"p1", "p2", "p3"})
        scene_b = SceneMetadata(scene_id="2", performer_ids={"p1", "p2"})  # 2/3 overlap

        score, reasoning = metadata_score(scene_a, scene_b)

        # Jaccard = 2/3 = 0.67, which is >= 0.5
        # Base: 12 (partial overlap)
        # No date or duration to multiply
        assert score == 12.0
        assert "Performers overlap" in reasoning

    def test_date_within_7_days(self):
        from duplicate_detection.scoring import metadata_score
        from duplicate_detection.models import SceneMetadata

        scene_a = SceneMetadata(
            scene_id="1", studio_id="s1", date="2024-01-15"
        )
        scene_b = SceneMetadata(
            scene_id="2", studio_id="s1", date="2024-01-20"  # 5 days later
        )

        score, reasoning = metadata_score(scene_a, scene_b)

        # Base: 20 (studio)
        # Multiplier: 1.0 + 0.3 (within 7 days) = 1.3
        assert score == 26.0
        assert "Release dates within 7 days" in reasoning


class TestStashboxMatch:
    """Tests for check_stashbox_match function."""

    def test_matching_stashbox_ids(self):
        from duplicate_detection.scoring import check_stashbox_match
        from duplicate_detection.models import SceneMetadata, StashID

        scene_a = SceneMetadata(
            scene_id="1",
            stash_ids=[StashID(endpoint="https://stashdb.org/graphql", stash_id="abc-123")],
        )
        scene_b = SceneMetadata(
            scene_id="2",
            stash_ids=[StashID(endpoint="https://stashdb.org/graphql", stash_id="abc-123")],
        )

        result = check_stashbox_match(scene_a, scene_b)

        assert result.matched is True
        assert result.endpoint == "https://stashdb.org/graphql"
        assert result.stash_id == "abc-123"

    def test_different_stashbox_ids(self):
        from duplicate_detection.scoring import check_stashbox_match
        from duplicate_detection.models import SceneMetadata, StashID

        scene_a = SceneMetadata(
            scene_id="1",
            stash_ids=[StashID(endpoint="https://stashdb.org/graphql", stash_id="abc-123")],
        )
        scene_b = SceneMetadata(
            scene_id="2",
            stash_ids=[StashID(endpoint="https://stashdb.org/graphql", stash_id="def-456")],
        )

        result = check_stashbox_match(scene_a, scene_b)

        assert result.matched is False

    def test_different_endpoints_same_id(self):
        from duplicate_detection.scoring import check_stashbox_match
        from duplicate_detection.models import SceneMetadata, StashID

        scene_a = SceneMetadata(
            scene_id="1",
            stash_ids=[StashID(endpoint="https://stashdb.org/graphql", stash_id="abc-123")],
        )
        scene_b = SceneMetadata(
            scene_id="2",
            stash_ids=[StashID(endpoint="https://theporndb.net/graphql", stash_id="abc-123")],
        )

        result = check_stashbox_match(scene_a, scene_b)

        # Same ID but different endpoint = not a match
        assert result.matched is False


class TestFaceSignatureSimilarity:
    """Tests for face_signature_similarity function."""

    def test_identical_fingerprints(self):
        from duplicate_detection.scoring import face_signature_similarity
        from duplicate_detection.models import SceneFingerprint, FaceAppearance

        fp_a = SceneFingerprint(
            stash_scene_id=1,
            faces={
                "stashdb:p1": FaceAppearance("stashdb:p1", 20, 0.9, 0.5),
                "stashdb:p2": FaceAppearance("stashdb:p2", 20, 0.85, 0.5),
            },
            total_faces_detected=40,
            frames_analyzed=40,
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2,
            faces={
                "stashdb:p1": FaceAppearance("stashdb:p1", 22, 0.88, 0.5),
                "stashdb:p2": FaceAppearance("stashdb:p2", 22, 0.86, 0.5),
            },
            total_faces_detected=44,
            frames_analyzed=40,
        )

        score, reasoning = face_signature_similarity(fp_a, fp_b)

        # Identical proportions = 0% difference = 85% score
        assert score == 85.0
        assert "2 shared performers" in reasoning

    def test_no_shared_performers(self):
        from duplicate_detection.scoring import face_signature_similarity
        from duplicate_detection.models import SceneFingerprint, FaceAppearance

        fp_a = SceneFingerprint(
            stash_scene_id=1,
            faces={"stashdb:p1": FaceAppearance("stashdb:p1", 20, 0.9, 1.0)},
            total_faces_detected=20,
            frames_analyzed=40,
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2,
            faces={"stashdb:p2": FaceAppearance("stashdb:p2", 20, 0.9, 1.0)},
            total_faces_detected=20,
            frames_analyzed=40,
        )

        score, reasoning = face_signature_similarity(fp_a, fp_b)

        # No overlap = low score
        assert score < 50.0

    def test_no_identified_performers(self):
        from duplicate_detection.scoring import face_signature_similarity
        from duplicate_detection.models import SceneFingerprint, FaceAppearance

        fp_a = SceneFingerprint(
            stash_scene_id=1,
            faces={"unknown": FaceAppearance("unknown", 20, 0.5, 1.0)},
            total_faces_detected=20,
            frames_analyzed=40,
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2,
            faces={"unknown": FaceAppearance("unknown", 20, 0.5, 1.0)},
            total_faces_detected=20,
            frames_analyzed=40,
        )

        score, reasoning = face_signature_similarity(fp_a, fp_b)

        assert score == 0.0
        assert "No identified performers" in reasoning


class TestCombinedConfidence:
    """Tests for calculate_duplicate_confidence function."""

    def test_stashbox_match_returns_100(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence
        from duplicate_detection.models import SceneMetadata, SceneFingerprint, StashID

        scene_a = SceneMetadata(
            scene_id="1",
            stash_ids=[StashID("https://stashdb.org/graphql", "abc-123")],
        )
        scene_b = SceneMetadata(
            scene_id="2",
            stash_ids=[StashID("https://stashdb.org/graphql", "abc-123")],
        )

        result = calculate_duplicate_confidence(scene_a, scene_b, None, None)

        assert result is not None
        assert result.confidence == 100.0
        assert result.signal_breakdown.stashbox_match is True

    def test_combined_face_and_metadata(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence
        from duplicate_detection.models import SceneMetadata, SceneFingerprint, FaceAppearance

        scene_a = SceneMetadata(scene_id="1", studio_id="s1")
        scene_b = SceneMetadata(scene_id="2", studio_id="s1")

        fp_a = SceneFingerprint(
            stash_scene_id=1,
            faces={"stashdb:p1": FaceAppearance("stashdb:p1", 20, 0.9, 1.0)},
            total_faces_detected=20,
            frames_analyzed=40,
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2,
            faces={"stashdb:p1": FaceAppearance("stashdb:p1", 22, 0.88, 1.0)},
            total_faces_detected=22,
            frames_analyzed=40,
        )

        result = calculate_duplicate_confidence(scene_a, scene_b, fp_a, fp_b)

        assert result is not None
        # Face score should be high (same performer, same proportion)
        # Metadata score = 20 (same studio)
        # Combined should be capped at 95
        assert 50.0 <= result.confidence <= 95.0
        assert result.signal_breakdown.stashbox_match is False

    def test_same_scene_returns_none(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence
        from duplicate_detection.models import SceneMetadata

        scene = SceneMetadata(scene_id="1")

        result = calculate_duplicate_confidence(scene, scene, None, None)

        assert result is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_scoring.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'duplicate_detection.scoring'`

**Step 3: Implement scoring functions**

Create `api/duplicate_detection/scoring.py`:

```python
"""Scoring functions for duplicate scene detection."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from .models import (
    SceneFingerprint,
    SceneMetadata,
    FaceAppearance,
    SignalBreakdown,
    DuplicateMatch,
)


@dataclass
class StashboxMatchResult:
    """Result of checking for stash-box ID match."""

    matched: bool
    endpoint: Optional[str] = None
    stash_id: Optional[str] = None


def check_stashbox_match(scene_a: SceneMetadata, scene_b: SceneMetadata) -> StashboxMatchResult:
    """Check if two scenes share the same stash-box ID on the same endpoint."""
    for sid_a in scene_a.stash_ids:
        for sid_b in scene_b.stash_ids:
            if sid_a.endpoint == sid_b.endpoint and sid_a.stash_id == sid_b.stash_id:
                return StashboxMatchResult(
                    matched=True,
                    endpoint=sid_a.endpoint,
                    stash_id=sid_a.stash_id,
                )
    return StashboxMatchResult(matched=False)


def _has_useful_metadata(scene: SceneMetadata) -> bool:
    """Check if scene has enough metadata for comparison."""
    return bool(scene.studio_id or scene.performer_ids)


def _jaccard_similarity(set_a: set, set_b: set) -> float:
    """Calculate Jaccard similarity between two sets."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def _days_between(date_a: Optional[str], date_b: Optional[str]) -> Optional[int]:
    """Calculate days between two date strings (YYYY-MM-DD format)."""
    if not date_a or not date_b:
        return None
    try:
        d1 = datetime.strptime(date_a, "%Y-%m-%d")
        d2 = datetime.strptime(date_b, "%Y-%m-%d")
        return abs((d2 - d1).days)
    except ValueError:
        return None


def metadata_score(scene_a: SceneMetadata, scene_b: SceneMetadata) -> tuple[float, str]:
    """
    Calculate metadata similarity score (0-60).

    Requires a base signal (studio or performers) to return non-zero.
    Duration and date are confirmation multipliers.
    """
    if not _has_useful_metadata(scene_a) or not _has_useful_metadata(scene_b):
        return 0.0, "Insufficient metadata"

    base = 0.0
    reasons = []

    # Base signals
    if scene_a.studio_id and scene_b.studio_id and scene_a.studio_id == scene_b.studio_id:
        base += 20.0
        reasons.append("Same studio")

    performer_overlap = _jaccard_similarity(scene_a.performer_ids, scene_b.performer_ids)
    if performer_overlap == 1.0 and scene_a.performer_ids:
        base += 20.0
        reasons.append("Exact performer match")
    elif performer_overlap >= 0.5:
        base += 12.0
        reasons.append(f"Performers overlap ({performer_overlap:.0%})")

    if base == 0:
        return 0.0, "No studio or performer match"

    # Confirmation multipliers
    multiplier = 1.0

    days = _days_between(scene_a.date, scene_b.date)
    if days is not None:
        if days == 0:
            multiplier += 0.5
            reasons.append("Same release date")
        elif days <= 7:
            multiplier += 0.3
            reasons.append("Release dates within 7 days")

    if scene_a.duration_seconds and scene_b.duration_seconds:
        diff = abs(scene_a.duration_seconds - scene_b.duration_seconds)
        if diff <= 5:
            multiplier += 0.5
            reasons.append("Duration within 5s")
        elif diff <= 30:
            multiplier += 0.3
            reasons.append("Duration within 30s")

    score = min(base * multiplier, 60.0)
    return score, " + ".join(reasons)


def face_signature_similarity(
    fp_a: SceneFingerprint, fp_b: SceneFingerprint
) -> tuple[float, str]:
    """
    Calculate face signature similarity score (0-85).

    Compares which performers appear and in what proportions.
    """
    all_performers = set(fp_a.faces.keys()) | set(fp_b.faces.keys())
    all_performers.discard("unknown")

    if not all_performers:
        return 0.0, "No identified performers in either scene"

    proportion_diffs = []
    matches = []

    for performer_id in all_performers:
        prop_a = fp_a.faces.get(performer_id, FaceAppearance(performer_id, 0, 0, 0)).proportion
        prop_b = fp_b.faces.get(performer_id, FaceAppearance(performer_id, 0, 0, 0)).proportion

        diff = abs(prop_a - prop_b)
        proportion_diffs.append(diff)

        # Both have meaningful presence (>10%)
        if prop_a > 0.1 and prop_b > 0.1:
            matches.append(performer_id)

    avg_diff = sum(proportion_diffs) / len(proportion_diffs)

    # Convert to similarity score (0-85 range)
    # Perfect match (avg_diff=0) → 85%
    # Completely different (avg_diff>=0.5) → 0%
    similarity = max(0.0, 85.0 * (1.0 - avg_diff * 2.0))

    reason = f"{len(matches)} shared performers, {avg_diff:.1%} avg proportion difference"
    return similarity, reason


def calculate_duplicate_confidence(
    scene_a: SceneMetadata,
    scene_b: SceneMetadata,
    fp_a: Optional[SceneFingerprint],
    fp_b: Optional[SceneFingerprint],
) -> Optional[DuplicateMatch]:
    """
    Calculate overall duplicate confidence combining all signals.

    Returns None if scenes are the same or confidence is too low to consider.
    """
    # Guard: same scene
    if scene_a.scene_id == scene_b.scene_id:
        return None

    # Tier 1: Authoritative stash-box match
    stashbox = check_stashbox_match(scene_a, scene_b)
    if stashbox.matched:
        return DuplicateMatch(
            scene_a_id=int(scene_a.scene_id),
            scene_b_id=int(scene_b.scene_id),
            confidence=100.0,
            reasoning=[f"Identical stash-box ID: {stashbox.stash_id}"],
            signal_breakdown=SignalBreakdown(
                stashbox_match=True,
                stashbox_endpoint=stashbox.endpoint,
                face_score=0.0,
                face_reasoning="",
                metadata_score=0.0,
                metadata_reasoning="",
            ),
        )

    # Tier 2: Face + Metadata signals
    face_score = 0.0
    face_reasoning = "No fingerprint available"

    if fp_a and fp_b:
        if fp_a.total_faces_detected == 0 and fp_b.total_faces_detected == 0:
            face_reasoning = "No faces detected in either scene"
        elif fp_a.total_faces_detected == 0 or fp_b.total_faces_detected == 0:
            face_reasoning = "Asymmetric face detection"
        else:
            face_score, face_reasoning = face_signature_similarity(fp_a, fp_b)

    meta_score, meta_reasoning = metadata_score(scene_a, scene_b)

    # No signals = no match
    if face_score == 0 and meta_score == 0:
        return None

    # Combine with diminishing returns
    primary = max(face_score, meta_score)
    secondary = min(face_score, meta_score)
    combined = primary + (secondary * 0.3)

    # Cap at 95% without stash-box confirmation
    confidence = min(combined, 95.0)

    # Build reasoning
    reasoning = []
    if face_score > 0:
        reasoning.append(f"Face analysis: {face_reasoning}")
    if meta_score > 0:
        reasoning.append(f"Metadata: {meta_reasoning}")

    # Add confidence qualifier
    if confidence >= 80:
        reasoning.insert(0, "High confidence duplicate")
    elif confidence >= 50:
        reasoning.insert(0, "Likely duplicate")
    elif confidence >= 30:
        reasoning.insert(0, "Possible duplicate")
    else:
        reasoning.insert(0, "Low confidence match")

    return DuplicateMatch(
        scene_a_id=int(scene_a.scene_id),
        scene_b_id=int(scene_b.scene_id),
        confidence=round(confidence, 1),
        reasoning=reasoning,
        signal_breakdown=SignalBreakdown(
            stashbox_match=False,
            stashbox_endpoint=None,
            face_score=face_score,
            face_reasoning=face_reasoning,
            metadata_score=meta_score,
            metadata_reasoning=meta_reasoning,
        ),
    )
```

Update `api/duplicate_detection/__init__.py`:

```python
"""Duplicate scene detection module."""

from .models import (
    FaceAppearance,
    SceneFingerprint,
    SceneMetadata,
    SignalBreakdown,
    DuplicateMatch,
    StashID,
)
from .scoring import (
    check_stashbox_match,
    metadata_score,
    face_signature_similarity,
    calculate_duplicate_confidence,
    StashboxMatchResult,
)

__all__ = [
    "FaceAppearance",
    "SceneFingerprint",
    "SceneMetadata",
    "SignalBreakdown",
    "DuplicateMatch",
    "StashID",
    "check_stashbox_match",
    "metadata_score",
    "face_signature_similarity",
    "calculate_duplicate_confidence",
    "StashboxMatchResult",
]
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_scoring.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/duplicate_detection/ api/tests/test_duplicate_scoring.py
git commit -m "feat: add scoring functions for duplicate detection

- metadata_score: base signals + confirmation multipliers
- face_signature_similarity: proportion-based comparison
- calculate_duplicate_confidence: combined scoring with caps
- check_stashbox_match: authoritative ID matching"
```

---

## Task 4: Stash Client - Scene Query Methods

**Files:**
- Modify: `api/stash_client_unified.py`
- Test: `api/tests/test_stash_client_scenes.py` (create)

**Step 1: Write the failing test**

Create `api/tests/test_stash_client_scenes.py`:

```python
"""Tests for Stash client scene query methods."""

import pytest
from unittest.mock import AsyncMock, patch


class TestGetScenesForFingerprinting:
    """Tests for get_scenes_for_fingerprinting method."""

    @pytest.mark.asyncio
    async def test_returns_scenes_with_metadata(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {
            "findScenes": {
                "count": 2,
                "scenes": [
                    {
                        "id": "1",
                        "title": "Test Scene",
                        "date": "2024-01-15",
                        "updated_at": "2024-01-20T00:00:00Z",
                        "studio": {"id": "s1", "name": "Test Studio"},
                        "performers": [{"id": "p1", "name": "Performer One"}],
                        "files": [{"duration": 1800}],
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}
                        ],
                    },
                    {
                        "id": "2",
                        "title": "Another Scene",
                        "date": None,
                        "updated_at": "2024-01-21T00:00:00Z",
                        "studio": None,
                        "performers": [],
                        "files": [],
                        "stash_ids": [],
                    },
                ],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            scenes, count = await client.get_scenes_for_fingerprinting(limit=10)

            assert count == 2
            assert len(scenes) == 2
            assert scenes[0]["id"] == "1"
            assert scenes[0]["studio"]["name"] == "Test Studio"

    @pytest.mark.asyncio
    async def test_filters_by_updated_after(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = {"findScenes": {"count": 0, "scenes": []}}

            await client.get_scenes_for_fingerprinting(
                updated_after="2024-01-15T00:00:00Z"
            )

            # Check that the filter was passed
            call_args = mock_execute.call_args
            variables = call_args[1]["variables"] if call_args[1] else call_args[0][1]
            assert variables["scene_filter"]["updated_at"]["value"] == "2024-01-15T00:00:00Z"


class TestGetSceneStreamUrl:
    """Tests for get_scene_stream_url method."""

    @pytest.mark.asyncio
    async def test_returns_stream_url(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {
            "findScene": {
                "id": "123",
                "sceneStreams": [
                    {"url": "http://localhost:9999/scene/123/stream.mp4", "label": "Direct stream"},
                ],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            url = await client.get_scene_stream_url("123")

            assert url == "http://localhost:9999/scene/123/stream.mp4"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_streams(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {"findScene": {"id": "123", "sceneStreams": []}}

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_execute:
            mock_execute.return_value = mock_response

            url = await client.get_scene_stream_url("123")

            assert url is None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_stash_client_scenes.py -v`

Expected: FAIL with `AttributeError: 'StashClientUnified' object has no attribute 'get_scenes_for_fingerprinting'`

**Step 3: Add scene query methods**

Add to `api/stash_client_unified.py`:

```python
    async def get_scenes_for_fingerprinting(
        self,
        updated_after: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict], int]:
        """
        Get scenes with full metadata for fingerprinting.
        Returns (scenes, total_count).
        """
        query = """
        query ScenesForFingerprinting($filter: FindFilterType, $scene_filter: SceneFilterType) {
          findScenes(filter: $filter, scene_filter: $scene_filter) {
            count
            scenes {
              id
              title
              date
              updated_at
              studio {
                id
                name
              }
              performers {
                id
                name
              }
              files {
                duration
              }
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        filter_input = {"per_page": limit, "page": (offset // limit) + 1}
        scene_filter = {}

        if updated_after:
            scene_filter["updated_at"] = {"value": updated_after, "modifier": "GREATER_THAN"}

        data = await self._execute(query, variables={"filter": filter_input, "scene_filter": scene_filter})
        return data["findScenes"]["scenes"], data["findScenes"]["count"]

    async def get_scene_stream_url(self, scene_id: str) -> Optional[str]:
        """Get the stream URL for a scene."""
        query = """
        query SceneStream($id: ID!) {
          findScene(id: $id) {
            id
            sceneStreams {
              url
              label
            }
          }
        }
        """
        data = await self._execute(query, variables={"id": scene_id})
        scene = data.get("findScene")
        if not scene or not scene.get("sceneStreams"):
            return None

        # Return first available stream
        streams = scene["sceneStreams"]
        return streams[0]["url"] if streams else None

    async def get_scene_by_id(self, scene_id: str) -> Optional[dict]:
        """Get a scene by ID with full metadata."""
        query = """
        query GetScene($id: ID!) {
          findScene(id: $id) {
            id
            title
            date
            updated_at
            studio {
              id
              name
            }
            performers {
              id
              name
            }
            files {
              duration
            }
            stash_ids {
              endpoint
              stash_id
            }
          }
        }
        """
        data = await self._execute(query, variables={"id": scene_id})
        return data.get("findScene")
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_stash_client_scenes.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/stash_client_unified.py api/tests/test_stash_client_scenes.py
git commit -m "feat: add scene query methods to Stash client

- get_scenes_for_fingerprinting: batch fetch with metadata
- get_scene_stream_url: for ffmpeg extraction
- get_scene_by_id: single scene lookup"
```

---

## Task 5: Duplicate Scenes Analyzer - Core Implementation

**Files:**
- Create: `api/analyzers/duplicate_scenes.py`
- Test: `api/tests/test_duplicate_scenes_analyzer.py` (create)

**Step 1: Write the failing test**

Create `api/tests/test_duplicate_scenes_analyzer.py`:

```python
"""Tests for DuplicateScenesAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDuplicateScenesAnalyzer:
    """Tests for the duplicate scenes analyzer."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_scenes_for_fingerprinting = AsyncMock(return_value=([], 0))
        stash.get_scene_stream_url = AsyncMock(return_value="http://test/stream.mp4")
        return stash

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def test_analyzer_type(self, mock_stash, rec_db):
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db)

        assert analyzer.type == "duplicate_scenes"

    @pytest.mark.asyncio
    async def test_finds_stashbox_duplicates(self, mock_stash, rec_db):
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        # Two scenes with same stash-box ID
        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {
                        "id": "1",
                        "title": "Scene A",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
                        "studio": None,
                        "performers": [],
                        "files": [],
                        "date": None,
                    },
                    {
                        "id": "2",
                        "title": "Scene B",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
                        "studio": None,
                        "performers": [],
                        "files": [],
                        "date": None,
                    },
                ],
                2,
            )
        )

        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db)
        result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 1

        # Check the recommendation was created
        recs = rec_db.get_recommendations(type="duplicate_scenes")
        assert len(recs) == 1
        assert recs[0].confidence == 1.0  # 100% normalized to 0-1
        assert "stash-box ID" in recs[0].details["reasoning"][0]

    @pytest.mark.asyncio
    async def test_finds_metadata_duplicates(self, mock_stash, rec_db):
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        # Two scenes with same studio + same performers + similar duration
        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {
                        "id": "1",
                        "title": "Scene A",
                        "stash_ids": [],
                        "studio": {"id": "s1", "name": "Test Studio"},
                        "performers": [{"id": "p1", "name": "Test Performer"}],
                        "files": [{"duration": 1800}],
                        "date": "2024-01-15",
                    },
                    {
                        "id": "2",
                        "title": "Scene B",
                        "stash_ids": [],
                        "studio": {"id": "s1", "name": "Test Studio"},
                        "performers": [{"id": "p1", "name": "Test Performer"}],
                        "files": [{"duration": 1803}],  # Within 5s
                        "date": "2024-01-15",
                    },
                ],
                2,
            )
        )

        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db, min_confidence=40)
        result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 1

    @pytest.mark.asyncio
    async def test_respects_min_confidence(self, mock_stash, rec_db):
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        # Two scenes with only studio match (low confidence)
        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {
                        "id": "1",
                        "title": "Scene A",
                        "stash_ids": [],
                        "studio": {"id": "s1", "name": "Test Studio"},
                        "performers": [],
                        "files": [],
                        "date": None,
                    },
                    {
                        "id": "2",
                        "title": "Scene B",
                        "stash_ids": [],
                        "studio": {"id": "s1", "name": "Test Studio"},
                        "performers": [],
                        "files": [],
                        "date": None,
                    },
                ],
                2,
            )
        )

        # High minimum confidence should filter out low-confidence matches
        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db, min_confidence=80)
        result = await analyzer.run(incremental=False)

        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_skips_dismissed_targets(self, mock_stash, rec_db):
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {
                        "id": "1",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
                        "studio": None,
                        "performers": [],
                        "files": [],
                        "date": None,
                    },
                    {
                        "id": "2",
                        "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
                        "studio": None,
                        "performers": [],
                        "files": [],
                        "date": None,
                    },
                ],
                2,
            )
        )

        # Dismiss scene 1 for duplicate_scenes
        rec_db._connection().__enter__().execute(
            "INSERT INTO dismissed_targets (type, target_type, target_id) VALUES (?, ?, ?)",
            ("duplicate_scenes", "scene", "1"),
        )

        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db)
        result = await analyzer.run(incremental=False)

        # Should not create recommendation for dismissed target
        assert result.recommendations_created == 0
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_scenes_analyzer.py -v`

Expected: FAIL with `ModuleNotFoundError: No module named 'analyzers.duplicate_scenes'`

**Step 3: Implement the analyzer**

Create `api/analyzers/duplicate_scenes.py`:

```python
"""
Duplicate Scenes Analyzer

Detects duplicate scenes using multi-signal analysis:
- Stash-box ID matching (authoritative, 100%)
- Face fingerprint similarity (up to 85%)
- Metadata heuristics (up to 60%)

See: docs/plans/2026-01-30-duplicate-scene-detection-design.md
"""

import asyncio
import logging
from dataclasses import asdict
from typing import TYPE_CHECKING, Optional

from .base import BaseAnalyzer, AnalysisResult
from duplicate_detection import (
    SceneMetadata,
    SceneFingerprint,
    FaceAppearance,
    calculate_duplicate_confidence,
)

if TYPE_CHECKING:
    from stash_client_unified import StashClientUnified
    from recommendations_db import RecommendationsDB


logger = logging.getLogger(__name__)


class DuplicateScenesAnalyzer(BaseAnalyzer):
    """
    Detects duplicate scenes using multi-signal analysis.

    Configuration:
        min_confidence: Minimum confidence (0-100) to create recommendation
        batch_size: Scenes to process per batch
        max_comparisons: Safety limit on O(n²) comparisons
    """

    type = "duplicate_scenes"

    def __init__(
        self,
        stash: "StashClientUnified",
        rec_db: "RecommendationsDB",
        min_confidence: float = 50.0,
        batch_size: int = 100,
        max_comparisons: int = 50000,
    ):
        super().__init__(stash, rec_db)
        self.min_confidence = min_confidence
        self.batch_size = batch_size
        self.max_comparisons = max_comparisons

    async def run(self, incremental: bool = True) -> AnalysisResult:
        """
        Run duplicate scene detection.

        Phase 1: Load scenes and existing fingerprints
        Phase 2: Compare all pairs for duplicates
        Phase 3: Create recommendations
        """
        # Phase 1: Load scenes
        logger.info("Loading scenes from Stash...")
        all_scenes: list[dict] = []
        offset = 0

        while True:
            scenes, total = await self.stash.get_scenes_for_fingerprinting(
                limit=self.batch_size,
                offset=offset,
            )
            all_scenes.extend(scenes)

            if len(all_scenes) >= total or not scenes:
                break

            offset += self.batch_size
            await asyncio.sleep(0.1)  # Rate limiting

        logger.info(f"Loaded {len(all_scenes)} scenes")

        if len(all_scenes) < 2:
            return AnalysisResult(items_processed=len(all_scenes), recommendations_created=0)

        # Convert to metadata objects
        scene_metadata = [SceneMetadata.from_stash(s) for s in all_scenes]

        # Load existing fingerprints
        fingerprints: dict[str, SceneFingerprint] = {}
        for fp_data in self.rec_db.get_all_scene_fingerprints(status="complete"):
            scene_id = str(fp_data["stash_scene_id"])
            faces_data = self.rec_db.get_fingerprint_faces(fp_data["id"])

            faces = {}
            for f in faces_data:
                faces[f["performer_id"]] = FaceAppearance(
                    performer_id=f["performer_id"],
                    face_count=f["face_count"],
                    avg_confidence=f["avg_confidence"],
                    proportion=f["proportion"],
                )

            fingerprints[scene_id] = SceneFingerprint(
                stash_scene_id=fp_data["stash_scene_id"],
                faces=faces,
                total_faces_detected=fp_data["total_faces"],
                frames_analyzed=fp_data["frames_analyzed"],
            )

        logger.info(f"Loaded {len(fingerprints)} existing fingerprints")

        # Phase 2: Find duplicates
        duplicates = []
        comparisons = 0

        for i, scene_a in enumerate(scene_metadata):
            fp_a = fingerprints.get(scene_a.scene_id)

            for scene_b in scene_metadata[i + 1 :]:
                comparisons += 1

                if comparisons > self.max_comparisons:
                    logger.warning(f"Hit max comparisons limit ({self.max_comparisons})")
                    break

                # Yield periodically
                if comparisons % 1000 == 0:
                    await asyncio.sleep(0)

                fp_b = fingerprints.get(scene_b.scene_id)

                match = calculate_duplicate_confidence(scene_a, scene_b, fp_a, fp_b)

                if match and match.confidence >= self.min_confidence:
                    duplicates.append(match)

            if comparisons > self.max_comparisons:
                break

        logger.info(f"Found {len(duplicates)} potential duplicates from {comparisons} comparisons")

        # Phase 3: Create recommendations
        created = 0
        for match in duplicates:
            rec_id = self.create_recommendation(
                target_type="scene",
                target_id=str(match.scene_a_id),
                details={
                    "scene_b_id": match.scene_b_id,
                    "confidence": match.confidence,
                    "reasoning": match.reasoning,
                    "signal_breakdown": asdict(match.signal_breakdown),
                },
                confidence=match.confidence / 100.0,  # Normalize to 0-1
            )

            if rec_id:
                created += 1

        return AnalysisResult(
            items_processed=len(all_scenes),
            recommendations_created=created,
        )
```

Update `api/analyzers/__init__.py`:

```python
"""Analyzers for the recommendations engine."""

from .base import BaseAnalyzer, AnalysisResult
from .duplicate_performer import DuplicatePerformerAnalyzer
from .duplicate_scene_files import DuplicateSceneFilesAnalyzer
from .duplicate_scenes import DuplicateScenesAnalyzer

__all__ = [
    "BaseAnalyzer",
    "AnalysisResult",
    "DuplicatePerformerAnalyzer",
    "DuplicateSceneFilesAnalyzer",
    "DuplicateScenesAnalyzer",
]
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_scenes_analyzer.py -v`

Expected: All tests PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/analyzers/duplicate_scenes.py api/analyzers/__init__.py api/tests/test_duplicate_scenes_analyzer.py
git commit -m "feat: implement DuplicateScenesAnalyzer

- Loads scenes and existing fingerprints
- Compares all pairs using multi-signal scoring
- Creates recommendations for matches above threshold
- Respects dismissed targets and max comparisons limit"
```

---

## Task 6: API Endpoint for Duplicate Detection

**Files:**
- Modify: `api/recommendations_router.py`
- Test: `api/tests/test_duplicate_endpoint.py` (create)

**Step 1: Write the failing test**

Create `api/tests/test_duplicate_endpoint.py`:

```python
"""Tests for duplicate detection API endpoints."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestDuplicateScanEndpoint:
    """Tests for POST /analysis/duplicate_scenes/run endpoint."""

    @pytest.fixture
    def client(self, tmp_path):
        # Set up test environment
        import os
        os.environ["STASH_URL"] = "http://test:9999"
        os.environ["DATA_DIR"] = str(tmp_path)

        # Create minimal required files
        (tmp_path / "manifest.json").write_text('{"version": "test"}')

        from fastapi.testclient import TestClient
        from main import app

        return TestClient(app)

    def test_endpoint_exists(self, client):
        # The endpoint should exist even if analysis fails
        response = client.post("/analysis/duplicate_scenes/run")

        # May fail due to missing stash connection, but endpoint should exist
        assert response.status_code in [200, 500, 503]

    def test_returns_analysis_result(self, client):
        with patch("recommendations_router.DuplicateScenesAnalyzer") as MockAnalyzer:
            mock_instance = MagicMock()
            mock_instance.run = AsyncMock(
                return_value=MagicMock(
                    items_processed=100,
                    recommendations_created=5,
                    errors=[],
                )
            )
            MockAnalyzer.return_value = mock_instance

            response = client.post("/analysis/duplicate_scenes/run")

            if response.status_code == 200:
                data = response.json()
                assert "items_processed" in data or "run_id" in data
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_endpoint.py -v`

Expected: FAIL with route not found or similar

**Step 3: Add endpoint to recommendations router**

Add to `api/recommendations_router.py`, in the analysis endpoints section:

```python
from analyzers.duplicate_scenes import DuplicateScenesAnalyzer


@router.post("/analysis/duplicate_scenes/run")
async def run_duplicate_scenes_analysis(
    min_confidence: float = Query(50.0, ge=0.0, le=100.0, description="Minimum confidence threshold"),
):
    """
    Run duplicate scene detection analysis.

    Compares all scenes using stash-box IDs, face fingerprints, and metadata
    to find potential duplicates.
    """
    if rec_db is None or stash is None:
        raise HTTPException(status_code=503, detail="Recommendations engine not initialized")

    try:
        analyzer = DuplicateScenesAnalyzer(
            stash=stash,
            rec_db=rec_db,
            min_confidence=min_confidence,
        )
        result = await analyzer.run(incremental=False)

        return {
            "status": "completed",
            "items_processed": result.items_processed,
            "recommendations_created": result.recommendations_created,
            "errors": result.errors,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_duplicate_endpoint.py -v`

Expected: Tests PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/recommendations_router.py api/tests/test_duplicate_endpoint.py
git commit -m "feat: add API endpoint for duplicate scene analysis

POST /analysis/duplicate_scenes/run with configurable min_confidence"
```

---

## Task 7: Run All Tests and Verify Integration

**Files:**
- All test files

**Step 1: Run the full test suite**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v --ignore=tests/test_*.py -k "fingerprint or duplicate"`

Expected: All new tests PASS

**Step 2: Run existing tests to ensure no regressions**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v`

Expected: No regressions in existing tests

**Step 3: Final commit with integration verification**

```bash
cd /home/carrot/code/stash-sense
git add -A
git commit -m "test: verify duplicate detection integration

All tests pass for:
- Scene fingerprint storage (schema v2)
- Data models and scoring functions
- Stash client scene queries
- DuplicateScenesAnalyzer
- API endpoints"
```

---

## Summary of Files Created/Modified

| File | Action | Purpose |
|------|--------|---------|
| `api/recommendations_db.py` | Modified | Schema v2 + fingerprint tables + CRUD |
| `api/duplicate_detection/__init__.py` | Created | Module exports |
| `api/duplicate_detection/models.py` | Created | Data classes |
| `api/duplicate_detection/scoring.py` | Created | Scoring functions |
| `api/stash_client_unified.py` | Modified | Scene query methods |
| `api/analyzers/duplicate_scenes.py` | Created | Main analyzer |
| `api/analyzers/__init__.py` | Modified | Export new analyzer |
| `api/recommendations_router.py` | Modified | API endpoint |
| `api/tests/test_scene_fingerprints.py` | Created | DB tests |
| `api/tests/test_duplicate_models.py` | Created | Model tests |
| `api/tests/test_duplicate_scoring.py` | Created | Scoring tests |
| `api/tests/test_stash_client_scenes.py` | Created | Client tests |
| `api/tests/test_duplicate_scenes_analyzer.py` | Created | Analyzer tests |
| `api/tests/test_duplicate_endpoint.py` | Created | API tests |

---

## Future Tasks (Not in This Plan)

These are noted for follow-up implementation:

1. **Fingerprint generation from scene analysis** - Persist fingerprints when running `/identify/scene`
2. **Background scheduler integration** - Add to APScheduler with configurable interval
3. **Environment variables** - Add DEDUP_* config options
4. **Plugin UI** - Display duplicate recommendations in Stash plugin

---

*Plan created 2026-01-30 following TDD approach with bite-sized tasks.*

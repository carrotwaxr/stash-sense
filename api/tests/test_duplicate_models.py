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

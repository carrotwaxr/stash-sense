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

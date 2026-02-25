"""Tests for duplicate detection scoring functions."""



class TestMetadataScore:
    """Tests for revised metadata scoring with date as strongest signal."""

    def _scene(self, **kwargs):
        from duplicate_detection.models import SceneMetadata

        defaults = dict(scene_id="1", studio_id=None, performer_ids=set(), date=None, duration_seconds=None)
        defaults.update(kwargs)
        return SceneMetadata(**defaults)

    def test_same_date_exact_performers_highest_score(self):
        from duplicate_detection.scoring import metadata_score

        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        score, reason = metadata_score(a, b)
        assert score >= 55.0  # 45 + 10 studio
        assert "Same date" in reason

    def test_same_date_partial_performers(self):
        from duplicate_detection.scoring import metadata_score

        # Jaccard = 2/3 = 67% (>= 50% threshold)
        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1", "p2"})
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1", "p2", "p3"})
        score, reason = metadata_score(a, b)
        assert 35.0 <= score <= 50.0

    def test_exact_performers_no_date(self):
        from duplicate_detection.scoring import metadata_score

        a = self._scene(scene_id="1", performer_ids={"p1", "p2"})
        b = self._scene(scene_id="2", performer_ids={"p1", "p2"})
        score, reason = metadata_score(a, b)
        assert 25.0 <= score <= 40.0

    def test_studio_only_mild_boost(self):
        from duplicate_detection.scoring import metadata_score

        a = self._scene(scene_id="1", studio_id="s1")
        b = self._scene(scene_id="2", studio_id="s1")
        score, reason = metadata_score(a, b)
        assert score == 10.0

    def test_no_metadata_returns_zero(self):
        from duplicate_detection.scoring import metadata_score

        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        score, _ = metadata_score(a, b)
        assert score == 0.0

    def test_duration_penalty_extreme(self):
        from duplicate_detection.scoring import metadata_score

        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1"}, duration_seconds=180)
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1"}, duration_seconds=2700)
        score_with_dur, reason = metadata_score(a, b)
        a_no_dur = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1"})
        b_no_dur = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1"})
        score_no_dur, _ = metadata_score(a_no_dur, b_no_dur)
        assert score_with_dur < score_no_dur

    def test_duration_no_penalty_web_vs_dvd(self):
        from duplicate_detection.scoring import metadata_score

        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1"}, duration_seconds=3000)
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1"}, duration_seconds=1500)
        score_with_dur, _ = metadata_score(a, b)
        a_no_dur = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1"})
        b_no_dur = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1"})
        score_no_dur, _ = metadata_score(a_no_dur, b_no_dur)
        assert score_with_dur == score_no_dur  # ratio 0.5, no penalty

    def test_cap_at_60(self):
        from duplicate_detection.scoring import metadata_score

        a = self._scene(scene_id="1", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        b = self._scene(scene_id="2", date="2024-01-15", performer_ids={"p1", "p2"}, studio_id="s1")
        score, _ = metadata_score(a, b)
        assert score <= 60.0


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

    def test_zero_shared_performers_returns_zero(self):
        """Scenes with no shared performers must score 0."""
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

        assert score == 0.0
        assert "No shared performers" in reasoning

    def test_identical_cast_scores_high(self):
        """Identical cast with identical proportions should score >= 70."""
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

        # Jaccard = 1.0 -> base = 60, proportion bonus = 15, total = 75 (capped)
        assert score >= 70.0
        assert "2 shared performers" in reasoning

    def test_partial_cast_overlap_scores_moderately(self):
        """Partial overlap (1 shared out of 3 total) scores in 15-40 range."""
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
                "stashdb:p1": FaceAppearance("stashdb:p1", 15, 0.88, 0.5),
                "stashdb:p3": FaceAppearance("stashdb:p3", 15, 0.80, 0.5),
            },
            total_faces_detected=30,
            frames_analyzed=40,
        )

        score, reasoning = face_signature_similarity(fp_a, fp_b)

        # Jaccard = 1/3 -> base = 20, proportion bonus up to 15
        assert 15.0 <= score <= 40.0
        assert "1 shared performers" in reasoning

    def test_unknown_performers_excluded(self):
        """'unknown' performer ID should be excluded from comparison."""
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

    def test_score_capped_at_75(self):
        """Score must never exceed 75."""
        from duplicate_detection.scoring import face_signature_similarity
        from duplicate_detection.models import SceneFingerprint, FaceAppearance

        fp_a = SceneFingerprint(
            stash_scene_id=1,
            faces={
                "stashdb:p1": FaceAppearance("stashdb:p1", 30, 0.95, 0.33),
                "stashdb:p2": FaceAppearance("stashdb:p2", 30, 0.95, 0.33),
                "stashdb:p3": FaceAppearance("stashdb:p3", 30, 0.95, 0.34),
            },
            total_faces_detected=90,
            frames_analyzed=40,
        )
        fp_b = SceneFingerprint(
            stash_scene_id=2,
            faces={
                "stashdb:p1": FaceAppearance("stashdb:p1", 30, 0.95, 0.33),
                "stashdb:p2": FaceAppearance("stashdb:p2", 30, 0.95, 0.33),
                "stashdb:p3": FaceAppearance("stashdb:p3", 30, 0.95, 0.34),
            },
            total_faces_detected=90,
            frames_analyzed=40,
        )

        score, reasoning = face_signature_similarity(fp_a, fp_b)

        assert score <= 75.0


class TestCombinedConfidence:
    """Tests for calculate_duplicate_confidence function."""

    def test_stashbox_match_returns_100(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence
        from duplicate_detection.models import SceneMetadata, StashID

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

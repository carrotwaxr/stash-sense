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


class TestPhashScore:
    def test_identical_phash(self):
        from duplicate_detection.scoring import phash_score

        score, reason = phash_score(0)
        assert score >= 85.0

    def test_very_close_phash(self):
        from duplicate_detection.scoring import phash_score

        score, _ = phash_score(3)
        assert 60.0 <= score <= 90.0

    def test_moderate_phash(self):
        from duplicate_detection.scoring import phash_score

        score, _ = phash_score(8)
        assert 20.0 <= score <= 45.0

    def test_distant_phash(self):
        from duplicate_detection.scoring import phash_score

        score, _ = phash_score(11)
        assert score == 0.0

    def test_none_returns_zero(self):
        from duplicate_detection.scoring import phash_score

        score, _ = phash_score(None)
        assert score == 0.0


class TestHammingDistance:
    def test_identical(self):
        from duplicate_detection.scoring import hamming_distance

        assert hamming_distance("eb716d2e0149f2d1", "eb716d2e0149f2d1") == 0

    def test_one_bit_diff(self):
        from duplicate_detection.scoring import hamming_distance

        assert hamming_distance("eb716d2e0149f2d1", "eb716d2e0149f2d0") == 1

    def test_completely_different(self):
        from duplicate_detection.scoring import hamming_distance

        assert hamming_distance("0000000000000000", "ffffffffffffffff") == 64

    def test_none_handling(self):
        from duplicate_detection.scoring import hamming_distance

        assert hamming_distance(None, "eb716d2e0149f2d1") is None
        assert hamming_distance("eb716d2e0149f2d1", None) is None


class TestCombinedConfidenceRedesign:
    def _scene(self, **kwargs):
        from duplicate_detection.models import SceneMetadata

        defaults = dict(scene_id="1", studio_id=None, performer_ids=set(), date=None, duration_seconds=None, stash_ids=[])
        defaults.update(kwargs)
        return SceneMetadata(**defaults)

    def test_stashbox_match_still_100(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence
        from duplicate_detection.models import StashID

        a = self._scene(scene_id="1", stash_ids=[StashID("https://stashdb.org", "abc")])
        b = self._scene(scene_id="2", stash_ids=[StashID("https://stashdb.org", "abc")])
        match = calculate_duplicate_confidence(a, b)
        assert match is not None
        assert match.confidence == 100.0

    def test_strong_phash_scores_high(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence

        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b, phash_distance=2)
        assert match is not None
        assert 70.0 <= match.confidence <= 95.0

    def test_moderate_phash_alone_scores_low(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence

        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b, phash_distance=8)
        # Without corroboration, score = phash * 0.6 ≈ 33 * 0.6 ≈ 20
        assert match is None or match.confidence < 50.0

    def test_moderate_phash_with_metadata_scores_well(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence

        a = self._scene(scene_id="1", studio_id="s1", performer_ids={"p1"}, date="2024-01-15")
        b = self._scene(scene_id="2", studio_id="s1", performer_ids={"p1"}, date="2024-01-15")
        match = calculate_duplicate_confidence(a, b, phash_distance=8)
        assert match is not None
        assert match.confidence >= 50.0

    def test_metadata_only_catches_different_intros(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence

        a = self._scene(scene_id="1", studio_id="s1", performer_ids={"p1", "p2"}, date="2024-01-15")
        b = self._scene(scene_id="2", studio_id="s1", performer_ids={"p1", "p2"}, date="2024-01-15")
        match = calculate_duplicate_confidence(a, b)
        assert match is not None
        assert match.confidence >= 50.0

    def test_no_signals_returns_none(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence

        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b)
        assert match is None

    def test_same_scene_returns_none(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence

        a = self._scene(scene_id="1")
        match = calculate_duplicate_confidence(a, a)
        assert match is None

    def test_signal_breakdown_includes_phash(self):
        from duplicate_detection.scoring import calculate_duplicate_confidence

        a = self._scene(scene_id="1")
        b = self._scene(scene_id="2")
        match = calculate_duplicate_confidence(a, b, phash_distance=2)
        assert match is not None
        assert match.signal_breakdown.phash_distance == 2

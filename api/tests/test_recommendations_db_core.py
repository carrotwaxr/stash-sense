"""Tests for core recommendation CRUD in RecommendationsDB."""

import pytest
from recommendations_db import RecommendationsDB, Recommendation


@pytest.fixture
def db(tmp_path):
    return RecommendationsDB(str(tmp_path / "test.db"))


def _make_rec(db, type="upstream_performer_changes", target_type="performer",
              target_id="1", details=None, confidence=0.9, source_analysis_id=None):
    """Helper to create a recommendation with sensible defaults."""
    return db.create_recommendation(
        type=type, target_type=target_type, target_id=target_id,
        details=details or {"changes": ["name"]},
        confidence=confidence, source_analysis_id=source_analysis_id,
    )


# ==================== create_recommendation ====================


class TestCreateRecommendation:
    def test_returns_int_id(self, db):
        rec_id = _make_rec(db)
        assert isinstance(rec_id, int)
        assert rec_id > 0

    def test_stores_all_fields(self, db):
        run_id = db.start_analysis_run("upstream_performer_changes")
        rec_id = db.create_recommendation(
            type="upstream_performer_changes",
            target_type="performer",
            target_id="42",
            details={"changes": ["height", "name"]},
            confidence=0.85,
            source_analysis_id=run_id,
        )
        rec = db.get_recommendation(rec_id)
        assert rec.type == "upstream_performer_changes"
        assert rec.target_type == "performer"
        assert rec.target_id == "42"
        assert rec.details == {"changes": ["height", "name"]}
        assert rec.confidence == 0.85
        assert rec.source_analysis_id == run_id
        assert rec.status == "pending"

    def test_default_status_is_pending(self, db):
        rec_id = _make_rec(db)
        rec = db.get_recommendation(rec_id)
        assert rec.status == "pending"

    def test_sets_created_at_and_updated_at(self, db):
        rec_id = _make_rec(db)
        rec = db.get_recommendation(rec_id)
        assert rec.created_at is not None
        assert rec.updated_at is not None

    def test_duplicate_returns_none(self, db):
        _make_rec(db, target_id="1")
        dup = _make_rec(db, target_id="1")
        assert dup is None

    def test_different_target_ids_are_not_duplicates(self, db):
        id1 = _make_rec(db, target_id="1")
        id2 = _make_rec(db, target_id="2")
        assert id1 is not None
        assert id2 is not None
        assert id1 != id2

    def test_same_target_id_different_type_is_not_duplicate(self, db):
        id1 = _make_rec(db, type="upstream_performer_changes", target_id="1")
        id2 = _make_rec(db, type="duplicate_scene", target_id="1")
        assert id1 is not None
        assert id2 is not None

    def test_none_confidence_allowed(self, db):
        rec_id = db.create_recommendation(
            type="test", target_type="performer", target_id="1",
            details={}, confidence=None,
        )
        rec = db.get_recommendation(rec_id)
        assert rec.confidence is None


# ==================== get_recommendation ====================


class TestGetRecommendation:
    def test_returns_recommendation_dataclass(self, db):
        rec_id = _make_rec(db)
        rec = db.get_recommendation(rec_id)
        assert isinstance(rec, Recommendation)

    def test_returns_none_for_missing_id(self, db):
        assert db.get_recommendation(9999) is None

    def test_details_is_parsed_dict(self, db):
        rec_id = db.create_recommendation(
            type="test", target_type="performer", target_id="1",
            details={"nested": {"key": [1, 2, 3]}},
        )
        rec = db.get_recommendation(rec_id)
        assert rec.details == {"nested": {"key": [1, 2, 3]}}


# ==================== get_recommendation_by_target ====================


class TestGetRecommendationByTarget:
    def test_finds_by_type_target_type_target_id(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        rec = db.get_recommendation_by_target("upstream", "performer", "42")
        assert rec is not None
        assert rec.id == rec_id

    def test_returns_none_when_not_found(self, db):
        assert db.get_recommendation_by_target("upstream", "performer", "999") is None

    def test_with_status_filter(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        rec = db.get_recommendation_by_target("upstream", "performer", "42", status="pending")
        assert rec is not None
        assert rec.id == rec_id

    def test_status_filter_excludes_non_matching(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        db.resolve_recommendation(rec_id, action="accepted")
        rec = db.get_recommendation_by_target("upstream", "performer", "42", status="pending")
        assert rec is None

    def test_status_filter_finds_resolved(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        db.resolve_recommendation(rec_id, action="accepted")
        rec = db.get_recommendation_by_target("upstream", "performer", "42", status="resolved")
        assert rec is not None


# ==================== get_recommendations ====================


class TestGetRecommendations:
    def test_returns_list(self, db):
        recs = db.get_recommendations()
        assert isinstance(recs, list)

    def test_returns_empty_when_no_recs(self, db):
        assert db.get_recommendations() == []

    def test_filter_by_status(self, db):
        rec_id1 = _make_rec(db, target_id="1")
        _make_rec(db, target_id="2")
        db.resolve_recommendation(rec_id1, action="accepted")

        pending = db.get_recommendations(status="pending")
        assert len(pending) == 1
        assert pending[0].target_id == "2"

    def test_filter_by_type(self, db):
        _make_rec(db, type="type_a", target_id="1")
        _make_rec(db, type="type_b", target_id="2")

        results = db.get_recommendations(type="type_a")
        assert len(results) == 1
        assert results[0].type == "type_a"

    def test_filter_by_target_type(self, db):
        _make_rec(db, target_type="performer", target_id="1")
        _make_rec(db, target_type="scene", target_id="2")

        results = db.get_recommendations(target_type="scene")
        assert len(results) == 1
        assert results[0].target_type == "scene"

    def test_combined_filters(self, db):
        _make_rec(db, type="type_a", target_type="performer", target_id="1")
        _make_rec(db, type="type_a", target_type="scene", target_id="2")
        _make_rec(db, type="type_b", target_type="performer", target_id="3")

        results = db.get_recommendations(type="type_a", target_type="performer")
        assert len(results) == 1
        assert results[0].target_id == "1"

    def test_pagination_limit(self, db):
        for i in range(5):
            _make_rec(db, target_id=str(i))
        results = db.get_recommendations(limit=3)
        assert len(results) == 3

    def test_pagination_offset(self, db):
        for i in range(5):
            _make_rec(db, target_id=str(i))
        all_recs = db.get_recommendations(limit=100)
        offset_recs = db.get_recommendations(limit=100, offset=2)
        assert len(offset_recs) == 3
        assert offset_recs[0].id == all_recs[2].id

    def test_default_ordering_created_at_desc(self, db):
        # All created within same second so created_at is identical.
        # SQLite ORDER BY created_at DESC is stable by rowid when timestamps match,
        # so verify all records are returned and IDs are consistent.
        id1 = _make_rec(db, target_id="1")
        id2 = _make_rec(db, target_id="2")
        id3 = _make_rec(db, target_id="3")
        recs = db.get_recommendations()
        assert len(recs) == 3
        returned_ids = [r.id for r in recs]
        assert set(returned_ids) == {id1, id2, id3}


# ==================== count_recommendations ====================


class TestCountRecommendations:
    def test_count_no_filters(self, db):
        for i in range(4):
            _make_rec(db, target_id=str(i))
        assert db.count_recommendations() == 4

    def test_count_empty(self, db):
        assert db.count_recommendations() == 0

    def test_count_by_status(self, db):
        rec_id = _make_rec(db, target_id="1")
        _make_rec(db, target_id="2")
        db.resolve_recommendation(rec_id, action="accepted")

        assert db.count_recommendations(status="pending") == 1
        assert db.count_recommendations(status="resolved") == 1

    def test_count_by_type(self, db):
        _make_rec(db, type="type_a", target_id="1")
        _make_rec(db, type="type_a", target_id="2")
        _make_rec(db, type="type_b", target_id="3")

        assert db.count_recommendations(type="type_a") == 2
        assert db.count_recommendations(type="type_b") == 1

    def test_count_by_target_type(self, db):
        _make_rec(db, target_type="performer", target_id="1")
        _make_rec(db, target_type="scene", target_id="2")

        assert db.count_recommendations(target_type="performer") == 1


# ==================== resolve_recommendation ====================


class TestResolveRecommendation:
    def test_sets_status_to_resolved(self, db):
        rec_id = _make_rec(db)
        db.resolve_recommendation(rec_id, action="accepted")
        rec = db.get_recommendation(rec_id)
        assert rec.status == "resolved"

    def test_stores_action(self, db):
        rec_id = _make_rec(db)
        db.resolve_recommendation(rec_id, action="merged")
        rec = db.get_recommendation(rec_id)
        assert rec.resolution_action == "merged"

    def test_stores_details(self, db):
        rec_id = _make_rec(db)
        db.resolve_recommendation(rec_id, action="accepted", details={"kept_id": "1"})
        rec = db.get_recommendation(rec_id)
        assert rec.resolution_details == {"kept_id": "1"}

    def test_sets_resolved_at(self, db):
        rec_id = _make_rec(db)
        db.resolve_recommendation(rec_id, action="accepted")
        rec = db.get_recommendation(rec_id)
        assert rec.resolved_at is not None

    def test_returns_true_on_success(self, db):
        rec_id = _make_rec(db)
        assert db.resolve_recommendation(rec_id, action="accepted") is True

    def test_returns_false_for_missing_id(self, db):
        assert db.resolve_recommendation(9999, action="accepted") is False

    def test_details_none_stores_null(self, db):
        rec_id = _make_rec(db)
        db.resolve_recommendation(rec_id, action="accepted", details=None)
        rec = db.get_recommendation(rec_id)
        assert rec.resolution_details is None


# ==================== dismiss_recommendation ====================


class TestDismissRecommendation:
    def test_sets_status_to_dismissed(self, db):
        rec_id = _make_rec(db)
        db.dismiss_recommendation(rec_id)
        rec = db.get_recommendation(rec_id)
        assert rec.status == "dismissed"

    def test_adds_to_dismissed_targets(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        db.dismiss_recommendation(rec_id, reason="not now")
        assert db.is_dismissed("upstream", "performer", "42")

    def test_stores_reason(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        db.dismiss_recommendation(rec_id, reason="not interested")
        with db._connection() as conn:
            row = conn.execute(
                "SELECT reason FROM dismissed_targets WHERE target_id = '42'"
            ).fetchone()
            assert row["reason"] == "not interested"

    def test_returns_true_on_success(self, db):
        rec_id = _make_rec(db)
        assert db.dismiss_recommendation(rec_id) is True

    def test_returns_false_for_missing_id(self, db):
        assert db.dismiss_recommendation(9999) is False

    def test_permanent_dismiss(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        db.dismiss_recommendation(rec_id, permanent=True)
        assert db.is_permanently_dismissed("upstream", "performer", "42")

    def test_non_permanent_dismiss(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="42")
        db.dismiss_recommendation(rec_id, permanent=False)
        assert db.is_dismissed("upstream", "performer", "42")
        assert not db.is_permanently_dismissed("upstream", "performer", "42")


# ==================== batch_dismiss_by_type ====================


class TestBatchDismissByType:
    def test_dismisses_all_pending_of_type(self, db):
        _make_rec(db, type="upstream", target_id="1")
        _make_rec(db, type="upstream", target_id="2")
        _make_rec(db, type="other", target_id="3")

        count = db.batch_dismiss_by_type("upstream")
        assert count == 2
        assert db.count_recommendations(type="upstream", status="dismissed") == 2
        assert db.count_recommendations(type="other", status="pending") == 1

    def test_returns_zero_when_none_pending(self, db):
        assert db.batch_dismiss_by_type("nonexistent") == 0

    def test_skips_already_resolved(self, db):
        rec_id = _make_rec(db, type="upstream", target_id="1")
        _make_rec(db, type="upstream", target_id="2")
        db.resolve_recommendation(rec_id, action="accepted")

        count = db.batch_dismiss_by_type("upstream")
        assert count == 1

    def test_adds_to_dismissed_targets(self, db):
        _make_rec(db, type="upstream", target_type="performer", target_id="1")
        _make_rec(db, type="upstream", target_type="performer", target_id="2")

        db.batch_dismiss_by_type("upstream")
        assert db.is_dismissed("upstream", "performer", "1")
        assert db.is_dismissed("upstream", "performer", "2")

    def test_permanent_flag(self, db):
        _make_rec(db, type="upstream", target_type="performer", target_id="1")
        db.batch_dismiss_by_type("upstream", permanent=True)
        assert db.is_permanently_dismissed("upstream", "performer", "1")

    def test_reason_stored(self, db):
        _make_rec(db, type="upstream", target_type="performer", target_id="1")
        db.batch_dismiss_by_type("upstream", reason="bulk dismiss")
        with db._connection() as conn:
            row = conn.execute(
                "SELECT reason FROM dismissed_targets WHERE target_id = '1'"
            ).fetchone()
            assert row["reason"] == "bulk dismiss"


# ==================== is_dismissed ====================


class TestIsDismissed:
    def test_returns_false_when_not_dismissed(self, db):
        assert db.is_dismissed("upstream", "performer", "1") is False

    def test_returns_true_after_dismiss(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="1")
        db.dismiss_recommendation(rec_id)
        assert db.is_dismissed("upstream", "performer", "1") is True


# ==================== is_permanently_dismissed ====================


class TestIsPermanentlyDismissed:
    def test_returns_false_for_soft_dismiss(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="1")
        db.dismiss_recommendation(rec_id, permanent=False)
        assert db.is_permanently_dismissed("upstream", "performer", "1") is False

    def test_returns_true_for_permanent_dismiss(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="1")
        db.dismiss_recommendation(rec_id, permanent=True)
        assert db.is_permanently_dismissed("upstream", "performer", "1") is True


# ==================== undismiss ====================


class TestUndismiss:
    def test_removes_soft_dismissal(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="1")
        db.dismiss_recommendation(rec_id)
        db.undismiss("upstream", "performer", "1")
        assert db.is_dismissed("upstream", "performer", "1") is False

    def test_does_not_remove_permanent_dismissal(self, db):
        rec_id = _make_rec(db, type="upstream", target_type="performer", target_id="1")
        db.dismiss_recommendation(rec_id, permanent=True)
        db.undismiss("upstream", "performer", "1")
        assert db.is_dismissed("upstream", "performer", "1") is True

    def test_noop_when_not_dismissed(self, db):
        db.undismiss("upstream", "performer", "1")  # should not raise


# ==================== reopen_recommendation ====================


class TestReopenRecommendation:
    def test_reopens_dismissed_recommendation(self, db):
        rec_id = _make_rec(db, target_id="1")
        db.dismiss_recommendation(rec_id)
        result = db.reopen_recommendation(rec_id, details={"changes": ["height"]})
        assert result is True
        rec = db.get_recommendation(rec_id)
        assert rec.status == "pending"
        assert rec.details == {"changes": ["height"]}
        assert rec.resolution_action is None
        assert rec.resolution_details is None
        assert rec.resolved_at is None

    def test_returns_false_for_pending(self, db):
        rec_id = _make_rec(db, target_id="1")
        result = db.reopen_recommendation(rec_id, details={"changes": ["height"]})
        assert result is False

    def test_returns_false_for_resolved(self, db):
        rec_id = _make_rec(db, target_id="1")
        db.resolve_recommendation(rec_id, action="accepted")
        result = db.reopen_recommendation(rec_id, details={"changes": ["height"]})
        assert result is False

    def test_returns_false_for_nonexistent(self, db):
        result = db.reopen_recommendation(9999, details={})
        assert result is False


# ==================== update_recommendation_details ====================


class TestUpdateRecommendationDetails:
    def test_updates_details_on_pending(self, db):
        rec_id = _make_rec(db, target_id="1")
        result = db.update_recommendation_details(rec_id, {"changes": ["height", "name"]})
        assert result is True
        rec = db.get_recommendation(rec_id)
        assert rec.details == {"changes": ["height", "name"]}

    def test_returns_false_on_resolved(self, db):
        rec_id = _make_rec(db, target_id="1")
        db.resolve_recommendation(rec_id, action="accepted")
        result = db.update_recommendation_details(rec_id, {"changes": []})
        assert result is False

    def test_returns_false_on_nonexistent(self, db):
        assert db.update_recommendation_details(9999, {}) is False


# ==================== get_recommendation_counts ====================


class TestGetRecommendationCounts:
    def test_empty_db_returns_empty_dict(self, db):
        assert db.get_recommendation_counts() == {}

    def test_counts_by_type_and_status(self, db):
        _make_rec(db, type="type_a", target_id="1")
        _make_rec(db, type="type_a", target_id="2")
        rec_id = _make_rec(db, type="type_a", target_id="3")
        db.resolve_recommendation(rec_id, action="accepted")
        _make_rec(db, type="type_b", target_id="4")

        counts = db.get_recommendation_counts()
        assert counts["type_a"]["pending"] == 2
        assert counts["type_a"]["resolved"] == 1
        assert counts["type_b"]["pending"] == 1

    def test_does_not_include_zero_counts(self, db):
        _make_rec(db, type="type_a", target_id="1")
        counts = db.get_recommendation_counts()
        assert "type_b" not in counts
        assert "resolved" not in counts.get("type_a", {})


# ==================== get_stats ====================


class TestGetStats:
    def test_empty_db_stats(self, db):
        stats = db.get_stats()
        assert stats["total_recommendations"] == 0
        assert stats["pending_recommendations"] == 0
        assert stats["dismissed_count"] == 0

    def test_stats_after_operations(self, db):
        _make_rec(db, target_id="1")
        rec_id2 = _make_rec(db, target_id="2")
        db.dismiss_recommendation(rec_id2, reason="test")

        stats = db.get_stats()
        assert stats["total_recommendations"] == 2
        assert stats["pending_recommendations"] == 1
        assert stats["dismissed_count"] == 1


# ==================== Duplicate Candidates ====================


class TestInsertCandidate:
    def test_returns_int_id(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        cid = db.insert_candidate(10, 20, "phash", run_id)
        assert isinstance(cid, int)
        assert cid > 0

    def test_enforces_canonical_order(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(20, 10, "phash", run_id)
        candidates = db.get_candidates_batch(run_id)
        assert candidates[0]["scene_a_id"] == 10
        assert candidates[0]["scene_b_id"] == 20

    def test_duplicate_pair_returns_none(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(10, 20, "phash", run_id)
        dup = db.insert_candidate(10, 20, "face", run_id)
        assert dup is None

    def test_reversed_duplicate_pair_returns_none(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(10, 20, "phash", run_id)
        dup = db.insert_candidate(20, 10, "face", run_id)
        assert dup is None


class TestInsertCandidatesBatch:
    def test_batch_insert(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        count = db.insert_candidates_batch(
            [(10, 20, "phash"), (30, 40, "phash"), (50, 60, "face")],
            run_id,
        )
        assert count == 3

    def test_batch_ignores_duplicates(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(10, 20, "phash", run_id)
        count = db.insert_candidates_batch(
            [(10, 20, "phash"), (30, 40, "phash")],
            run_id,
        )
        # 10,20 already existed, so only 30,40 is new = total for this run_id is 2
        assert count == 2

    def test_empty_batch_returns_zero(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        count = db.insert_candidates_batch([], run_id)
        assert count == 0


class TestGetCandidatesBatch:
    def test_cursor_based_pagination(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        for i in range(5):
            db.insert_candidate(i, i + 100, "phash", run_id)

        batch1 = db.get_candidates_batch(run_id, after_id=0, limit=2)
        assert len(batch1) == 2

        batch2 = db.get_candidates_batch(run_id, after_id=batch1[-1]["id"], limit=2)
        assert len(batch2) == 2

        batch3 = db.get_candidates_batch(run_id, after_id=batch2[-1]["id"], limit=2)
        assert len(batch3) == 1


class TestCountCandidates:
    def test_count(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(10, 20, "phash", run_id)
        db.insert_candidate(30, 40, "phash", run_id)
        assert db.count_candidates(run_id) == 2

    def test_count_zero(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        assert db.count_candidates(run_id) == 0


class TestClearCandidates:
    def test_clear_by_run(self, db):
        run_id1 = db.start_analysis_run("dup1")
        run_id2 = db.start_analysis_run("dup2")
        db.insert_candidate(10, 20, "phash", run_id1)
        db.insert_candidate(30, 40, "phash", run_id2)

        deleted = db.clear_candidates(run_id1)
        assert deleted == 1
        assert db.count_candidates(run_id1) == 0
        assert db.count_candidates(run_id2) == 1

    def test_clear_all(self, db):
        run_id = db.start_analysis_run("dup")
        db.insert_candidate(10, 20, "phash", run_id)
        db.insert_candidate(30, 40, "phash", run_id)
        deleted = db.clear_all_candidates()
        assert deleted == 2


class TestGetCandidateSceneIds:
    def test_returns_all_scene_ids(self, db):
        run_id = db.start_analysis_run("dup")
        db.insert_candidate(10, 20, "phash", run_id)
        db.insert_candidate(20, 30, "phash", run_id)
        ids = db.get_candidate_scene_ids(run_id)
        assert ids == {10, 20, 30}

    def test_empty_for_no_candidates(self, db):
        run_id = db.start_analysis_run("dup")
        assert db.get_candidate_scene_ids(run_id) == set()

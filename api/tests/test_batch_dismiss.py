"""Tests for batch dismiss functionality."""

import pytest
from recommendations_db import RecommendationsDB


class TestBatchDismissByType:
    """Tests for RecommendationsDB.batch_dismiss_by_type()."""

    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def _create_rec(self, db, rec_type, target_id, status="pending"):
        """Helper to create a recommendation and optionally set its status."""
        rec_id = db.create_recommendation(
            type=rec_type,
            target_type="performer",
            target_id=target_id,
            details={"test": True},
        )
        if status != "pending":
            with db._connection() as conn:
                conn.execute(
                    "UPDATE recommendations SET status = ? WHERE id = ?",
                    (status, rec_id),
                )
        return rec_id

    def test_dismisses_all_pending_of_type(self, db):
        """All pending recs of the specified type should be dismissed."""
        self._create_rec(db, "upstream_performer_changes", "1")
        self._create_rec(db, "upstream_performer_changes", "2")
        self._create_rec(db, "upstream_performer_changes", "3")

        count = db.batch_dismiss_by_type("upstream_performer_changes")
        assert count == 3

        recs = db.get_recommendations(type="upstream_performer_changes")
        for rec in recs:
            assert rec.status == "dismissed"

    def test_does_not_affect_other_types(self, db):
        """Recs of other types should remain untouched."""
        self._create_rec(db, "upstream_performer_changes", "1")
        tag_id = self._create_rec(db, "upstream_tag_changes", "2")

        db.batch_dismiss_by_type("upstream_performer_changes")

        tag_rec = db.get_recommendation(tag_id)
        assert tag_rec.status == "pending"

    def test_only_dismisses_pending(self, db):
        """Already resolved/dismissed recs should not be affected."""
        pending_id = self._create_rec(db, "upstream_performer_changes", "1")
        resolved_id = self._create_rec(db, "upstream_performer_changes", "2", status="resolved")
        dismissed_id = self._create_rec(db, "upstream_performer_changes", "3", status="dismissed")

        count = db.batch_dismiss_by_type("upstream_performer_changes")
        assert count == 1

        assert db.get_recommendation(pending_id).status == "dismissed"
        assert db.get_recommendation(resolved_id).status == "resolved"
        assert db.get_recommendation(dismissed_id).status == "dismissed"

    def test_temporary_dismiss_adds_to_dismissed_targets(self, db):
        """Temporary dismiss should add to dismissed_targets with permanent=0."""
        self._create_rec(db, "upstream_performer_changes", "1")

        db.batch_dismiss_by_type("upstream_performer_changes", permanent=False)

        assert db.is_dismissed("upstream_performer_changes", "performer", "1")
        assert not db.is_permanently_dismissed("upstream_performer_changes", "performer", "1")

    def test_permanent_dismiss_adds_permanent_target(self, db):
        """Permanent dismiss should add to dismissed_targets with permanent=1."""
        self._create_rec(db, "upstream_performer_changes", "1")

        db.batch_dismiss_by_type("upstream_performer_changes", permanent=True)

        assert db.is_permanently_dismissed("upstream_performer_changes", "performer", "1")

    def test_returns_zero_when_no_pending(self, db):
        """Should return 0 when no pending recs of type exist."""
        self._create_rec(db, "upstream_performer_changes", "1", status="resolved")

        count = db.batch_dismiss_by_type("upstream_performer_changes")
        assert count == 0

    def test_returns_zero_for_unknown_type(self, db):
        """Should return 0 for a type with no recs at all."""
        count = db.batch_dismiss_by_type("nonexistent_type")
        assert count == 0

    def test_stores_reason(self, db):
        """Should store the dismiss reason in dismissed_targets."""
        self._create_rec(db, "upstream_performer_changes", "1")

        db.batch_dismiss_by_type("upstream_performer_changes", reason="Batch dismissed by user")

        with db._connection() as conn:
            row = conn.execute(
                "SELECT reason FROM dismissed_targets WHERE target_id = '1'"
            ).fetchone()
            assert row["reason"] == "Batch dismissed by user"

    def test_duplicate_dismissed_targets_ignored(self, db):
        """If a target is already in dismissed_targets, should not fail."""
        rec_id = self._create_rec(db, "upstream_performer_changes", "1")
        # Dismiss individually first
        db.dismiss_recommendation(rec_id)

        # Re-create as pending (simulate re-analysis finding changes)
        new_id = self._create_rec(db, "upstream_performer_changes", "1-new")

        # Batch dismiss should still work
        count = db.batch_dismiss_by_type("upstream_performer_changes")
        assert count == 1

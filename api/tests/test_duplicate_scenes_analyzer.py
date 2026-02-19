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

        run_id = rec_db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db, run_id=run_id)
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

        run_id = rec_db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db, min_confidence=40, run_id=run_id)
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
        run_id = rec_db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db, min_confidence=80, run_id=run_id)
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
        with rec_db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id) VALUES (?, ?, ?)",
                ("duplicate_scenes", "scene", "1"),
            )

        run_id = rec_db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db, run_id=run_id)
        result = await analyzer.run(incremental=False)

        # Should not create recommendation for dismissed target
        assert result.recommendations_created == 0


class TestDuplicateCandidatesDB:
    """Tests for duplicate_candidates table and DB methods."""

    @pytest.fixture
    def db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def test_table_exists(self, db):
        """The duplicate_candidates table should be created on init."""
        with db._connection() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='duplicate_candidates'"
            ).fetchone()
            assert row is not None

    def test_schema_version_is_8(self, db):
        with db._connection() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 8

    def test_insert_candidate(self, db):
        """Insert a candidate pair."""
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(scene_a_id=1, scene_b_id=2, source="stashbox", run_id=run_id)

        candidates = db.get_candidates_batch(run_id, after_id=0, limit=10)
        assert len(candidates) == 1
        assert candidates[0]["scene_a_id"] == 1
        assert candidates[0]["scene_b_id"] == 2
        assert candidates[0]["source"] == "stashbox"

    def test_insert_candidate_deduplicates(self, db):
        """Duplicate pairs are silently ignored."""
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(1, 2, "stashbox", run_id)
        db.insert_candidate(1, 2, "face", run_id)  # same pair, different source

        candidates = db.get_candidates_batch(run_id, after_id=0, limit=10)
        assert len(candidates) == 1  # deduped

    def test_insert_candidate_enforces_canonical_order(self, db):
        """scene_a_id should always be less than scene_b_id."""
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(scene_a_id=5, scene_b_id=3, source="face", run_id=run_id)

        candidates = db.get_candidates_batch(run_id, after_id=0, limit=10)
        assert candidates[0]["scene_a_id"] == 3
        assert candidates[0]["scene_b_id"] == 5

    def test_count_candidates(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(1, 2, "stashbox", run_id)
        db.insert_candidate(3, 4, "face", run_id)

        assert db.count_candidates(run_id) == 2

    def test_clear_candidates(self, db):
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(1, 2, "stashbox", run_id)
        db.clear_candidates(run_id)

        assert db.count_candidates(run_id) == 0

    def test_get_candidates_batch_cursor_pagination(self, db):
        """Cursor-based pagination: after_id returns only rows with id > after_id."""
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(1, 2, "stashbox", run_id)
        db.insert_candidate(3, 4, "face", run_id)
        db.insert_candidate(5, 6, "metadata", run_id)

        batch1 = db.get_candidates_batch(run_id, after_id=0, limit=2)
        assert len(batch1) == 2

        batch2 = db.get_candidates_batch(run_id, after_id=batch1[-1]["id"], limit=2)
        assert len(batch2) == 1

    def test_get_candidate_scene_ids(self, db):
        """Get distinct scene IDs from candidates."""
        run_id = db.start_analysis_run("duplicate_scenes")
        db.insert_candidate(1, 2, "stashbox", run_id)
        db.insert_candidate(2, 3, "face", run_id)

        scene_ids = db.get_candidate_scene_ids(run_id)
        assert scene_ids == {1, 2, 3}

    def test_insert_candidates_batch(self, db):
        """Batch insert multiple candidates."""
        run_id = db.start_analysis_run("duplicate_scenes")
        pairs = [(1, 2, "stashbox"), (3, 4, "face"), (5, 6, "metadata")]
        db.insert_candidates_batch(pairs, run_id)

        assert db.count_candidates(run_id) == 3


class TestFingerprintJoinQuery:
    """Tests for get_fingerprints_with_faces JOIN method."""

    @pytest.fixture
    def db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def test_returns_fingerprints_with_faces(self, db):
        """JOIN query returns fingerprints grouped with their face entries."""
        fp1_id = db.create_scene_fingerprint(stash_scene_id=100, total_faces=3, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp1_id, "performer_1", face_count=10, avg_confidence=0.8, proportion=0.6)
        db.add_fingerprint_face(fp1_id, "performer_2", face_count=5, avg_confidence=0.7, proportion=0.3)

        fp2_id = db.create_scene_fingerprint(stash_scene_id=200, total_faces=1, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp2_id, "performer_1", face_count=8, avg_confidence=0.9, proportion=1.0)

        result = db.get_fingerprints_with_faces()

        assert len(result) == 2
        assert "100" in result
        assert len(result["100"]["faces"]) == 2
        assert "performer_1" in result["100"]["faces"]
        assert "performer_2" in result["100"]["faces"]

    def test_excludes_non_complete_fingerprints(self, db):
        """Only complete fingerprints are returned."""
        fp1_id = db.create_scene_fingerprint(stash_scene_id=100, total_faces=3, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp1_id, "performer_1", face_count=10, avg_confidence=0.8, proportion=0.6)

        db.create_scene_fingerprint(
            stash_scene_id=200, total_faces=0, frames_analyzed=0,
            fingerprint_status="error"
        )

        result = db.get_fingerprints_with_faces()
        assert len(result) == 1

    def test_fingerprint_with_no_faces(self, db):
        """A complete fingerprint with 0 faces should still appear."""
        db.create_scene_fingerprint(stash_scene_id=100, total_faces=0, frames_analyzed=60, fingerprint_status="complete")

        result = db.get_fingerprints_with_faces()
        assert len(result) == 1
        assert len(result["100"]["faces"]) == 0

    def test_filtered_by_scene_ids(self, db):
        """Can filter to specific scene IDs."""
        fp1_id = db.create_scene_fingerprint(stash_scene_id=100, total_faces=1, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp1_id, "performer_1", face_count=5, avg_confidence=0.8, proportion=1.0)

        fp2_id = db.create_scene_fingerprint(stash_scene_id=200, total_faces=1, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp2_id, "performer_1", face_count=3, avg_confidence=0.7, proportion=1.0)

        result = db.get_fingerprints_with_faces(scene_ids={100})
        assert len(result) == 1


class TestFaceCandidateGeneration:
    """Tests for SQL self-join face candidate generation."""

    @pytest.fixture
    def db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def _add_fingerprint_with_faces(self, db, scene_id, performer_ids):
        """Helper: create a complete fingerprint with face entries."""
        fp_id = db.create_scene_fingerprint(
            stash_scene_id=scene_id, total_faces=len(performer_ids), frames_analyzed=60,
            fingerprint_status="complete"
        )
        for i, pid in enumerate(performer_ids):
            db.add_fingerprint_face(
                fp_id, pid, face_count=5,
                avg_confidence=0.8, proportion=1.0 / len(performer_ids)
            )
        return fp_id

    def test_finds_scenes_sharing_performer(self, db):
        """Two scenes with the same identified performer should be candidates."""
        self._add_fingerprint_with_faces(db, 100, ["performer_1"])
        self._add_fingerprint_with_faces(db, 200, ["performer_1"])

        pairs = db.generate_face_candidates()
        assert len(pairs) == 1
        assert pairs[0] == (100, 200)

    def test_excludes_unknown_performers(self, db):
        """Scenes sharing only 'unknown' faces should not be candidates."""
        self._add_fingerprint_with_faces(db, 100, ["unknown"])
        self._add_fingerprint_with_faces(db, 200, ["unknown"])

        pairs = db.generate_face_candidates()
        assert len(pairs) == 0

    def test_canonical_order(self, db):
        """Pairs should always have scene_a_id < scene_b_id."""
        self._add_fingerprint_with_faces(db, 300, ["performer_1"])
        self._add_fingerprint_with_faces(db, 100, ["performer_1"])

        pairs = db.generate_face_candidates()
        assert pairs[0] == (100, 300)

    def test_deduplicates_across_performers(self, db):
        """Two scenes sharing multiple performers should appear once."""
        self._add_fingerprint_with_faces(db, 100, ["performer_1", "performer_2"])
        self._add_fingerprint_with_faces(db, 200, ["performer_1", "performer_2"])

        pairs = db.generate_face_candidates()
        assert len(pairs) == 1

    def test_excludes_incomplete_fingerprints(self, db):
        """Only complete fingerprints participate."""
        self._add_fingerprint_with_faces(db, 100, ["performer_1"])
        # Scene 200 has an error fingerprint
        db.create_scene_fingerprint(
            stash_scene_id=200, total_faces=0, frames_analyzed=0,
            fingerprint_status="error"
        )

        pairs = db.generate_face_candidates()
        assert len(pairs) == 0

    def test_many_scenes_same_performer(self, db):
        """N scenes with same performer = N*(N-1)/2 pairs."""
        for sid in [100, 200, 300, 400]:
            self._add_fingerprint_with_faces(db, sid, ["performer_1"])

        pairs = db.generate_face_candidates()
        assert len(pairs) == 6  # 4*3/2


class TestCandidateGeneration:
    """Tests for Phase 1: candidate generation."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_scenes_for_fingerprinting = AsyncMock(return_value=([], 0))
        return stash

    @pytest.fixture
    def db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.mark.asyncio
    async def test_generates_stashbox_candidates(self, mock_stash, db):
        """Scenes sharing a stash-box ID become candidates with source='stashbox'."""
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {"id": "1", "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc"}],
                     "studio": None, "performers": [], "files": [], "date": None},
                    {"id": "2", "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc"}],
                     "studio": None, "performers": [], "files": [], "date": None},
                    {"id": "3", "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "xyz"}],
                     "studio": None, "performers": [], "files": [], "date": None},
                ],
                3,
            )
        )

        run_id = db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, db, run_id=run_id)
        count = await analyzer._generate_candidates(run_id)

        assert count >= 1
        candidates = db.get_candidates_batch(run_id, after_id=0, limit=100)
        stashbox_candidates = [c for c in candidates if c["source"] == "stashbox"]
        assert len(stashbox_candidates) == 1
        assert stashbox_candidates[0]["scene_a_id"] == 1
        assert stashbox_candidates[0]["scene_b_id"] == 2

    @pytest.mark.asyncio
    async def test_generates_face_candidates(self, mock_stash, db):
        """Scenes sharing an identified performer in fingerprints become candidates."""
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        fp1 = db.create_scene_fingerprint(stash_scene_id=10, total_faces=1, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp1, "performer_1", face_count=5, avg_confidence=0.8, proportion=1.0)
        fp2 = db.create_scene_fingerprint(stash_scene_id=20, total_faces=1, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp2, "performer_1", face_count=3, avg_confidence=0.7, proportion=1.0)

        mock_stash.get_scenes_for_fingerprinting = AsyncMock(return_value=([], 0))

        run_id = db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, db, run_id=run_id)
        count = await analyzer._generate_candidates(run_id)

        assert count >= 1
        candidates = db.get_candidates_batch(run_id, after_id=0, limit=100)
        face_candidates = [c for c in candidates if c["source"] == "face"]
        assert len(face_candidates) == 1

    @pytest.mark.asyncio
    async def test_generates_metadata_candidates(self, mock_stash, db):
        """Scenes sharing studio AND performer become candidates with source='metadata'."""
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {"id": "1", "stash_ids": [], "studio": {"id": "s1", "name": "Studio"},
                     "performers": [{"id": "p1", "name": "Performer"}], "files": [], "date": None},
                    {"id": "2", "stash_ids": [], "studio": {"id": "s1", "name": "Studio"},
                     "performers": [{"id": "p1", "name": "Performer"}], "files": [], "date": None},
                ],
                2,
            )
        )

        run_id = db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, db, run_id=run_id)
        count = await analyzer._generate_candidates(run_id)

        candidates = db.get_candidates_batch(run_id, after_id=0, limit=100)
        metadata_candidates = [c for c in candidates if c["source"] == "metadata"]
        assert len(metadata_candidates) == 1

    @pytest.mark.asyncio
    async def test_no_metadata_candidates_without_shared_performer(self, mock_stash, db):
        """Same studio but no shared performer should NOT produce metadata candidates."""
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {"id": "1", "stash_ids": [], "studio": {"id": "s1", "name": "Studio"},
                     "performers": [{"id": "p1", "name": "Perf A"}], "files": [], "date": None},
                    {"id": "2", "stash_ids": [], "studio": {"id": "s1", "name": "Studio"},
                     "performers": [{"id": "p2", "name": "Perf B"}], "files": [], "date": None},
                ],
                2,
            )
        )

        run_id = db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, db, run_id=run_id)
        count = await analyzer._generate_candidates(run_id)

        candidates = db.get_candidates_batch(run_id, after_id=0, limit=100)
        metadata_candidates = [c for c in candidates if c["source"] == "metadata"]
        assert len(metadata_candidates) == 0

    @pytest.mark.asyncio
    async def test_deduplicates_across_sources(self, mock_stash, db):
        """A pair found by both stashbox and face should appear only once."""
        from analyzers.duplicate_scenes import DuplicateScenesAnalyzer

        mock_stash.get_scenes_for_fingerprinting = AsyncMock(
            return_value=(
                [
                    {"id": "10", "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "same"}],
                     "studio": None, "performers": [], "files": [], "date": None},
                    {"id": "20", "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "same"}],
                     "studio": None, "performers": [], "files": [], "date": None},
                ],
                2,
            )
        )
        fp1 = db.create_scene_fingerprint(stash_scene_id=10, total_faces=1, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp1, "performer_1", face_count=5, avg_confidence=0.8, proportion=1.0)
        fp2 = db.create_scene_fingerprint(stash_scene_id=20, total_faces=1, frames_analyzed=60, fingerprint_status="complete")
        db.add_fingerprint_face(fp2, "performer_1", face_count=3, avg_confidence=0.7, proportion=1.0)

        run_id = db.start_analysis_run("duplicate_scenes")
        analyzer = DuplicateScenesAnalyzer(mock_stash, db, run_id=run_id)
        count = await analyzer._generate_candidates(run_id)

        assert count == 1

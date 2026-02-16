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
        with rec_db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id) VALUES (?, ?, ?)",
                ("duplicate_scenes", "scene", "1"),
            )

        analyzer = DuplicateScenesAnalyzer(mock_stash, rec_db)
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

    def test_schema_version_is_7(self, db):
        with db._connection() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 7

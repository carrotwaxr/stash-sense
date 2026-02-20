"""Tests for upstream scene sync."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestStashBoxSceneQuery:
    @pytest.mark.asyncio
    async def test_get_scene_returns_scene_data(self):
        """get_scene fetches a scene by ID with all required fields."""
        from stashbox_client import StashBoxClient

        mock_response = {
            "findScene": {
                "id": "scene-uuid-1",
                "title": "Test Scene",
                "details": "A test scene",
                "date": "2025-01-15",
                "urls": [{"url": "https://example.com/scene1", "site": {"name": "Example"}}],
                "studio": {"id": "studio-uuid-1", "name": "Test Studio"},
                "tags": [{"id": "tag-uuid-1", "name": "HD"}],
                "performers": [
                    {"performer": {"id": "perf-uuid-1", "name": "Jane Doe"}, "as": "Jane Smith"}
                ],
                "director": "John Director",
                "code": "TS-001",
                "deleted": False,
                "created": "2025-01-01T00:00:00Z",
                "updated": "2025-01-15T00:00:00Z",
            }
        }

        client = StashBoxClient("https://test.box/graphql", "key")
        client._execute = AsyncMock(return_value=mock_response)

        result = await client.get_scene("scene-uuid-1")
        assert result is not None
        assert result["title"] == "Test Scene"
        assert result["studio"]["id"] == "studio-uuid-1"
        assert len(result["performers"]) == 1
        assert result["performers"][0]["as"] == "Jane Smith"

    @pytest.mark.asyncio
    async def test_get_scene_returns_none_for_missing(self):
        """get_scene returns None when scene not found."""
        from stashbox_client import StashBoxClient

        client = StashBoxClient("https://test.box/graphql", "key")
        client._execute = AsyncMock(return_value={"findScene": None})

        result = await client.get_scene("nonexistent")
        assert result is None


class TestStashSceneQueries:
    @pytest.mark.asyncio
    async def test_get_scenes_for_endpoint(self):
        """get_scenes_for_endpoint returns scenes linked to a specific endpoint."""
        from stash_client_unified import StashClientUnified

        mock_response = {
            "findScenes": {
                "scenes": [
                    {
                        "id": "1",
                        "title": "Scene One",
                        "date": "2025-01-01",
                        "details": "Details here",
                        "director": "Director",
                        "code": "SC-001",
                        "urls": ["https://example.com/1"],
                        "studio": {"id": "10", "name": "Studio A", "stash_ids": []},
                        "performers": [{"id": "20", "name": "Perf A", "stash_ids": []}],
                        "tags": [{"id": "30", "name": "Tag A", "stash_ids": []}],
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                        ],
                    }
                ]
            }
        }

        client = StashClientUnified("http://localhost:9999", "key")
        client._execute = AsyncMock(return_value=mock_response)

        scenes = await client.get_scenes_for_endpoint("https://stashdb.org/graphql")
        assert len(scenes) == 1
        assert scenes[0]["title"] == "Scene One"
        assert scenes[0]["stash_ids"][0]["stash_id"] == "sb-scene-1"

    @pytest.mark.asyncio
    async def test_update_scene(self):
        """update_scene sends mutation with correct fields."""
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999", "key")
        client._execute = AsyncMock(return_value={"sceneUpdate": {"id": "1"}})

        result = await client.update_scene("1", title="New Title", date="2025-02-01")
        assert result["id"] == "1"
        # Verify the input dict was passed correctly
        call_args = client._execute.call_args
        # _execute is called as (query, {"input": input_dict}, priority=Priority.CRITICAL)
        # positional args: call_args[0][1] is the variables dict
        variables = call_args[0][1]
        input_dict = variables["input"]
        assert input_dict["id"] == "1"
        assert input_dict["title"] == "New Title"
        assert input_dict["date"] == "2025-02-01"


class TestSceneFieldMapper:
    def test_scene_field_config_registered(self):
        """Scene fields are registered in ENTITY_FIELD_CONFIGS."""
        from upstream_field_mapper import ENTITY_FIELD_CONFIGS
        assert "scene" in ENTITY_FIELD_CONFIGS
        cfg = ENTITY_FIELD_CONFIGS["scene"]
        assert "title" in cfg["default_fields"]
        assert "date" in cfg["default_fields"]
        assert "studio" in cfg["default_fields"]
        assert "performers" in cfg["default_fields"]
        assert "tags" in cfg["default_fields"]

    def test_normalize_upstream_scene_simple_fields(self):
        """normalize_upstream_scene extracts simple scalar fields."""
        from upstream_field_mapper import normalize_upstream_scene

        upstream = {
            "title": "Test Scene",
            "details": "Some details",
            "date": "2025-01-15",
            "director": "John",
            "code": "TS-001",
            "urls": [{"url": "https://example.com/1", "site": {"name": "Example"}}],
            "studio": {"id": "studio-1", "name": "Studio A"},
            "tags": [{"id": "tag-1", "name": "HD"}],
            "performers": [
                {"performer": {"id": "perf-1", "name": "Jane"}, "as": "Jane Smith"}
            ],
        }

        result = normalize_upstream_scene(upstream)
        assert result["title"] == "Test Scene"
        assert result["date"] == "2025-01-15"
        assert result["details"] == "Some details"
        assert result["director"] == "John"
        assert result["code"] == "TS-001"
        assert result["urls"] == ["https://example.com/1"]

    def test_normalize_upstream_scene_relational_fields(self):
        """normalize_upstream_scene extracts relational entity data."""
        from upstream_field_mapper import normalize_upstream_scene

        upstream = {
            "title": "Test",
            "details": None,
            "date": None,
            "director": None,
            "code": None,
            "urls": [],
            "studio": {"id": "studio-1", "name": "Studio A"},
            "tags": [
                {"id": "tag-1", "name": "HD"},
                {"id": "tag-2", "name": "POV"},
            ],
            "performers": [
                {"performer": {"id": "perf-1", "name": "Jane"}, "as": "Jane Smith"},
                {"performer": {"id": "perf-2", "name": "John"}, "as": None},
            ],
        }

        result = normalize_upstream_scene(upstream)
        assert result["studio"] == {"id": "studio-1", "name": "Studio A"}
        assert len(result["performers"]) == 2
        assert result["performers"][0] == {"id": "perf-1", "name": "Jane", "as": "Jane Smith"}
        assert result["performers"][1] == {"id": "perf-2", "name": "John", "as": None}
        assert len(result["tags"]) == 2
        assert result["tags"][0] == {"id": "tag-1", "name": "HD"}

    def test_diff_scene_simple_fields(self):
        """diff_scene_fields detects simple scalar changes."""
        from upstream_field_mapper import diff_scene_fields

        local = {"title": "Old Title", "date": "2025-01-01", "details": "", "director": "", "code": "", "urls": [],
                 "studio": None, "performers": [], "tags": []}
        upstream = {"title": "New Title", "date": "2025-01-01", "details": "", "director": "", "code": "", "urls": [],
                    "studio": None, "performers": [], "tags": []}

        result = diff_scene_fields(local, upstream, None, {"title", "date"})
        assert len(result["changes"]) == 1
        assert result["changes"][0]["field"] == "title"
        assert result["changes"][0]["upstream_value"] == "New Title"

    def test_diff_scene_relational_performers(self):
        """diff_scene_fields detects added/removed performers."""
        from upstream_field_mapper import diff_scene_fields

        local = {
            "title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
            "studio": None,
            "performers": [{"id": "perf-1", "name": "Jane", "as": None}],
            "tags": [],
        }
        upstream = {
            "title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
            "studio": None,
            "performers": [
                {"id": "perf-1", "name": "Jane", "as": None},
                {"id": "perf-2", "name": "John", "as": "Johnny"},
            ],
            "tags": [],
        }

        result = diff_scene_fields(local, upstream, None, {"performers"})
        assert len(result["performer_changes"]["added"]) == 1
        assert result["performer_changes"]["added"][0]["id"] == "perf-2"
        assert len(result["performer_changes"]["removed"]) == 0

    def test_diff_scene_no_changes(self):
        """diff_scene_fields returns empty results when nothing changed."""
        from upstream_field_mapper import diff_scene_fields

        data = {"title": "Same", "date": "2025-01-01", "details": "", "director": "", "code": "", "urls": [],
                "studio": {"id": "s1", "name": "S"}, "performers": [{"id": "p1", "name": "P", "as": None}],
                "tags": [{"id": "t1", "name": "T"}]}

        result = diff_scene_fields(data, data, None, {"title", "date", "studio", "performers", "tags"})
        assert result["changes"] == []
        assert result["studio_change"] is None
        assert result["performer_changes"]["added"] == []
        assert result["performer_changes"]["removed"] == []
        assert result["tag_changes"]["added"] == []
        assert result["tag_changes"]["removed"] == []

    def test_diff_scene_has_any_changes(self):
        """has_any_scene_changes returns True when there are changes."""
        from upstream_field_mapper import diff_scene_fields

        local = {"title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
                 "studio": None, "performers": [], "tags": []}
        upstream = {"title": "Scene", "date": "", "details": "", "director": "", "code": "", "urls": [],
                    "studio": None, "performers": [], "tags": [{"id": "t1", "name": "New Tag"}]}

        result = diff_scene_fields(local, upstream, None, {"tags"})
        assert len(result["tag_changes"]["added"]) == 1


class TestUpstreamSceneAnalyzer:
    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key"},
        ])
        stash.get_scenes_for_endpoint = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Local Title",
                "date": "2025-01-01",
                "details": "",
                "director": "",
                "code": "",
                "urls": [],
                "studio": None,
                "performers": [],
                "tags": [],
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "sb-scene-1"}
                ],
            }
        ])
        return stash

    @pytest.mark.asyncio
    async def test_detects_title_change(self, mock_stash, rec_db):
        """Analyzer detects when upstream scene title differs from local."""
        upstream_data = {
            "title": "Upstream Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1
        assert recs[0].details["scene_name"] == "Local Title"
        changes = recs[0].details.get("changes", [])
        title_change = next((c for c in changes if c["field"] == "title"), None)
        assert title_change is not None
        assert title_change["upstream_value"] == "Upstream Title"

    @pytest.mark.asyncio
    async def test_no_changes_creates_no_recommendation(self, mock_stash, rec_db):
        """Analyzer creates no recommendation when upstream matches local."""
        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 0

    @pytest.mark.asyncio
    async def test_detects_performer_addition(self, mock_stash, rec_db):
        """Analyzer detects when upstream has additional performers."""
        upstream_data = {
            "title": "Local Title",
            "details": "",
            "date": "2025-01-01",
            "director": "",
            "code": "",
            "urls": [],
            "studio": None,
            "tags": [],
            "performers": [
                {"performer": {"id": "perf-1", "name": "Jane"}, "as": None}
            ],
            "deleted": False,
            "updated": "2025-01-15T00:00:00Z",
        }

        with patch("stashbox_client.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.get_scene = AsyncMock(return_value=upstream_data)
            MockSBC.return_value = mock_sbc

            from analyzers.upstream_scene import UpstreamSceneAnalyzer
            analyzer = UpstreamSceneAnalyzer(mock_stash, rec_db)
            result = await analyzer.run()

        recs = rec_db.get_recommendations(type="upstream_scene_changes", status="pending")
        assert len(recs) == 1
        assert len(recs[0].details["performer_changes"]["added"]) == 1

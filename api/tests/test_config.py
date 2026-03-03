"""Tests for config.py - configuration dataclasses and utility functions."""

import os
from unittest.mock import patch

from config import (
    StashConfig,
    DatabaseConfig,
    MultiSignalConfig,
    get_stashbox_shortname,
    STASHBOX_ENDPOINTS,
    FACENET_DIM,
    ARCFACE_DIM,
)


class TestStashConfig:
    def test_from_env_defaults(self):
        with patch.dict(os.environ, {}, clear=True):
            config = StashConfig.from_env()
            assert config.url == "http://localhost:9999"
            assert config.api_key == ""

    def test_from_env_with_vars(self):
        with patch.dict(os.environ, {"STASH_URL": "http://10.0.0.4:6969", "STASH_API_KEY": "my-key"}):
            config = StashConfig.from_env()
            assert config.url == "http://10.0.0.4:6969"
            assert config.api_key == "my-key"


class TestDatabaseConfig:
    def test_post_init_sets_paths(self, tmp_path):
        config = DatabaseConfig(data_dir=tmp_path)

        assert config.facenet_index_path == tmp_path / "face_facenet.voy"
        assert config.arcface_index_path == tmp_path / "face_arcface.voy"
        assert config.adaface_index_path == tmp_path / "face_adaface.voy"
        assert config.tattoo_index_path == tmp_path / "tattoo_embeddings.voy"
        assert config.sqlite_db_path == tmp_path / "performers.db"
        assert config.faces_json_path == tmp_path / "faces.json"
        assert config.performers_json_path == tmp_path / "performers.json"
        assert config.manifest_json_path == tmp_path / "manifest.json"

    def test_creates_data_dir(self, tmp_path):
        new_dir = tmp_path / "nested" / "data"
        config = DatabaseConfig(data_dir=new_dir)
        assert new_dir.exists()
        assert config.data_dir == new_dir

    def test_custom_path_preserved(self, tmp_path):
        custom_facenet = tmp_path / "custom_facenet.voy"
        config = DatabaseConfig(data_dir=tmp_path, facenet_index_path=custom_facenet)
        assert config.facenet_index_path == custom_facenet
        # Other paths should still use defaults
        assert config.arcface_index_path == tmp_path / "face_arcface.voy"


class TestMultiSignalConfig:
    def test_defaults(self):
        config = MultiSignalConfig()
        assert config.enable_body is True
        assert config.enable_tattoo == "auto"
        assert config.face_candidates == 20


class TestGetStashboxShortname:
    def test_known_stashdb(self):
        assert get_stashbox_shortname("https://stashdb.org/graphql") == "stashdb.org"

    def test_known_fansdb(self):
        assert get_stashbox_shortname("https://fansdb.cc/graphql") == "fansdb.cc"

    def test_known_pmvstash(self):
        assert get_stashbox_shortname("https://pmvstash.org/graphql") == "pmvstash.org"

    def test_known_theporndb(self):
        assert get_stashbox_shortname("https://theporndb.net/graphql") == "theporndb.net"

    def test_unknown_strips_protocol_and_path(self):
        result = get_stashbox_shortname("https://custom-stashbox.example.com/graphql")
        assert result == "custom-stashbox.example.com"

    def test_unknown_no_graphql_path(self):
        result = get_stashbox_shortname("https://example.com/api")
        assert result == "example.com/api"


class TestStashboxEndpointsDict:
    def test_expected_entries_exist(self):
        assert "https://stashdb.org/graphql" in STASHBOX_ENDPOINTS
        assert "https://fansdb.cc/graphql" in STASHBOX_ENDPOINTS
        assert "https://pmvstash.org/graphql" in STASHBOX_ENDPOINTS
        assert "https://javstash.org/graphql" in STASHBOX_ENDPOINTS
        assert "https://theporndb.net/graphql" in STASHBOX_ENDPOINTS

    def test_dimensions(self):
        assert FACENET_DIM == 512
        assert ARCFACE_DIM == 512

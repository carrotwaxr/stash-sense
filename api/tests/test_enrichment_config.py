"""Tests for enrichment configuration loading."""
import pytest
import tempfile
from pathlib import Path


class TestEnrichmentConfig:
    """Test configuration loading and CLI overrides."""

    def test_load_default_config(self):
        """Config loads with sensible defaults when no file provided."""
        from enrichment_config import EnrichmentConfig

        config = EnrichmentConfig()

        assert config.global_settings.max_faces_per_performer == 20
        assert config.global_settings.default_rate_limit == 60

    def test_load_yaml_config(self, tmp_path):
        """Config loads from YAML file."""
        from enrichment_config import EnrichmentConfig

        yaml_content = """
global:
  max_faces_per_performer: 15
  default_rate_limit: 120

stash_boxes:
  stashdb:
    enabled: true
    url: "https://stashdb.org/graphql"
    rate_limit: 240
    max_faces: 5
    priority: 1
"""
        config_file = tmp_path / "sources.yaml"
        config_file.write_text(yaml_content)

        config = EnrichmentConfig(config_path=config_file)

        assert config.global_settings.max_faces_per_performer == 15
        assert config.get_source("stashdb").rate_limit == 240
        assert config.get_source("stashdb").max_faces == 5

    def test_cli_override_sources(self, tmp_path):
        """CLI --sources flag limits which sources are enabled."""
        from enrichment_config import EnrichmentConfig

        yaml_content = """
stash_boxes:
  stashdb:
    enabled: true
  theporndb:
    enabled: true
  pmvstash:
    enabled: true
"""
        config_file = tmp_path / "sources.yaml"
        config_file.write_text(yaml_content)

        config = EnrichmentConfig(
            config_path=config_file,
            cli_sources=["stashdb", "theporndb"]
        )

        enabled = config.get_enabled_sources()
        assert "stashdb" in enabled
        assert "theporndb" in enabled
        assert "pmvstash" not in enabled

    def test_cli_override_disable_source(self, tmp_path):
        """CLI --disable-source removes a source."""
        from enrichment_config import EnrichmentConfig

        yaml_content = """
stash_boxes:
  stashdb:
    enabled: true
  theporndb:
    enabled: true
"""
        config_file = tmp_path / "sources.yaml"
        config_file.write_text(yaml_content)

        config = EnrichmentConfig(
            config_path=config_file,
            cli_disabled_sources=["theporndb"]
        )

        enabled = config.get_enabled_sources()
        assert "stashdb" in enabled
        assert "theporndb" not in enabled

    def test_cli_override_max_faces(self, tmp_path):
        """CLI --source-max-faces overrides per-source limit."""
        from enrichment_config import EnrichmentConfig

        yaml_content = """
stash_boxes:
  stashdb:
    enabled: true
    max_faces: 5
"""
        config_file = tmp_path / "sources.yaml"
        config_file.write_text(yaml_content)

        config = EnrichmentConfig(
            config_path=config_file,
            cli_source_max_faces={"stashdb": 10}
        )

        assert config.get_source("stashdb").max_faces == 10

    def test_source_not_found_raises(self):
        """Accessing unknown source raises KeyError."""
        from enrichment_config import EnrichmentConfig

        config = EnrichmentConfig()

        with pytest.raises(KeyError):
            config.get_source("nonexistent")

    def test_gender_filter(self, tmp_path):
        """Gender filter is respected."""
        from enrichment_config import EnrichmentConfig

        yaml_content = """
reference_sites:
  babepedia:
    enabled: true
    gender_filter: female
"""
        config_file = tmp_path / "sources.yaml"
        config_file.write_text(yaml_content)

        config = EnrichmentConfig(config_path=config_file)
        source = config.get_source("babepedia")

        assert source.gender_filter == "female"
        assert source.should_process_performer(gender="FEMALE") is True
        assert source.should_process_performer(gender="MALE") is False
        assert source.should_process_performer(gender=None) is True  # Unknown gender = process

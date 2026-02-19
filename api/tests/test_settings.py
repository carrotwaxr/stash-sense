"""Tests for the settings system."""

import json
import os
import pytest
import sqlite3
from pathlib import Path
from unittest.mock import patch

from recommendations_db import RecommendationsDB
from settings import (
    SettingsManager,
    SettingType,
    SETTING_DEFS,
    TIER_DEFAULTS,
    ENV_VAR_MIGRATION,
    migrate_env_vars,
)


@pytest.fixture
def db(tmp_path):
    """Create a fresh RecommendationsDB for testing."""
    db_path = tmp_path / "test_settings.db"
    return RecommendationsDB(str(db_path))


@pytest.fixture
def mgr_gpu_high(db):
    """Settings manager with gpu-high tier."""
    return SettingsManager(db, "gpu-high")


@pytest.fixture
def mgr_cpu(db):
    """Settings manager with cpu tier."""
    return SettingsManager(db, "cpu")


class TestSettingDefinitions:
    """Test that setting definitions are well-formed."""

    def test_all_settings_have_unique_keys(self):
        keys = list(SETTING_DEFS.keys())
        assert len(keys) == len(set(keys))

    def test_all_settings_have_categories(self):
        for key, defn in SETTING_DEFS.items():
            assert defn.category, f"{key} has no category"

    def test_all_settings_have_descriptions(self):
        for key, defn in SETTING_DEFS.items():
            assert defn.description, f"{key} has no description"

    def test_tier_defaults_reference_valid_settings(self):
        for tier, defaults in TIER_DEFAULTS.items():
            for key in defaults:
                assert key in SETTING_DEFS, f"{tier} has unknown setting: {key}"

    def test_env_var_migration_references_valid_settings(self):
        for env_var, setting_key in ENV_VAR_MIGRATION.items():
            assert setting_key in SETTING_DEFS, f"Migration target {setting_key} not in SETTING_DEFS"


class TestResolution:
    """Test settings resolution: user override > tier default > fallback."""

    def test_fallback_when_no_tier_default(self, mgr_gpu_high):
        # stash_api_rate has no tier default, should use fallback
        assert mgr_gpu_high.get("stash_api_rate") == 5.0

    def test_tier_default_gpu_high(self, mgr_gpu_high):
        assert mgr_gpu_high.get("embedding_batch_size") == 32

    def test_tier_default_cpu(self, mgr_cpu):
        assert mgr_cpu.get("embedding_batch_size") == 4

    def test_tier_default_num_frames_differs(self, mgr_gpu_high, mgr_cpu):
        assert mgr_gpu_high.get("num_frames") == 60
        assert mgr_cpu.get("num_frames") == 30

    def test_user_override_wins(self, mgr_gpu_high):
        mgr_gpu_high.set("embedding_batch_size", 64)
        assert mgr_gpu_high.get("embedding_batch_size") == 64

    def test_unknown_key_raises(self, mgr_gpu_high):
        with pytest.raises(KeyError, match="Unknown setting"):
            mgr_gpu_high.get("nonexistent_setting")


class TestPersistence:
    """Test write/read/delete cycle."""

    def test_set_and_get(self, mgr_gpu_high):
        mgr_gpu_high.set("stash_api_rate", 10.0)
        assert mgr_gpu_high.get("stash_api_rate") == 10.0

    def test_delete_reverts_to_default(self, mgr_gpu_high):
        mgr_gpu_high.set("embedding_batch_size", 64)
        assert mgr_gpu_high.get("embedding_batch_size") == 64
        mgr_gpu_high.delete("embedding_batch_size")
        assert mgr_gpu_high.get("embedding_batch_size") == 32  # tier default

    def test_delete_unknown_key_raises(self, mgr_gpu_high):
        with pytest.raises(KeyError):
            mgr_gpu_high.delete("nonexistent")

    def test_set_unknown_key_raises(self, mgr_gpu_high):
        with pytest.raises(KeyError):
            mgr_gpu_high.set("nonexistent", 42)

    def test_persistence_across_managers(self, db):
        """Settings persist even with a new SettingsManager instance."""
        mgr1 = SettingsManager(db, "gpu-high")
        mgr1.set("stash_api_rate", 10.0)

        mgr2 = SettingsManager(db, "gpu-high")
        assert mgr2.get("stash_api_rate") == 10.0

    def test_bulk_set(self, mgr_gpu_high):
        result = mgr_gpu_high.set_bulk({
            "stash_api_rate": 10.0,
            "num_frames": 45,
        })
        assert result["stash_api_rate"] == 10.0
        assert result["num_frames"] == 45
        assert mgr_gpu_high.get("stash_api_rate") == 10.0
        assert mgr_gpu_high.get("num_frames") == 45


class TestValidation:
    """Test type coercion and range validation."""

    def test_int_coercion(self, mgr_gpu_high):
        mgr_gpu_high.set("embedding_batch_size", "16")
        assert mgr_gpu_high.get("embedding_batch_size") == 16

    def test_float_coercion(self, mgr_gpu_high):
        mgr_gpu_high.set("stash_api_rate", "7.5")
        assert mgr_gpu_high.get("stash_api_rate") == 7.5

    def test_bool_coercion_string_true(self, mgr_gpu_high):
        mgr_gpu_high.set("gpu_enabled", "true")
        assert mgr_gpu_high.get("gpu_enabled") is True

    def test_bool_coercion_string_false(self, mgr_gpu_high):
        mgr_gpu_high.set("gpu_enabled", "false")
        assert mgr_gpu_high.get("gpu_enabled") is False

    def test_below_minimum_raises(self, mgr_gpu_high):
        with pytest.raises(ValueError, match="below minimum"):
            mgr_gpu_high.set("embedding_batch_size", 0)

    def test_above_maximum_raises(self, mgr_gpu_high):
        with pytest.raises(ValueError, match="above maximum"):
            mgr_gpu_high.set("embedding_batch_size", 200)

    def test_at_minimum_ok(self, mgr_gpu_high):
        mgr_gpu_high.set("embedding_batch_size", 1)
        assert mgr_gpu_high.get("embedding_batch_size") == 1

    def test_at_maximum_ok(self, mgr_gpu_high):
        mgr_gpu_high.set("embedding_batch_size", 128)
        assert mgr_gpu_high.get("embedding_batch_size") == 128


class TestGetAll:
    """Test get_all and get_all_with_metadata."""

    def test_get_all_returns_all_settings(self, mgr_gpu_high):
        all_settings = mgr_gpu_high.get_all()
        for key in SETTING_DEFS:
            assert key in all_settings

    def test_get_all_reflects_overrides(self, mgr_gpu_high):
        mgr_gpu_high.set("stash_api_rate", 10.0)
        all_settings = mgr_gpu_high.get_all()
        assert all_settings["stash_api_rate"] == 10.0

    def test_get_all_with_metadata_structure(self, mgr_gpu_high):
        result = mgr_gpu_high.get_all_with_metadata()
        assert "hardware_tier" in result
        assert result["hardware_tier"] == "gpu-high"
        assert "categories" in result

        # Each category has label, order, settings
        for cat_key, cat in result["categories"].items():
            assert "label" in cat
            assert "order" in cat
            assert "settings" in cat

            # Each setting has required fields
            for skey, sinfo in cat["settings"].items():
                assert "value" in sinfo
                assert "default" in sinfo
                assert "is_override" in sinfo
                assert "type" in sinfo

    def test_metadata_shows_override_flag(self, mgr_gpu_high):
        mgr_gpu_high.set("stash_api_rate", 10.0)
        result = mgr_gpu_high.get_all_with_metadata()

        # Find stash_api_rate in the result
        rate_setting = result["categories"]["rate_limits"]["settings"]["stash_api_rate"]
        assert rate_setting["is_override"] is True
        assert rate_setting["value"] == 10.0

    def test_get_all_caching(self, mgr_gpu_high):
        """get_all should cache results."""
        result1 = mgr_gpu_high.get_all()
        result2 = mgr_gpu_high.get_all()
        assert result1 == result2

    def test_cache_invalidated_on_set(self, mgr_gpu_high):
        mgr_gpu_high.get_all()
        mgr_gpu_high.set("stash_api_rate", 10.0)
        result = mgr_gpu_high.get_all()
        assert result["stash_api_rate"] == 10.0


class TestEnvVarMigration:
    """Test env var migration to settings."""

    def test_migrates_rate_limit(self, mgr_gpu_high):
        with patch.dict(os.environ, {"STASH_RATE_LIMIT": "10.0"}):
            count = migrate_env_vars(mgr_gpu_high)
        assert count == 1
        assert mgr_gpu_high.get("stash_api_rate") == 10.0

    def test_migrates_body_signal(self, mgr_gpu_high):
        with patch.dict(os.environ, {"ENABLE_BODY_SIGNAL": "false"}):
            count = migrate_env_vars(mgr_gpu_high)
        assert count == 1
        assert mgr_gpu_high.get("body_signal_enabled") is False

    def test_does_not_overwrite_existing_override(self, mgr_gpu_high):
        mgr_gpu_high.set("stash_api_rate", 7.5)
        with patch.dict(os.environ, {"STASH_RATE_LIMIT": "10.0"}):
            count = migrate_env_vars(mgr_gpu_high)
        assert count == 0
        assert mgr_gpu_high.get("stash_api_rate") == 7.5

    def test_ignores_unset_env_vars(self, mgr_gpu_high):
        # Clear the env vars we're testing
        env = {k: None for k in ENV_VAR_MIGRATION}
        with patch.dict(os.environ, {}, clear=False):
            # Ensure none of the migration vars are set
            for k in ENV_VAR_MIGRATION:
                os.environ.pop(k, None)
            count = migrate_env_vars(mgr_gpu_high)
        assert count == 0

    def test_handles_invalid_value_gracefully(self, mgr_gpu_high):
        with patch.dict(os.environ, {"STASH_RATE_LIMIT": "not_a_number"}):
            count = migrate_env_vars(mgr_gpu_high)
        assert count == 0

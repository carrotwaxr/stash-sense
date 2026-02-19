"""Sidecar settings system.

Defines all configurable parameters, their defaults (per hardware tier),
validation rules, and grouping for the UI. Settings are persisted as user
overrides in the user_settings table of stash_sense.db.

Resolution order:
    1. User override (from DB)  — highest priority
    2. Hardware-tier defaults   — based on detected profile
    3. Hardcoded fallbacks      — always present

Only user overrides are stored in the DB. Absence of a key means "use tier
default." This keeps the table sparse and resets easy (delete the row).
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Setting definitions
# ============================================================================

class SettingType(str, Enum):
    INT = "int"
    FLOAT = "float"
    BOOL = "bool"
    STRING = "string"


@dataclass(frozen=True)
class SettingDef:
    """Definition of a single configurable setting."""
    key: str
    label: str
    description: str
    category: str
    type: SettingType
    fallback: Any  # Hardcoded fallback if not in tier defaults
    min_val: Optional[float] = None
    max_val: Optional[float] = None


# All settings, keyed by setting key
SETTING_DEFS: dict[str, SettingDef] = {}

# Category metadata for UI rendering
CATEGORIES = {
    "performance": {"label": "Performance", "order": 0},
    "rate_limits": {"label": "Rate Limits", "order": 1},
    "recognition": {"label": "Recognition", "order": 2},
    "signals": {"label": "Signals", "order": 3},
}


def _define(
    key: str,
    label: str,
    description: str,
    category: str,
    type: SettingType,
    fallback: Any,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
) -> SettingDef:
    """Register a setting definition."""
    defn = SettingDef(
        key=key, label=label, description=description,
        category=category, type=type, fallback=fallback,
        min_val=min_val, max_val=max_val,
    )
    SETTING_DEFS[key] = defn
    return defn


# -- Performance --
_define("embedding_batch_size", "Embedding Batch Size",
        "Faces processed per GPU inference call",
        "performance", SettingType.INT, fallback=16, min_val=1, max_val=128)

_define("frame_extraction_concurrency", "Frame Extraction Workers",
        "Parallel ffmpeg processes for frame extraction",
        "performance", SettingType.INT, fallback=4, min_val=1, max_val=16)

_define("detection_size", "Detection Resolution",
        "Face detection input resolution (pixels). Higher = more accurate but slower",
        "performance", SettingType.INT, fallback=640, min_val=160, max_val=1280)

# -- Rate Limits --
_define("stash_api_rate", "Stash API Rate",
        "Maximum requests per second to local Stash instance",
        "rate_limits", SettingType.FLOAT, fallback=5.0, min_val=0.5, max_val=50.0)

# -- Recognition --
_define("gpu_enabled", "GPU Acceleration",
        "Use GPU for face detection and embedding. Disable to force CPU mode",
        "recognition", SettingType.BOOL, fallback=True)

_define("num_frames", "Frames Per Scene",
        "Number of frames to sample from each scene for face recognition",
        "recognition", SettingType.INT, fallback=60, min_val=10, max_val=200)

_define("face_candidates", "Face Candidates",
        "Number of candidate matches to retrieve from the vector index per face",
        "recognition", SettingType.INT, fallback=20, min_val=5, max_val=100)

# -- Signals --
_define("body_signal_enabled", "Body Proportions",
        "Use body proportion analysis as a supplementary identification signal",
        "signals", SettingType.BOOL, fallback=True)

_define("tattoo_signal_enabled", "Tattoo Detection",
        "Use tattoo detection as a supplementary identification signal (requires model)",
        "signals", SettingType.BOOL, fallback=False)


# ============================================================================
# Tier defaults
# ============================================================================

TIER_DEFAULTS: dict[str, dict[str, Any]] = {
    "gpu-high": {
        "embedding_batch_size": 32,
        "frame_extraction_concurrency": 8,
        "detection_size": 640,
        "num_frames": 60,
        "gpu_enabled": True,
    },
    "gpu-low": {
        "embedding_batch_size": 16,
        "frame_extraction_concurrency": 6,
        "detection_size": 640,
        "num_frames": 60,
        "gpu_enabled": True,
    },
    "cpu": {
        "embedding_batch_size": 4,
        "frame_extraction_concurrency": 2,
        "detection_size": 320,
        "num_frames": 30,
        "gpu_enabled": False,
    },
}


# ============================================================================
# Env var migration mapping
# ============================================================================

ENV_VAR_MIGRATION: dict[str, str] = {
    # env_var_name -> setting_key
    "STASH_RATE_LIMIT": "stash_api_rate",
    "ENABLE_BODY_SIGNAL": "body_signal_enabled",
    "ENABLE_TATTOO_SIGNAL": "tattoo_signal_enabled",
    "FACE_CANDIDATES": "face_candidates",
}


# ============================================================================
# Settings manager
# ============================================================================

class SettingsManager:
    """Resolves, validates, and persists settings.

    Uses the existing user_settings table in stash_sense.db via the
    RecommendationsDB instance.
    """

    # Key prefix to namespace settings in user_settings table
    PREFIX = "settings."

    def __init__(self, db, tier: str):
        """
        Args:
            db: RecommendationsDB instance (has get/set/delete user_setting methods)
            tier: Hardware tier string ("gpu-high", "gpu-low", "cpu")
        """
        self._db = db
        self._tier = tier
        self._tier_defaults = TIER_DEFAULTS.get(tier, {})
        self._cache: Optional[dict[str, Any]] = None

    @property
    def tier(self) -> str:
        return self._tier

    def _db_key(self, key: str) -> str:
        """Prefix a setting key for storage in user_settings table."""
        return f"{self.PREFIX}{key}"

    def _invalidate_cache(self):
        self._cache = None

    def get_default(self, key: str) -> Any:
        """Get the effective default for a setting (tier default or fallback)."""
        if key in self._tier_defaults:
            return self._tier_defaults[key]
        defn = SETTING_DEFS.get(key)
        if defn:
            return defn.fallback
        raise KeyError(f"Unknown setting: {key}")

    def has_override(self, key: str) -> bool:
        """Check if a setting has a user override stored in the DB."""
        if key not in SETTING_DEFS:
            raise KeyError(f"Unknown setting: {key}")
        return self._db.get_user_setting(self._db_key(key)) is not None

    def get(self, key: str) -> Any:
        """Get the resolved value for a setting."""
        if key not in SETTING_DEFS:
            raise KeyError(f"Unknown setting: {key}")

        # Check user override
        override = self._db.get_user_setting(self._db_key(key))
        if override is not None:
            return override

        return self.get_default(key)

    def get_all(self) -> dict[str, Any]:
        """Get all resolved settings as a flat dict."""
        if self._cache is not None:
            return dict(self._cache)

        result = {}
        for key in SETTING_DEFS:
            result[key] = self.get(key)
        self._cache = result
        return dict(result)

    def get_all_with_metadata(self) -> dict:
        """Get all settings grouped by category with metadata for UI rendering."""
        # Fetch all overrides at once
        all_db_settings = self._db.get_all_user_settings()
        overrides = {
            k[len(self.PREFIX):]: v
            for k, v in all_db_settings.items()
            if k.startswith(self.PREFIX)
        }

        categories = {}
        for key, defn in SETTING_DEFS.items():
            cat = defn.category
            if cat not in categories:
                cat_meta = CATEGORIES.get(cat, {"label": cat.title(), "order": 99})
                categories[cat] = {
                    "label": cat_meta["label"],
                    "order": cat_meta["order"],
                    "settings": {},
                }

            default = self.get_default(key)
            is_override = key in overrides
            value = overrides[key] if is_override else default

            setting_info = {
                "value": value,
                "default": default,
                "is_override": is_override,
                "type": defn.type.value,
                "label": defn.label,
                "description": defn.description,
            }
            if defn.min_val is not None:
                setting_info["min"] = defn.min_val
            if defn.max_val is not None:
                setting_info["max"] = defn.max_val

            categories[cat]["settings"][key] = setting_info

        return {
            "hardware_tier": self._tier,
            "categories": categories,
        }

    def set(self, key: str, value: Any) -> Any:
        """Set a user override. Validates and returns the stored value."""
        if key not in SETTING_DEFS:
            raise KeyError(f"Unknown setting: {key}")

        defn = SETTING_DEFS[key]
        value = self._coerce_and_validate(defn, value)
        self._db.set_user_setting(self._db_key(key), value)
        self._invalidate_cache()
        return value

    def set_bulk(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Set multiple user overrides. Returns all stored values."""
        result = {}
        for key, value in updates.items():
            result[key] = self.set(key, value)
        return result

    def delete(self, key: str):
        """Remove a user override, reverting to tier default."""
        if key not in SETTING_DEFS:
            raise KeyError(f"Unknown setting: {key}")
        self._db.delete_user_setting(self._db_key(key))
        self._invalidate_cache()

    def _coerce_and_validate(self, defn: SettingDef, value: Any) -> Any:
        """Coerce to correct type and validate constraints."""
        # Type coercion
        if defn.type == SettingType.INT:
            value = int(value)
        elif defn.type == SettingType.FLOAT:
            value = float(value)
        elif defn.type == SettingType.BOOL:
            if isinstance(value, str):
                value = value.lower() in ("true", "1", "yes")
            value = bool(value)
        elif defn.type == SettingType.STRING:
            value = str(value)

        # Range validation
        if defn.min_val is not None and isinstance(value, (int, float)):
            if value < defn.min_val:
                raise ValueError(f"{defn.key}: {value} below minimum {defn.min_val}")
        if defn.max_val is not None and isinstance(value, (int, float)):
            if value > defn.max_val:
                raise ValueError(f"{defn.key}: {value} above maximum {defn.max_val}")

        return value


# ============================================================================
# Module-level singleton
# ============================================================================

_settings_manager: Optional[SettingsManager] = None


def init_settings(db, tier: str) -> SettingsManager:
    """Initialize the global settings manager. Called once at startup."""
    global _settings_manager
    _settings_manager = SettingsManager(db, tier)
    return _settings_manager


def get_settings_manager() -> SettingsManager:
    """Get the global settings manager. Must be called after init_settings()."""
    if _settings_manager is None:
        raise RuntimeError("Settings not initialized. Call init_settings() during startup.")
    return _settings_manager


def get_setting(key: str) -> Any:
    """Convenience: get a single resolved setting value."""
    return get_settings_manager().get(key)


def migrate_env_vars(mgr: SettingsManager) -> int:
    """Migrate deprecated env vars to settings overrides.

    Runs once at startup. Only migrates if the env var is set AND no
    user override already exists for that setting.

    Returns the number of env vars migrated.
    """
    import os

    migrated = 0
    for env_var, setting_key in ENV_VAR_MIGRATION.items():
        env_value = os.environ.get(env_var)
        if env_value is None:
            continue

        # Don't overwrite existing user override
        override = mgr._db.get_user_setting(mgr._db_key(setting_key))
        if override is not None:
            continue

        try:
            mgr.set(setting_key, env_value)
            migrated += 1
            logger.warning(
                f"Migrated env var {env_var}={env_value} → setting '{setting_key}'. "
                f"You can remove {env_var} from your environment."
            )
        except (ValueError, KeyError) as e:
            logger.warning(f"Failed to migrate env var {env_var}: {e}")

    return migrated

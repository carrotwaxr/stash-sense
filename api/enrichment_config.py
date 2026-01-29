"""
Configuration system for multi-source enrichment.

Loads from YAML config file with CLI override support.

See: docs/plans/2026-01-29-multi-source-enrichment-design.md
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class QualityFilters:
    """Quality filter settings."""
    min_face_size: int = 80
    min_image_size: int = 400
    min_detection_confidence: float = 0.8
    max_face_angle: float = 45.0
    prefer_single_face: bool = True


@dataclass
class GlobalSettings:
    """Global enrichment settings."""
    max_faces_per_performer: int = 20
    default_rate_limit: int = 60  # requests per minute
    quality_filters: QualityFilters = field(default_factory=QualityFilters)


@dataclass
class SourceConfig:
    """Configuration for a single source."""
    name: str
    enabled: bool = True
    url: Optional[str] = None
    rate_limit: int = 60  # requests per minute
    max_faces: int = 5
    priority: int = 10
    trust_level: str = "medium"  # high, medium, low
    gender_filter: Optional[str] = None  # None, "female", "male"
    needs_flaresolverr: bool = False
    source_type: str = "stash_box"  # stash_box or reference_site

    def should_process_performer(self, gender: Optional[str]) -> bool:
        """Check if this source should process a performer based on gender."""
        if self.gender_filter is None:
            return True
        if gender is None:
            return True  # Unknown gender = process anyway
        return gender.upper() == self.gender_filter.upper()


class EnrichmentConfig:
    """
    Configuration manager for multi-source enrichment.

    Loads configuration from YAML file with CLI override support.

    Usage:
        # Load from default location
        config = EnrichmentConfig()

        # Load from specific file with CLI overrides
        config = EnrichmentConfig(
            config_path="custom_sources.yaml",
            cli_sources=["stashdb", "babepedia"],
            cli_disabled_sources=["pornpics"],
            cli_source_max_faces={"stashdb": 10}
        )

        # Get enabled sources
        for name in config.get_enabled_sources():
            source = config.get_source(name)
            print(f"{name}: {source.rate_limit} req/min")
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        cli_sources: Optional[list[str]] = None,
        cli_disabled_sources: Optional[list[str]] = None,
        cli_source_max_faces: Optional[dict[str, int]] = None,
        cli_max_faces_total: Optional[int] = None,
    ):
        self.config_path = Path(config_path) if config_path else None
        self.cli_sources = cli_sources
        self.cli_disabled_sources = cli_disabled_sources or []
        self.cli_source_max_faces = cli_source_max_faces or {}
        self.cli_max_faces_total = cli_max_faces_total

        self.global_settings = GlobalSettings()
        self._sources: dict[str, SourceConfig] = {}

        self._load_config()
        self._apply_cli_overrides()

    def _load_config(self):
        """Load configuration from YAML file or use defaults."""
        if self.config_path and self.config_path.exists():
            with open(self.config_path) as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        # Load global settings
        global_data = data.get("global", {})
        quality_data = global_data.get("quality_filters", {})

        self.global_settings = GlobalSettings(
            max_faces_per_performer=global_data.get("max_faces_per_performer", 20),
            default_rate_limit=global_data.get("default_rate_limit", 60),
            quality_filters=QualityFilters(
                min_face_size=quality_data.get("min_face_size", 80),
                min_image_size=quality_data.get("min_image_size", 400),
                min_detection_confidence=quality_data.get("min_detection_confidence", 0.8),
                max_face_angle=quality_data.get("max_face_angle", 45.0),
                prefer_single_face=quality_data.get("prefer_single_face", True),
            ),
        )

        # Load stash-box sources
        for name, source_data in data.get("stash_boxes", {}).items():
            self._sources[name] = SourceConfig(
                name=name,
                source_type="stash_box",
                enabled=source_data.get("enabled", True),
                url=source_data.get("url"),
                rate_limit=source_data.get("rate_limit", self.global_settings.default_rate_limit),
                max_faces=source_data.get("max_faces", 5),
                priority=source_data.get("priority", 10),
                trust_level=source_data.get("trust_level", "high"),
                gender_filter=source_data.get("gender_filter"),
                needs_flaresolverr=source_data.get("needs_flaresolverr", False),
            )

        # Load reference site sources
        for name, source_data in data.get("reference_sites", {}).items():
            self._sources[name] = SourceConfig(
                name=name,
                source_type="reference_site",
                enabled=source_data.get("enabled", True),
                url=source_data.get("url"),
                rate_limit=source_data.get("rate_limit", self.global_settings.default_rate_limit),
                max_faces=source_data.get("max_faces", 5),
                priority=source_data.get("priority", 10),
                trust_level=source_data.get("trust_level", "medium"),
                gender_filter=source_data.get("gender_filter"),
                needs_flaresolverr=source_data.get("needs_flaresolverr", False),
            )

    def _apply_cli_overrides(self):
        """Apply CLI argument overrides to configuration."""
        # Override global max faces
        if self.cli_max_faces_total is not None:
            self.global_settings.max_faces_per_performer = self.cli_max_faces_total

        # Override per-source max faces
        for source_name, max_faces in self.cli_source_max_faces.items():
            if source_name in self._sources:
                self._sources[source_name].max_faces = max_faces

    def get_source(self, name: str) -> SourceConfig:
        """Get configuration for a specific source."""
        if name not in self._sources:
            raise KeyError(f"Unknown source: {name}")
        return self._sources[name]

    def get_enabled_sources(self, source_type: Optional[str] = None) -> list[str]:
        """
        Get list of enabled source names.

        Args:
            source_type: Filter by type ("stash_box" or "reference_site")

        Returns:
            List of enabled source names, sorted by priority
        """
        enabled = []

        for name, source in self._sources.items():
            # Check if source type matches filter
            if source_type and source.source_type != source_type:
                continue

            # Check if enabled in config
            if not source.enabled:
                continue

            # Check CLI disabled list
            if name in self.cli_disabled_sources:
                continue

            # Check CLI sources whitelist (if provided)
            if self.cli_sources is not None and name not in self.cli_sources:
                continue

            enabled.append(name)

        # Sort by priority
        enabled.sort(key=lambda n: self._sources[n].priority)
        return enabled

    def get_stash_box_sources(self) -> list[str]:
        """Get enabled stash-box sources."""
        return self.get_enabled_sources(source_type="stash_box")

    def get_reference_site_sources(self) -> list[str]:
        """Get enabled reference site sources."""
        return self.get_enabled_sources(source_type="reference_site")

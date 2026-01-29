# Multi-Source Enrichment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build infrastructure to enrich performer database with faces from multiple sources in parallel.

**Architecture:** Two-phase approach - stash-boxes first (create performers), reference sites second (enrich existing). Single write queue per phase to prevent race conditions. Config file + CLI overrides for source management.

**Tech Stack:** Python 3.11+, asyncio for concurrency, YAML for config, SQLite for persistence, pytest for testing.

**Design Doc:** [2026-01-29-multi-source-enrichment-design.md](2026-01-29-multi-source-enrichment-design.md)

---

## Task 1: Source Configuration System

**Files:**
- Create: `api/enrichment_config.py`
- Create: `api/sources.yaml` (default config)
- Test: `api/tests/test_enrichment_config.py`

**Step 1: Create test directory and write failing test**

```bash
mkdir -p api/tests
touch api/tests/__init__.py
```

```python
# api/tests/test_enrichment_config.py
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
```

**Step 2: Run test to verify it fails**

```bash
cd api && python -m pytest tests/test_enrichment_config.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'enrichment_config'`

**Step 3: Write minimal implementation**

```python
# api/enrichment_config.py
"""
Configuration system for multi-source enrichment.

Loads from YAML config file with CLI override support.
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
```

**Step 4: Run tests to verify they pass**

```bash
cd api && python -m pytest tests/test_enrichment_config.py -v
```

Expected: All tests PASS

**Step 5: Create default config file**

```yaml
# api/sources.yaml
# Multi-source enrichment configuration
# See: docs/plans/2026-01-29-multi-source-enrichment-design.md

global:
  max_faces_per_performer: 20
  default_rate_limit: 60

  quality_filters:
    min_face_size: 80
    min_image_size: 400
    min_detection_confidence: 0.8
    max_face_angle: 45.0
    prefer_single_face: true

stash_boxes:
  stashdb:
    enabled: true
    url: "https://stashdb.org/graphql"
    rate_limit: 240
    max_faces: 5
    priority: 1
    trust_level: high

  theporndb:
    enabled: true
    url: "https://api.theporndb.net"
    rate_limit: 240
    max_faces: 5
    priority: 2
    trust_level: high

  pmvstash:
    enabled: false  # Enable after testing
    url: "https://pmvstash.org/graphql"
    rate_limit: 300
    max_faces: 3
    priority: 3
    trust_level: high

  javstash:
    enabled: false  # Enable after testing
    url: "https://javstash.org/graphql"
    rate_limit: 300
    max_faces: 3
    priority: 4
    trust_level: high

  fansdb:
    enabled: false  # Enable after testing
    url: "https://fansdb.cc/graphql"
    rate_limit: 240
    max_faces: 3
    priority: 5
    trust_level: high

reference_sites:
  babepedia:
    enabled: false  # Enable after testing
    trust_level: high
    rate_limit: 60
    max_faces: 5
    gender_filter: female
    needs_flaresolverr: true

  iafd:
    enabled: false
    trust_level: high
    rate_limit: 120
    max_faces: 3
    needs_flaresolverr: true

  freeones:
    enabled: false
    trust_level: medium
    rate_limit: 30
    max_faces: 5
    gender_filter: female
    needs_flaresolverr: false

  boobpedia:
    enabled: false
    trust_level: medium
    rate_limit: 60
    max_faces: 3
    gender_filter: female
    needs_flaresolverr: false

  pornpics:
    enabled: false
    trust_level: low
    rate_limit: 60
    max_faces: 5
    needs_flaresolverr: false

  elitebabes:
    enabled: false
    trust_level: low
    rate_limit: 60
    max_faces: 5
    gender_filter: female
    needs_flaresolverr: false

  javdatabase:
    enabled: false
    trust_level: medium
    rate_limit: 60
    max_faces: 3
    needs_flaresolverr: false
```

**Step 6: Commit**

```bash
git add api/enrichment_config.py api/sources.yaml api/tests/
git commit -m "feat: add source configuration system for multi-source enrichment

- YAML config file with global settings and per-source configuration
- CLI override support (--sources, --disable-source, --source-max-faces)
- Trust levels (high/medium/low) and gender filters
- Quality filter settings (face size, image size, confidence, angle)"
```

---

## Task 2: Quality Filters Module

**Files:**
- Create: `api/quality_filters.py`
- Test: `api/tests/test_quality_filters.py`
- Modify: `api/embeddings.py` (add face angle estimation)

**Step 1: Write failing test**

```python
# api/tests/test_quality_filters.py
"""Tests for face quality filtering."""
import pytest
import numpy as np
from unittest.mock import MagicMock


class TestQualityFilters:
    """Test quality filter logic."""

    def test_face_too_small_rejected(self):
        """Faces below minimum size are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_face_size=80)
        qf = QualityFilter(filters)

        # 50x50 face should be rejected
        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 50, "h": 50}
        face.confidence = 0.95

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is False
        assert "face_too_small" in result.rejection_reason

    def test_face_large_enough_accepted(self):
        """Faces at or above minimum size pass."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_face_size=80)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 120}
        face.confidence = 0.95

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is True

    def test_image_too_small_rejected(self):
        """Images below minimum resolution are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_image_size=400)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95

        # 300x200 image should be rejected
        result = qf.check_face(face, image_width=300, image_height=200)
        assert result.passed is False
        assert "image_too_small" in result.rejection_reason

    def test_low_confidence_rejected(self):
        """Faces with low detection confidence are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(min_detection_confidence=0.8)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.6

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is False
        assert "low_confidence" in result.rejection_reason

    def test_extreme_angle_rejected(self):
        """Faces at extreme angles are rejected."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(max_face_angle=45.0)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95
        face.yaw = 60.0  # Looking sideways

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is False
        assert "extreme_angle" in result.rejection_reason

    def test_multi_face_rejected_for_high_trust(self):
        """Multiple faces in image rejected for high-trust sources."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(prefer_single_face=True)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95

        result = qf.check_face(
            face,
            image_width=800,
            image_height=600,
            total_faces_in_image=3,
            trust_level="high"
        )
        assert result.passed is False
        assert "multi_face" in result.rejection_reason

    def test_multi_face_allowed_for_low_trust(self):
        """Multiple faces in image allowed for low-trust (will be clustered)."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(prefer_single_face=True)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95

        result = qf.check_face(
            face,
            image_width=800,
            image_height=600,
            total_faces_in_image=3,
            trust_level="low"
        )
        # Low trust doesn't apply single-face filter
        assert result.passed is True

    def test_no_angle_info_passes(self):
        """Faces without angle info pass angle check."""
        from quality_filters import QualityFilter, QualityFilters

        filters = QualityFilters(max_face_angle=45.0)
        qf = QualityFilter(filters)

        face = MagicMock()
        face.bbox = {"x": 0, "y": 0, "w": 100, "h": 100}
        face.confidence = 0.95
        face.yaw = None  # No angle info

        result = qf.check_face(face, image_width=800, image_height=600)
        assert result.passed is True
```

**Step 2: Run test to verify it fails**

```bash
cd api && python -m pytest tests/test_quality_filters.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'quality_filters'`

**Step 3: Write minimal implementation**

```python
# api/quality_filters.py
"""
Quality filters for face detection during enrichment.

Filters out low-quality faces that would degrade recognition accuracy.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class QualityFilters:
    """Quality filter configuration."""
    min_face_size: int = 80
    min_image_size: int = 400
    min_detection_confidence: float = 0.8
    max_face_angle: float = 45.0
    prefer_single_face: bool = True


@dataclass
class FilterResult:
    """Result of quality filter check."""
    passed: bool
    rejection_reason: Optional[str] = None
    details: Optional[dict] = None


class QualityFilter:
    """
    Applies quality filters to detected faces.

    Usage:
        filters = QualityFilters(min_face_size=80)
        qf = QualityFilter(filters)

        result = qf.check_face(detected_face, image_width, image_height)
        if result.passed:
            # Process face
        else:
            print(f"Rejected: {result.rejection_reason}")
    """

    def __init__(self, filters: QualityFilters):
        self.filters = filters

    def check_face(
        self,
        face,
        image_width: int,
        image_height: int,
        total_faces_in_image: int = 1,
        trust_level: str = "medium",
    ) -> FilterResult:
        """
        Check if a detected face passes quality filters.

        Args:
            face: DetectedFace-like object with bbox, confidence, and optional yaw
            image_width: Width of source image
            image_height: Height of source image
            total_faces_in_image: Number of faces detected in the image
            trust_level: Source trust level ("high", "medium", "low")

        Returns:
            FilterResult with passed status and rejection reason if failed
        """
        # Check image resolution
        min_dimension = min(image_width, image_height)
        if min_dimension < self.filters.min_image_size:
            return FilterResult(
                passed=False,
                rejection_reason="image_too_small",
                details={"min_dimension": min_dimension, "required": self.filters.min_image_size},
            )

        # Check face size
        face_width = face.bbox.get("w", 0)
        face_height = face.bbox.get("h", 0)
        face_size = min(face_width, face_height)

        if face_size < self.filters.min_face_size:
            return FilterResult(
                passed=False,
                rejection_reason="face_too_small",
                details={"face_size": face_size, "required": self.filters.min_face_size},
            )

        # Check detection confidence
        if face.confidence < self.filters.min_detection_confidence:
            return FilterResult(
                passed=False,
                rejection_reason="low_confidence",
                details={"confidence": face.confidence, "required": self.filters.min_detection_confidence},
            )

        # Check face angle (if available)
        yaw = getattr(face, "yaw", None)
        if yaw is not None and abs(yaw) > self.filters.max_face_angle:
            return FilterResult(
                passed=False,
                rejection_reason="extreme_angle",
                details={"yaw": yaw, "max_allowed": self.filters.max_face_angle},
            )

        # Check single face preference (only for high-trust sources)
        if (
            self.filters.prefer_single_face
            and trust_level == "high"
            and total_faces_in_image > 1
        ):
            return FilterResult(
                passed=False,
                rejection_reason="multi_face",
                details={"faces_in_image": total_faces_in_image},
            )

        return FilterResult(passed=True)

    def check_image(self, image_width: int, image_height: int) -> FilterResult:
        """Quick check if image resolution is acceptable."""
        min_dimension = min(image_width, image_height)
        if min_dimension < self.filters.min_image_size:
            return FilterResult(
                passed=False,
                rejection_reason="image_too_small",
                details={"min_dimension": min_dimension, "required": self.filters.min_image_size},
            )
        return FilterResult(passed=True)
```

**Step 4: Run tests to verify they pass**

```bash
cd api && python -m pytest tests/test_quality_filters.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/quality_filters.py api/tests/test_quality_filters.py
git commit -m "feat: add quality filters for face detection

- Minimum face size filter (80px default)
- Minimum image resolution filter (400px default)
- Detection confidence threshold (0.8 default)
- Face angle filter (45° max from frontal)
- Single-face preference for high-trust sources"
```

---

## Task 3: Write Queue Infrastructure

**Files:**
- Create: `api/write_queue.py`
- Test: `api/tests/test_write_queue.py`

**Step 1: Write failing test**

```python
# api/tests/test_write_queue.py
"""Tests for async write queue."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock


class TestWriteQueue:
    """Test write queue behavior."""

    @pytest.mark.asyncio
    async def test_queue_processes_messages_in_order(self):
        """Messages are processed in FIFO order."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        processed = []

        async def handler(msg):
            processed.append(msg.performer_id)
            await asyncio.sleep(0.01)  # Simulate work

        queue = WriteQueue(handler)
        await queue.start()

        # Enqueue messages
        for i in range(5):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        # Wait for processing
        await queue.wait_until_empty()
        await queue.stop()

        assert processed == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_queue_handles_errors_gracefully(self):
        """Errors in handler don't crash the queue."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        call_count = 0

        async def failing_handler(msg):
            nonlocal call_count
            call_count += 1
            if msg.performer_id == 1:
                raise ValueError("Simulated error")

        queue = WriteQueue(failing_handler)
        await queue.start()

        # Enqueue messages - middle one will fail
        for i in range(3):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        await queue.wait_until_empty()
        await queue.stop()

        # All messages were attempted despite error
        assert call_count == 3
        assert queue.stats.errors == 1

    @pytest.mark.asyncio
    async def test_queue_tracks_statistics(self):
        """Queue tracks processing statistics."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        async def handler(msg):
            await asyncio.sleep(0.001)

        queue = WriteQueue(handler)
        await queue.start()

        for i in range(10):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        await queue.wait_until_empty()
        await queue.stop()

        assert queue.stats.processed == 10
        assert queue.stats.errors == 0

    @pytest.mark.asyncio
    async def test_queue_graceful_shutdown(self):
        """Queue processes remaining messages on shutdown."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        processed = []

        async def handler(msg):
            processed.append(msg.performer_id)
            await asyncio.sleep(0.05)

        queue = WriteQueue(handler)
        await queue.start()

        # Enqueue many messages
        for i in range(5):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        # Stop with grace period
        await queue.stop(timeout=5.0)

        # All messages should be processed
        assert len(processed) == 5
```

**Step 2: Run test to verify it fails**

```bash
cd api && python -m pytest tests/test_write_queue.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'write_queue'`

**Step 3: Write minimal implementation**

```python
# api/write_queue.py
"""
Async write queue for serializing database writes.

Multiple scrapers can enqueue writes concurrently.
Single consumer processes them sequentially to prevent race conditions.
"""
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable
import numpy as np

logger = logging.getLogger(__name__)


class WriteOperation(Enum):
    """Types of write operations."""
    CREATE_PERFORMER = "create_performer"
    UPDATE_PERFORMER = "update_performer"
    ADD_EMBEDDING = "add_embedding"
    ADD_STASH_ID = "add_stash_id"
    ADD_EXTERNAL_URL = "add_external_url"
    ADD_ALIAS = "add_alias"


@dataclass
class WriteMessage:
    """Message for the write queue."""
    operation: WriteOperation
    source: str

    # For CREATE/UPDATE performer
    performer_data: Optional[dict] = None

    # For ADD_EMBEDDING
    performer_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    embedding_type: Optional[str] = None
    image_url: Optional[str] = None
    quality_score: Optional[float] = None

    # For ADD_STASH_ID
    endpoint: Optional[str] = None
    stashbox_id: Optional[str] = None

    # For ADD_EXTERNAL_URL
    url: Optional[str] = None
    site: Optional[str] = None

    # For ADD_ALIAS
    alias: Optional[str] = None


@dataclass
class QueueStats:
    """Statistics for queue monitoring."""
    enqueued: int = 0
    processed: int = 0
    errors: int = 0
    by_operation: dict = field(default_factory=dict)
    by_source: dict = field(default_factory=dict)

    def record_processed(self, msg: WriteMessage):
        """Record a processed message."""
        self.processed += 1

        op = msg.operation.value
        self.by_operation[op] = self.by_operation.get(op, 0) + 1

        self.by_source[msg.source] = self.by_source.get(msg.source, 0) + 1

    def record_error(self, msg: WriteMessage):
        """Record an error."""
        self.errors += 1


class WriteQueue:
    """
    Async queue for serializing database writes.

    Usage:
        async def handle_write(msg: WriteMessage):
            if msg.operation == WriteOperation.ADD_EMBEDDING:
                db.add_embedding(...)

        queue = WriteQueue(handle_write)
        await queue.start()

        # From multiple scrapers concurrently:
        await queue.enqueue(WriteMessage(...))

        # Shutdown
        await queue.stop()
    """

    def __init__(
        self,
        handler: Callable[[WriteMessage], Awaitable[None]],
        max_size: int = 10000,
    ):
        self.handler = handler
        self.max_size = max_size
        self._queue: asyncio.Queue[WriteMessage] = asyncio.Queue(maxsize=max_size)
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
        self.stats = QueueStats()

    async def start(self):
        """Start the queue consumer."""
        if self._running:
            return

        self._running = True
        self._consumer_task = asyncio.create_task(self._consume())
        logger.info("Write queue started")

    async def stop(self, timeout: float = 30.0):
        """Stop the queue, processing remaining messages."""
        if not self._running:
            return

        self._running = False

        # Wait for queue to drain
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Queue drain timeout after {timeout}s, {self._queue.qsize()} messages remaining")

        # Cancel consumer
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Write queue stopped. Stats: {self.stats.processed} processed, {self.stats.errors} errors")

    async def enqueue(self, message: WriteMessage):
        """Add a message to the queue."""
        await self._queue.put(message)
        self.stats.enqueued += 1

    async def wait_until_empty(self):
        """Wait until all queued messages are processed."""
        await self._queue.join()

    @property
    def size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    async def _consume(self):
        """Consumer loop - processes messages sequentially."""
        while self._running or not self._queue.empty():
            try:
                # Wait for message with timeout to allow checking _running flag
                try:
                    message = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process message
                try:
                    await self.handler(message)
                    self.stats.record_processed(message)
                except Exception as e:
                    logger.error(f"Error processing {message.operation.value}: {e}")
                    self.stats.record_error(message)
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")
```

**Step 4: Run tests to verify they pass**

```bash
cd api && python -m pytest tests/test_write_queue.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/write_queue.py api/tests/test_write_queue.py
git commit -m "feat: add async write queue for serialized database writes

- AsyncIO-based queue with single consumer
- Graceful shutdown with timeout
- Statistics tracking (processed, errors, by operation/source)
- Handles errors without crashing consumer loop"
```

---

## Task 4: Per-Source Face Tracking

**Files:**
- Modify: `api/database.py` (add source tracking to faces, add query methods)
- Test: `api/tests/test_database_source_tracking.py`

**Step 1: Write failing test**

```python
# api/tests/test_database_source_tracking.py
"""Tests for per-source face tracking in database."""
import pytest
import tempfile
from pathlib import Path


class TestSourceTracking:
    """Test source tracking for faces."""

    @pytest.fixture
    def db(self):
        """Create a temporary database."""
        from database import PerformerDatabase

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = PerformerDatabase(db_path)
        yield db

        # Cleanup
        Path(db_path).unlink(missing_ok=True)

    def test_get_face_count_by_source(self, db):
        """Can get face count per source for a performer."""
        # Create performer
        performer_id = db.add_performer(
            canonical_name="Test Performer",
            gender="FEMALE",
        )
        db.add_stashbox_id(performer_id, "stashdb", "test-uuid")

        # Add faces from different sources
        db.add_face(performer_id, facenet_index=0, arcface_index=0, source="stashdb")
        db.add_face(performer_id, facenet_index=1, arcface_index=1, source="stashdb")
        db.add_face(performer_id, facenet_index=2, arcface_index=2, source="babepedia")
        db.add_face(performer_id, facenet_index=3, arcface_index=3, source="babepedia")
        db.add_face(performer_id, facenet_index=4, arcface_index=4, source="freeones")

        counts = db.get_face_counts_by_source(performer_id)

        assert counts["stashdb"] == 2
        assert counts["babepedia"] == 2
        assert counts["freeones"] == 1

    def test_check_source_limit(self, db):
        """Can check if source limit is reached."""
        performer_id = db.add_performer(canonical_name="Test", gender="FEMALE")

        # Add 5 faces from stashdb
        for i in range(5):
            db.add_face(performer_id, facenet_index=i, arcface_index=i, source="stashdb")

        # Check limits
        assert db.source_limit_reached(performer_id, "stashdb", max_faces=5) is True
        assert db.source_limit_reached(performer_id, "stashdb", max_faces=10) is False
        assert db.source_limit_reached(performer_id, "babepedia", max_faces=5) is False

    def test_add_face_with_source(self, db):
        """Faces are stored with their source."""
        performer_id = db.add_performer(canonical_name="Test", gender="FEMALE")

        db.add_face(
            performer_id,
            facenet_index=0,
            arcface_index=0,
            source="stashdb",
            image_url="https://example.com/image.jpg",
            quality_score=0.95,
        )

        faces = db.get_performer_faces(performer_id)
        assert len(faces) == 1
        assert faces[0].source_endpoint == "stashdb"
        assert faces[0].quality_score == 0.95
```

**Step 2: Run test to verify it fails**

```bash
cd api && python -m pytest tests/test_database_source_tracking.py -v
```

Expected: FAIL (methods don't exist yet)

**Step 3: Add methods to database.py**

Add to `api/database.py` after the existing `add_face` method (around line 500):

```python
    def add_face(
        self,
        performer_id: int,
        facenet_index: int,
        arcface_index: int,
        source: str,
        image_url: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> int:
        """
        Add a face embedding reference for a performer.

        Args:
            performer_id: Performer's database ID
            facenet_index: Index in Voyager facenet index
            arcface_index: Index in Voyager arcface index
            source: Source that provided this face (stashdb, babepedia, etc.)
            image_url: URL of source image
            quality_score: Quality score for this face

        Returns:
            Face ID
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO faces (performer_id, facenet_index, arcface_index, source_endpoint, image_url, quality_score)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (performer_id, facenet_index, arcface_index, source, image_url, quality_score),
            )
            face_id = cursor.lastrowid

            # Update face count
            conn.execute(
                "UPDATE performers SET face_count = face_count + 1, updated_at = datetime('now') WHERE id = ?",
                (performer_id,),
            )

            return face_id

    def get_face_counts_by_source(self, performer_id: int) -> dict[str, int]:
        """
        Get face counts per source for a performer.

        Returns:
            Dict mapping source name to face count
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT source_endpoint, COUNT(*) as count
                FROM faces
                WHERE performer_id = ?
                GROUP BY source_endpoint
                """,
                (performer_id,),
            )
            return {row["source_endpoint"]: row["count"] for row in cursor.fetchall()}

    def source_limit_reached(self, performer_id: int, source: str, max_faces: int) -> bool:
        """
        Check if a source has reached its face limit for a performer.

        Args:
            performer_id: Performer's database ID
            source: Source name to check
            max_faces: Maximum faces allowed from this source

        Returns:
            True if limit reached, False otherwise
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM faces
                WHERE performer_id = ? AND source_endpoint = ?
                """,
                (performer_id, source),
            )
            count = cursor.fetchone()["count"]
            return count >= max_faces

    def get_performer_faces(self, performer_id: int) -> list[Face]:
        """
        Get all faces for a performer.

        Returns:
            List of Face objects
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT id, performer_id, facenet_index, arcface_index, image_url,
                       source_endpoint, quality_score, created_at
                FROM faces
                WHERE performer_id = ?
                ORDER BY created_at
                """,
                (performer_id,),
            )
            return [
                Face(
                    id=row["id"],
                    performer_id=row["performer_id"],
                    facenet_index=row["facenet_index"],
                    arcface_index=row["arcface_index"],
                    image_url=row["image_url"],
                    source_endpoint=row["source_endpoint"],
                    quality_score=row["quality_score"],
                    created_at=row["created_at"],
                )
                for row in cursor.fetchall()
            ]
```

**Step 4: Run tests to verify they pass**

```bash
cd api && python -m pytest tests/test_database_source_tracking.py -v
```

Expected: All tests PASS

**Step 5: Commit**

```bash
git add api/database.py api/tests/test_database_source_tracking.py
git commit -m "feat: add per-source face tracking to database

- get_face_counts_by_source() returns dict of source -> count
- source_limit_reached() checks if source hit per-source max
- get_performer_faces() retrieves all faces for a performer
- add_face() now properly tracks source_endpoint"
```

---

## Task 5: Database Schema Migration for Source Tracking

**Files:**
- Modify: `api/database.py` (add migration for existing faces)
- Create: `api/migrate_face_sources.py` (one-time migration script)

**Step 1: Add schema migration**

Add to `api/database.py` in the `_migrate_schema` method:

```python
        if from_version < 4:
            # Ensure all existing faces have source_endpoint set
            # (existing faces are from StashDB)
            conn.executescript("""
                UPDATE faces SET source_endpoint = 'stashdb' WHERE source_endpoint IS NULL;
                UPDATE schema_version SET version = 4;
            """)
```

Update `SCHEMA_VERSION = 4` at the top of the file.

**Step 2: Create migration verification test**

```python
# api/tests/test_schema_migration.py
"""Tests for schema migrations."""
import pytest
import tempfile
import sqlite3
from pathlib import Path


class TestSchemaMigration:
    """Test database schema migrations."""

    def test_migrate_sets_source_on_existing_faces(self):
        """Migration sets source_endpoint='stashdb' on existing faces."""
        from database import PerformerDatabase

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        try:
            # Create old schema manually
            conn = sqlite3.connect(db_path)
            conn.executescript("""
                CREATE TABLE schema_version (version INTEGER PRIMARY KEY);
                INSERT INTO schema_version (version) VALUES (3);

                CREATE TABLE performers (
                    id INTEGER PRIMARY KEY,
                    canonical_name TEXT,
                    face_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE faces (
                    id INTEGER PRIMARY KEY,
                    performer_id INTEGER,
                    facenet_index INTEGER,
                    arcface_index INTEGER,
                    source_endpoint TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                INSERT INTO performers (id, canonical_name, face_count) VALUES (1, 'Test', 2);
                INSERT INTO faces (performer_id, facenet_index, arcface_index) VALUES (1, 0, 0);
                INSERT INTO faces (performer_id, facenet_index, arcface_index) VALUES (1, 1, 1);
            """)
            conn.close()

            # Open with our class - should trigger migration
            db = PerformerDatabase(db_path)

            # Check that faces now have source
            faces = db.get_performer_faces(1)
            assert len(faces) == 2
            assert all(f.source_endpoint == "stashdb" for f in faces)

        finally:
            Path(db_path).unlink(missing_ok=True)
```

**Step 3: Run test**

```bash
cd api && python -m pytest tests/test_schema_migration.py -v
```

**Step 4: Commit**

```bash
git add api/database.py api/tests/test_schema_migration.py
git commit -m "feat: add schema migration to set source on existing faces

- Migrates v3 -> v4 setting source_endpoint='stashdb' on existing faces
- Ensures backward compatibility with existing databases"
```

---

## Task 6: Scraper Progress Tracking Table

**Files:**
- Modify: `api/database.py` (add scrape_progress table and methods)
- Test: `api/tests/test_scrape_progress.py`

**Step 1: Write failing test**

```python
# api/tests/test_scrape_progress.py
"""Tests for scrape progress tracking."""
import pytest
import tempfile
from pathlib import Path
from datetime import datetime


class TestScrapeProgress:
    """Test scrape progress persistence."""

    @pytest.fixture
    def db(self):
        from database import PerformerDatabase

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        db = PerformerDatabase(db_path)
        yield db

        Path(db_path).unlink(missing_ok=True)

    def test_save_and_load_progress(self, db):
        """Can save and retrieve scrape progress."""
        db.save_scrape_progress(
            source="stashdb",
            last_processed_id="abc-123",
            performers_processed=1000,
            faces_added=2500,
            errors=5,
        )

        progress = db.get_scrape_progress("stashdb")

        assert progress is not None
        assert progress["last_processed_id"] == "abc-123"
        assert progress["performers_processed"] == 1000
        assert progress["faces_added"] == 2500
        assert progress["errors"] == 5

    def test_update_progress(self, db):
        """Progress can be updated."""
        db.save_scrape_progress(source="babepedia", last_processed_id="p1", performers_processed=100)
        db.save_scrape_progress(source="babepedia", last_processed_id="p2", performers_processed=200)

        progress = db.get_scrape_progress("babepedia")

        assert progress["last_processed_id"] == "p2"
        assert progress["performers_processed"] == 200

    def test_no_progress_returns_none(self, db):
        """Unknown source returns None."""
        progress = db.get_scrape_progress("unknown_source")
        assert progress is None

    def test_clear_progress(self, db):
        """Can clear progress for a source."""
        db.save_scrape_progress(source="test", last_processed_id="x", performers_processed=50)
        db.clear_scrape_progress("test")

        progress = db.get_scrape_progress("test")
        assert progress is None
```

**Step 2: Run test to verify it fails**

```bash
cd api && python -m pytest tests/test_scrape_progress.py -v
```

**Step 3: Add scrape_progress table and methods to database.py**

Add to schema in `_create_schema`:

```sql
            -- Scrape progress per source (for resume capability)
            CREATE TABLE IF NOT EXISTS scrape_progress (
                source TEXT PRIMARY KEY,
                last_processed_id TEXT,
                last_processed_time TEXT DEFAULT (datetime('now')),
                performers_processed INTEGER DEFAULT 0,
                faces_added INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0
            );
```

Add methods:

```python
    def save_scrape_progress(
        self,
        source: str,
        last_processed_id: str,
        performers_processed: int = 0,
        faces_added: int = 0,
        errors: int = 0,
    ):
        """Save scrape progress for resume capability."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO scrape_progress (source, last_processed_id, performers_processed, faces_added, errors)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    last_processed_id = excluded.last_processed_id,
                    last_processed_time = datetime('now'),
                    performers_processed = excluded.performers_processed,
                    faces_added = excluded.faces_added,
                    errors = excluded.errors
                """,
                (source, last_processed_id, performers_processed, faces_added, errors),
            )

    def get_scrape_progress(self, source: str) -> Optional[dict]:
        """Get scrape progress for a source."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT last_processed_id, last_processed_time, performers_processed, faces_added, errors
                FROM scrape_progress
                WHERE source = ?
                """,
                (source,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "last_processed_id": row["last_processed_id"],
                "last_processed_time": row["last_processed_time"],
                "performers_processed": row["performers_processed"],
                "faces_added": row["faces_added"],
                "errors": row["errors"],
            }

    def clear_scrape_progress(self, source: str):
        """Clear progress for a source (for fresh start)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM scrape_progress WHERE source = ?", (source,))
```

Add migration:

```python
        if from_version < 5:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scrape_progress (
                    source TEXT PRIMARY KEY,
                    last_processed_id TEXT,
                    last_processed_time TEXT DEFAULT (datetime('now')),
                    performers_processed INTEGER DEFAULT 0,
                    faces_added INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0
                );
                UPDATE schema_version SET version = 5;
            """)
```

Update `SCHEMA_VERSION = 5`.

**Step 4: Run tests**

```bash
cd api && python -m pytest tests/test_scrape_progress.py -v
```

**Step 5: Commit**

```bash
git add api/database.py api/tests/test_scrape_progress.py
git commit -m "feat: add scrape progress tracking for resume capability

- scrape_progress table tracks last_processed_id per source
- save/get/clear methods for progress management
- Enables resuming interrupted enrichment runs"
```

---

## Summary: Infrastructure Tasks

| Task | Description | Files |
|------|-------------|-------|
| 1 | Source configuration system | `enrichment_config.py`, `sources.yaml` |
| 2 | Quality filters module | `quality_filters.py` |
| 3 | Write queue infrastructure | `write_queue.py` |
| 4 | Per-source face tracking | `database.py` updates |
| 5 | Schema migration | `database.py` migration |
| 6 | Scrape progress tracking | `database.py` scrape_progress table |

---

## Next Phase: Scraper Implementation

After infrastructure is complete, the next tasks would be:

- Task 7: Base scraper interface
- Task 8: Refactor StashDB scraper to new interface
- Task 9: Scraper coordinator for parallel execution
- Task 10: ThePornDB scraper implementation
- Task 11: Reference site scraper (Babepedia)
- Task 12: Trust-level validation logic
- Task 13: CLI entry point

These will be detailed in a follow-up plan once infrastructure is validated.

---

*Plan created: 2026-01-29*

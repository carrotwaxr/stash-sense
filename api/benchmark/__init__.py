"""Benchmark framework for performer identification."""

from api.benchmark.models import (
    ExpectedPerformer,
    TestScene,
    BenchmarkParams,
    SceneResult,
    PerformerResult,
    AggregateMetrics,
    BenchmarkState,
)
from api.benchmark.scene_selector import (
    SceneSelector,
    STASHDB_ENDPOINT,
    MIN_RESOLUTION_WIDTH,
    MIN_RESOLUTION_HEIGHT,
    MIN_PERFORMERS,
    WELL_COVERED_THRESHOLD,
)
from api.benchmark.test_executor import TestExecutor
from api.benchmark.analyzer import Analyzer, MIN_DB_COVERAGE
from api.benchmark.reporter import Reporter
from api.benchmark.runner import BenchmarkRunner
from api.benchmark.config import BenchmarkConfig

__all__ = [
    # Models
    "ExpectedPerformer",
    "TestScene",
    "BenchmarkParams",
    "SceneResult",
    "PerformerResult",
    "AggregateMetrics",
    "BenchmarkState",
    # Scene selector
    "SceneSelector",
    "STASHDB_ENDPOINT",
    "MIN_RESOLUTION_WIDTH",
    "MIN_RESOLUTION_HEIGHT",
    "MIN_PERFORMERS",
    "WELL_COVERED_THRESHOLD",
    # Test executor
    "TestExecutor",
    # Analyzer
    "Analyzer",
    "MIN_DB_COVERAGE",
    # Reporter
    "Reporter",
    # Runner
    "BenchmarkRunner",
    # Config
    "BenchmarkConfig",
]

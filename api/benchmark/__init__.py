"""Benchmark framework for performer identification."""

from benchmark.models import (
    ExpectedPerformer,
    TestScene,
    BenchmarkParams,
    SceneResult,
    PerformerResult,
    AggregateMetrics,
    BenchmarkState,
)
from benchmark.scene_selector import (
    SceneSelector,
    STASHDB_ENDPOINT,
    MIN_RESOLUTION_WIDTH,
    MIN_RESOLUTION_HEIGHT,
    MIN_PERFORMERS,
    WELL_COVERED_THRESHOLD,
)
from benchmark.test_executor import TestExecutor
from benchmark.analyzer import Analyzer, MIN_DB_COVERAGE
from benchmark.reporter import Reporter
from benchmark.runner import BenchmarkRunner
from benchmark.config import BenchmarkConfig

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

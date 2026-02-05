"""Configuration for the benchmark framework.

This module provides the BenchmarkConfig dataclass for configuring
benchmark runs, including scene selection, iteration control, and output settings.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution.

    Attributes:
        min_scenes: Minimum number of scenes to select for testing.
        scene_ids: Optional list of specific scene IDs to test. If set, only these scenes are tested.
        max_rounds: Maximum number of benchmark rounds to run.
        start_round: Round number to start from (for resuming).
        sample_fraction: Fraction of scenes to sample for parameter variations.
        output_dir: Directory path for benchmark output files.
    """

    # Scene selection
    min_scenes: int = 100
    scene_ids: Optional[list[str]] = None  # If set, only test these scenes

    # Iteration control
    max_rounds: int = 4
    start_round: int = 1
    sample_fraction: float = 0.3

    # Output
    output_dir: str = "benchmark_results"

    # Quick mode settings (private)
    _quick_min_scenes: int = field(default=20, repr=False)
    _quick_max_rounds: int = field(default=2, repr=False)

    @classmethod
    def quick(cls) -> "BenchmarkConfig":
        """Create a quick test configuration.

        Returns a BenchmarkConfig with reduced scene count and rounds
        for fast testing.

        Returns:
            BenchmarkConfig configured for quick testing (20 scenes, 2 rounds).
        """
        return cls(min_scenes=20, max_rounds=2)

    @classmethod
    def from_args(cls, args) -> "BenchmarkConfig":
        """Create a BenchmarkConfig from parsed command line arguments.

        Args:
            args: Namespace from argparse containing CLI arguments.

        Returns:
            BenchmarkConfig initialized from the provided arguments.
        """
        if args.quick:
            config = cls.quick()
        else:
            config = cls()

        if args.scenes:
            config.scene_ids = [s.strip() for s in args.scenes.split(",")]
            config.min_scenes = len(config.scene_ids)

        if hasattr(args, "start_round") and args.start_round:
            config.start_round = args.start_round

        if hasattr(args, "output_dir") and args.output_dir:
            config.output_dir = args.output_dir

        return config

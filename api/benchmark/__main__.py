"""CLI entry point for the benchmark framework.

This module provides the command-line interface for running benchmarks
to evaluate the performer identification system.

Usage:
    python -m api.benchmark [options]

Examples:
    # Run full benchmark
    python -m api.benchmark

    # Quick test mode
    python -m api.benchmark --quick

    # Resume from checkpoint
    python -m api.benchmark --resume

    # Test specific scenes
    python -m api.benchmark --scenes "scene1,scene2,scene3"
"""

import argparse
import asyncio
import sys

from api.benchmark.config import BenchmarkConfig
from api.benchmark.runner import BenchmarkRunner


def parse_args(args: list[str] = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: List of arguments to parse. If None, uses sys.argv.

    Returns:
        Namespace containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Benchmark performer identification system"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick test mode (20 scenes, 2 rounds)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--scenes",
        type=str,
        help="Comma-separated scene IDs to test",
    )
    parser.add_argument(
        "--start-round",
        type=int,
        help="Start from specific round (requires --resume)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark_results",
        help="Output directory",
    )
    return parser.parse_args(args)


def create_runner(config: BenchmarkConfig) -> BenchmarkRunner:
    """Create a BenchmarkRunner with all dependencies.

    Args:
        config: BenchmarkConfig with configuration settings.

    Returns:
        BenchmarkRunner ready for execution.

    Note:
        For now, creates with None placeholders - will be wired up in integration.
    """
    # Import dependencies
    from api.benchmark.scene_selector import SceneSelector
    from api.benchmark.test_executor import TestExecutor
    from api.benchmark.analyzer import Analyzer
    from api.benchmark.reporter import Reporter

    # Create instances
    # Note: SceneSelector, TestExecutor may require additional setup
    # This will be completed in the integration task
    scene_selector = SceneSelector(db_reader=None)
    executor = TestExecutor(recognizer=None)
    analyzer = Analyzer()
    reporter = Reporter()

    return BenchmarkRunner(
        scene_selector=scene_selector,
        executor=executor,
        analyzer=analyzer,
        reporter=reporter,
        output_dir=config.output_dir,
    )


async def main(args: list[str] = None) -> int:
    """Main entry point for the benchmark CLI.

    Args:
        args: List of command line arguments. If None, uses sys.argv.

    Returns:
        Exit code (0 for success, 1 for error/interrupt).
    """
    parsed = parse_args(args)
    config = BenchmarkConfig.from_args(parsed)

    print("Benchmark Configuration:")
    print(f"  Min scenes: {config.min_scenes}")
    print(f"  Max rounds: {config.max_rounds}")
    print(f"  Output: {config.output_dir}")

    runner = create_runner(config)

    try:
        state = await runner.run(
            min_scenes=config.min_scenes,
            max_rounds=config.max_rounds,
            resume=parsed.resume,
        )
        print(f"\nBenchmark complete. Final accuracy: {state.current_best_accuracy*100:.1f}%")
        return 0
    except KeyboardInterrupt:
        print("\nBenchmark interrupted. Progress saved to checkpoint.")
        return 1
    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

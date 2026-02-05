"""Tests for benchmark CLI interface and configuration."""

import argparse
import pytest

from benchmark.config import BenchmarkConfig
from benchmark.__main__ import parse_args


class TestParseArgs:
    """Tests for CLI argument parsing."""

    def test_default_args(self):
        """Test default argument values."""
        args = parse_args([])
        assert args.quick is False
        assert args.resume is False
        assert args.scenes is None
        assert args.start_round is None
        assert args.output_dir == "benchmark_results"

    def test_quick_mode(self):
        """Test --quick flag sets quick mode."""
        args = parse_args(["--quick"])
        assert args.quick is True

    def test_resume(self):
        """Test --resume flag."""
        args = parse_args(["--resume"])
        assert args.resume is True

    def test_specific_scenes(self):
        """Test --scenes with comma-separated IDs."""
        args = parse_args(["--scenes", "123,456,789"])
        assert args.scenes == "123,456,789"

    def test_start_round(self):
        """Test --start-round option."""
        args = parse_args(["--start-round", "3"])
        assert args.start_round == 3

    def test_output_dir(self):
        """Test --output-dir option."""
        args = parse_args(["--output-dir", "/custom/path"])
        assert args.output_dir == "/custom/path"

    def test_combined_args(self):
        """Test multiple arguments combined."""
        args = parse_args([
            "--quick",
            "--resume",
            "--scenes", "abc,def",
            "--start-round", "2",
            "--output-dir", "/my/output"
        ])
        assert args.quick is True
        assert args.resume is True
        assert args.scenes == "abc,def"
        assert args.start_round == 2
        assert args.output_dir == "/my/output"


class TestBenchmarkConfig:
    """Tests for BenchmarkConfig dataclass."""

    def test_BenchmarkConfig_default(self):
        """Test default BenchmarkConfig values."""
        config = BenchmarkConfig()
        assert config.min_scenes == 100
        assert config.scene_ids is None
        assert config.max_rounds == 4
        assert config.start_round == 1
        assert config.sample_fraction == 0.3
        assert config.output_dir == "benchmark_results"

    def test_BenchmarkConfig_quick(self):
        """Test BenchmarkConfig.quick() factory method."""
        config = BenchmarkConfig.quick()
        assert config.min_scenes == 20
        assert config.max_rounds == 2
        # Other values should remain default
        assert config.scene_ids is None
        assert config.start_round == 1
        assert config.sample_fraction == 0.3
        assert config.output_dir == "benchmark_results"

    def test_BenchmarkConfig_from_args_default(self):
        """Test BenchmarkConfig.from_args with default args."""
        args = parse_args([])
        config = BenchmarkConfig.from_args(args)
        assert config.min_scenes == 100
        assert config.max_rounds == 4
        assert config.scene_ids is None
        assert config.output_dir == "benchmark_results"

    def test_BenchmarkConfig_from_args_quick(self):
        """Test BenchmarkConfig.from_args with quick mode."""
        args = parse_args(["--quick"])
        config = BenchmarkConfig.from_args(args)
        assert config.min_scenes == 20
        assert config.max_rounds == 2

    def test_BenchmarkConfig_from_args_scenes(self):
        """Test BenchmarkConfig.from_args with specific scenes."""
        args = parse_args(["--scenes", "123,456,789"])
        config = BenchmarkConfig.from_args(args)
        assert config.scene_ids == ["123", "456", "789"]
        assert config.min_scenes == 3  # Length of scene_ids list

    def test_BenchmarkConfig_from_args_scenes_with_spaces(self):
        """Test BenchmarkConfig.from_args handles spaces in scene IDs."""
        args = parse_args(["--scenes", " 123 , 456 , 789 "])
        config = BenchmarkConfig.from_args(args)
        assert config.scene_ids == ["123", "456", "789"]

    def test_BenchmarkConfig_from_args_start_round(self):
        """Test BenchmarkConfig.from_args with start round."""
        args = parse_args(["--start-round", "3"])
        config = BenchmarkConfig.from_args(args)
        assert config.start_round == 3

    def test_BenchmarkConfig_from_args_output_dir(self):
        """Test BenchmarkConfig.from_args with custom output directory."""
        args = parse_args(["--output-dir", "/custom/output"])
        config = BenchmarkConfig.from_args(args)
        assert config.output_dir == "/custom/output"

    def test_BenchmarkConfig_from_args_combined(self):
        """Test BenchmarkConfig.from_args with multiple options."""
        args = parse_args([
            "--quick",
            "--scenes", "a,b,c",
            "--start-round", "2",
            "--output-dir", "/my/path"
        ])
        config = BenchmarkConfig.from_args(args)
        # Quick mode values
        assert config.max_rounds == 2
        # But min_scenes is overridden by scenes list
        assert config.min_scenes == 3
        assert config.scene_ids == ["a", "b", "c"]
        assert config.start_round == 2
        assert config.output_dir == "/my/path"

    def test_BenchmarkConfig_private_fields(self):
        """Test private quick mode settings exist."""
        config = BenchmarkConfig()
        assert config._quick_min_scenes == 20
        assert config._quick_max_rounds == 2

    def test_BenchmarkConfig_repr_excludes_private(self):
        """Test that repr excludes private fields."""
        config = BenchmarkConfig()
        repr_str = repr(config)
        assert "_quick_min_scenes" not in repr_str
        assert "_quick_max_rounds" not in repr_str

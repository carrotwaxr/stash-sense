"""
Stash Sense Analyzers

Each analyzer detects a specific type of issue in the user's library
and creates recommendations for resolution.
"""

from .base import BaseAnalyzer, AnalysisResult
from .base_upstream import BaseUpstreamAnalyzer
from .duplicate_performer import DuplicatePerformerAnalyzer
from .duplicate_scene_files import DuplicateSceneFilesAnalyzer
from .duplicate_scenes import DuplicateScenesAnalyzer
from .upstream_performer import UpstreamPerformerAnalyzer

__all__ = [
    "BaseAnalyzer",
    "BaseUpstreamAnalyzer",
    "AnalysisResult",
    "DuplicatePerformerAnalyzer",
    "DuplicateSceneFilesAnalyzer",
    "DuplicateScenesAnalyzer",
    "UpstreamPerformerAnalyzer",
]

"""Pytest configuration for async tests."""

import sys
from unittest.mock import MagicMock

import pytest

# Provide dummy modules for ML/GPU dependencies that aren't available in CI.
# This allows pytest to collect (import) heavy-marked test files without error;
# the @pytest.mark.heavy marker then skips them at runtime.
# Packages with submodule imports (e.g., insightface.app) need each level mocked.
_ML_MODULES = [
    "cv2",
    "onnxruntime",
    "mediapipe",
    "voyager",
    "insightface",
    "insightface.app",
    "insightface.utils",
    "insightface.utils.face_align",
]
for _mod in _ML_MODULES:
    if _mod not in sys.modules:
        try:
            __import__(_mod)
        except (ImportError, ModuleNotFoundError):
            sys.modules[_mod] = MagicMock()


# Configure pytest-asyncio
pytest_plugins = ('pytest_asyncio',)


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()


class _StaticExclusion:
    def __init__(self, value):
        self.value = value

    def invalidate(self):
        pass

    async def get(self, stash, *, max_age=None, allow_stale=False):
        return self.value


@pytest.fixture(autouse=True)
def _empty_exclusion_list(monkeypatch):
    """Existing tests run with an empty #125 exclusion list.

    The real code fails closed when the list hasn't been loaded; tests of
    the exclusion list itself install a real ExclusionCache over FakeStash.
    """
    import curation_settings
    from analyzers.base import BaseAnalyzer

    monkeypatch.setattr(curation_settings, "_cache", _StaticExclusion(curation_settings.EMPTY))
    monkeypatch.setattr(BaseAnalyzer, "exclusion", curation_settings.EMPTY)

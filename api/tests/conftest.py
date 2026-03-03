"""Pytest configuration for async tests."""

import sys
from unittest.mock import MagicMock

import pytest

# Provide dummy modules for ML/GPU dependencies that aren't available in CI.
# This allows pytest to collect (import) heavy-marked test files without error;
# the @pytest.mark.heavy marker then skips them at runtime.
for _mod in ["cv2", "onnxruntime", "insightface", "mediapipe", "voyager"]:
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

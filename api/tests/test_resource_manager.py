"""Tests for the resource manager."""

import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from resource_manager import (
    ResourceGroup,
    ResourceManager,
    get_resource_manager,
    init_resource_manager,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def mgr():
    """Create a ResourceManager with a short idle timeout for testing."""
    return ResourceManager(idle_timeout_seconds=0.1)


def _make_loader(data="loaded_data"):
    """Create a mock loader that returns the given data."""
    loader = MagicMock(return_value=data)
    return loader


def _make_unloader():
    """Create a mock unloader."""
    return MagicMock()


# ============================================================================
# ResourceGroup dataclass tests
# ============================================================================

class TestResourceGroup:
    """Test the ResourceGroup dataclass."""

    def test_defaults(self):
        """ResourceGroup has correct defaults."""
        group = ResourceGroup(
            name="test",
            loader=lambda: None,
            unloader=lambda: None,
        )
        assert group.name == "test"
        assert group.data is None
        assert group.loaded is False
        assert group.last_access == 0.0
        assert group.load_time_seconds == 0.0


# ============================================================================
# Registration tests
# ============================================================================

class TestRegister:
    """Test resource group registration."""

    def test_register_creates_group(self, mgr):
        """register() creates a resource group that is not loaded."""
        loader = _make_loader()
        unloader = _make_unloader()
        mgr.register("face_data", loader, unloader)

        assert not mgr.is_loaded("face_data")

    def test_register_multiple_groups(self, mgr):
        """Can register multiple independent resource groups."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.register("tattoo_data", _make_loader(), _make_unloader())

        assert not mgr.is_loaded("face_data")
        assert not mgr.is_loaded("tattoo_data")


# ============================================================================
# require() tests
# ============================================================================

class TestRequire:
    """Test the require() method for lazy loading."""

    def test_not_loaded_initially(self, mgr):
        """Resource is not loaded after registration."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        assert not mgr.is_loaded("face_data")

    def test_require_calls_loader_and_marks_loaded(self, mgr):
        """require() calls loader and marks the resource as loaded."""
        loader = _make_loader("my_data")
        mgr.register("face_data", loader, _make_unloader())

        result = mgr.require("face_data")

        assert result == "my_data"
        assert mgr.is_loaded("face_data")
        loader.assert_called_once()

    def test_require_twice_only_calls_loader_once(self, mgr):
        """require() called twice only invokes the loader once (caching)."""
        loader = _make_loader("cached")
        mgr.register("face_data", loader, _make_unloader())

        result1 = mgr.require("face_data")
        result2 = mgr.require("face_data")

        assert result1 == "cached"
        assert result2 == "cached"
        loader.assert_called_once()

    def test_require_unknown_raises_key_error(self, mgr):
        """require() raises KeyError for unregistered resource group."""
        with pytest.raises(KeyError, match="not registered"):
            mgr.require("nonexistent")

    def test_require_with_failing_loader(self, mgr):
        """require() propagates loader exceptions, resource stays unloaded."""
        loader = MagicMock(side_effect=RuntimeError("disk full"))
        mgr.register("broken", loader, _make_unloader())

        with pytest.raises(RuntimeError, match="disk full"):
            mgr.require("broken")

        # Resource should still be unloaded
        assert not mgr.is_loaded("broken")

    def test_require_after_failed_loader_retries(self, mgr):
        """After a failed load, require() tries the loader again."""
        call_count = 0

        def flaky_loader():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient error")
            return "recovered_data"

        mgr.register("flaky", flaky_loader, _make_unloader())

        # First call fails
        with pytest.raises(RuntimeError):
            mgr.require("flaky")

        # Second call succeeds
        result = mgr.require("flaky")
        assert result == "recovered_data"
        assert mgr.is_loaded("flaky")
        assert call_count == 2

    def test_require_returns_loader_data(self, mgr):
        """require() returns exactly what the loader returns."""
        complex_data = {"models": [1, 2, 3], "index": object()}
        loader = _make_loader(complex_data)
        mgr.register("complex", loader, _make_unloader())

        result = mgr.require("complex")
        assert result is complex_data


# ============================================================================
# Idle unload tests
# ============================================================================

class TestCheckIdle:
    """Test idle timeout and automatic unloading."""

    def test_idle_unload_after_timeout(self, mgr):
        """check_idle() unloads resources that exceed the idle timeout."""
        unloader = _make_unloader()
        mgr.register("face_data", _make_loader(), unloader)
        mgr.require("face_data")
        assert mgr.is_loaded("face_data")

        # Wait for the idle timeout to expire
        time.sleep(0.15)
        mgr.check_idle()

        assert not mgr.is_loaded("face_data")
        unloader.assert_called_once()

    def test_access_resets_idle_timer(self, mgr):
        """Accessing a resource resets the idle timer."""
        unloader = _make_unloader()
        mgr.register("face_data", _make_loader(), unloader)
        mgr.require("face_data")

        # Wait partway through timeout
        time.sleep(0.06)
        mgr.require("face_data")  # Reset timer

        # Wait partway again (total from last access < timeout)
        time.sleep(0.06)
        mgr.check_idle()

        # Should still be loaded because timer was reset
        assert mgr.is_loaded("face_data")
        unloader.assert_not_called()

    def test_idle_unload_calls_gc_collect(self, mgr):
        """check_idle() calls gc.collect() after unloading."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.require("face_data")
        time.sleep(0.15)

        with patch("resource_manager.gc.collect") as mock_gc:
            mgr.check_idle()
            mock_gc.assert_called_once()

    def test_no_gc_collect_when_nothing_unloaded(self, mgr):
        """check_idle() does NOT call gc.collect() if nothing was unloaded."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        # Don't load it

        with patch("resource_manager.gc.collect") as mock_gc:
            mgr.check_idle()
            mock_gc.assert_not_called()

    def test_check_idle_skips_unloaded_groups(self, mgr):
        """check_idle() ignores groups that are not loaded."""
        unloader = _make_unloader()
        mgr.register("face_data", _make_loader(), unloader)
        # Don't load it

        time.sleep(0.15)
        mgr.check_idle()

        # Unloader should not be called for unloaded groups
        unloader.assert_not_called()


# ============================================================================
# unload_all() tests
# ============================================================================

class TestUnloadAll:
    """Test the unload_all() method."""

    def test_unload_all_unloads_everything(self, mgr):
        """unload_all() unloads all loaded resource groups."""
        unloader_a = _make_unloader()
        unloader_b = _make_unloader()

        mgr.register("face_data", _make_loader(), unloader_a)
        mgr.register("tattoo_data", _make_loader(), unloader_b)

        mgr.require("face_data")
        mgr.require("tattoo_data")
        assert mgr.is_loaded("face_data")
        assert mgr.is_loaded("tattoo_data")

        mgr.unload_all()

        assert not mgr.is_loaded("face_data")
        assert not mgr.is_loaded("tattoo_data")
        unloader_a.assert_called_once()
        unloader_b.assert_called_once()

    def test_unload_all_calls_gc_collect(self, mgr):
        """unload_all() calls gc.collect() after unloading."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.require("face_data")

        with patch("resource_manager.gc.collect") as mock_gc:
            mgr.unload_all()
            mock_gc.assert_called_once()

    def test_unload_all_no_gc_when_nothing_loaded(self, mgr):
        """unload_all() does NOT call gc.collect() if nothing was loaded."""
        mgr.register("face_data", _make_loader(), _make_unloader())

        with patch("resource_manager.gc.collect") as mock_gc:
            mgr.unload_all()
            mock_gc.assert_not_called()

    def test_unload_all_skips_unloaded_groups(self, mgr):
        """unload_all() only calls unloader for loaded groups."""
        unloader_a = _make_unloader()
        unloader_b = _make_unloader()

        mgr.register("face_data", _make_loader(), unloader_a)
        mgr.register("tattoo_data", _make_loader(), unloader_b)

        # Only load face_data
        mgr.require("face_data")

        mgr.unload_all()

        unloader_a.assert_called_once()
        unloader_b.assert_not_called()

    def test_unload_all_clears_data(self, mgr):
        """unload_all() sets data to None on unloaded groups."""
        mgr.register("face_data", _make_loader("important"), _make_unloader())
        mgr.require("face_data")

        mgr.unload_all()

        # After unloading, require should call the loader again
        loader = _make_loader("reloaded")
        mgr.register("face_data", loader, _make_unloader())
        result = mgr.require("face_data")
        assert result == "reloaded"


# ============================================================================
# get_status() tests
# ============================================================================

class TestGetStatus:
    """Test the get_status() method."""

    def test_status_unloaded_resource(self, mgr):
        """Status shows unloaded resource correctly."""
        mgr.register("face_data", _make_loader(), _make_unloader())

        status = mgr.get_status()

        assert "face_data" in status
        assert status["face_data"]["loaded"] is False
        assert status["face_data"]["idle_seconds"] is None
        assert status["face_data"]["load_time_seconds"] == 0.0

    def test_status_loaded_resource(self, mgr):
        """Status shows loaded resource with idle time and load time."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.require("face_data")

        status = mgr.get_status()

        assert status["face_data"]["loaded"] is True
        assert status["face_data"]["idle_seconds"] is not None
        assert status["face_data"]["idle_seconds"] >= 0.0
        assert status["face_data"]["load_time_seconds"] >= 0.0

    def test_status_multiple_groups(self, mgr):
        """Status includes all registered groups."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.register("tattoo_data", _make_loader(), _make_unloader())
        mgr.require("face_data")

        status = mgr.get_status()

        assert len(status) == 2
        assert status["face_data"]["loaded"] is True
        assert status["tattoo_data"]["loaded"] is False

    def test_status_load_time_recorded(self, mgr):
        """Load time is recorded after require()."""
        def slow_loader():
            time.sleep(0.05)
            return "data"

        mgr.register("slow", slow_loader, _make_unloader())
        mgr.require("slow")

        status = mgr.get_status()
        assert status["slow"]["load_time_seconds"] >= 0.05


# ============================================================================
# Thread safety tests
# ============================================================================

class TestThreadSafety:
    """Test thread safety of concurrent operations."""

    def test_concurrent_require_calls_loader_once(self):
        """Multiple threads calling require() simultaneously only load once."""
        mgr = ResourceManager(idle_timeout_seconds=60.0)
        load_count = 0
        load_lock = threading.Lock()

        def counting_loader():
            nonlocal load_count
            with load_lock:
                load_count += 1
            # Small delay to increase chance of race condition
            time.sleep(0.01)
            return "shared_data"

        mgr.register("shared", counting_loader, _make_unloader())

        barrier = threading.Barrier(10)
        results = [None] * 10
        errors = []

        def worker(idx):
            try:
                barrier.wait()
                results[idx] = mgr.require("shared")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Thread errors: {errors}"
        assert load_count == 1, f"Loader called {load_count} times, expected 1"
        assert all(r == "shared_data" for r in results)

    def test_concurrent_require_different_groups(self):
        """Multiple threads can load different groups concurrently."""
        mgr = ResourceManager(idle_timeout_seconds=60.0)

        results = {}
        errors = []

        def make_loader(name):
            def loader():
                time.sleep(0.01)
                return f"data_{name}"
            return loader

        for i in range(5):
            name = f"group_{i}"
            mgr.register(name, make_loader(name), _make_unloader())

        def worker(group_name):
            try:
                results[group_name] = mgr.require(group_name)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(f"group_{i}",))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)

        assert not errors, f"Thread errors: {errors}"
        for i in range(5):
            name = f"group_{i}"
            assert results[name] == f"data_{name}"
            assert mgr.is_loaded(name)


# ============================================================================
# is_loaded() tests
# ============================================================================

class TestIsLoaded:
    """Test the is_loaded() method."""

    def test_unknown_group_raises(self, mgr):
        """is_loaded() raises KeyError for unknown group."""
        with pytest.raises(KeyError, match="not registered"):
            mgr.is_loaded("nonexistent")

    def test_false_before_require(self, mgr):
        """is_loaded() returns False before require()."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        assert not mgr.is_loaded("face_data")

    def test_true_after_require(self, mgr):
        """is_loaded() returns True after require()."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.require("face_data")
        assert mgr.is_loaded("face_data")

    def test_false_after_unload(self, mgr):
        """is_loaded() returns False after unload_all()."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.require("face_data")
        mgr.unload_all()
        assert not mgr.is_loaded("face_data")

    def test_false_after_single_unload(self, mgr):
        """is_loaded() returns False after unload(name)."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.require("face_data")
        mgr.unload("face_data")
        assert not mgr.is_loaded("face_data")


# ============================================================================
# unload() tests (single group)
# ============================================================================

class TestUnload:
    """Test the unload(name) method for single-group unloading."""

    def test_unload_calls_unloader(self, mgr):
        """unload() calls the registered unloader."""
        unloader = _make_unloader()
        mgr.register("face_data", _make_loader(), unloader)
        mgr.require("face_data")

        mgr.unload("face_data")

        unloader.assert_called_once()
        assert not mgr.is_loaded("face_data")

    def test_unload_noop_when_not_loaded(self, mgr):
        """unload() is a no-op when the resource is not loaded."""
        unloader = _make_unloader()
        mgr.register("face_data", _make_loader(), unloader)

        mgr.unload("face_data")

        unloader.assert_not_called()
        assert not mgr.is_loaded("face_data")

    def test_unload_unknown_raises(self, mgr):
        """unload() raises KeyError for unknown group."""
        with pytest.raises(KeyError, match="not registered"):
            mgr.unload("nonexistent")

    def test_unload_then_require_reloads(self, mgr):
        """After unload(), require() calls the loader again."""
        call_count = 0

        def counting_loader():
            nonlocal call_count
            call_count += 1
            return f"data_v{call_count}"

        mgr.register("face_data", counting_loader, _make_unloader())

        result1 = mgr.require("face_data")
        assert result1 == "data_v1"
        assert call_count == 1

        mgr.unload("face_data")

        result2 = mgr.require("face_data")
        assert result2 == "data_v2"
        assert call_count == 2

    def test_unload_only_affects_target_group(self, mgr):
        """unload() does not affect other loaded groups."""
        mgr.register("face_data", _make_loader("face"), _make_unloader())
        mgr.register("tattoo_data", _make_loader("tattoo"), _make_unloader())

        mgr.require("face_data")
        mgr.require("tattoo_data")

        mgr.unload("face_data")

        assert not mgr.is_loaded("face_data")
        assert mgr.is_loaded("tattoo_data")

    def test_unload_calls_gc_collect(self, mgr):
        """unload() calls gc.collect() after unloading."""
        mgr.register("face_data", _make_loader(), _make_unloader())
        mgr.require("face_data")

        with patch("resource_manager.gc.collect") as mock_gc:
            mgr.unload("face_data")
            mock_gc.assert_called_once()


# ============================================================================
# Singleton tests
# ============================================================================

class TestSingleton:
    """Test module-level singleton functions."""

    def test_get_before_init_raises(self):
        """get_resource_manager raises before init_resource_manager."""
        import resource_manager as rm
        original = rm._resource_manager
        try:
            rm._resource_manager = None
            with pytest.raises(RuntimeError, match="not initialized"):
                get_resource_manager()
        finally:
            rm._resource_manager = original

    def test_init_and_get(self):
        """init_resource_manager creates a manager, get_resource_manager returns it."""
        import resource_manager as rm
        original = rm._resource_manager
        try:
            mgr = init_resource_manager(idle_timeout_seconds=600.0)
            assert mgr is get_resource_manager()
            assert isinstance(mgr, ResourceManager)
        finally:
            rm._resource_manager = original

    def test_init_returns_instance(self):
        """init_resource_manager returns the created instance."""
        import resource_manager as rm
        original = rm._resource_manager
        try:
            mgr = init_resource_manager()
            assert isinstance(mgr, ResourceManager)
        finally:
            rm._resource_manager = original

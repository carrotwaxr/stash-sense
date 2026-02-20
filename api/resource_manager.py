"""Resource manager for lazy loading and idle unloading of heavy resources.

Manages resource groups (face recognition data, tattoo data, body proportions)
with lazy loading on first access and automatic unloading after idle timeout.

Usage:
    # At startup
    mgr = init_resource_manager(idle_timeout_seconds=1800.0)
    mgr.register("face_data", loader=load_face_data, unloader=unload_face_data)

    # When needed
    mgr = get_resource_manager()
    data = mgr.require("face_data")  # Loads on first call, cached after

    # Periodically (called by background task in main.py)
    mgr.check_idle()

    # At shutdown
    mgr.unload_all()
"""

import gc
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


# ============================================================================
# Data types
# ============================================================================

@dataclass
class ResourceGroup:
    """A managed resource group with lazy loading and idle tracking."""
    name: str
    loader: Callable[[], Any]
    unloader: Callable[[], None]
    data: Any = None
    loaded: bool = False
    last_access: float = 0.0
    load_time_seconds: float = 0.0


# ============================================================================
# ResourceManager
# ============================================================================

class ResourceManager:
    """Manages lazy loading and idle unloading of heavy resources.

    Thread-safe. Resources are loaded on first require() call and unloaded
    after being idle beyond the configured timeout.
    """

    def __init__(self, idle_timeout_seconds: float = 1800.0):
        """Initialize the resource manager.

        Args:
            idle_timeout_seconds: Seconds of idle time before a resource group
                is unloaded. Default is 30 minutes (1800 seconds).
        """
        self._idle_timeout = idle_timeout_seconds
        self._groups: dict[str, ResourceGroup] = {}
        self._lock = threading.Lock()

    def register(self, name: str, loader: Callable[[], Any], unloader: Callable[[], None]) -> None:
        """Register a resource group with its loader and unloader functions.

        Args:
            name: Unique name for the resource group.
            loader: Callable that loads and returns the resource data.
            unloader: Callable that cleans up the resource (e.g. closes files,
                releases memory).
        """
        with self._lock:
            self._groups[name] = ResourceGroup(
                name=name,
                loader=loader,
                unloader=unloader,
            )

    def require(self, name: str) -> Any:
        """Ensure a resource group is loaded and return its data.

        Loads the resource on first call via the registered loader. On
        subsequent calls, returns the cached data and resets the idle timer.

        Args:
            name: Name of the resource group to load/access.

        Returns:
            The data returned by the resource group's loader.

        Raises:
            KeyError: If the resource group name is not registered.
            Exception: Any exception raised by the loader propagates to the
                caller. The resource remains unloaded on failure.
        """
        with self._lock:
            group = self._groups.get(name)
            if group is None:
                raise KeyError(f"Resource group not registered: {name}")

            if group.loaded:
                group.last_access = time.monotonic()
                return group.data

            # Load the resource (may raise)
            logger.warning(f"Loading resource group: {name}")
            start = time.monotonic()
            try:
                data = group.loader()
            except Exception:
                logger.warning(f"Failed to load resource group: {name}")
                raise

            elapsed = time.monotonic() - start
            group.data = data
            group.loaded = True
            group.last_access = time.monotonic()
            group.load_time_seconds = elapsed
            logger.warning(f"Loaded resource group: {name} in {elapsed:.2f}s")
            return group.data

    def is_loaded(self, name: str) -> bool:
        """Check if a resource group is currently loaded.

        Args:
            name: Name of the resource group.

        Returns:
            True if the resource group is loaded, False otherwise.

        Raises:
            KeyError: If the resource group name is not registered.
        """
        with self._lock:
            group = self._groups.get(name)
            if group is None:
                raise KeyError(f"Resource group not registered: {name}")
            return group.loaded

    def check_idle(self) -> None:
        """Unload resource groups that have been idle beyond the timeout.

        Iterates all registered groups and unloads any that are loaded and
        have not been accessed within idle_timeout_seconds. Calls gc.collect()
        after unloading any resources.
        """
        now = time.monotonic()
        unloaded_any = False

        with self._lock:
            for group in self._groups.values():
                if not group.loaded:
                    continue
                idle_seconds = now - group.last_access
                if idle_seconds >= self._idle_timeout:
                    logger.warning(
                        f"Unloading idle resource group: {group.name} "
                        f"(idle {idle_seconds:.0f}s)"
                    )
                    self._unload_group(group)
                    unloaded_any = True

        if unloaded_any:
            gc.collect()

    def unload_all(self) -> None:
        """Unload all resource groups. Intended for clean shutdown."""
        unloaded_any = False

        with self._lock:
            for group in self._groups.values():
                if group.loaded:
                    logger.warning(f"Unloading resource group: {group.name}")
                    self._unload_group(group)
                    unloaded_any = True

        if unloaded_any:
            gc.collect()

    def get_status(self) -> dict[str, dict]:
        """Get status of all resource groups.

        Returns:
            Dict keyed by group name with:
                - loaded (bool): Whether the group is currently loaded.
                - idle_seconds (float | None): Seconds since last access, or
                    None if not loaded.
                - load_time_seconds (float): Time taken to load, or 0.0 if
                    never loaded.
        """
        now = time.monotonic()
        result = {}

        with self._lock:
            for name, group in self._groups.items():
                if group.loaded:
                    idle_seconds = now - group.last_access
                else:
                    idle_seconds = None

                result[name] = {
                    "loaded": group.loaded,
                    "idle_seconds": idle_seconds,
                    "load_time_seconds": group.load_time_seconds,
                }

        return result

    def _unload_group(self, group: ResourceGroup) -> None:
        """Unload a single resource group. Must be called under self._lock.

        Args:
            group: The resource group to unload.
        """
        try:
            group.unloader()
        except Exception as e:
            logger.warning(f"Error unloading resource group {group.name}: {e}")
        group.data = None
        group.loaded = False


# ============================================================================
# Module-level singleton
# ============================================================================

_resource_manager: Optional[ResourceManager] = None


def init_resource_manager(
    idle_timeout_seconds: float = 1800.0,
) -> ResourceManager:
    """Initialize the global resource manager. Called once at startup.

    Args:
        idle_timeout_seconds: Seconds of idle time before auto-unload.

    Returns:
        The initialized ResourceManager instance.
    """
    global _resource_manager
    _resource_manager = ResourceManager(idle_timeout_seconds=idle_timeout_seconds)
    return _resource_manager


def get_resource_manager() -> ResourceManager:
    """Get the global resource manager.

    Must be called after init_resource_manager().

    Raises:
        RuntimeError: If init_resource_manager() has not been called.
    """
    if _resource_manager is None:
        raise RuntimeError(
            "ResourceManager not initialized. Call init_resource_manager() during startup."
        )
    return _resource_manager

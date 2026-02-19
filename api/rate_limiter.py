"""
Rate Limiter for Stash API Requests

Provides a centralized rate limiter with priority queue support.
All Stash API calls should go through this to prevent overwhelming
the local Stash instance during analysis operations.

Usage:
    limiter = RateLimiter.get_instance()
    async with limiter.acquire():
        result = await stash_client.some_query()

    # Or with priority (lower = higher priority)
    async with limiter.acquire(priority=Priority.HIGH):
        result = await stash_client.important_query()
"""

import asyncio
import os
import time
from dataclasses import dataclass, field
from enum import IntEnum
from heapq import heappush, heappop
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class Priority(IntEnum):
    """Request priority levels. Lower value = higher priority."""
    CRITICAL = 0   # User-initiated actions (merge, delete)
    HIGH = 10      # Interactive requests (single scene lookup)
    NORMAL = 50    # Background analysis (default)
    LOW = 100      # Bulk/batch operations


@dataclass(order=True)
class PriorityRequest:
    """A request waiting in the priority queue."""
    priority: int
    timestamp: float = field(compare=False)
    event: asyncio.Event = field(compare=False)


class RateLimiter:
    """
    Async rate limiter with priority queue.

    Features:
    - Configurable requests per second
    - Priority queue (lower priority value = served first)
    - Thread-safe singleton pattern
    - Metrics tracking
    """

    _instance: Optional["RateLimiter"] = None
    _lock = asyncio.Lock()

    def __init__(self, requests_per_second: float = 5.0):
        """
        Initialize rate limiter.

        Args:
            requests_per_second: Maximum requests per second (default: 5.0)
        """
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second

        self._last_request_time: float = 0.0
        self._queue: list[PriorityRequest] = []
        self._processing_lock = asyncio.Lock()
        self._queue_lock = asyncio.Lock()

        # Metrics
        self._total_requests = 0
        self._total_wait_time = 0.0

        logger.info(f"RateLimiter initialized: {requests_per_second} req/sec")

    @classmethod
    async def get_instance(cls) -> "RateLimiter":
        """Get or create the singleton rate limiter instance."""
        if cls._instance is None:
            async with cls._lock:
                if cls._instance is None:
                    # Try settings system first, fall back to env var
                    try:
                        from settings import get_setting
                        rate = float(get_setting("stash_api_rate"))
                    except (RuntimeError, KeyError):
                        rate = float(os.environ.get("STASH_RATE_LIMIT", "5.0"))
                    cls._instance = cls(requests_per_second=rate)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset the singleton instance (mainly for testing)."""
        cls._instance = None

    def acquire(self, priority: Priority = Priority.NORMAL):
        """
        Context manager to acquire rate limit slot.

        Args:
            priority: Request priority (default: NORMAL)

        Usage:
            async with limiter.acquire(Priority.HIGH):
                await do_request()
        """
        return _RateLimitContext(self, priority)

    async def _wait_for_slot(self, priority: Priority) -> float:
        """
        Wait for a rate limit slot to become available.

        Returns the time spent waiting.
        """
        start_wait = time.monotonic()

        # Create our request and add to priority queue
        request = PriorityRequest(
            priority=int(priority),
            timestamp=time.monotonic(),
            event=asyncio.Event(),
        )

        async with self._queue_lock:
            heappush(self._queue, request)

        # Process queue
        await self._process_queue()

        # Wait for our turn
        await request.event.wait()

        wait_time = time.monotonic() - start_wait
        self._total_wait_time += wait_time
        self._total_requests += 1

        return wait_time

    async def _process_queue(self):
        """Process the priority queue, releasing highest priority request when ready."""
        async with self._processing_lock:
            if not self._queue:
                return

            # Calculate time until next slot available
            now = time.monotonic()
            elapsed = now - self._last_request_time
            wait_needed = max(0, self.min_interval - elapsed)

            if wait_needed > 0:
                await asyncio.sleep(wait_needed)

            # Pop highest priority request and signal it
            async with self._queue_lock:
                if self._queue:
                    request = heappop(self._queue)
                    self._last_request_time = time.monotonic()
                    request.event.set()

    def get_metrics(self) -> dict:
        """Get rate limiter metrics."""
        return {
            "total_requests": self._total_requests,
            "total_wait_time": round(self._total_wait_time, 3),
            "avg_wait_time": round(self._total_wait_time / max(1, self._total_requests), 3),
            "requests_per_second": self.requests_per_second,
            "queue_depth": len(self._queue),
        }

    def update_rate(self, requests_per_second: float):
        """Update the rate limit dynamically."""
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        logger.info(f"RateLimiter updated: {requests_per_second} req/sec")


class _RateLimitContext:
    """Context manager for rate-limited operations."""

    def __init__(self, limiter: RateLimiter, priority: Priority):
        self.limiter = limiter
        self.priority = priority
        self.wait_time: float = 0.0

    async def __aenter__(self):
        self.wait_time = await self.limiter._wait_for_slot(self.priority)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Could add post-request hooks here if needed
        pass


# Convenience function for simple usage
async def rate_limited(priority: Priority = Priority.NORMAL):
    """
    Convenience function to acquire rate limit.

    Usage:
        async with rate_limited():
            await do_request()
    """
    limiter = await RateLimiter.get_instance()
    return limiter.acquire(priority)

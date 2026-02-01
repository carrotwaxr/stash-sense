"""Tests for the rate limiter module."""

import asyncio
import time
import pytest

from rate_limiter import RateLimiter, Priority


class TestRateLimiter:
    """Tests for RateLimiter class."""

    @pytest.fixture(autouse=True)
    def reset_singleton(self):
        """Reset the singleton before each test."""
        RateLimiter.reset_instance()
        yield
        RateLimiter.reset_instance()

    @pytest.mark.asyncio
    async def test_respects_rate_limit(self):
        """Requests should be spaced by min_interval."""
        limiter = RateLimiter(requests_per_second=10.0)  # 0.1s interval

        start = time.monotonic()
        for _ in range(3):
            async with limiter.acquire():
                pass
        elapsed = time.monotonic() - start

        # 3 requests at 10/sec should take at least 0.2s (2 intervals)
        assert elapsed >= 0.18  # Allow small timing variance

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Higher priority requests should be served first when queued."""
        limiter = RateLimiter(requests_per_second=5.0)  # Slow enough to queue
        order = []

        # First, saturate the limiter with a request
        async with limiter.acquire(Priority.NORMAL):
            pass

        async def make_request(priority: Priority, label: str):
            async with limiter.acquire(priority):
                order.append(label)

        # Queue up requests with different priorities simultaneously
        # All should queue behind the rate limit
        tasks = [
            asyncio.create_task(make_request(Priority.LOW, "low")),
            asyncio.create_task(make_request(Priority.CRITICAL, "critical")),
            asyncio.create_task(make_request(Priority.NORMAL, "normal")),
        ]

        # Give tasks time to queue before processing starts
        await asyncio.sleep(0.01)

        await asyncio.gather(*tasks)

        # Critical (0) should come before normal (50), normal before low (100)
        critical_idx = order.index("critical")
        normal_idx = order.index("normal")
        low_idx = order.index("low")

        assert critical_idx < normal_idx, f"Expected critical before normal, got {order}"
        assert normal_idx < low_idx, f"Expected normal before low, got {order}"

    @pytest.mark.asyncio
    async def test_singleton_pattern(self):
        """get_instance should return same instance."""
        instance1 = await RateLimiter.get_instance()
        instance2 = await RateLimiter.get_instance()

        assert instance1 is instance2

    @pytest.mark.asyncio
    async def test_metrics_tracking(self):
        """Metrics should track requests and wait time."""
        limiter = RateLimiter(requests_per_second=100.0)

        for _ in range(5):
            async with limiter.acquire():
                pass

        metrics = limiter.get_metrics()

        assert metrics["total_requests"] == 5
        assert metrics["requests_per_second"] == 100.0
        assert "avg_wait_time" in metrics

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Multiple concurrent requests should queue properly."""
        limiter = RateLimiter(requests_per_second=10.0)
        completed = []

        async def make_request(idx: int):
            async with limiter.acquire():
                completed.append(idx)

        # Fire 5 requests concurrently
        tasks = [asyncio.create_task(make_request(i)) for i in range(5)]
        await asyncio.gather(*tasks)

        # All should complete
        assert len(completed) == 5
        assert set(completed) == {0, 1, 2, 3, 4}

    @pytest.mark.asyncio
    async def test_update_rate(self):
        """Rate can be updated dynamically."""
        limiter = RateLimiter(requests_per_second=5.0)
        assert limiter.min_interval == 0.2

        limiter.update_rate(10.0)
        assert limiter.min_interval == 0.1
        assert limiter.requests_per_second == 10.0

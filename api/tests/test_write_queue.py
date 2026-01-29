"""Tests for async write queue."""
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock


class TestWriteQueue:
    """Test write queue behavior."""

    @pytest.mark.asyncio
    async def test_queue_processes_messages_in_order(self):
        """Messages are processed in FIFO order."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        processed = []

        async def handler(msg):
            processed.append(msg.performer_id)
            await asyncio.sleep(0.01)  # Simulate work

        queue = WriteQueue(handler)
        await queue.start()

        # Enqueue messages
        for i in range(5):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        # Wait for processing
        await queue.wait_until_empty()
        await queue.stop()

        assert processed == [0, 1, 2, 3, 4]

    @pytest.mark.asyncio
    async def test_queue_handles_errors_gracefully(self):
        """Errors in handler don't crash the queue."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        call_count = 0

        async def failing_handler(msg):
            nonlocal call_count
            call_count += 1
            if msg.performer_id == 1:
                raise ValueError("Simulated error")

        queue = WriteQueue(failing_handler)
        await queue.start()

        # Enqueue messages - middle one will fail
        for i in range(3):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        await queue.wait_until_empty()
        await queue.stop()

        # All messages were attempted despite error
        assert call_count == 3
        assert queue.stats.errors == 1

    @pytest.mark.asyncio
    async def test_queue_tracks_statistics(self):
        """Queue tracks processing statistics."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        async def handler(msg):
            await asyncio.sleep(0.001)

        queue = WriteQueue(handler)
        await queue.start()

        for i in range(10):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        await queue.wait_until_empty()
        await queue.stop()

        assert queue.stats.processed == 10
        assert queue.stats.errors == 0

    @pytest.mark.asyncio
    async def test_queue_graceful_shutdown(self):
        """Queue processes remaining messages on shutdown."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        processed = []

        async def handler(msg):
            processed.append(msg.performer_id)
            await asyncio.sleep(0.05)

        queue = WriteQueue(handler)
        await queue.start()

        # Enqueue many messages
        for i in range(5):
            await queue.enqueue(WriteMessage(
                operation=WriteOperation.ADD_EMBEDDING,
                source="test",
                performer_id=i,
            ))

        # Stop with grace period
        await queue.stop(timeout=5.0)

        # All messages should be processed
        assert len(processed) == 5

    @pytest.mark.asyncio
    async def test_queue_tracks_by_operation(self):
        """Stats track operations by type."""
        from write_queue import WriteQueue, WriteMessage, WriteOperation

        async def handler(msg):
            pass

        queue = WriteQueue(handler)
        await queue.start()

        await queue.enqueue(WriteMessage(operation=WriteOperation.ADD_EMBEDDING, source="s1"))
        await queue.enqueue(WriteMessage(operation=WriteOperation.ADD_EMBEDDING, source="s1"))
        await queue.enqueue(WriteMessage(operation=WriteOperation.ADD_STASH_ID, source="s2"))

        await queue.wait_until_empty()
        await queue.stop()

        assert queue.stats.by_operation["add_embedding"] == 2
        assert queue.stats.by_operation["add_stash_id"] == 1
        assert queue.stats.by_source["s1"] == 2
        assert queue.stats.by_source["s2"] == 1

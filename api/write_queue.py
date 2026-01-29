"""
Async write queue for serializing database writes.

Multiple scrapers can enqueue writes concurrently.
Single consumer processes them sequentially to prevent race conditions.

See: docs/plans/2026-01-29-multi-source-enrichment-design.md
"""
import asyncio
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable, Awaitable
import numpy as np

logger = logging.getLogger(__name__)


class WriteOperation(Enum):
    """Types of write operations."""
    CREATE_PERFORMER = "create_performer"
    UPDATE_PERFORMER = "update_performer"
    ADD_EMBEDDING = "add_embedding"
    ADD_STASH_ID = "add_stash_id"
    ADD_EXTERNAL_URL = "add_external_url"
    ADD_ALIAS = "add_alias"


@dataclass
class WriteMessage:
    """Message for the write queue."""
    operation: WriteOperation
    source: str

    # For CREATE/UPDATE performer
    performer_data: Optional[dict] = None

    # For ADD_EMBEDDING
    performer_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    embedding_type: Optional[str] = None
    image_url: Optional[str] = None
    quality_score: Optional[float] = None

    # For ADD_STASH_ID
    endpoint: Optional[str] = None
    stashbox_id: Optional[str] = None

    # For ADD_EXTERNAL_URL
    url: Optional[str] = None
    site: Optional[str] = None

    # For ADD_ALIAS
    alias: Optional[str] = None


@dataclass
class QueueStats:
    """Statistics for queue monitoring."""
    enqueued: int = 0
    processed: int = 0
    errors: int = 0
    by_operation: dict = field(default_factory=dict)
    by_source: dict = field(default_factory=dict)

    def record_processed(self, msg: WriteMessage):
        """Record a processed message."""
        self.processed += 1

        op = msg.operation.value
        self.by_operation[op] = self.by_operation.get(op, 0) + 1

        self.by_source[msg.source] = self.by_source.get(msg.source, 0) + 1

    def record_error(self, msg: WriteMessage):
        """Record an error."""
        self.errors += 1


class WriteQueue:
    """
    Async queue for serializing database writes.

    Usage:
        async def handle_write(msg: WriteMessage):
            if msg.operation == WriteOperation.ADD_EMBEDDING:
                db.add_embedding(...)

        queue = WriteQueue(handle_write)
        await queue.start()

        # From multiple scrapers concurrently:
        await queue.enqueue(WriteMessage(...))

        # Shutdown
        await queue.stop()
    """

    def __init__(
        self,
        handler: Callable[[WriteMessage], Awaitable[None]],
        max_size: int = 10000,
    ):
        self.handler = handler
        self.max_size = max_size
        self._queue: asyncio.Queue[WriteMessage] = asyncio.Queue(maxsize=max_size)
        self._consumer_task: Optional[asyncio.Task] = None
        self._running = False
        self.stats = QueueStats()

    async def start(self):
        """Start the queue consumer."""
        if self._running:
            return

        self._running = True
        self._consumer_task = asyncio.create_task(self._consume())
        logger.info("Write queue started")

    async def stop(self, timeout: float = 30.0):
        """Stop the queue, processing remaining messages."""
        if not self._running:
            return

        self._running = False

        # Wait for queue to drain
        try:
            await asyncio.wait_for(self._queue.join(), timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(f"Queue drain timeout after {timeout}s, {self._queue.qsize()} messages remaining")

        # Cancel consumer
        if self._consumer_task:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass

        logger.info(f"Write queue stopped. Stats: {self.stats.processed} processed, {self.stats.errors} errors")

    async def enqueue(self, message: WriteMessage):
        """Add a message to the queue."""
        await self._queue.put(message)
        self.stats.enqueued += 1

    async def wait_until_empty(self):
        """Wait until all queued messages are processed."""
        await self._queue.join()

    @property
    def size(self) -> int:
        """Current queue size."""
        return self._queue.qsize()

    async def _consume(self):
        """Consumer loop - processes messages sequentially."""
        while self._running or not self._queue.empty():
            try:
                # Wait for message with timeout to allow checking _running flag
                try:
                    message = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process message
                try:
                    await self.handler(message)
                    self.stats.record_processed(message)
                except Exception as e:
                    logger.error(f"Error processing {message.operation.value}: {e}")
                    self.stats.record_error(message)
                finally:
                    self._queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Consumer loop error: {e}")

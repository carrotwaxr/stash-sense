"""
Enrichment coordinator for multi-source database building.

Runs multiple scrapers concurrently, funneling results through
a single write queue for serialized database writes.
"""
import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from base_scraper import BaseScraper, ScrapedPerformer
from database import PerformerDatabase
from write_queue import WriteQueue, WriteMessage, WriteOperation

logger = logging.getLogger(__name__)


@dataclass
class CoordinatorStats:
    """Statistics for monitoring."""
    performers_processed: int = 0
    faces_added: int = 0
    images_processed: int = 0
    errors: int = 0
    by_source: dict = field(default_factory=dict)

    def record_performer(self, source: str):
        """Record a processed performer."""
        self.performers_processed += 1
        self.by_source[source] = self.by_source.get(source, 0) + 1


class EnrichmentCoordinator:
    """
    Coordinates multi-source enrichment.

    Features:
    - Runs scrapers concurrently (one thread per scraper)
    - Single write queue for serialized database writes
    - Per-source and total face limits
    - Resume capability via scrape_progress table
    """

    def __init__(
        self,
        database: PerformerDatabase,
        scrapers: list[BaseScraper],
        max_faces_per_source: int = 5,
        max_faces_total: int = 20,
        dry_run: bool = False,
    ):
        self.database = database
        self.scrapers = scrapers
        self.max_faces_per_source = max_faces_per_source
        self.max_faces_total = max_faces_total
        self.dry_run = dry_run

        self.stats = CoordinatorStats()

        # Write queue with handler
        self.write_queue = WriteQueue(self._handle_write)

        # Thread pool for running sync scrapers
        self._executor = ThreadPoolExecutor(max_workers=max(1, len(scrapers)))

        # Event loop reference for cross-thread communication
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def run(self):
        """Run all scrapers and process results."""
        self._loop = asyncio.get_event_loop()
        await self.write_queue.start()

        try:
            # Run each scraper in a thread
            tasks = [
                self._loop.run_in_executor(self._executor, self._run_scraper, scraper)
                for scraper in self.scrapers
            ]

            # Wait for all scrapers to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Wait for write queue to drain
            await self.write_queue.wait_until_empty()

        finally:
            await self.write_queue.stop()
            self._executor.shutdown(wait=True)

        logger.info(f"Enrichment complete: {self.stats.performers_processed} performers processed")

    def _run_scraper(self, scraper: BaseScraper):
        """Run a single scraper (called from thread pool)."""
        source = scraper.source_name
        logger.info(f"Starting scraper: {source}")

        # Get resume point
        progress = self.database.get_scrape_progress(source)
        last_id = progress['last_processed_id'] if progress else None

        processed = 0
        faces_added = 0

        try:
            for performer in scraper.iter_performers_after(last_id):
                # Process performer
                self._process_performer(scraper, performer)
                processed += 1

                # Save progress periodically
                if processed % 100 == 0:
                    self.database.save_scrape_progress(
                        source=source,
                        last_processed_id=performer.id,
                        performers_processed=processed,
                        faces_added=faces_added,
                    )
                    logger.info(f"{source}: {processed} performers processed")

                # Record stats
                self.stats.record_performer(source)

        except Exception as e:
            logger.error(f"Scraper {source} error: {e}")
            self.stats.errors += 1

        # Final progress save
        self.database.save_scrape_progress(
            source=source,
            last_processed_id=performer.id if processed > 0 else last_id,
            performers_processed=processed,
            faces_added=faces_added,
        )

        logger.info(f"Scraper {source} complete: {processed} performers")

    def _process_performer(self, scraper: BaseScraper, performer: ScrapedPerformer):
        """Process a single performer."""
        # Queue performer creation/update
        future = asyncio.run_coroutine_threadsafe(
            self.write_queue.enqueue(WriteMessage(
                operation=WriteOperation.CREATE_PERFORMER,
                source=scraper.source_name,
                performer_data=self._performer_to_dict(performer),
            )),
            self._loop,
        )
        # Wait for enqueue to complete
        future.result(timeout=10.0)

    def _performer_to_dict(self, performer: ScrapedPerformer) -> dict:
        """Convert performer to dict for write queue."""
        return {
            'id': performer.id,
            'name': performer.name,
            'aliases': performer.aliases,
            'gender': performer.gender,
            'country': performer.country,
            'birth_date': performer.birth_date,
            'death_date': performer.death_date,
            'image_urls': performer.image_urls,
            'stash_ids': performer.stash_ids,
            'external_urls': performer.external_urls,
            'ethnicity': performer.ethnicity,
            'height_cm': performer.height_cm,
            'eye_color': performer.eye_color,
            'hair_color': performer.hair_color,
            'career_start_year': performer.career_start_year,
            'career_end_year': performer.career_end_year,
            'disambiguation': performer.disambiguation,
        }

    async def _handle_write(self, message: WriteMessage):
        """Handle write queue messages."""
        if self.dry_run:
            return

        if message.operation == WriteOperation.CREATE_PERFORMER:
            await self._handle_create_performer(message)
        elif message.operation == WriteOperation.ADD_EMBEDDING:
            await self._handle_add_embedding(message)
        elif message.operation == WriteOperation.ADD_STASH_ID:
            await self._handle_add_stash_id(message)
        elif message.operation == WriteOperation.ADD_ALIAS:
            await self._handle_add_alias(message)
        elif message.operation == WriteOperation.ADD_EXTERNAL_URL:
            await self._handle_add_url(message)

    async def _handle_create_performer(self, message: WriteMessage):
        """Create or update performer in database."""
        data = message.performer_data
        source = message.source

        # Check if performer exists by stash ID
        stash_id = data.get('stash_ids', {}).get(source)
        if stash_id:
            existing = self.database.get_performer_by_stashbox_id(source, stash_id)
        else:
            existing = None

        if existing:
            # Update existing performer
            self.database.update_performer(
                existing.id,
                canonical_name=data.get('name'),
                gender=data.get('gender'),
                country=data.get('country'),
            )
            performer_id = existing.id
        else:
            # Create new performer
            performer_id = self.database.add_performer(
                canonical_name=data['name'],
                gender=data.get('gender'),
                country=data.get('country'),
                birth_date=data.get('birth_date'),
            )
            # Add stash ID for this source
            if stash_id:
                self.database.add_stashbox_id(performer_id, source, stash_id)

        # Add aliases
        for alias in data.get('aliases', []):
            try:
                self.database.add_alias(performer_id, alias, source)
            except Exception:
                pass  # Ignore duplicate aliases

        # Add URLs
        for site, urls in data.get('external_urls', {}).items():
            for url in urls:
                try:
                    self.database.add_url(performer_id, url, source)
                except Exception:
                    pass  # Ignore duplicate URLs

    async def _handle_add_embedding(self, message: WriteMessage):
        """Add face embedding to database."""
        # This would add to Voyager index and database
        # Implementation depends on index management
        pass

    async def _handle_add_stash_id(self, message: WriteMessage):
        """Add stash ID to performer."""
        if message.performer_id and message.endpoint and message.stashbox_id:
            try:
                self.database.add_stashbox_id(
                    message.performer_id,
                    message.endpoint,
                    message.stashbox_id,
                )
            except Exception:
                pass

    async def _handle_add_alias(self, message: WriteMessage):
        """Add alias to performer."""
        if message.performer_id and message.alias:
            try:
                self.database.add_alias(
                    message.performer_id,
                    message.alias,
                    message.source,
                )
            except Exception:
                pass

    async def _handle_add_url(self, message: WriteMessage):
        """Add external URL to performer."""
        if message.performer_id and message.url:
            try:
                self.database.add_url(
                    message.performer_id,
                    message.url,
                    message.source,
                )
            except Exception:
                pass

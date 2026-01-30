"""
Enrichment coordinator for multi-source database building.

Runs multiple scrapers concurrently, funneling results through
a single write queue for serialized database writes.
"""
import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

from base_scraper import BaseScraper, ScrapedPerformer
from database import PerformerDatabase
from write_queue import WriteQueue, WriteMessage, WriteOperation

logger = logging.getLogger(__name__)


class ReferenceSiteMode(Enum):
    """Mode for reference site scraper iteration."""
    URL_LOOKUP = "url"   # Look up performers by existing URLs
    NAME_LOOKUP = "name" # Try all performers by name


@dataclass
class CoordinatorStats:
    """Statistics for monitoring."""
    performers_processed: int = 0
    faces_added: int = 0
    images_processed: int = 0
    images_skipped: int = 0
    faces_rejected: int = 0
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

    SAVE_INTERVAL = 1000  # Save indices every N faces

    def __init__(
        self,
        database: PerformerDatabase,
        scrapers: list[BaseScraper],
        data_dir: Optional[Path] = None,
        max_faces_per_source: int = 5,
        max_faces_total: int = 20,
        dry_run: bool = False,
        enable_face_processing: bool = False,
        source_trust_levels: Optional[dict[str, str]] = None,
        reference_site_mode: ReferenceSiteMode = ReferenceSiteMode.URL_LOOKUP,
    ):
        self.database = database
        self.scrapers = scrapers
        self.data_dir = Path(data_dir) if data_dir else None
        self.max_faces_per_source = max_faces_per_source
        self.max_faces_total = max_faces_total
        self.dry_run = dry_run
        self.enable_face_processing = enable_face_processing
        self.source_trust_levels = source_trust_levels or {}
        self.reference_site_mode = reference_site_mode

        self.stats = CoordinatorStats()

        # Write queue with handler
        self.write_queue = WriteQueue(self._handle_write)

        # Thread pool for running sync scrapers
        self._executor = ThreadPoolExecutor(max_workers=max(1, len(scrapers)))

        # Event loop reference for cross-thread communication
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Face processing components (lazy loaded)
        self._index_manager = None
        self._face_processor = None
        self._face_validator = None
        self._generator = None

        # Lock for index writes
        self._index_lock = threading.Lock()

        # Track faces added since last save
        self._faces_since_save = 0

    def _init_face_processing(self):
        """Initialize face processing components."""
        if not self.enable_face_processing or self.data_dir is None:
            return

        from index_manager import IndexManager
        from face_processor import FaceProcessor
        from face_validator import FaceValidator
        from embeddings import FaceEmbeddingGenerator
        from quality_filters import QualityFilters

        logger.info("Initializing face processing components...")

        self._index_manager = IndexManager(self.data_dir)
        self._generator = FaceEmbeddingGenerator()
        self._face_processor = FaceProcessor(
            generator=self._generator,
            quality_filters=QualityFilters(),
        )
        self._face_validator = FaceValidator()

        logger.info(f"Loaded {self._index_manager.current_index} existing embeddings")

    async def run(self):
        """Run all scrapers and process results."""
        self._loop = asyncio.get_event_loop()
        self._init_face_processing()
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

            # Save indices on shutdown
            if self._index_manager:
                self._index_manager.save()

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
                faces_from_performer = self._process_performer(scraper, performer)
                faces_added += faces_from_performer
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

    def _process_performer(self, scraper: BaseScraper, performer: ScrapedPerformer) -> int:
        """Process a single performer. Returns number of faces added."""
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

        # Process images for faces if enabled
        if not self.enable_face_processing or not performer.image_urls:
            return 0

        # Wait for the write queue to process the CREATE_PERFORMER before proceeding
        # This ensures the performer exists in the database before we try to add faces
        future = asyncio.run_coroutine_threadsafe(
            self.write_queue.wait_until_empty(),
            self._loop,
        )
        future.result(timeout=30.0)

        return self._process_performer_images(scraper, performer)

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

    def _process_performer_images(self, scraper: BaseScraper, performer: ScrapedPerformer) -> int:
        """Download and process images for a performer. Returns faces added."""
        source = scraper.source_name
        trust_level = self.source_trust_levels.get(source, "medium")

        # Get performer ID from database
        stash_id = performer.stash_ids.get(source)
        if not stash_id:
            return 0

        db_performer = self.database.get_performer_by_stashbox_id(source, stash_id)
        if not db_performer:
            return 0

        performer_id = db_performer.id

        # Get current face counts from database
        existing_source_faces = len([
            f for f in self.database.get_faces(performer_id)
            if f.source_endpoint == source
        ])
        existing_total_faces = len(self.database.get_faces(performer_id))

        # Check if already at total limit
        if existing_total_faces >= self.max_faces_total:
            return 0

        # Get existing embeddings for validation
        existing_embeddings = self._get_existing_embeddings(performer_id)

        faces_added = 0

        for image_url in performer.image_urls:
            # Check per-source limit (using local count + added faces)
            if existing_source_faces + faces_added >= self.max_faces_per_source:
                break

            # Check total limit (using local count + added faces)
            if existing_total_faces + faces_added >= self.max_faces_total:
                break

            # Download image
            try:
                image_data = scraper.download_image(image_url)
                if not image_data:
                    continue
            except Exception as e:
                logger.debug(f"Failed to download {image_url}: {e}")
                continue

            self.stats.images_processed += 1

            # Process image for faces
            processed_faces = self._face_processor.process_image(image_data, trust_level)

            if not processed_faces:
                continue

            # Validate and add each face
            for face in processed_faces:
                # Validate against existing embeddings
                validation = self._face_validator.validate(
                    new_embedding=face.embedding.facenet,
                    existing_embeddings=existing_embeddings,
                    trust_level=trust_level,
                )

                if not validation.accepted:
                    self.stats.faces_rejected += 1
                    logger.debug(f"Face rejected for {performer.name}: {validation.reason}")
                    continue

                # Add to index and database
                with self._index_lock:
                    face_index = self._index_manager.add_embedding(face.embedding)
                    self._faces_since_save += 1

                    # Periodic save
                    if self._faces_since_save >= self.SAVE_INTERVAL:
                        self._index_manager.save()
                        self._faces_since_save = 0

                # Queue database write
                future = asyncio.run_coroutine_threadsafe(
                    self.write_queue.enqueue(WriteMessage(
                        operation=WriteOperation.ADD_EMBEDDING,
                        source=source,
                        performer_id=performer_id,
                        image_url=image_url,
                        quality_score=face.quality_score,
                        embedding_type=str(face_index),  # Store index in this field
                    )),
                    self._loop,
                )
                future.result(timeout=10.0)

                faces_added += 1
                self.stats.faces_added += 1

                # Update existing embeddings for subsequent validation
                existing_embeddings.append(face.embedding.facenet)

        return faces_added

    def _get_existing_embeddings(self, performer_id: int) -> list:
        """Get existing face embeddings for a performer."""
        import numpy as np

        if not self._index_manager:
            return []

        faces = self.database.get_faces(performer_id)
        embeddings = []

        for face in faces:
            try:
                embedding = self._index_manager.get_embedding(face.facenet_index)
                embeddings.append(embedding.facenet)
            except Exception:
                pass

        return embeddings

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
        """Add face embedding record to database."""
        if not message.performer_id or not message.image_url:
            return

        # The index was stored in embedding_type field
        try:
            face_index = int(message.embedding_type)
        except (ValueError, TypeError):
            return

        self.database.add_face(
            performer_id=message.performer_id,
            facenet_index=face_index,
            arcface_index=face_index,
            image_url=message.image_url,
            source_endpoint=message.source,
            quality_score=message.quality_score,
        )

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

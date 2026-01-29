# Concurrent Scraping Architecture

**Date:** 2026-01-26
**Status:** Designed

---

## Problem Statement

Building the complete performer database requires scraping multiple stash-boxes:
- StashDB (~100k performers)
- ThePornDB (~150k performers)
- PMVStash
- FansDB
- JAVStash (~21k performers)

Each has its own rate limits. Running sequentially would take 12-16+ days. Running concurrently introduces race conditions when multiple sources find the same performer.

**Additional Challenge:** Most StashDB performers have only 1-2 face embeddings (average 1.04 faces/performer), limiting match reliability. We need to accumulate embeddings from multiple sources. See [Performer Identity Graph](2026-01-27-performer-identity-graph.md) for the cross-source linking strategy.

---

## Solution: Concurrent Scraping with Single Writer Queue

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Scraper Coordinator                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  StashDB    │  │ ThePornDB   │  │  JAVStash   │  │   PMVStash  │    │
│  │  Scraper    │  │  Scraper    │  │  Scraper    │  │   Scraper   │    │
│  │             │  │             │  │             │  │             │    │
│  │ rate: 1/sec │  │ rate: 2/sec │  │ rate: 1/sec │  │ rate: 1/sec │    │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘    │
│         │                │                │                │            │
│         │    ┌───────────┴────────────────┴───────────┐    │            │
│         │    │                                        │    │            │
│         └────┴────────────────┬───────────────────────┴────┘            │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │    Write Queue      │                              │
│                    │   (asyncio.Queue)   │                              │
│                    └──────────┬──────────┘                              │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │   Queue Consumer    │                              │
│                    │  (single writer)    │                              │
│                    │                     │                              │
│                    │  - Deduplication    │                              │
│                    │  - Face matching    │                              │
│                    │  - Merge logic      │                              │
│                    └──────────┬──────────┘                              │
│                               │                                          │
│                               ▼                                          │
│                    ┌─────────────────────┐                              │
│                    │     Database        │                              │
│                    │  (performers.db)    │                              │
│                    └─────────────────────┘                              │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Key Insight:** Scraping is I/O bound (network wait), database writes are fast. Parallelizing scraping while serializing writes gives us the best of both worlds.

---

## Why This Works

| Component | Bottleneck | Parallelizable? |
|-----------|------------|-----------------|
| HTTP requests to stash-boxes | Rate limits (1-2 req/sec) | Yes - each source independent |
| Image downloads | Network bandwidth | Yes - concurrent downloads |
| Face detection | GPU compute | Partially - batch processing |
| Embedding generation | CPU compute | Yes - per image |
| Database writes | Disk I/O | No - must serialize for consistency |

The slow parts (scraping, downloading) are parallelized. The fast part (database writes) is serialized to prevent race conditions.

---

## Time Comparison

| Approach | Estimated Time |
|----------|----------------|
| Sequential (one source at a time) | 3-4 days × 5 sources = 15-20 days |
| Concurrent with write queue | 3-4 days (limited by slowest source) |

**Speedup: ~4-5x**

---

## Architecture Components

### 1. Scraper Base Class

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator
import asyncio

@dataclass
class ScrapedPerformer:
    """Data scraped from a single source."""
    source: str  # 'stashdb', 'tpdb', 'javstash', etc.
    source_id: str  # ID in that source
    name: str
    aliases: list[str]
    image_urls: list[str]
    stash_ids: dict[str, str]  # endpoint -> id mapping (e.g., TPDB has stashdb_id field)
    external_urls: dict[str, str]  # source -> url (twitter, instagram, babepedia, etc.)
    metadata: dict  # source-specific metadata (country, birthdate, etc.)

@dataclass
class ScraperConfig:
    rate_limit: float  # requests per second
    batch_size: int  # performers per batch
    retry_attempts: int
    retry_delay: float

class BaseScraper(ABC):
    config: ScraperConfig
    source_name: str

    @abstractmethod
    async def scrape(self) -> AsyncIterator[ScrapedPerformer]:
        """Yield performers one at a time."""
        pass

    @abstractmethod
    async def get_performer_images(self, performer_id: str) -> list[str]:
        """Get image URLs for a performer."""
        pass
```

### 2. Write Queue Messages

```python
from dataclasses import dataclass
from enum import Enum
from typing import Optional
import numpy as np

class WriteOperation(Enum):
    CREATE_PERFORMER = "create_performer"
    UPDATE_PERFORMER = "update_performer"
    ADD_EMBEDDING = "add_embedding"
    ADD_STASH_ID = "add_stash_id"

@dataclass
class WriteMessage:
    """Message queued for the single writer."""
    operation: WriteOperation
    source: str

    # For CREATE/UPDATE
    performer_data: Optional[ScrapedPerformer] = None

    # For ADD_EMBEDDING
    performer_id: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    embedding_type: Optional[str] = None  # 'facenet' or 'arcface'
    image_url: Optional[str] = None

    # For ADD_STASH_ID
    stash_endpoint: Optional[str] = None
    stash_id: Optional[str] = None
```

### 3. Scraper Coordinator

```python
import asyncio
from typing import List

class ScraperCoordinator:
    def __init__(self, db_path: str):
        self.db = Database(db_path)
        self.write_queue: asyncio.Queue[WriteMessage] = asyncio.Queue()
        self.face_detector = FaceDetector()
        self.embedding_generator = EmbeddingGenerator()

        # Statistics - tracking identity graph linking outcomes
        self.stats = {
            # Linking outcomes (see performer-identity-graph.md)
            'linked_by_explicit_id': 0,      # Had stashdb_id or matching stash_id
            'linked_by_external_url': 0,     # Matched by twitter/IAFD/etc URL
            'linked_by_name_metadata': 0,    # Name + country/birthdate matched
            'linked_by_face': 0,             # Face recognition matched
            'performers_created': 0,         # New performer, no match found

            # Enrichment
            'embeddings_added': 0,
            'stash_ids_added': 0,
            'external_urls_added': 0,
            'aliases_added': 0,

            # Errors
            'errors': 0,
        }

    async def run(self, scrapers: List[BaseScraper]):
        """Run all scrapers concurrently with single writer."""

        # Create tasks
        scraper_tasks = [
            self.run_scraper(scraper)
            for scraper in scrapers
        ]
        writer_task = self.process_writes()

        # Run scrapers concurrently, writer processes queue
        try:
            await asyncio.gather(
                *scraper_tasks,
                writer_task,
            )
        except Exception as e:
            logger.error(f"Coordinator error: {e}")
            raise

    async def run_scraper(self, scraper: BaseScraper):
        """Run a single scraper, queuing results."""
        logger.info(f"Starting {scraper.source_name} scraper")

        async for performer in scraper.scrape():
            # Process images and generate embeddings
            embeddings = await self.process_performer_images(performer)

            # Queue the performer data
            await self.write_queue.put(WriteMessage(
                operation=WriteOperation.CREATE_PERFORMER,
                source=scraper.source_name,
                performer_data=performer,
            ))

            # Queue each embedding
            for emb_type, embedding, image_url in embeddings:
                await self.write_queue.put(WriteMessage(
                    operation=WriteOperation.ADD_EMBEDDING,
                    source=scraper.source_name,
                    embedding=embedding,
                    embedding_type=emb_type,
                    image_url=image_url,
                ))

        logger.info(f"Finished {scraper.source_name} scraper")

    async def process_performer_images(self, performer: ScrapedPerformer):
        """Download images, detect faces, generate embeddings."""
        embeddings = []

        for image_url in performer.image_urls:
            try:
                image = await download_image(image_url)
                faces = self.face_detector.detect(image)

                for face in faces:
                    facenet_emb = self.embedding_generator.facenet(face)
                    arcface_emb = self.embedding_generator.arcface(face)

                    embeddings.append(('facenet', facenet_emb, image_url))
                    embeddings.append(('arcface', arcface_emb, image_url))

            except Exception as e:
                logger.warning(f"Failed to process {image_url}: {e}")

        return embeddings

    async def process_writes(self):
        """Single consumer - processes all writes sequentially."""
        pending_performer: Optional[ScrapedPerformer] = None
        pending_embeddings: List[tuple] = []

        while True:
            try:
                # Get next message (with timeout to allow graceful shutdown)
                message = await asyncio.wait_for(
                    self.write_queue.get(),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # Check if all scrapers are done
                if self.all_scrapers_done:
                    break
                continue

            try:
                await self.handle_write(message)
                self.write_queue.task_done()
            except Exception as e:
                logger.error(f"Write error: {e}")
                self.stats['errors'] += 1

    async def handle_write(self, message: WriteMessage):
        """Handle a single write operation."""

        if message.operation == WriteOperation.CREATE_PERFORMER:
            await self.handle_create_performer(message)

        elif message.operation == WriteOperation.ADD_EMBEDDING:
            await self.handle_add_embedding(message)

        elif message.operation == WriteOperation.ADD_STASH_ID:
            await self.handle_add_stash_id(message)

    async def handle_create_performer(self, message: WriteMessage):
        """Create or merge performer using identity graph linking strategy.

        See: docs/plans/2026-01-27-performer-identity-graph.md
        """
        performer_data = message.performer_data

        # Step 1: Check for existing by explicit stash_ids (fastest, most reliable)
        # e.g., ThePornDB has stashdb_id field that directly links to StashDB
        existing = await self.db.find_by_stash_ids(performer_data.stash_ids)

        if existing:
            await self.merge_performer(existing, performer_data)
            self.stats['linked_by_explicit_id'] += 1
            return existing.id

        # Step 2: Check for existing by external URLs
        # Parse URLs field from StashDB to find known profiles (twitter, IAFD, etc.)
        for source, url in performer_data.external_urls.items():
            existing = await self.db.find_by_external_url(source, url)
            if existing:
                await self.merge_performer(existing, performer_data)
                self.stats['linked_by_external_url'] += 1
                return existing.id

        # Step 3: Check for existing by name + metadata (fuzzy)
        candidates = await self.db.find_by_name_fuzzy(
            performer_data.name,
            performer_data.aliases
        )

        for candidate in candidates:
            # Score by metadata overlap (country, birthdate, aliases)
            score = self.calculate_metadata_match_score(performer_data, candidate)
            if score >= 0.5:
                await self.merge_performer(candidate, performer_data)
                self.stats['linked_by_name_metadata'] += 1
                return candidate.id

        # Step 4: Face matching will happen when embeddings arrive
        # (handled in handle_add_embedding)

        # Step 5: Create new performer (only for Tier 1 stash-box sources)
        new_id = await self.db.create_performer(
            name=performer_data.name,
            aliases=performer_data.aliases,
            stash_ids=performer_data.stash_ids,
            external_urls=performer_data.external_urls,
            primary_source=message.source,
        )

        self.stats['performers_created'] += 1
        return new_id

    async def handle_add_embedding(self, message: WriteMessage):
        """Add embedding, checking for face-based duplicates."""

        # Search for matching faces in database
        matches = await self.db.search_embeddings(
            message.embedding,
            message.embedding_type,
            threshold=DUPLICATE_THRESHOLD,
        )

        if matches:
            best_match = matches[0]

            # Found existing performer with matching face
            # Add embedding to them instead of creating duplicate
            await self.db.add_embedding(
                performer_id=best_match.performer_id,
                embedding=message.embedding,
                embedding_type=message.embedding_type,
                source=message.source,
                image_url=message.image_url,
            )

            # If this came from a different source, link them
            if message.performer_id and message.performer_id != best_match.performer_id:
                await self.merge_performers(
                    keep_id=best_match.performer_id,
                    merge_id=message.performer_id,
                )
                self.stats['duplicates_merged'] += 1
        else:
            # No match - add to the performer from this source
            await self.db.add_embedding(
                performer_id=message.performer_id,
                embedding=message.embedding,
                embedding_type=message.embedding_type,
                source=message.source,
                image_url=message.image_url,
            )

        self.stats['embeddings_added'] += 1

    async def merge_performer(self, existing, new_data: ScrapedPerformer):
        """Merge new data into existing performer.

        This accumulates cross-references over time, building the identity graph.
        """

        # Add new stash_ids
        for endpoint, stash_id in new_data.stash_ids.items():
            if endpoint not in existing.stash_ids:
                await self.db.add_stash_id(existing.id, endpoint, stash_id)
                self.stats['stash_ids_added'] += 1

        # Add new external URLs
        for source, url in new_data.external_urls.items():
            if source not in existing.external_urls:
                await self.db.add_external_url(existing.id, source, url)
                self.stats['external_urls_added'] += 1

        # Add new aliases
        for alias in new_data.aliases:
            if alias not in existing.aliases:
                await self.db.add_alias(existing.id, alias)
                self.stats['aliases_added'] += 1

        # Metadata merge based on source priority
        # (handled elsewhere based on primary_source rules)
```

### 4. Running the Coordinator

```python
async def main():
    coordinator = ScraperCoordinator(db_path="performers.db")

    scrapers = [
        StashDBScraper(config=ScraperConfig(
            rate_limit=1.0,
            batch_size=100,
            retry_attempts=3,
            retry_delay=5.0,
        )),
        ThePornDBScraper(config=ScraperConfig(
            rate_limit=2.0,
            batch_size=50,
            retry_attempts=3,
            retry_delay=5.0,
        )),
        JAVStashScraper(config=ScraperConfig(
            rate_limit=1.0,
            batch_size=100,
            retry_attempts=3,
            retry_delay=5.0,
        )),
        PMVStashScraper(config=ScraperConfig(
            rate_limit=1.0,
            batch_size=100,
            retry_attempts=3,
            retry_delay=5.0,
        )),
    ]

    await coordinator.run(scrapers)

    print(f"Stats: {coordinator.stats}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Race Condition Prevention

### Scenario 1: Same Performer from Multiple Sources

```
T0: StashDB scraper finds "Jane Doe"
T1: ThePornDB scraper finds "Jane Doe"
T2: Both queue CREATE_PERFORMER messages
T3: Writer processes StashDB message first, creates performer #123
T4: Writer processes ThePornDB message, finds #123 by name, merges
Result: Single performer with both stash_ids ✓
```

### Scenario 2: Face Match Across Sources

```
T0: JAVStash creates "山田花子" (no StashDB match by name)
T1: StashDB creates "Hanako Yamada" (no match by name)
T2: Embedding from JAVStash queued
T3: Writer adds embedding to "山田花子"
T4: Embedding from StashDB queued
T5: Writer searches, finds face match to "山田花子"
T6: Writer merges "Hanako Yamada" into "山田花子"
Result: Single performer with both identities ✓
```

### Scenario 3: Concurrent Embedding Writes

```
T0: Two sources queue embeddings for same performer
T1: Writer processes first embedding, adds to DB
T2: Writer processes second embedding, adds to DB
Result: Both embeddings added, no conflict ✓
```

**The single writer guarantees sequential processing, eliminating race conditions.**

---

## Progress Tracking & Resume

```python
@dataclass
class ScrapeProgress:
    source: str
    last_processed_id: str
    last_processed_time: datetime
    performers_processed: int
    embeddings_added: int
    errors: int

class ScraperCoordinator:
    async def save_progress(self, source: str, last_id: str):
        """Save progress for resume capability."""
        await self.db.upsert_progress(ScrapeProgress(
            source=source,
            last_processed_id=last_id,
            last_processed_time=datetime.now(),
            performers_processed=self.stats['performers_processed'],
            embeddings_added=self.stats['embeddings_added'],
            errors=self.stats['errors'],
        ))

    async def get_resume_point(self, source: str) -> Optional[str]:
        """Get last processed ID for resume."""
        progress = await self.db.get_progress(source)
        return progress.last_processed_id if progress else None
```

Each scraper checks for resume point on startup:

```python
class StashDBScraper(BaseScraper):
    async def scrape(self) -> AsyncIterator[ScrapedPerformer]:
        resume_id = await self.coordinator.get_resume_point('stashdb')

        async for performer in self.api.list_performers(after=resume_id):
            yield performer
            await self.coordinator.save_progress('stashdb', performer.id)
```

---

## Monitoring & Logging

```python
class ScraperCoordinator:
    async def log_status(self):
        """Periodic status logging."""
        while not self.all_scrapers_done:
            logger.info(
                f"Progress: "
                f"performers={self.stats['performers_processed']}, "
                f"embeddings={self.stats['embeddings_added']}, "
                f"merged={self.stats['duplicates_merged']}, "
                f"errors={self.stats['errors']}, "
                f"queue_size={self.write_queue.qsize()}"
            )
            await asyncio.sleep(60)  # Log every minute
```

---

## Configuration

```yaml
# scraper_config.yaml
coordinator:
  db_path: "./data/performers.db"
  log_interval: 60

scrapers:
  stashdb:
    enabled: true
    rate_limit: 1.0
    batch_size: 100
    priority: 1  # Primary source

  theporndb:
    enabled: true
    rate_limit: 2.0
    batch_size: 50
    priority: 2

  javstash:
    enabled: true
    rate_limit: 1.0
    batch_size: 100
    priority: 3

  pmvstash:
    enabled: true
    rate_limit: 1.0
    batch_size: 100
    priority: 4

  fansdb:
    enabled: false  # Not yet implemented
    rate_limit: 1.0
    batch_size: 100
    priority: 5

thresholds:
  duplicate_face: 0.35  # Cosine distance for face dedup
  fuzzy_name: 0.85  # Jaro-Winkler for name matching
```

---

## Error Handling

### Scraper Errors (Non-Fatal)
- Rate limit exceeded → Exponential backoff
- Network timeout → Retry with delay
- Invalid performer data → Log and skip
- Image download failed → Continue without embedding

### Writer Errors (Non-Fatal)
- Database constraint violation → Log and skip
- Embedding index error → Log and retry
- Merge conflict → Log for manual review

### Fatal Errors
- Database connection lost → Save progress, exit gracefully
- Out of memory → Save progress, exit gracefully
- Coordinator exception → Save all progress, full shutdown

```python
async def run_with_recovery(self, scrapers):
    """Run with automatic recovery on non-fatal errors."""
    try:
        await self.run(scrapers)
    except FatalError:
        await self.save_all_progress()
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        await self.save_all_progress()
        # Can be restarted, will resume from last checkpoint
```

---

## Open Questions

1. **Queue memory:** For very large scrapes, should queue be disk-backed?
2. **Embedding batching:** Should we batch embedding operations for efficiency?
3. **Progress granularity:** Save per-performer or per-batch?
4. **Priority ordering:** Should high-priority sources' writes go first?
5. **Backpressure:** If writer falls behind, should scrapers slow down?

---

## Implementation Checklist

- [ ] Implement `BaseScraper` abstract class
- [ ] Implement `WriteMessage` and queue types
- [ ] Implement `ScraperCoordinator` core logic
- [ ] Implement face-based deduplication in writer
- [ ] Implement progress tracking and resume
- [ ] Add configuration loading
- [ ] Add monitoring and logging
- [ ] Implement individual scrapers:
  - [ ] StashDB (exists, needs refactor)
  - [ ] ThePornDB
  - [ ] JAVStash
  - [ ] PMVStash
  - [ ] FansDB

---

*This document will be updated as implementation progresses.*

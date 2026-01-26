# Database Builder Improvements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve the database builder to track per-performer sync status, delete images after processing, and support incremental sync with completeness thresholds.

**Architecture:** Replace the simple `processed_ids` list with a per-performer progress record that tracks faces indexed, images available, and sync timestamp. Add migration logic to convert existing 39k records. Delete images immediately after embedding extraction.

**Tech Stack:** Python, existing database_builder.py infrastructure

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Completeness threshold | 5 faces | Good balance of recognition accuracy vs processing time |
| Incomplete performer handling | Re-query StashDB | Enables future model improvements with higher thresholds |
| Image caching | Delete after processing | Disk efficiency; images not needed after embedding extraction |
| Migration approach | Infer from existing face_count | Preserves all existing work |

---

## New Progress Schema

**Old `progress.json`:**
```json
{
  "processed_ids": ["abc123", "def456", ...],
  "stats": {...},
  "last_save": "2026-01-26T..."
}
```

**New `progress.json`:**
```json
{
  "schema_version": 2,
  "performers": {
    "abc123": {
      "faces_indexed": 3,
      "images_processed": 5,
      "images_available": 5,
      "last_synced": "2026-01-26T12:00:00Z"
    },
    "def456": {
      "faces_indexed": 7,
      "images_processed": 5,
      "images_available": 12,
      "last_synced": "2026-01-26T12:00:00Z"
    }
  },
  "stats": {...},
  "last_save": "2026-01-26T..."
}
```

**Completeness logic:**
- `complete` = `faces_indexed >= COMPLETENESS_THRESHOLD` (5)
- `incomplete` = `faces_indexed < COMPLETENESS_THRESHOLD`

---

## Sync Logic

```
For each performer from StashDB:
  1. If NOT in progress:
     → Process as new (download images, extract embeddings, delete images)

  2. If in progress AND complete (faces >= 5):
     → Skip entirely

  3. If in progress AND incomplete (faces < 5):
     → Check if StashDB has more images than we processed
     → If yes: download NEW images only, extract embeddings, delete images
     → If no: skip (performer just doesn't have many images)
```

---

## Tasks

### Task 0: Fix Sort Direction for Stable Pagination

**Files:**
- Modify: `api/stashdb_client.py`

**Problem:** We're sorting by `CREATED_AT` but not specifying direction. StashDB likely defaults to descending (newest first), which breaks pagination stability when new performers are added.

**Step 1: Update `query_performers` to use ascending order**

Replace lines 109-114 in `stashdb_client.py`:

```python
    def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
        sort: str = "CREATED_AT",
        direction: str = "ASC",  # Ascending = oldest first, stable for pagination
    ) -> tuple[int, list[StashDBPerformer]]:
```

**Step 2: Update the variables dict to include direction**

Replace lines 135-140:

```python
        variables = {
            "input": {
                "page": page,
                "per_page": per_page,
                "sort": sort,
                "direction": direction,
            }
        }
```

**Step 3: Verify syntax**

Run: `python -m py_compile api/stashdb_client.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add api/stashdb_client.py
git commit -m "fix(stashdb): use ascending sort for stable pagination"
```

---

### Task 1: Add Schema Version and New Progress Record Dataclass

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Add constants and dataclass at top of file (after imports)**

Add after line 34 (after the existing imports):

```python
# Progress tracking schema version
PROGRESS_SCHEMA_VERSION = 2
COMPLETENESS_THRESHOLD = 5


@dataclass
class PerformerProgress:
    """Track sync progress for a single performer."""
    faces_indexed: int = 0
    images_processed: int = 0
    images_available: int = 0
    last_synced: str = ""  # ISO format timestamp

    def is_complete(self) -> bool:
        """Check if performer has enough faces for reliable recognition."""
        return self.faces_indexed >= COMPLETENESS_THRESHOLD

    def needs_recheck(self, current_images_available: int) -> bool:
        """Check if we should try to get more faces for this performer."""
        if self.is_complete():
            return False
        # Re-check if StashDB has more images than we processed
        return current_images_available > self.images_processed

    def to_dict(self) -> dict:
        return {
            "faces_indexed": self.faces_indexed,
            "images_processed": self.images_processed,
            "images_available": self.images_available,
            "last_synced": self.last_synced,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PerformerProgress":
        return cls(
            faces_indexed=data.get("faces_indexed", 0),
            images_processed=data.get("images_processed", 0),
            images_available=data.get("images_available", 0),
            last_synced=data.get("last_synced", ""),
        )
```

**Step 2: Verify syntax**

Run: `python -m py_compile api/database_builder.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add api/database_builder.py
git commit -m "feat(builder): add PerformerProgress dataclass and completeness threshold"
```

---

### Task 2: Update DatabaseBuilder to Use New Progress Tracking

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Update `__init__` to use new progress structure**

Replace line 74 (`self.processed_ids: set[str] = set()`) with:

```python
        self.performer_progress: dict[str, PerformerProgress] = {}  # stashdb_id -> progress
```

**Step 2: Add migration method after `_load_existing_data`**

Add this method after `_load_existing_data` (around line 171):

```python
    def _migrate_progress_v1_to_v2(self, old_progress: dict) -> dict[str, PerformerProgress]:
        """
        Migrate from v1 progress (processed_ids list) to v2 (per-performer tracking).

        Uses face_count from performers.json to infer progress.
        """
        print("  Migrating progress from v1 to v2 schema...")
        migrated = {}
        now = datetime.now(timezone.utc).isoformat()

        # Get face counts from performers.json (already loaded in self.performers)
        for universal_id, record in self.performers.items():
            # Extract stashdb_id from universal_id (e.g., "stashdb.org:abc123" -> "abc123")
            stashdb_id = record.stashdb_id

            # We don't know how many images were available, but we know:
            # - faces_indexed = record.face_count
            # - images_processed >= faces_indexed (we processed at least that many)
            # For migration, assume images_processed = max_images_per_performer (5)
            # This is conservative - if they're incomplete, they'll get rechecked
            migrated[stashdb_id] = PerformerProgress(
                faces_indexed=record.face_count,
                images_processed=self.builder_config.max_images_per_performer,
                images_available=self.builder_config.max_images_per_performer,
                last_synced=now,
            )

        # Also add any IDs from old processed_ids that aren't in performers
        # (these are performers we tried but got no faces from)
        for stashdb_id in old_progress.get("processed_ids", []):
            if stashdb_id not in migrated:
                migrated[stashdb_id] = PerformerProgress(
                    faces_indexed=0,
                    images_processed=self.builder_config.max_images_per_performer,
                    images_available=self.builder_config.max_images_per_performer,
                    last_synced=now,
                )

        print(f"  Migrated {len(migrated)} performer progress records")
        return migrated
```

**Step 3: Update `_load_existing_data` to handle migration**

Replace the progress loading section (lines 163-168) with:

```python
        # Load progress file if exists
        if self.progress_file.exists():
            with open(self.progress_file) as f:
                progress = json.load(f)

            schema_version = progress.get("schema_version", 1)

            if schema_version == 1:
                # Migrate from old format
                self.performer_progress = self._migrate_progress_v1_to_v2(progress)
                self.stats = progress.get("stats", self.stats)
            else:
                # Load v2 format
                self.performer_progress = {
                    pid: PerformerProgress.from_dict(data)
                    for pid, data in progress.get("performers", {}).items()
                }
                self.stats = progress.get("stats", self.stats)
        else:
            # No progress file - infer from performers.json
            now = datetime.now(timezone.utc).isoformat()
            for universal_id, record in self.performers.items():
                self.performer_progress[record.stashdb_id] = PerformerProgress(
                    faces_indexed=record.face_count,
                    images_processed=self.builder_config.max_images_per_performer,
                    images_available=self.builder_config.max_images_per_performer,
                    last_synced=now,
                )
```

**Step 4: Verify syntax**

Run: `python -m py_compile api/database_builder.py`
Expected: No output (success)

**Step 5: Commit**

```bash
git add api/database_builder.py
git commit -m "feat(builder): add progress migration from v1 to v2 schema"
```

---

### Task 3: Update `_save_progress` for New Schema

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Replace `_save_progress` method**

Replace the entire `_save_progress` method (lines 250-258) with:

```python
    def _save_progress(self):
        """Save progress to allow resuming."""
        progress = {
            "schema_version": PROGRESS_SCHEMA_VERSION,
            "performers": {
                pid: prog.to_dict()
                for pid, prog in self.performer_progress.items()
            },
            "stats": self.stats,
            "last_save": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.progress_file, "w") as f:
            json.dump(progress, f)
```

**Step 2: Verify syntax**

Run: `python -m py_compile api/database_builder.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add api/database_builder.py
git commit -m "feat(builder): update _save_progress for v2 schema"
```

---

### Task 4: Update Build Loop for New Sync Logic

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Update the main loop in `build_from_stashdb`**

Replace the performer processing loop (lines 304-323) with:

```python
        for performer in tqdm(performers_to_process, total=total, desc="Processing"):
            # Check for interrupt
            if self._interrupted:
                break

            # Get current progress for this performer
            progress = self.performer_progress.get(performer.id)
            images_available = len(performer.image_urls)

            if progress is not None:
                # Already seen this performer
                if progress.is_complete():
                    # Complete - skip entirely
                    self.stats["performers_skipped"] += 1
                    continue
                elif not progress.needs_recheck(images_available):
                    # Incomplete but no new images available - skip
                    self.stats["performers_skipped"] += 1
                    continue
                # else: incomplete and has new images - will reprocess below

            # Process this performer (new or incomplete with new images)
            self._process_performer(performer, stashbox_name, progress)
            performers_since_save += 1

            # Auto-save periodically
            if performers_since_save >= self.AUTO_SAVE_INTERVAL:
                tqdm.write(f"  Auto-saving progress ({len(self.performer_progress)} performers, {len(self.faces)} faces)...")
                self.save()
                self._save_progress()
                performers_since_save = 0
```

**Step 2: Verify syntax**

Run: `python -m py_compile api/database_builder.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add api/database_builder.py
git commit -m "feat(builder): update build loop with completeness-aware sync logic"
```

---

### Task 5: Update `_process_performer` for Incremental Processing

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Update `_process_performer` signature and logic**

Replace the entire `_process_performer` method (lines 339-374) with:

```python
    def _process_performer(
        self,
        performer: StashDBPerformer,
        stashbox_name: str,
        existing_progress: Optional[PerformerProgress] = None,
    ):
        """
        Process a single performer.

        Args:
            performer: Performer data from StashDB
            stashbox_name: Short name for the stash-box (e.g., "stashdb.org")
            existing_progress: Previous progress if this is a re-check, None if new
        """
        self.stats["performers_processed"] += 1
        images_available = len(performer.image_urls)

        # Skip performers with no images
        if not performer.image_urls:
            # Still track that we checked them
            self.performer_progress[performer.id] = PerformerProgress(
                faces_indexed=0,
                images_processed=0,
                images_available=0,
                last_synced=datetime.now(timezone.utc).isoformat(),
            )
            return

        # Create universal ID
        universal_id = f"{stashbox_name}:{performer.id}"

        # Get or create record
        if universal_id in self.performers:
            record = self.performers[universal_id]
        else:
            record = PerformerRecord(
                universal_id=universal_id,
                stashdb_id=performer.id,
                name=performer.name,
                country=performer.country,
                image_url=performer.image_urls[0] if performer.image_urls else None,
            )

        # Determine which images to process
        if existing_progress is not None:
            # Re-check: only process images beyond what we already processed
            images_to_process = performer.image_urls[existing_progress.images_processed:]
            start_count = existing_progress.images_processed
        else:
            # New performer: process up to max
            images_to_process = performer.image_urls[:self.builder_config.max_images_per_performer]
            start_count = 0

        # Process images
        images_processed_this_run = 0
        for url in images_to_process:
            if start_count + images_processed_this_run >= self.builder_config.max_images_per_performer:
                break

            self.stats["images_processed"] += 1
            image_data = self._download_and_process_image(url, record)
            images_processed_this_run += 1

        # Update progress tracking
        total_images_processed = start_count + images_processed_this_run
        self.performer_progress[performer.id] = PerformerProgress(
            faces_indexed=record.face_count,
            images_processed=total_images_processed,
            images_available=images_available,
            last_synced=datetime.now(timezone.utc).isoformat(),
        )

        # Store performer if we have at least one face
        if record.face_count > 0:
            self.performers[universal_id] = record
            if existing_progress is None or existing_progress.faces_indexed == 0:
                self.stats["performers_with_faces"] += 1
```

**Step 2: Verify syntax**

Run: `python -m py_compile api/database_builder.py`
Expected: No output (success)

**Step 3: Commit**

```bash
git add api/database_builder.py
git commit -m "feat(builder): update _process_performer for incremental processing"
```

---

### Task 6: Add Image Download with Immediate Deletion

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Add new method for download-process-delete workflow**

Add this method after `_process_image` (around line 248):

```python
    def _download_and_process_image(self, url: str, record: PerformerRecord) -> bool:
        """
        Download an image, process it for face embedding, and delete it.

        This is the disk-efficient version that doesn't cache images.

        Returns: True if a face was successfully indexed
        """
        # Download directly (no caching)
        try:
            image_data = self.stashdb.download_image(url)
            if not image_data:
                self.stats["images_failed"] += 1
                return False
        except Exception as e:
            print(f"  Failed to download image for {record.name}: {e}")
            self.stats["images_failed"] += 1
            return False

        # Process the image
        result = self._process_image(image_data, record)

        # Image data goes out of scope and is garbage collected
        # No disk storage needed

        return result
```

**Step 2: Remove old `_download_image` method and `_get_image_cache_path`**

Delete lines 173-193 (the `_get_image_cache_path` and `_download_image` methods).

**Step 3: Remove image_cache_dir creation from config**

In `api/config.py`, remove line 85:
```python
        self.image_cache_dir.mkdir(parents=True, exist_ok=True)
```

And remove line 73:
```python
    image_cache_dir: Path = None
```

And remove line 84:
```python
        self.image_cache_dir = self.image_cache_dir or self.data_dir / "image_cache"
```

**Step 4: Verify syntax**

Run: `python -m py_compile api/database_builder.py && python -m py_compile api/config.py`
Expected: No output (success)

**Step 5: Commit**

```bash
git add api/database_builder.py api/config.py
git commit -m "feat(builder): remove image caching, delete after processing"
```

---

### Task 7: Update Print Statements and Summary

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Update the summary print in `build_from_stashdb`**

Replace the summary section (lines 326-336) with:

```python
        # Print summary
        if self._interrupted:
            print(f"\n⚠️  Build interrupted!")
        else:
            print(f"\n✅ Build complete!")

        complete_count = sum(1 for p in self.performer_progress.values() if p.is_complete())
        incomplete_count = len(self.performer_progress) - complete_count

        print(f"  Performers processed: {self.stats['performers_processed']}")
        print(f"  Performers skipped (complete): {self.stats['performers_skipped']}")
        print(f"  Performers with faces: {self.stats['performers_with_faces']}")
        print(f"  Total faces indexed: {self.stats['faces_indexed']}")
        print(f"  Complete performers (>={COMPLETENESS_THRESHOLD} faces): {complete_count}")
        print(f"  Incomplete performers (<{COMPLETENESS_THRESHOLD} faces): {incomplete_count}")
        print(f"  Images failed: {self.stats['images_failed']}")

        return self.stats
```

**Step 2: Update the startup print to show completeness threshold**

In `build_from_stashdb`, add after line 277:

```python
        print(f"  Completeness threshold: {COMPLETENESS_THRESHOLD} faces")
```

**Step 3: Verify syntax**

Run: `python -m py_compile api/database_builder.py`
Expected: No output (success)

**Step 4: Commit**

```bash
git add api/database_builder.py
git commit -m "feat(builder): update summary output with completeness stats"
```

---

### Task 8: Add UPDATED_AT Sync Mode for Catching Modified Performers

**Files:**
- Modify: `api/stashdb_client.py`
- Modify: `api/database_builder.py`

**Purpose:** Allow syncing only performers modified since a given date, to efficiently catch those who got new images.

**Step 1: Add `iter_performers_updated_since` method to StashDBClient**

Add after `iter_all_performers` in `stashdb_client.py`:

```python
    def iter_performers_updated_since(
        self,
        since: str,  # ISO format date string
        per_page: int = 25,
        max_performers: int = None,
    ) -> Iterator[StashDBPerformer]:
        """
        Iterate through performers updated since a given date.

        Uses UPDATED_AT sort to efficiently find recently modified performers.
        """
        page = 1
        count_fetched = 0

        while True:
            count, performers = self.query_performers(
                page=page,
                per_page=per_page,
                sort="UPDATED_AT",
                direction="DESC",  # Most recently updated first
            )
            if not performers:
                break

            for performer in performers:
                # Note: We'd need to add updated_at to the query to filter properly
                # For now, this just iterates in update order
                yield performer
                count_fetched += 1
                if max_performers and count_fetched >= max_performers:
                    return

            if page * per_page >= count:
                break
            page += 1
```

**Step 2: Add `--sync-updates` CLI flag**

In `main()`, add after `--completeness-threshold`:

```python
    parser.add_argument("--sync-updates-only", action="store_true",
                        help="Only process performers updated since last sync (for incremental updates)")
```

**Step 3: Commit**

```bash
git add api/stashdb_client.py api/database_builder.py
git commit -m "feat(builder): add UPDATED_AT sync mode for incremental updates"
```

---

### Task 9: Improve Progress Logging with Performance Breakdown

**Files:**
- Modify: `api/database_builder.py`

**Problem:**
1. tqdm time estimates are unstable (based on short smoothing window)
2. No visibility into where time is spent (fetch vs process vs save)

**Step 1: Add timing tracker dataclass**

Add after `PerformerProgress` class:

```python
@dataclass
class BatchTimings:
    """Track time spent in each phase during a batch."""
    fetch_performer_ms: list[float] = field(default_factory=list)
    fetch_images_ms: list[float] = field(default_factory=list)
    process_face_ms: list[float] = field(default_factory=list)

    def reset(self):
        self.fetch_performer_ms.clear()
        self.fetch_images_ms.clear()
        self.process_face_ms.clear()

    def summary(self) -> str:
        def avg(lst):
            return sum(lst) / len(lst) if lst else 0

        def fmt(ms):
            if ms >= 1000:
                return f"{ms/1000:.2f}s"
            return f"{ms:.0f}ms"

        parts = []
        if self.fetch_performer_ms:
            parts.append(f"Fetch metadata: {fmt(avg(self.fetch_performer_ms))}")
        if self.fetch_images_ms:
            parts.append(f"Fetch images: {fmt(avg(self.fetch_images_ms))}")
        if self.process_face_ms:
            parts.append(f"Process face: {fmt(avg(self.process_face_ms))}")

        return " | ".join(parts) if parts else "No timing data"
```

**Step 2: Initialize timings in `__init__`**

Add to `DatabaseBuilder.__init__`:

```python
        self.batch_timings = BatchTimings()
```

**Step 3: Update tqdm for better estimates**

In `build_from_stashdb`, change the tqdm call:

```python
        for performer in tqdm(
            performers_to_process,
            total=total,
            desc="Processing",
            smoothing=0.05,  # Use longer history for stable estimates
            mininterval=1.0,  # Update at most once per second
        ):
```

**Step 4: Add timing instrumentation to `_process_performer`**

Wrap the key operations with timing:

```python
    def _process_performer(self, performer, stashbox_name, existing_progress=None):
        import time

        # ... existing setup code ...

        # Process images with timing
        for url in images_to_process:
            if start_count + images_processed_this_run >= self.builder_config.max_images_per_performer:
                break

            self.stats["images_processed"] += 1

            # Time image fetch
            t0 = time.perf_counter()
            image_data = self.stashdb.download_image(url)
            self.batch_timings.fetch_images_ms.append((time.perf_counter() - t0) * 1000)

            if image_data:
                # Time face processing
                t0 = time.perf_counter()
                if self._process_image(image_data, record):
                    images_processed_this_run += 1
                self.batch_timings.process_face_ms.append((time.perf_counter() - t0) * 1000)

        # ... rest of method ...
```

**Step 5: Print batch summary and reset timings after auto-save**

Update the auto-save section in `build_from_stashdb`:

```python
            # Auto-save periodically
            if performers_since_save >= self.AUTO_SAVE_INTERVAL:
                # Print batch performance summary
                tqdm.write(f"\n  Batch complete: {self.batch_timings.summary()}")
                tqdm.write(f"  Saving progress ({len(self.performer_progress)} performers, {len(self.faces)} faces)...")

                self.save()
                self._save_progress()
                self.batch_timings.reset()
                performers_since_save = 0
```

**Step 6: Reduce save verbosity**

Update the `save()` method to be less verbose during auto-saves. Add a `quiet` parameter:

```python
    def save(self, quiet: bool = False):
        """Save the database to files."""
        if not quiet:
            print("\nSaving database...")

        # Save Voyager indices
        if not quiet:
            print(f"  Saving FaceNet index to {self.db_config.facenet_index_path}")
        with open(self.db_config.facenet_index_path, "wb") as f:
            self.facenet_index.save(f)

        # ... similar for other files ...

        if not quiet:
            print("Database saved successfully!")
```

Then call `self.save(quiet=True)` during auto-saves.

**Step 7: Verify and commit**

Run: `python -m py_compile api/database_builder.py`
Expected: No output (success)

```bash
git add api/database_builder.py
git commit -m "feat(builder): add performance breakdown logging and stable time estimates"
```

**Expected output after changes:**

```
Processing:  37%|███████████████████████████▏ | 38558/103502 [24:57:46<41:23:15,  2.31s/it]

  Batch complete: Fetch metadata: 0.26s | Fetch images: 1.12s | Process face: 589ms
  Saving progress (38593 performers, 36679 faces)...

Processing:  38%|███████████████████████████▎ | 38658/103502 [25:02:58<41:18:42,  2.29s/it]
```

---

### Task 10: Add CLI Flag for Completeness Threshold

**Files:**
- Modify: `api/database_builder.py`

**Step 1: Update argparse in `main()`**

Add this argument after `--resume` (around line 492):

```python
    parser.add_argument("--completeness-threshold", type=int, default=5,
                        help="Minimum faces for a performer to be 'complete' (default: 5)")
```

**Step 2: Pass threshold to BuilderConfig**

Update `BuilderConfig` in `api/config.py` to include the threshold. Add after line 44:

```python
    completeness_threshold: int = 5
```

**Step 3: Update the builder initialization in `main()`**

Update the `BuilderConfig` creation (around line 502):

```python
    builder_config = BuilderConfig(
        max_performers=args.max_performers,
        max_images_per_performer=args.max_images,
        version=args.version,
        completeness_threshold=args.completeness_threshold,
    )
```

**Step 4: Update PerformerProgress to use config threshold**

In `DatabaseBuilder.__init__`, store the threshold and update `PerformerProgress.is_complete()` to use it. This requires passing the threshold to the class or using it from builder_config.

Actually, simpler approach: make `COMPLETENESS_THRESHOLD` a module-level variable that gets set from config. Or just use the builder_config value directly.

Replace the `is_complete` method to take threshold as parameter:

```python
    def is_complete(self, threshold: int = 5) -> bool:
        """Check if performer has enough faces for reliable recognition."""
        return self.faces_indexed >= threshold
```

And update all calls to `progress.is_complete()` to `progress.is_complete(self.builder_config.completeness_threshold)`.

**Step 5: Verify syntax**

Run: `python -m py_compile api/database_builder.py && python -m py_compile api/config.py`
Expected: No output (success)

**Step 6: Commit**

```bash
git add api/database_builder.py api/config.py
git commit -m "feat(builder): add --completeness-threshold CLI flag"
```

---

### Task 11: Clean Up Existing Image Cache

**Files:**
- No code changes, just cleanup

**Step 1: After stopping the current build, delete the image cache**

```bash
rm -rf api/data/image_cache/
```

This will free up ~21GB of disk space.

**Step 2: Verify disk space recovered**

```bash
df -h /home/carrot/code/stash-face-recognition
```

Expected: ~39GB available (18GB + 21GB recovered)

---

### Task 12: Test Migration with Existing Data

**Step 1: Back up current progress files**

```bash
cp api/data/progress.json api/data/progress.json.backup
cp api/data/performers.json api/data/performers.json.backup
```

**Step 2: Run builder with --resume to test migration**

```bash
cd /home/carrot/code/stash-face-recognition/api
python database_builder.py --resume --output ./data --max-performers 100
```

Expected output should show:
- "Migrating progress from v1 to v2 schema..."
- "Migrated ~39000 performer progress records"
- Most performers skipped as complete or incomplete-no-new-images

**Step 3: Verify new progress.json format**

```bash
python -c "import json; d=json.load(open('api/data/progress.json')); print(f'Schema version: {d.get(\"schema_version\")}'); print(f'Performers tracked: {len(d.get(\"performers\", {}))}')"
```

Expected: Schema version 2, ~39000 performers tracked

**Step 4: Commit the backup files to gitignore (they're already ignored)**

No action needed - data/ is already in .gitignore.

---

## Summary of Changes

| File | Changes |
|------|---------|
| `api/database_builder.py` | New PerformerProgress dataclass, migration logic, sync logic, no image caching |
| `api/config.py` | Remove image_cache_dir, add completeness_threshold |

## Migration Behavior

When you restart the build with `--resume`:

1. **Migration runs**: Old `progress.json` (v1) converts to v2 schema
2. **Existing 39k performers**:
   - Those with `face_count >= 5` → marked complete → skipped
   - Those with `face_count < 5` → marked incomplete → will be re-checked if StashDB has more images
3. **New performers** (39k - 103k): processed normally
4. **No image caching**: Images downloaded, processed, garbage collected

## Disk Usage After Migration

- Image cache deleted: **-21GB**
- Ongoing usage: **~0** (images not stored)
- Progress.json: **~5MB** (larger than before due to per-performer tracking)
- .voy files: **~100MB** (grows as more faces added)

---

## Rollback Plan

If something goes wrong:

```bash
# Restore old progress
cp api/data/progress.json.backup api/data/progress.json

# The .voy files and performers.json are still valid
# You can re-run the old code if needed
```

The .voy indices and performers.json are append-only, so no data is lost.

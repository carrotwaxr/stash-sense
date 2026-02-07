# Gallery & Image Identification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add face recognition identification to Stash Gallery and Image pages, with fingerprint storage and per-performer tagging controls.

**Architecture:** New `/identify/image` and `/identify/gallery` sidecar endpoints fetch images from Stash via GraphQL, run through the existing `recognize_image()` pipeline, store fingerprints in SQLite, and return results. The plugin injects "Identify Performers" buttons on image/gallery pages, shows results in modals, and applies tags via Stash GraphQL mutations.

**Tech Stack:** Python/FastAPI (sidecar), SQLite (fingerprints), JavaScript (plugin UI), Stash GraphQL API

---

### Task 1: Database Schema — Image Fingerprint Tables

**Files:**
- Modify: `api/recommendations_db.py:19` (SCHEMA_VERSION), `:107-245` (_create_schema), `:247-333` (_migrate_schema)
- Test: `api/tests/test_image_fingerprints.py` (create)

**Step 1: Write the failing test**

Create `api/tests/test_image_fingerprints.py`:

```python
"""Tests for image fingerprint storage in recommendations DB."""

import pytest


class TestImageFingerprintSchema:
    """Tests for image fingerprint table operations."""

    def test_create_image_fingerprint(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_image_fingerprint(
            stash_image_id="27839",
            gallery_id=None,
            faces_detected=3,
            db_version="v1.0",
        )

        assert fp_id is not None
        assert fp_id > 0

    def test_get_image_fingerprint(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(
            stash_image_id="100",
            gallery_id="50",
            faces_detected=2,
        )

        fp = db.get_image_fingerprint(stash_image_id="100")

        assert fp is not None
        assert fp["stash_image_id"] == "100"
        assert fp["gallery_id"] == "50"
        assert fp["faces_detected"] == 2

    def test_image_fingerprint_upsert(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        id1 = db.create_image_fingerprint(stash_image_id="200", faces_detected=1)
        id2 = db.create_image_fingerprint(stash_image_id="200", faces_detected=5)

        assert id1 == id2

        fp = db.get_image_fingerprint(stash_image_id="200")
        assert fp["faces_detected"] == 5

    def test_add_image_fingerprint_face(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        fp_id = db.create_image_fingerprint(stash_image_id="300", faces_detected=1)

        face_id = db.add_image_fingerprint_face(
            stash_image_id="300",
            performer_id="stashdb:abc-123",
            confidence=0.85,
            distance=0.15,
            bbox_x=0.1, bbox_y=0.2, bbox_w=0.3, bbox_h=0.4,
        )

        assert face_id is not None

        faces = db.get_image_fingerprint_faces("300")
        assert len(faces) == 1
        assert faces[0]["performer_id"] == "stashdb:abc-123"
        assert faces[0]["confidence"] == 0.85
        assert faces[0]["bbox_x"] == 0.1

    def test_get_gallery_image_fingerprints(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="401", gallery_id="50", faces_detected=2)
        db.create_image_fingerprint(stash_image_id="402", gallery_id="50", faces_detected=1)
        db.create_image_fingerprint(stash_image_id="403", gallery_id="99", faces_detected=3)
        db.create_image_fingerprint(stash_image_id="404", gallery_id=None, faces_detected=0)

        fps = db.get_gallery_image_fingerprints(gallery_id="50")
        assert len(fps) == 2

    def test_delete_image_fingerprint_faces(self, tmp_path):
        from recommendations_db import RecommendationsDB

        db = RecommendationsDB(tmp_path / "test.db")

        db.create_image_fingerprint(stash_image_id="500", faces_detected=2)
        db.add_image_fingerprint_face(stash_image_id="500", performer_id="p1", confidence=0.9, distance=0.1)
        db.add_image_fingerprint_face(stash_image_id="500", performer_id="p2", confidence=0.8, distance=0.2)

        count = db.delete_image_fingerprint_faces("500")
        assert count == 2

        faces = db.get_image_fingerprint_faces("500")
        assert len(faces) == 0

    def test_schema_migration_from_v5_to_v6(self, tmp_path):
        from recommendations_db import RecommendationsDB

        # Create a v5 database
        import sqlite3
        db_path = tmp_path / "migrate.db"
        conn = sqlite3.connect(db_path)
        conn.execute("CREATE TABLE schema_version (version INTEGER PRIMARY KEY)")
        conn.execute("INSERT INTO schema_version (version) VALUES (5)")
        # Create minimal v5 tables so migration doesn't fail
        conn.execute("""CREATE TABLE recommendations (
            id INTEGER PRIMARY KEY, type TEXT, status TEXT DEFAULT 'pending',
            target_type TEXT, target_id TEXT, details JSON,
            resolution_action TEXT, resolution_details JSON, resolved_at TEXT,
            confidence REAL, source_analysis_id INTEGER,
            created_at TEXT, updated_at TEXT,
            UNIQUE(type, target_type, target_id))""")
        conn.execute("""CREATE TABLE analysis_runs (
            id INTEGER PRIMARY KEY, type TEXT, status TEXT, started_at TEXT,
            completed_at TEXT, items_total INTEGER, items_processed INTEGER,
            recommendations_created INTEGER DEFAULT 0, cursor TEXT, error_message TEXT)""")
        conn.execute("""CREATE TABLE recommendation_settings (
            type TEXT PRIMARY KEY, enabled INTEGER DEFAULT 1,
            auto_dismiss_threshold REAL, notify INTEGER DEFAULT 1,
            interval_hours INTEGER, last_run_at TEXT, next_run_at TEXT, config JSON)""")
        conn.execute("""CREATE TABLE dismissed_targets (
            type TEXT, target_type TEXT, target_id TEXT, dismissed_at TEXT,
            reason TEXT, permanent INTEGER DEFAULT 0,
            PRIMARY KEY (type, target_type, target_id))""")
        conn.execute("""CREATE TABLE analysis_watermarks (
            type TEXT PRIMARY KEY, last_completed_at TEXT, last_cursor TEXT, last_stash_updated_at TEXT)""")
        conn.execute("""CREATE TABLE scene_fingerprints (
            id INTEGER PRIMARY KEY, stash_scene_id INTEGER NOT NULL UNIQUE,
            total_faces INTEGER DEFAULT 0, frames_analyzed INTEGER DEFAULT 0,
            fingerprint_status TEXT DEFAULT 'pending', db_version TEXT,
            created_at TEXT, updated_at TEXT)""")
        conn.execute("""CREATE TABLE scene_fingerprint_faces (
            id INTEGER PRIMARY KEY, fingerprint_id INTEGER, performer_id TEXT,
            face_count INTEGER DEFAULT 0, avg_confidence REAL, proportion REAL,
            created_at TEXT, UNIQUE(fingerprint_id, performer_id))""")
        conn.execute("""CREATE TABLE upstream_snapshots (
            id INTEGER PRIMARY KEY, entity_type TEXT, local_entity_id TEXT,
            endpoint TEXT, stash_box_id TEXT, upstream_data JSON,
            upstream_updated_at TEXT, fetched_at TEXT,
            UNIQUE(entity_type, endpoint, stash_box_id))""")
        conn.execute("""CREATE TABLE upstream_field_config (
            endpoint TEXT, entity_type TEXT, field_name TEXT, enabled INTEGER DEFAULT 1,
            PRIMARY KEY (endpoint, entity_type, field_name))""")
        conn.execute("""CREATE TABLE user_settings (
            key TEXT PRIMARY KEY, value JSON, updated_at TEXT)""")
        conn.commit()
        conn.close()

        # Open with new code — should migrate
        db = RecommendationsDB(db_path)

        # Verify image fingerprint tables work
        fp_id = db.create_image_fingerprint(stash_image_id="1", faces_detected=2)
        assert fp_id is not None
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense && .venv/bin/python -m pytest api/tests/test_image_fingerprints.py -v`
Expected: FAIL — `create_image_fingerprint` method doesn't exist

**Step 3: Implement schema changes**

In `api/recommendations_db.py`:

1. Change `SCHEMA_VERSION = 5` to `SCHEMA_VERSION = 6` (line 19)

2. Add to `_create_schema` (after the `user_settings` seed, before the closing `"""`):

```python
            -- Image fingerprints for gallery/image identification
            CREATE TABLE image_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_image_id TEXT NOT NULL UNIQUE,
                gallery_id TEXT,
                faces_detected INTEGER NOT NULL DEFAULT 0,
                db_version TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_image_fp_gallery ON image_fingerprints(gallery_id);

            -- Face entries within image fingerprints
            CREATE TABLE image_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_image_id TEXT NOT NULL REFERENCES image_fingerprints(stash_image_id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                confidence REAL,
                distance REAL,
                bbox_x REAL,
                bbox_y REAL,
                bbox_w REAL,
                bbox_h REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(stash_image_id, performer_id)
            );
            CREATE INDEX idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
            CREATE INDEX idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);
```

3. Add migration in `_migrate_schema` (after the `from_version < 5` block):

```python
        if from_version < 6:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS image_fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_image_id TEXT NOT NULL UNIQUE,
                    gallery_id TEXT,
                    faces_detected INTEGER NOT NULL DEFAULT 0,
                    db_version TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_image_fp_gallery ON image_fingerprints(gallery_id);

                CREATE TABLE IF NOT EXISTS image_fingerprint_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_image_id TEXT NOT NULL REFERENCES image_fingerprints(stash_image_id) ON DELETE CASCADE,
                    performer_id TEXT NOT NULL,
                    confidence REAL,
                    distance REAL,
                    bbox_x REAL,
                    bbox_y REAL,
                    bbox_w REAL,
                    bbox_h REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(stash_image_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);

                UPDATE schema_version SET version = 6;
            """)
```

4. Add CRUD methods to `RecommendationsDB` class (after the Scene Fingerprints section, around line 1148):

```python
    # ==================== Image Fingerprints ====================

    def create_image_fingerprint(
        self,
        stash_image_id: str,
        faces_detected: int = 0,
        gallery_id: Optional[str] = None,
        db_version: Optional[str] = None,
    ) -> int:
        """Create or update an image fingerprint. Returns the fingerprint ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_fingerprints (stash_image_id, gallery_id, faces_detected, db_version)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(stash_image_id) DO UPDATE SET
                    gallery_id = COALESCE(excluded.gallery_id, gallery_id),
                    faces_detected = excluded.faces_detected,
                    db_version = excluded.db_version,
                    updated_at = datetime('now')
                RETURNING id
                """,
                (stash_image_id, gallery_id, faces_detected, db_version)
            )
            return cursor.fetchone()[0]

    def get_image_fingerprint(self, stash_image_id: str) -> Optional[dict]:
        """Get an image fingerprint by stash image ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM image_fingerprints WHERE stash_image_id = ?",
                (stash_image_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_gallery_image_fingerprints(self, gallery_id: str) -> list[dict]:
        """Get all image fingerprints for a gallery."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM image_fingerprints WHERE gallery_id = ?",
                (gallery_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def add_image_fingerprint_face(
        self,
        stash_image_id: str,
        performer_id: str,
        confidence: Optional[float] = None,
        distance: Optional[float] = None,
        bbox_x: Optional[float] = None,
        bbox_y: Optional[float] = None,
        bbox_w: Optional[float] = None,
        bbox_h: Optional[float] = None,
    ) -> int:
        """Add or update a face entry in an image fingerprint. Returns the face entry ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO image_fingerprint_faces (
                    stash_image_id, performer_id, confidence, distance,
                    bbox_x, bbox_y, bbox_w, bbox_h
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(stash_image_id, performer_id) DO UPDATE SET
                    confidence = excluded.confidence,
                    distance = excluded.distance,
                    bbox_x = excluded.bbox_x,
                    bbox_y = excluded.bbox_y,
                    bbox_w = excluded.bbox_w,
                    bbox_h = excluded.bbox_h
                RETURNING id
                """,
                (stash_image_id, performer_id, confidence, distance,
                 bbox_x, bbox_y, bbox_w, bbox_h)
            )
            return cursor.fetchone()[0]

    def get_image_fingerprint_faces(self, stash_image_id: str) -> list[dict]:
        """Get all face entries for an image fingerprint."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM image_fingerprint_faces WHERE stash_image_id = ?",
                (stash_image_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_image_fingerprint_faces(self, stash_image_id: str) -> int:
        """Delete all face entries for an image fingerprint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM image_fingerprint_faces WHERE stash_image_id = ?",
                (stash_image_id,)
            )
            return cursor.rowcount
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense && .venv/bin/python -m pytest api/tests/test_image_fingerprints.py -v`
Expected: All 7 tests PASS

**Step 5: Run existing tests to verify no regressions**

Run: `cd /home/carrot/code/stash-sense && .venv/bin/python -m pytest api/tests/test_scene_fingerprints.py -v`
Expected: All existing tests still PASS

**Step 6: Commit**

```bash
git add api/recommendations_db.py api/tests/test_image_fingerprints.py
git commit -m "feat: add image fingerprint tables for gallery/image identification"
```

---

### Task 2: Stash GraphQL Queries for Images & Galleries

**Files:**
- Modify: `api/stash_client_unified.py:648` (end of file — add new methods)

**Step 1: Add `get_image_by_id` and `get_gallery_by_id` methods**

Add at end of `StashClientUnified` class in `api/stash_client_unified.py`:

```python
    # ==================== Images ====================

    async def get_image_by_id(self, image_id: str) -> Optional[dict]:
        """Get an image by ID with paths and performers."""
        query = """
        query GetImage($id: ID!) {
          findImage(id: $id) {
            id
            title
            paths {
              image
              thumbnail
            }
            performers {
              id
              name
              image_path
            }
          }
        }
        """
        data = await self._execute(query, {"id": image_id})
        return data.get("findImage")

    # ==================== Galleries ====================

    async def get_gallery_by_id(self, gallery_id: str) -> Optional[dict]:
        """Get a gallery by ID with all images and performers."""
        query = """
        query GetGallery($id: ID!) {
          findGallery(id: $id) {
            id
            title
            image_count
            performers {
              id
              name
              image_path
            }
            images {
              id
              title
              paths {
                image
                thumbnail
              }
              performers {
                id
                name
              }
            }
          }
        }
        """
        data = await self._execute(query, {"id": gallery_id})
        return data.get("findGallery")
```

**Step 2: Commit**

```bash
git add api/stash_client_unified.py
git commit -m "feat: add GraphQL queries for image and gallery lookup"
```

---

### Task 3: `/identify/image` Endpoint

**Files:**
- Modify: `api/main.py:86-92` (add new Pydantic models after `IdentifyResponse`), `:368-380` (add endpoint after `/identify/url`)
- Modify: `api/recommendations_router.py` (import `save_image_fingerprint` function or add it inline)

**Step 1: Add Pydantic models and endpoint**

In `api/main.py`, add after the `IdentifyResponse` model (line 92):

```python
class ImageIdentifyRequest(BaseModel):
    """Request to identify performers in a Stash image by ID."""
    image_id: str = Field(description="Stash image ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")
```

Add the endpoint after the `/identify/url` endpoint (after line 380):

```python
@app.post("/identify/image", response_model=IdentifyResponse)
async def identify_image(request: ImageIdentifyRequest):
    """
    Identify performers in a Stash image by image ID.

    Fetches the image from Stash, runs face recognition, and stores fingerprint.
    """
    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")

    base_url = STASH_URL.rstrip("/")
    api_key = STASH_API_KEY

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    # Fetch image info from Stash
    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(base_url, api_key)
    image_data = await stash_client.get_image_by_id(request.image_id)

    if not image_data:
        raise HTTPException(status_code=404, detail="Image not found")

    image_url = image_data.get("paths", {}).get("image")
    if not image_url:
        raise HTTPException(status_code=400, detail="Image has no image path")

    # Fetch the image
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            headers = {"ApiKey": api_key} if api_key else {}
            response = await client.get(image_url, headers=headers)
            response.raise_for_status()
            image = load_image(response.content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {e}")

    # Run recognition
    try:
        results = recognizer.recognize_image(
            image,
            top_k=request.top_k,
            max_distance=request.max_distance,
            min_face_confidence=request.min_face_confidence,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {e}")

    # Convert to response format (same as /identify)
    faces = []
    img_h, img_w = image.shape[:2]

    for result in results:
        box = result.face.box
        face_box = FaceBox(
            x=int(box[0]),
            y=int(box[1]),
            width=int(box[2] - box[0]),
            height=int(box[3] - box[1]),
            confidence=result.face.confidence,
        )

        matches = [
            PerformerMatchResponse(
                stashdb_id=m.stashdb_id,
                name=m.name,
                confidence=distance_to_confidence(m.combined_score),
                distance=m.combined_score,
                facenet_distance=m.facenet_distance,
                arcface_distance=m.arcface_distance,
                country=m.country,
                image_url=m.image_url,
            )
            for m in result.matches
        ]

        faces.append(FaceResult(box=face_box, matches=matches))

    # Save fingerprint
    try:
        from recommendations_router import save_image_fingerprint
        save_image_fingerprint(
            image_id=request.image_id,
            gallery_id=None,
            faces=results,
            image_shape=(img_h, img_w),
            db_version=db_manifest.get("version"),
        )
    except Exception as e:
        print(f"[identify_image] Failed to save fingerprint: {e}")

    return IdentifyResponse(faces=faces, face_count=len(faces))
```

**Step 2: Add `save_image_fingerprint` helper in `api/recommendations_router.py`**

Add near the existing `save_scene_fingerprint` function:

```python
def save_image_fingerprint(
    image_id: str,
    gallery_id: Optional[str],
    faces: list,
    image_shape: tuple[int, int],
    db_version: Optional[str] = None,
) -> tuple[Optional[int], Optional[str]]:
    """Save image identification results as a fingerprint."""
    if _db is None:
        return None, "Database not initialized"

    try:
        img_h, img_w = image_shape

        fp_id = _db.create_image_fingerprint(
            stash_image_id=image_id,
            gallery_id=gallery_id,
            faces_detected=len(faces),
            db_version=db_version,
        )

        # Clear old face data
        _db.delete_image_fingerprint_faces(image_id)

        # Save each detected face's best match
        for result in faces:
            if result.matches:
                best = result.matches[0]
                box = result.face.box
                _db.add_image_fingerprint_face(
                    stash_image_id=image_id,
                    performer_id=best.stashdb_id,
                    confidence=distance_to_confidence(best.combined_score),
                    distance=best.combined_score,
                    bbox_x=box[0] / img_w if img_w > 0 else 0,
                    bbox_y=box[1] / img_h if img_h > 0 else 0,
                    bbox_w=(box[2] - box[0]) / img_w if img_w > 0 else 0,
                    bbox_h=(box[3] - box[1]) / img_h if img_h > 0 else 0,
                )

        return fp_id, None
    except Exception as e:
        return None, str(e)
```

Note: Import `distance_to_confidence` from `main` or duplicate the simple helper inline. Since recommendations_router already imports from main via the router pattern, add a local copy:

```python
def distance_to_confidence(distance: float) -> float:
    """Convert distance score to confidence (0-1)."""
    return max(0.0, min(1.0, 1.0 - distance))
```

**Step 3: Commit**

```bash
git add api/main.py api/recommendations_router.py
git commit -m "feat: add /identify/image endpoint for single image identification"
```

---

### Task 4: `/identify/gallery` Endpoint

**Files:**
- Modify: `api/main.py` (add models and endpoint after `/identify/image`)

**Step 1: Add Pydantic models**

Add after `ImageIdentifyRequest`:

```python
class GalleryPerformerResult(BaseModel):
    """A performer identified across a gallery."""
    performer_id: str = Field(description="StashDB performer UUID")
    name: str
    best_distance: float
    avg_distance: float
    confidence: float = Field(description="Best match confidence (0-1)")
    image_count: int = Field(description="Number of images this performer appeared in")
    image_ids: list[str] = Field(description="Stash image IDs where performer was found")
    country: Optional[str] = None
    image_url: Optional[str] = Field(None, description="StashDB profile image URL")


class GalleryIdentifyRequest(BaseModel):
    """Request to identify performers in a Stash gallery."""
    gallery_id: str = Field(description="Stash gallery ID")
    top_k: int = Field(5, ge=1, le=20, description="Number of matches per face")
    max_distance: float = Field(0.6, ge=0.0, le=2.0, description="Maximum distance threshold")
    min_face_confidence: float = Field(0.5, ge=0.0, le=1.0, description="Minimum face detection confidence")


class GalleryIdentifyResponse(BaseModel):
    """Response with gallery identification results."""
    gallery_id: str
    total_images: int
    images_processed: int
    faces_detected: int
    performers: list[GalleryPerformerResult]
    errors: list[str] = []
```

**Step 2: Add the endpoint**

Add after the `/identify/image` endpoint:

```python
@app.post("/identify/gallery", response_model=GalleryIdentifyResponse)
async def identify_gallery(request: GalleryIdentifyRequest):
    """
    Identify all performers across a gallery.

    Processes each image, aggregates results per-performer, and stores fingerprints.
    """
    import time
    t_start = time.time()

    if recognizer is None:
        raise HTTPException(status_code=503, detail="Database not loaded")

    base_url = STASH_URL.rstrip("/")
    api_key = STASH_API_KEY

    if not base_url:
        raise HTTPException(status_code=400, detail="STASH_URL env var not set")

    # Fetch gallery info from Stash
    from stash_client_unified import StashClientUnified
    stash_client = StashClientUnified(base_url, api_key)
    gallery_data = await stash_client.get_gallery_by_id(request.gallery_id)

    if not gallery_data:
        raise HTTPException(status_code=404, detail="Gallery not found")

    images = gallery_data.get("images", [])
    if not images:
        return GalleryIdentifyResponse(
            gallery_id=request.gallery_id,
            total_images=0,
            images_processed=0,
            faces_detected=0,
            performers=[],
        )

    total_images = len(images)
    print(f"[identify_gallery] === START gallery_id={request.gallery_id}, {total_images} images ===")

    # Process each image
    from recommendations_router import save_image_fingerprint
    performer_appearances: dict[str, list[dict]] = defaultdict(list)
    performer_info: dict[str, dict] = {}
    total_faces = 0
    images_processed = 0
    errors = []

    for i, img in enumerate(images):
        img_id = img["id"]
        img_url = img.get("paths", {}).get("image")

        if not img_url:
            errors.append(f"Image {img_id} has no URL")
            continue

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {"ApiKey": api_key} if api_key else {}
                resp = await client.get(img_url, headers=headers)
                resp.raise_for_status()
                image = load_image(resp.content)

            results = recognizer.recognize_image(
                image,
                top_k=request.top_k,
                max_distance=request.max_distance,
                min_face_confidence=request.min_face_confidence,
            )

            img_h, img_w = image.shape[:2]
            total_faces += len(results)
            images_processed += 1

            # Save per-image fingerprint
            try:
                save_image_fingerprint(
                    image_id=img_id,
                    gallery_id=request.gallery_id,
                    faces=results,
                    image_shape=(img_h, img_w),
                    db_version=db_manifest.get("version"),
                )
            except Exception as e:
                print(f"[identify_gallery] Failed to save fingerprint for image {img_id}: {e}")

            # Collect per-performer data
            for result in results:
                if result.matches:
                    best = result.matches[0]
                    pid = best.stashdb_id

                    performer_appearances[pid].append({
                        "image_id": img_id,
                        "distance": best.combined_score,
                    })

                    # Keep best info
                    if pid not in performer_info or best.combined_score < performer_info[pid]["distance"]:
                        performer_info[pid] = {
                            "name": best.name,
                            "distance": best.combined_score,
                            "country": best.country,
                            "image_url": best.image_url,
                        }

            if (i + 1) % 10 == 0:
                print(f"[identify_gallery] [{time.time()-t_start:.1f}s] Processed {i+1}/{total_images} images")

        except Exception as e:
            errors.append(f"Image {img_id}: {str(e)[:100]}")
            print(f"[identify_gallery] Error processing image {img_id}: {e}")

    # Aggregate results
    performers = []
    for pid, appearances in performer_appearances.items():
        distances = [a["distance"] for a in appearances]
        image_ids = [a["image_id"] for a in appearances]
        image_count = len(set(image_ids))
        best_distance = min(distances)
        avg_distance = sum(distances) / len(distances)

        # Filter: 2+ appearances OR single match with distance < 0.4
        if image_count < 2 and best_distance >= 0.4:
            continue

        info = performer_info[pid]
        performers.append(GalleryPerformerResult(
            performer_id=pid,
            name=info["name"],
            best_distance=best_distance,
            avg_distance=avg_distance,
            confidence=max(0.0, min(1.0, 1.0 - best_distance)),
            image_count=image_count,
            image_ids=list(set(image_ids)),
            country=info.get("country"),
            image_url=info.get("image_url"),
        ))

    # Sort by image count desc, then best distance asc
    performers.sort(key=lambda p: (-p.image_count, p.best_distance))

    top_names = [p.name for p in performers[:3]]
    print(f"[identify_gallery] [{time.time()-t_start:.1f}s] === DONE === "
          f"{images_processed}/{total_images} images, {total_faces} faces, "
          f"{len(performers)} performers: {', '.join(top_names)}")

    return GalleryIdentifyResponse(
        gallery_id=request.gallery_id,
        total_images=total_images,
        images_processed=images_processed,
        faces_detected=total_faces,
        performers=performers,
        errors=errors[:10],
    )
```

**Step 3: Commit**

```bash
git add api/main.py
git commit -m "feat: add /identify/gallery endpoint for gallery-wide identification"
```

---

### Task 5: Backend Proxy — New Modes

**Files:**
- Modify: `plugin/stash_sense_backend.py:24-43` (dispatcher), add new functions

**Step 1: Add `identify_image` and `identify_gallery` modes**

In `plugin/stash_sense_backend.py`, add to the dispatcher in `main()` (after the `identify_scene` elif, before the `database_info` elif):

```python
    elif mode == "identify_image":
        image_id = args.get("image_id")
        result = identify_image(sidecar_url, image_id)
    elif mode == "identify_gallery":
        gallery_id = args.get("gallery_id")
        result = identify_gallery(sidecar_url, gallery_id)
```

Add the two proxy functions (after `identify_scene`, around line 122):

```python
def identify_image(sidecar_url, image_id):
    """Identify performers in a single image."""
    if not image_id:
        return {"error": "No image_id provided"}

    try:
        payload = {"image_id": str(image_id)}
        log(f"Identifying image {image_id}")

        response = requests.post(
            f"{sidecar_url}/identify/image",
            json=payload,
            timeout=30,
        )

        if response.ok:
            result = response.json()
            log(f"Image {image_id}: {result.get('face_count', 0)} faces")
            return result

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Request timed out"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}


def identify_gallery(sidecar_url, gallery_id):
    """Identify performers across an entire gallery."""
    if not gallery_id:
        return {"error": "No gallery_id provided"}

    try:
        payload = {"gallery_id": str(gallery_id)}
        log(f"Identifying gallery {gallery_id}")

        response = requests.post(
            f"{sidecar_url}/identify/gallery",
            json=payload,
            timeout=300,
        )

        if response.ok:
            result = response.json()
            log(f"Gallery {gallery_id}: {result.get('images_processed', 0)} images, "
                f"{len(result.get('performers', []))} performers")
            return result

        try:
            error_detail = response.json().get("detail", response.text)
        except Exception:
            error_detail = response.text or f"HTTP {response.status_code}"
        return {"error": f"Identification failed: {error_detail}"}

    except requests.ConnectionError:
        return {"error": "Connection refused - is Stash Sense running?"}
    except requests.Timeout:
        return {"error": "Gallery identification timed out - gallery may be too large"}
    except requests.RequestException as e:
        return {"error": f"Request failed: {e}"}
```

**Step 2: Commit**

```bash
git add plugin/stash_sense_backend.py
git commit -m "feat: add identify_image and identify_gallery proxy modes"
```

---

### Task 6: Plugin Route Detection — Image & Gallery Pages

**Files:**
- Modify: `plugin/stash-sense-core.js:264-286` (`getRoute` function)
- Modify: `plugin/stash-sense-core.js:403-438` (window.StashSense export — add `getImage`, `getGallery` helpers)

**Step 1: Add image and gallery routes to `getRoute()`**

In `plugin/stash-sense-core.js`, modify the `getRoute()` function (lines 264-286). Add after the scene match block and before the performer match block:

```javascript
    // Image page
    const imageMatch = path.match(/\/images\/(\d+)/);
    if (imageMatch) {
      return { type: 'image', id: imageMatch[1] };
    }

    // Gallery page
    const galleryMatch = path.match(/\/galleries\/(\d+)/);
    if (galleryMatch) {
      return { type: 'gallery', id: galleryMatch[1] };
    }
```

**Step 2: Add GraphQL helper functions**

Add after `getScene()` (line 257), before the URL Routing section:

```javascript
  /**
   * Get image details by ID
   */
  async function getImage(id) {
    const query = `
      query GetImage($id: ID!) {
        findImage(id: $id) {
          id
          title
          paths {
            image
            thumbnail
          }
          performers {
            id
            name
            image_path
          }
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findImage;
  }

  /**
   * Get gallery details by ID
   */
  async function getGallery(id) {
    const query = `
      query GetGallery($id: ID!) {
        findGallery(id: $id) {
          id
          title
          image_count
          performers {
            id
            name
            image_path
          }
        }
      }
    `;
    const data = await stashQuery(query, { id });
    return data?.findGallery;
  }
```

**Step 3: Export the new functions**

In the `window.StashSense` export object (line 403-438), add to the Stash GraphQL section:

```javascript
    getImage,
    getGallery,
```

**Step 4: Commit**

```bash
git add plugin/stash-sense-core.js
git commit -m "feat: add image/gallery route detection and GraphQL helpers"
```

---

### Task 7: Plugin UI — Image Page Identification

**Files:**
- Modify: `plugin/stash-sense.js:26-403` (FaceRecognition module — add image identification methods)
- Modify: `plugin/stash-sense.js:407-442` (init — add image page handling)

**Step 1: Add image identification to FaceRecognition module**

In `plugin/stash-sense.js`, add these methods inside the `FaceRecognition` object (after `addPerformerToScene`, around line 86):

```javascript
      // Call the face recognition API for a single image
      async identifyImage(imageId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.('Connecting to Stash Sense...');

        const result = await SS.runPluginOperation('identify_image', {
          image_id: imageId,
          sidecar_url: settings.sidecarUrl,
        });

        if (result.error) {
          throw new Error(result.error);
        }

        return result;
      },

      // Add performer to image
      async addPerformerToImage(imageId, performerId) {
        const getQuery = `
          query GetImage($id: ID!) {
            findImage(id: $id) {
              performers { id }
            }
          }
        `;

        const updateQuery = `
          mutation UpdateImage($id: ID!, $performer_ids: [ID!]) {
            imageUpdate(input: { id: $id, performer_ids: $performer_ids }) {
              id
            }
          }
        `;

        try {
          const getResult = await SS.stashQuery(getQuery, { id: imageId });
          const currentPerformers = getResult?.findImage?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: imageId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer to image:', e);
          return false;
        }
      },

      async handleIdentifyImage() {
        const route = SS.getRoute();
        if (route.type !== 'image') return;

        const imageId = route.id;
        const modal = this.createModal();

        try {
          this.updateLoading(modal, 'Analyzing image...', 'Detecting faces');

          const results = await this.identifyImage(imageId, (stage) => {
            this.updateLoading(modal, stage);
          });

          this.updateLoading(modal, 'Processing results...');
          // Reuse renderResults but pass imageId instead of sceneId
          await this.renderImageResults(modal, results, imageId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Image analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      async renderImageResults(modal, results, imageId) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        if (!results.faces || results.faces.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No faces detected</p>
            <p class="ss-error-hint">The image may not contain clear face shots.</p>
          `;
          errorDiv.style.display = 'block';
          return;
        }

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Detected <strong>${results.face_count}</strong> face(s) in image.
          </p>
          <div class="ss-persons"></div>
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        for (let i = 0; i < results.faces.length; i++) {
          const face = results.faces[i];
          const personDiv = document.createElement('div');
          personDiv.className = 'ss-person';

          if (!face.matches || face.matches.length === 0) {
            personDiv.innerHTML = `
              <div class="ss-person-header">
                <span class="ss-person-label">Face ${i + 1}</span>
              </div>
              <p class="ss-no-match">No match found in database</p>
            `;
          } else {
            const match = face.matches[0];
            const confidence = this.distanceToConfidence(match.distance);
            const confidenceClass = SS.getConfidenceClass(confidence);
            const localPerformer = await SS.findPerformerByStashDBId(match.stashdb_id);

            personDiv.innerHTML = `
              <div class="ss-person-header">
                <span class="ss-person-label">Face ${i + 1}</span>
              </div>
              <div class="ss-match">
                <div class="ss-match-image">
                  ${match.image_url ? `<img src="${match.image_url}" alt="${match.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
                </div>
                <div class="ss-match-info">
                  <h4>${match.name}</h4>
                  <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
                  ${match.country ? `<div class="ss-country">${match.country}</div>` : ''}
                  <div class="ss-links">
                    <a href="https://stashdb.org/performers/${match.stashdb_id}" target="_blank" rel="noopener" class="ss-link">
                      View on StashDB
                    </a>
                  </div>
                  <div class="ss-actions">
                    ${localPerformer
                      ? `<button class="ss-btn ss-btn-add" data-performer-id="${localPerformer.id}" data-image-id="${imageId}">
                           Add to Image
                         </button>
                         <span class="ss-local-status">In library as: ${localPerformer.name}</span>`
                      : `<span class="ss-local-status ss-not-in-library">Not in library</span>`
                    }
                  </div>
                </div>
              </div>
              ${face.matches.length > 1 ? `
                <details class="ss-other-matches">
                  <summary>Other possible matches (${face.matches.length - 1})</summary>
                  <ul>
                    ${face.matches.slice(1).map(m => {
                      const altConf = this.distanceToConfidence(m.distance);
                      return `<li>
                        <a href="https://stashdb.org/performers/${m.stashdb_id}" target="_blank" rel="noopener">${m.name}</a>
                        <span class="ss-alt-confidence">${altConf}%</span>
                      </li>`;
                    }).join('')}
                  </ul>
                </details>
              ` : ''}
            `;
          }

          personsDiv.appendChild(personDiv);
        }

        // Add click handlers for "Add to Image" buttons
        resultsDiv.querySelectorAll('.ss-btn-add').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const performerId = e.target.dataset.performerId;
            const targetImageId = e.target.dataset.imageId;
            btn.disabled = true;
            btn.textContent = 'Adding...';

            const success = await this.addPerformerToImage(targetImageId, performerId);
            if (success) {
              btn.textContent = 'Added!';
              btn.classList.add('ss-btn-success');
            } else {
              btn.textContent = 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
            }
          });
        });

        resultsDiv.style.display = 'block';
      },

      createImageButton() {
        const status = SS.getSidecarStatus();
        const btn = SS.createElement('button', {
          className: 'ss-identify-btn btn btn-secondary',
          attrs: {
            title: status === false ? 'Stash Sense: Not connected' : 'Identify performers using face recognition',
          },
          innerHTML: `
            <span class="ss-btn-icon ${status === true ? 'ss-connected' : status === false ? 'ss-disconnected' : ''}">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
              </svg>
            </span>
            <span class="ss-btn-text">Identify Performers</span>
          `,
        });
        btn.addEventListener('click', () => this.handleIdentifyImage());
        return btn;
      },

      injectImageButton() {
        const route = SS.getRoute();
        if (route.type !== 'image') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.image-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.image-header .btn-group',
          '.detail-header-buttons',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createImageButton());
            console.log(`[${SS.PLUGIN_NAME}] Image button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createImageButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },
```

**Step 2: Update init to handle image pages**

In the `init()` function (lines 407-442), modify:

1. After `setTimeout(() => FaceRecognition.injectSceneButton(), 500);` add:
```javascript
      setTimeout(() => FaceRecognition.injectImageButton(), 500);
```

2. In the `onNavigate` callback, add:
```javascript
        if (route.type === 'image') {
          setTimeout(() => FaceRecognition.injectImageButton(), 300);
        }
```

**Step 3: Commit**

```bash
git add plugin/stash-sense.js
git commit -m "feat: add Identify Performers button and modal for image pages"
```

---

### Task 8: Plugin UI — Gallery Page Identification

**Files:**
- Modify: `plugin/stash-sense.js` (add gallery methods to FaceRecognition, update init)
- Modify: `plugin/stash-sense.css` (add gallery-specific styles)

**Step 1: Add gallery identification methods to FaceRecognition**

Add these methods inside the `FaceRecognition` object (after the image methods):

```javascript
      // Call the gallery identification API
      async identifyGallery(galleryId, onProgress) {
        const settings = await SS.getSettings();
        onProgress?.('Connecting to Stash Sense...');

        const result = await SS.runPluginOperation('identify_gallery', {
          gallery_id: galleryId,
          sidecar_url: settings.sidecarUrl,
        });

        if (result.error) {
          throw new Error(result.error);
        }

        return result;
      },

      // Add performer to gallery
      async addPerformerToGallery(galleryId, performerId) {
        const getQuery = `
          query GetGallery($id: ID!) {
            findGallery(id: $id) {
              performers { id }
            }
          }
        `;

        const updateQuery = `
          mutation UpdateGallery($id: ID!, $performer_ids: [ID!]) {
            galleryUpdate(input: { id: $id, performer_ids: $performer_ids }) {
              id
            }
          }
        `;

        try {
          const getResult = await SS.stashQuery(getQuery, { id: galleryId });
          const currentPerformers = getResult?.findGallery?.performers || [];
          const currentIds = currentPerformers.map(p => p.id);

          if (!currentIds.includes(performerId)) {
            currentIds.push(performerId);
          }

          await SS.stashQuery(updateQuery, { id: galleryId, performer_ids: currentIds });
          return true;
        } catch (e) {
          console.error('Failed to add performer to gallery:', e);
          return false;
        }
      },

      async handleIdentifyGallery() {
        const route = SS.getRoute();
        if (route.type !== 'gallery') return;

        const galleryId = route.id;
        const modal = this.createModal();

        try {
          this.updateLoading(modal, 'Identifying performers in gallery...', 'This may take a while for large galleries');

          const results = await this.identifyGallery(galleryId, (stage) => {
            this.updateLoading(modal, stage);
          });

          this.updateLoading(modal, 'Processing results...');
          await this.renderGalleryResults(modal, results, galleryId);
        } catch (error) {
          console.error(`[${SS.PLUGIN_NAME}] Gallery analysis failed:`, error);
          this.showError(modal, error.message);
        }
      },

      async renderGalleryResults(modal, results, galleryId) {
        const loading = modal.querySelector('.ss-loading');
        const resultsDiv = modal.querySelector('.ss-results');
        const errorDiv = modal.querySelector('.ss-error');

        loading.style.display = 'none';

        if (!results.performers || results.performers.length === 0) {
          errorDiv.innerHTML = `
            <div class="ss-error-icon">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="48" height="48" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-13h2v6h-2zm0 8h2v2h-2z"/>
              </svg>
            </div>
            <p class="ss-error-title">No performers identified</p>
            <p class="ss-error-hint">
              Processed ${results.images_processed || 0}/${results.total_images || 0} images
              but no confident matches were found.
            </p>
          `;
          errorDiv.style.display = 'block';
          return;
        }

        resultsDiv.innerHTML = `
          <p class="ss-summary">
            Processed <strong>${results.images_processed}</strong>/${results.total_images} images,
            detected <strong>${results.faces_detected}</strong> faces,
            identified <strong>${results.performers.length}</strong> performer(s).
          </p>
          <div class="ss-gallery-actions-bar">
            <button class="ss-btn ss-btn-primary ss-accept-all-btn">Accept All</button>
          </div>
          <div class="ss-persons"></div>
        `;

        const personsDiv = resultsDiv.querySelector('.ss-persons');

        for (const performer of results.performers) {
          const personDiv = document.createElement('div');
          personDiv.className = 'ss-person';

          const confidence = this.distanceToConfidence(performer.best_distance);
          const confidenceClass = SS.getConfidenceClass(confidence);

          const localPerformer = await SS.findPerformerByStashDBId(performer.performer_id);

          personDiv.innerHTML = `
            <div class="ss-person-header">
              <span class="ss-person-label">${performer.name}</span>
              <span class="ss-person-frames">Found in ${performer.image_count}/${results.total_images} images</span>
            </div>
            <div class="ss-match">
              <div class="ss-match-image">
                ${performer.image_url ? `<img src="${performer.image_url}" alt="${performer.name}" loading="lazy" />` : '<div class="ss-no-image">No image</div>'}
              </div>
              <div class="ss-match-info">
                <div class="ss-confidence ${confidenceClass}">${confidence}% match</div>
                ${performer.country ? `<div class="ss-country">${performer.country}</div>` : ''}
                <div class="ss-links">
                  <a href="https://stashdb.org/performers/${performer.performer_id}" target="_blank" rel="noopener" class="ss-link">
                    View on StashDB
                  </a>
                </div>
                ${localPerformer ? `
                  <div class="ss-gallery-performer-actions" data-performer-id="${localPerformer.id}" data-stashdb-id="${performer.performer_id}">
                    <div class="ss-gallery-tag-toggle">
                      <label class="ss-toggle-label">
                        <input type="checkbox" class="ss-tag-images-toggle" />
                        <span>Also tag individual images</span>
                      </label>
                    </div>
                    <div class="ss-actions">
                      <button class="ss-btn ss-btn-add ss-gallery-accept-btn"
                              data-performer-id="${localPerformer.id}"
                              data-gallery-id="${galleryId}"
                              data-image-ids='${JSON.stringify(performer.image_ids)}'>
                        Add to Gallery
                      </button>
                      <span class="ss-local-status">In library as: ${localPerformer.name}</span>
                    </div>
                  </div>
                ` : `
                  <div class="ss-actions">
                    <span class="ss-local-status ss-not-in-library">Not in library</span>
                  </div>
                `}
              </div>
            </div>
          `;

          personsDiv.appendChild(personDiv);
        }

        // Click handlers for individual accept buttons
        resultsDiv.querySelectorAll('.ss-gallery-accept-btn').forEach(btn => {
          btn.addEventListener('click', async (e) => {
            const performerId = btn.dataset.performerId;
            const targetGalleryId = btn.dataset.galleryId;
            const imageIds = JSON.parse(btn.dataset.imageIds);
            const tagImages = btn.closest('.ss-gallery-performer-actions')
              ?.querySelector('.ss-tag-images-toggle')?.checked || false;

            btn.disabled = true;
            btn.textContent = 'Adding...';

            let success = await this.addPerformerToGallery(targetGalleryId, performerId);

            if (success && tagImages) {
              btn.textContent = `Tagging images...`;
              for (const imgId of imageIds) {
                await this.addPerformerToImage(imgId, performerId);
              }
            }

            if (success) {
              btn.textContent = tagImages ? `Added to gallery + ${imageIds.length} images` : 'Added to gallery!';
              btn.classList.add('ss-btn-success');
            } else {
              btn.textContent = 'Failed';
              btn.classList.add('ss-btn-error');
              btn.disabled = false;
            }
          });
        });

        // Accept All handler
        resultsDiv.querySelector('.ss-accept-all-btn')?.addEventListener('click', async (e) => {
          const acceptAllBtn = e.target;
          acceptAllBtn.disabled = true;
          acceptAllBtn.textContent = 'Accepting...';

          const buttons = resultsDiv.querySelectorAll('.ss-gallery-accept-btn:not(:disabled)');
          for (const btn of buttons) {
            btn.click();
            // Small delay between operations
            await new Promise(r => setTimeout(r, 200));
          }

          acceptAllBtn.textContent = 'All accepted!';
          acceptAllBtn.classList.add('ss-btn-success');
        });

        resultsDiv.style.display = 'block';
      },

      createGalleryButton() {
        const status = SS.getSidecarStatus();
        const btn = SS.createElement('button', {
          className: 'ss-identify-btn btn btn-secondary',
          attrs: {
            title: status === false ? 'Stash Sense: Not connected' : 'Identify all performers in this gallery',
          },
          innerHTML: `
            <span class="ss-btn-icon ${status === true ? 'ss-connected' : status === false ? 'ss-disconnected' : ''}">
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="16" height="16" fill="currentColor">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 3c1.66 0 3 1.34 3 3s-1.34 3-3 3-3-1.34-3-3 1.34-3 3-3zm0 14.2c-2.5 0-4.71-1.28-6-3.22.03-1.99 4-3.08 6-3.08 1.99 0 5.97 1.09 6 3.08-1.29 1.94-3.5 3.22-6 3.22z"/>
              </svg>
            </span>
            <span class="ss-btn-text">Identify Performers</span>
          `,
        });
        btn.addEventListener('click', () => this.handleIdentifyGallery());
        return btn;
      },

      injectGalleryButton() {
        const route = SS.getRoute();
        if (route.type !== 'gallery') return;
        if (document.querySelector('.ss-identify-btn')) return;

        const buttonContainers = [
          '.gallery-toolbar .btn-group',
          '.detail-header .ml-auto .btn-group',
          '.gallery-header .btn-group',
          '.detail-header-buttons',
          '.ml-auto.btn-group',
        ];

        for (const selector of buttonContainers) {
          const container = document.querySelector(selector);
          if (container) {
            container.appendChild(this.createGalleryButton());
            console.log(`[${SS.PLUGIN_NAME}] Gallery button injected into ${selector}`);
            return;
          }
        }

        // Fallback: floating button
        const floatingBtn = this.createGalleryButton();
        floatingBtn.classList.add('ss-floating-btn');
        document.body.appendChild(floatingBtn);
      },
```

**Step 2: Update init for gallery pages**

In the `init()` function, add alongside the image button injection:

```javascript
      setTimeout(() => FaceRecognition.injectGalleryButton(), 500);
```

And in the `onNavigate` callback:

```javascript
        if (route.type === 'gallery') {
          setTimeout(() => FaceRecognition.injectGalleryButton(), 300);
        }
```

**Step 3: Add gallery-specific CSS**

In `plugin/stash-sense.css`, add at the end:

```css
/* ==================== Gallery Identification ==================== */

.ss-gallery-actions-bar {
  display: flex;
  justify-content: flex-end;
  margin-bottom: 16px;
}

.ss-gallery-tag-toggle {
  margin-bottom: 8px;
}

.ss-toggle-label {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  color: var(--bs-secondary-color, #888);
  cursor: pointer;
}

.ss-toggle-label input[type="checkbox"] {
  accent-color: var(--bs-primary, #0d6efd);
}
```

**Step 4: Commit**

```bash
git add plugin/stash-sense.js plugin/stash-sense.css
git commit -m "feat: add Identify Performers button and modal for gallery pages"
```

---

### Task 9: Deploy & Smoke Test

**Step 1: Deploy plugin to Stash**

```bash
scp plugin/* root@10.0.0.4:/mnt/nvme_cache/appdata/stash/config/plugins/stash-sense/
```

**Step 2: Start sidecar**

```bash
cd /home/carrot/code/stash-sense/api && source ../.venv/bin/activate && make sidecar
```

**Step 3: Manual smoke test**

1. Hard refresh Stash UI (Ctrl+Shift+R)
2. Navigate to an image page (e.g. `http://10.0.0.4:6969/images/27839`)
   - Verify "Identify Performers" button appears
   - Click it, verify modal shows with face results
   - Accept a match, verify performer is tagged on the image
3. Navigate to a gallery page (e.g. `http://10.0.0.4:6969/galleries/797`)
   - Verify "Identify Performers" button appears
   - Click it, verify progress spinner then results modal
   - Verify per-performer toggle ("Also tag individual images")
   - Accept with toggle off → verify gallery tagged, images not
   - Accept with toggle on → verify gallery + images tagged
   - Test "Accept All"

**Step 4: Commit any fixes from smoke testing**

```bash
git add -A
git commit -m "fix: address issues found during gallery/image identification smoke test"
```

# Upstream Performer Sync Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a new recommendation type that detects upstream stash-box changes to linked performers and presents field-level merge controls.

**Architecture:** New `UpstreamPerformerAnalyzer` follows the existing analyzer pattern (extends `BaseAnalyzer`). A separate `StashBoxClient` handles stash-box API calls. A field mapping/diff engine compares upstream vs local data using 3-way snapshots. The plugin JS renders inline diff cards with per-field merge controls.

**Tech Stack:** Python (FastAPI, httpx, SQLite), JavaScript (Stash plugin UI), GraphQL (both Stash and stash-box APIs)

**Design doc:** `docs/plans/2026-02-06-upstream-performer-sync-design.md`

---

### Task 1: Database Schema Migration (V4)

Add new tables for upstream sync tracking and modify dismissed_targets.

**Files:**
- Modify: `api/recommendations_db.py`

**Step 1: Write the failing test**

Create test file:

```python
# api/tests/test_upstream_sync_db.py
"""Tests for upstream sync database tables and methods."""

import pytest
from recommendations_db import RecommendationsDB


class TestUpstreamSyncSchema:
    """Tests for V4 schema additions."""

    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_upstream_snapshots_table_exists(self, db):
        with db._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='upstream_snapshots'"
            )
            assert cursor.fetchone() is not None

    def test_upstream_field_config_table_exists(self, db):
        with db._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='upstream_field_config'"
            )
            assert cursor.fetchone() is not None

    def test_dismissed_targets_has_permanent_column(self, db):
        with db._connection() as conn:
            # Insert a row with permanent flag
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id, permanent) VALUES (?, ?, ?, ?)",
                ("test", "performer", "1", 1),
            )
            row = conn.execute(
                "SELECT permanent FROM dismissed_targets WHERE target_id = '1'"
            ).fetchone()
            assert row["permanent"] == 1

    def test_dismissed_targets_permanent_defaults_to_zero(self, db):
        with db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id) VALUES (?, ?, ?)",
                ("test", "performer", "2"),
            )
            row = conn.execute(
                "SELECT permanent FROM dismissed_targets WHERE target_id = '2'"
            ).fetchone()
            assert row["permanent"] == 0

    def test_schema_version_is_4(self, db):
        with db._connection() as conn:
            version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
            assert version == 4
```

**Step 2: Run test to verify it fails**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_sync_db.py -v`
Expected: FAIL - tables don't exist, schema is V3

**Step 3: Implement the schema migration**

In `api/recommendations_db.py`:

1. Change `SCHEMA_VERSION = 3` to `SCHEMA_VERSION = 4`
2. Add the new tables to `_create_schema()` (after the `scene_fingerprint_faces` table):

```python
            -- Upstream sync snapshots
            CREATE TABLE upstream_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_type TEXT NOT NULL,
                local_entity_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                stash_box_id TEXT NOT NULL,
                upstream_data JSON NOT NULL,
                upstream_updated_at TEXT NOT NULL,
                fetched_at TEXT DEFAULT (datetime('now')),
                UNIQUE(entity_type, endpoint, stash_box_id)
            );
            CREATE INDEX idx_upstream_entity ON upstream_snapshots(entity_type, endpoint);
            CREATE INDEX idx_upstream_stash_box_id ON upstream_snapshots(stash_box_id);

            -- Per-field monitoring configuration
            CREATE TABLE upstream_field_config (
                endpoint TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                field_name TEXT NOT NULL,
                enabled INTEGER DEFAULT 1,
                PRIMARY KEY (endpoint, entity_type, field_name)
            );
```

3. Add `permanent INTEGER DEFAULT 0` column to `dismissed_targets` in `_create_schema()`
4. Add migration case in `_migrate_schema()` for `from_version == 3` (or `< 4`):

```python
        if from_version < 4:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS upstream_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    entity_type TEXT NOT NULL,
                    local_entity_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    stash_box_id TEXT NOT NULL,
                    upstream_data JSON NOT NULL,
                    upstream_updated_at TEXT NOT NULL,
                    fetched_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(entity_type, endpoint, stash_box_id)
                );
                CREATE INDEX IF NOT EXISTS idx_upstream_entity ON upstream_snapshots(entity_type, endpoint);
                CREATE INDEX IF NOT EXISTS idx_upstream_stash_box_id ON upstream_snapshots(stash_box_id);

                CREATE TABLE IF NOT EXISTS upstream_field_config (
                    endpoint TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    enabled INTEGER DEFAULT 1,
                    PRIMARY KEY (endpoint, entity_type, field_name)
                );

                ALTER TABLE dismissed_targets ADD COLUMN permanent INTEGER DEFAULT 0;

                UPDATE schema_version SET version = 4;
            """)
```

**Step 4: Run test to verify it passes**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_sync_db.py -v`
Expected: PASS

**Step 5: Run full test suite to check no regressions**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v --tb=short`
Expected: All existing tests still PASS

**Step 6: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/recommendations_db.py api/tests/test_upstream_sync_db.py
git commit -m "feat: add V4 schema for upstream sync tables"
```

---

### Task 2: Upstream Snapshot & Field Config DB Methods

Add CRUD methods for the new tables.

**Files:**
- Modify: `api/recommendations_db.py`
- Modify: `api/tests/test_upstream_sync_db.py`

**Step 1: Write failing tests**

Append to `api/tests/test_upstream_sync_db.py`:

```python
class TestUpstreamSnapshots:
    """Tests for upstream snapshot CRUD."""

    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_upsert_and_get_snapshot(self, db):
        db.upsert_upstream_snapshot(
            entity_type="performer",
            local_entity_id="42",
            endpoint="https://stashdb.org/graphql",
            stash_box_id="abc-123",
            upstream_data={"name": "Jane Doe", "height": 165},
            upstream_updated_at="2026-01-15T10:00:00Z",
        )
        snapshot = db.get_upstream_snapshot(
            entity_type="performer",
            endpoint="https://stashdb.org/graphql",
            stash_box_id="abc-123",
        )
        assert snapshot is not None
        assert snapshot["local_entity_id"] == "42"
        assert snapshot["upstream_data"]["name"] == "Jane Doe"
        assert snapshot["upstream_updated_at"] == "2026-01-15T10:00:00Z"

    def test_upsert_snapshot_updates_existing(self, db):
        db.upsert_upstream_snapshot(
            entity_type="performer",
            local_entity_id="42",
            endpoint="https://stashdb.org/graphql",
            stash_box_id="abc-123",
            upstream_data={"name": "Jane Doe"},
            upstream_updated_at="2026-01-15T10:00:00Z",
        )
        db.upsert_upstream_snapshot(
            entity_type="performer",
            local_entity_id="42",
            endpoint="https://stashdb.org/graphql",
            stash_box_id="abc-123",
            upstream_data={"name": "Jane Smith"},
            upstream_updated_at="2026-01-16T10:00:00Z",
        )
        snapshot = db.get_upstream_snapshot(
            entity_type="performer",
            endpoint="https://stashdb.org/graphql",
            stash_box_id="abc-123",
        )
        assert snapshot["upstream_data"]["name"] == "Jane Smith"
        assert snapshot["upstream_updated_at"] == "2026-01-16T10:00:00Z"

    def test_get_snapshot_returns_none_when_missing(self, db):
        snapshot = db.get_upstream_snapshot(
            entity_type="performer",
            endpoint="https://stashdb.org/graphql",
            stash_box_id="nonexistent",
        )
        assert snapshot is None


class TestUpstreamFieldConfig:
    """Tests for field monitoring configuration."""

    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_get_enabled_fields_returns_defaults_when_no_config(self, db):
        fields = db.get_enabled_fields(
            endpoint="https://stashdb.org/graphql",
            entity_type="performer",
        )
        # No config rows exist yet, so should return None (caller uses defaults)
        assert fields is None

    def test_set_and_get_field_config(self, db):
        db.set_field_config(
            endpoint="https://stashdb.org/graphql",
            entity_type="performer",
            field_configs={"name": True, "height": True, "tattoos": False},
        )
        fields = db.get_enabled_fields(
            endpoint="https://stashdb.org/graphql",
            entity_type="performer",
        )
        assert fields is not None
        assert "name" in fields
        assert "height" in fields
        assert "tattoos" not in fields

    def test_set_field_config_replaces_existing(self, db):
        db.set_field_config(
            endpoint="https://stashdb.org/graphql",
            entity_type="performer",
            field_configs={"name": True, "height": True},
        )
        db.set_field_config(
            endpoint="https://stashdb.org/graphql",
            entity_type="performer",
            field_configs={"name": True, "height": False, "gender": True},
        )
        fields = db.get_enabled_fields(
            endpoint="https://stashdb.org/graphql",
            entity_type="performer",
        )
        assert "name" in fields
        assert "gender" in fields
        assert "height" not in fields


class TestDismissedTargetsPermanent:
    """Tests for permanent dismissal flag."""

    @pytest.fixture
    def db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    def test_dismiss_with_permanent_flag(self, db):
        # Create a recommendation first
        rec_id = db.create_recommendation(
            type="upstream_performer_changes",
            target_type="performer",
            target_id="42",
            details={"changes": []},
            confidence=1.0,
        )
        db.dismiss_recommendation(rec_id, reason="not interested", permanent=True)

        assert db.is_dismissed("upstream_performer_changes", "performer", "42")
        assert db.is_permanently_dismissed("upstream_performer_changes", "performer", "42")

    def test_dismiss_without_permanent_is_soft(self, db):
        rec_id = db.create_recommendation(
            type="upstream_performer_changes",
            target_type="performer",
            target_id="43",
            details={"changes": []},
            confidence=1.0,
        )
        db.dismiss_recommendation(rec_id, reason="not now")

        assert db.is_dismissed("upstream_performer_changes", "performer", "43")
        assert not db.is_permanently_dismissed("upstream_performer_changes", "performer", "43")

    def test_undismiss_soft_dismissed(self, db):
        rec_id = db.create_recommendation(
            type="upstream_performer_changes",
            target_type="performer",
            target_id="44",
            details={"changes": []},
            confidence=1.0,
        )
        db.dismiss_recommendation(rec_id)
        db.undismiss("upstream_performer_changes", "performer", "44")

        assert not db.is_dismissed("upstream_performer_changes", "performer", "44")
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_sync_db.py -v`
Expected: FAIL - methods don't exist yet

**Step 3: Implement the DB methods**

Add to `api/recommendations_db.py` (after the Watermarks section, before Statistics):

```python
    # ==================== Upstream Snapshots ====================

    def upsert_upstream_snapshot(
        self,
        entity_type: str,
        local_entity_id: str,
        endpoint: str,
        stash_box_id: str,
        upstream_data: dict,
        upstream_updated_at: str,
    ) -> int:
        """Create or update an upstream snapshot. Returns the snapshot ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO upstream_snapshots
                    (entity_type, local_entity_id, endpoint, stash_box_id, upstream_data, upstream_updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, endpoint, stash_box_id) DO UPDATE SET
                    local_entity_id = excluded.local_entity_id,
                    upstream_data = excluded.upstream_data,
                    upstream_updated_at = excluded.upstream_updated_at,
                    fetched_at = datetime('now')
                RETURNING id
                """,
                (entity_type, local_entity_id, endpoint, stash_box_id,
                 json.dumps(upstream_data), upstream_updated_at),
            )
            return cursor.fetchone()[0]

    def get_upstream_snapshot(
        self,
        entity_type: str,
        endpoint: str,
        stash_box_id: str,
    ) -> Optional[dict]:
        """Get an upstream snapshot. Returns dict with upstream_data parsed from JSON."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM upstream_snapshots
                WHERE entity_type = ? AND endpoint = ? AND stash_box_id = ?
                """,
                (entity_type, endpoint, stash_box_id),
            ).fetchone()
            if row:
                result = dict(row)
                result["upstream_data"] = json.loads(result["upstream_data"])
                return result
        return None

    # ==================== Field Config ====================

    def get_enabled_fields(
        self,
        endpoint: str,
        entity_type: str,
    ) -> Optional[set[str]]:
        """Get set of enabled field names. Returns None if no config exists (use defaults)."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT field_name, enabled FROM upstream_field_config
                WHERE endpoint = ? AND entity_type = ?
                """,
                (endpoint, entity_type),
            ).fetchall()
            if not rows:
                return None
            return {row["field_name"] for row in rows if row["enabled"]}

    def set_field_config(
        self,
        endpoint: str,
        entity_type: str,
        field_configs: dict[str, bool],
    ):
        """Set field monitoring config. field_configs maps field_name -> enabled."""
        with self._connection() as conn:
            # Delete existing config for this endpoint/entity
            conn.execute(
                "DELETE FROM upstream_field_config WHERE endpoint = ? AND entity_type = ?",
                (endpoint, entity_type),
            )
            # Insert new config
            for field_name, enabled in field_configs.items():
                conn.execute(
                    """
                    INSERT INTO upstream_field_config (endpoint, entity_type, field_name, enabled)
                    VALUES (?, ?, ?, ?)
                    """,
                    (endpoint, entity_type, field_name, int(enabled)),
                )
```

Also modify `dismiss_recommendation` to accept a `permanent` parameter:

```python
    def dismiss_recommendation(self, rec_id: int, reason: Optional[str] = None, permanent: bool = False) -> bool:
```

And update the INSERT in that method:

```python
                conn.execute(
                    """
                    INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row['type'], row['target_type'], row['target_id'], reason, int(permanent))
                )
```

Add `is_permanently_dismissed` and `undismiss` methods:

```python
    def is_permanently_dismissed(self, type: str, target_type: str, target_id: str) -> bool:
        """Check if a target has been permanently dismissed."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT permanent FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ?
                """,
                (type, target_type, target_id),
            ).fetchone()
            return row is not None and bool(row["permanent"])

    def undismiss(self, type: str, target_type: str, target_id: str):
        """Remove a dismissal (for soft-dismissed targets that get new upstream changes)."""
        with self._connection() as conn:
            conn.execute(
                """
                DELETE FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ? AND permanent = 0
                """,
                (type, target_type, target_id),
            )
```

Also add `update_recommendation_details` for updating existing pending recs:

```python
    def update_recommendation_details(self, rec_id: int, details: dict) -> bool:
        """Update the details JSON of an existing recommendation."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET details = ?, updated_at = datetime('now')
                WHERE id = ? AND status = 'pending'
                """,
                (json.dumps(details), rec_id),
            )
            return cursor.rowcount > 0
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_sync_db.py -v`
Expected: PASS

**Step 5: Run full test suite**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 6: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/recommendations_db.py api/tests/test_upstream_sync_db.py
git commit -m "feat: add upstream snapshot and field config DB methods"
```

---

### Task 3: Stash-Box GraphQL Client

A new client for querying stash-box endpoints (separate from `StashClientUnified` which only talks to local Stash).

**Files:**
- Create: `api/stashbox_client.py`
- Create: `api/tests/test_stashbox_client.py`

**Step 1: Write the failing test**

```python
# api/tests/test_stashbox_client.py
"""Tests for StashBoxClient."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock


class TestStashBoxClient:
    """Tests for the stash-box GraphQL client."""

    @pytest.fixture
    def client(self):
        from stashbox_client import StashBoxClient
        return StashBoxClient(
            endpoint="https://stashdb.org/graphql",
            api_key="test-key",
        )

    def test_init_sets_endpoint(self, client):
        assert client.endpoint == "https://stashdb.org/graphql"

    def test_init_sets_headers(self, client):
        assert client.headers["ApiKey"] == "test-key"
        assert client.headers["Content-Type"] == "application/json"

    @pytest.mark.asyncio
    async def test_query_performers_returns_performers(self, client):
        mock_response = {
            "queryPerformers": {
                "count": 2,
                "performers": [
                    {
                        "id": "abc-123",
                        "name": "Jane Doe",
                        "aliases": ["JD"],
                        "gender": "FEMALE",
                        "birth_date": "1990-01-15",
                        "death_date": None,
                        "ethnicity": "CAUCASIAN",
                        "country": "US",
                        "eye_color": "BROWN",
                        "hair_color": "BROWN",
                        "height": 165,
                        "cup_size": "B",
                        "band_size": 32,
                        "waist_size": 24,
                        "hip_size": 34,
                        "breast_type": "NATURAL",
                        "career_start_year": 2015,
                        "career_end_year": None,
                        "tattoos": [{"location": "Left arm", "description": "Dragon"}],
                        "piercings": [],
                        "urls": [{"url": "https://example.com", "type": "HOME"}],
                        "is_favorite": False,
                        "deleted": False,
                        "merged_into_id": None,
                        "disambiguation": "",
                        "created": "2024-01-01T00:00:00Z",
                        "updated": "2026-01-15T10:00:00Z",
                    },
                ],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            performers, count = await client.query_performers(page=1, per_page=25)

            assert count == 2
            assert len(performers) == 1
            assert performers[0]["name"] == "Jane Doe"
            assert performers[0]["updated"] == "2026-01-15T10:00:00Z"

    @pytest.mark.asyncio
    async def test_query_performers_sorts_by_updated(self, client):
        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"queryPerformers": {"count": 0, "performers": []}}
            await client.query_performers(page=1, per_page=25)

            call_args = mock_exec.call_args
            query = call_args[0][0]
            variables = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("variables")
            assert variables["input"]["sort"] == "UPDATED_AT"
            assert variables["input"]["direction"] == "DESC"

    @pytest.mark.asyncio
    async def test_query_performer_by_id(self, client):
        mock_response = {
            "findPerformer": {
                "id": "abc-123",
                "name": "Jane Doe",
                "aliases": [],
                "updated": "2026-01-15T10:00:00Z",
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            performer = await client.get_performer("abc-123")

            assert performer["name"] == "Jane Doe"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_stashbox_client.py -v`
Expected: FAIL - module doesn't exist

**Step 3: Implement StashBoxClient**

```python
# api/stashbox_client.py
"""
Stash-Box GraphQL Client

Queries stash-box endpoints (StashDB, FansDB, etc.) for performer data.
Separate from StashClientUnified which only talks to the local Stash instance.
"""

import httpx
from typing import Optional

from rate_limiter import RateLimiter, Priority


class StashBoxClient:
    """Client for querying a stash-box GraphQL API endpoint."""

    def __init__(self, endpoint: str, api_key: str = ""):
        self.endpoint = endpoint
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        if api_key:
            self.headers["ApiKey"] = api_key

    async def _execute(
        self,
        query: str,
        variables: Optional[dict] = None,
        priority: Priority = Priority.LOW,
    ) -> dict:
        """Execute a GraphQL query against this stash-box endpoint."""
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        limiter = await RateLimiter.get_instance()
        async with limiter.acquire(priority):
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.endpoint, json=payload, headers=self.headers
                )
                response.raise_for_status()
                result = response.json()
                if "errors" in result:
                    raise RuntimeError(f"StashBox GraphQL error: {result['errors']}")
                return result["data"]

    async def query_performers(
        self,
        page: int = 1,
        per_page: int = 25,
    ) -> tuple[list[dict], int]:
        """
        Query performers sorted by UPDATED_AT DESC.
        Returns (performers, total_count).
        """
        query = """
        query QueryPerformers($input: PerformerQueryInput!) {
          queryPerformers(input: $input) {
            count
            performers {
              id
              name
              disambiguation
              aliases
              gender
              birth_date
              death_date
              ethnicity
              country
              eye_color
              hair_color
              height
              cup_size
              band_size
              waist_size
              hip_size
              breast_type
              career_start_year
              career_end_year
              tattoos { location description }
              piercings { location description }
              urls { url type }
              is_favorite
              deleted
              merged_into_id
              created
              updated
            }
          }
        }
        """
        variables = {
            "input": {
                "page": page,
                "per_page": per_page,
                "sort": "UPDATED_AT",
                "direction": "DESC",
            }
        }
        data = await self._execute(query, variables)
        result = data["queryPerformers"]
        return result["performers"], result["count"]

    async def get_performer(self, performer_id: str) -> Optional[dict]:
        """Get a single performer by stash-box ID."""
        query = """
        query FindPerformer($id: ID!) {
          findPerformer(id: $id) {
            id
            name
            disambiguation
            aliases
            gender
            birth_date
            death_date
            ethnicity
            country
            eye_color
            hair_color
            height
            cup_size
            band_size
            waist_size
            hip_size
            breast_type
            career_start_year
            career_end_year
            tattoos { location description }
            piercings { location description }
            urls { url type }
            is_favorite
            deleted
            merged_into_id
            created
            updated
          }
        }
        """
        data = await self._execute(query, {"id": performer_id})
        return data.get("findPerformer")
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_stashbox_client.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/stashbox_client.py api/tests/test_stashbox_client.py
git commit -m "feat: add StashBoxClient for querying stash-box endpoints"
```

---

### Task 4: Field Mapping & Diff Engine

Maps fields between stash-box schema and local Stash schema, normalizes values, and produces a structured diff.

**Files:**
- Create: `api/upstream_field_mapper.py`
- Create: `api/tests/test_upstream_field_mapper.py`

**Step 1: Write the failing tests**

```python
# api/tests/test_upstream_field_mapper.py
"""Tests for upstream field mapping and diff engine."""

import pytest
from upstream_field_mapper import (
    normalize_upstream_performer,
    diff_performer_fields,
    DEFAULT_PERFORMER_FIELDS,
    FIELD_MERGE_TYPES,
)


class TestNormalizeUpstreamPerformer:
    """Tests for converting stash-box performer data to comparable format."""

    def test_maps_basic_fields(self):
        upstream = {
            "name": "Jane Doe",
            "disambiguation": "actress",
            "aliases": ["JD", "Janey"],
            "gender": "FEMALE",
            "height": 165,
            "country": "US",
        }
        result = normalize_upstream_performer(upstream)
        assert result["name"] == "Jane Doe"
        assert result["disambiguation"] == "actress"
        assert result["aliases"] == ["JD", "Janey"]
        assert result["gender"] == "FEMALE"
        assert result["height"] == 165
        assert result["country"] == "US"

    def test_maps_birth_date_field_name(self):
        upstream = {"birth_date": "1990-01-15", "death_date": "2025-12-01"}
        result = normalize_upstream_performer(upstream)
        assert result["birthdate"] == "1990-01-15"
        assert result["death_date"] == "2025-12-01"

    def test_formats_tattoos_from_body_modifications(self):
        upstream = {
            "tattoos": [
                {"location": "Left arm", "description": "Dragon"},
                {"location": "Back", "description": "Wings"},
            ]
        }
        result = normalize_upstream_performer(upstream)
        assert result["tattoos"] == "Left arm: Dragon; Back: Wings"

    def test_formats_piercings_from_body_modifications(self):
        upstream = {
            "piercings": [
                {"location": "Navel", "description": ""},
                {"location": "Left ear", "description": "Stud"},
            ]
        }
        result = normalize_upstream_performer(upstream)
        assert result["piercings"] == "Navel; Left ear: Stud"

    def test_extracts_urls_from_objects(self):
        upstream = {
            "urls": [
                {"url": "https://twitter.com/jane", "type": "TWITTER"},
                {"url": "https://example.com", "type": "HOME"},
            ]
        }
        result = normalize_upstream_performer(upstream)
        assert result["urls"] == ["https://twitter.com/jane", "https://example.com"]

    def test_maps_is_favorite_to_favorite(self):
        upstream = {"is_favorite": True}
        result = normalize_upstream_performer(upstream)
        assert result["favorite"] is True

    def test_handles_none_values(self):
        upstream = {"name": "Jane", "height": None, "aliases": None}
        result = normalize_upstream_performer(upstream)
        assert result["name"] == "Jane"
        assert result["height"] is None
        assert result["aliases"] == []

    def test_maps_measurements(self):
        upstream = {
            "cup_size": "B",
            "band_size": 32,
            "waist_size": 24,
            "hip_size": 34,
            "breast_type": "NATURAL",
        }
        result = normalize_upstream_performer(upstream)
        assert result["cup_size"] == "B"
        assert result["band_size"] == 32
        assert result["waist_size"] == 24
        assert result["hip_size"] == 34
        assert result["breast_type"] == "NATURAL"


class TestDiffPerformerFields:
    """Tests for 3-way diff logic."""

    def test_detects_simple_change(self):
        local = {"name": "Jane Doe", "height": 165}
        upstream = {"name": "Jane Doe", "height": 168}
        snapshot = {"name": "Jane Doe", "height": 165}

        changes = diff_performer_fields(local, upstream, snapshot, enabled_fields={"name", "height"})

        assert len(changes) == 1
        assert changes[0]["field"] == "height"
        assert changes[0]["local_value"] == 165
        assert changes[0]["upstream_value"] == 168

    def test_skips_unchanged_upstream(self):
        # User set height locally to 170, upstream is still 165 (same as snapshot)
        local = {"height": 170}
        upstream = {"height": 165}
        snapshot = {"height": 165}

        changes = diff_performer_fields(local, upstream, snapshot, enabled_fields={"height"})
        assert len(changes) == 0  # upstream didn't change, user's local choice stands

    def test_first_run_no_snapshot_flags_all_differences(self):
        local = {"name": "Jane Doe", "height": 165}
        upstream = {"name": "Jane Smith", "height": 168}

        changes = diff_performer_fields(local, upstream, snapshot=None, enabled_fields={"name", "height"})
        assert len(changes) == 2

    def test_skips_fields_already_in_sync(self):
        local = {"name": "Jane Doe"}
        upstream = {"name": "Jane Doe"}
        snapshot = {"name": "Jane Doe"}

        changes = diff_performer_fields(local, upstream, snapshot, enabled_fields={"name"})
        assert len(changes) == 0

    def test_respects_enabled_fields_filter(self):
        local = {"name": "Jane Doe", "height": 165}
        upstream = {"name": "Jane Smith", "height": 168}
        snapshot = {"name": "Jane Doe", "height": 165}

        changes = diff_performer_fields(local, upstream, snapshot, enabled_fields={"height"})
        assert len(changes) == 1
        assert changes[0]["field"] == "height"

    def test_assigns_correct_merge_types(self):
        local = {"name": "Old", "aliases": ["A"], "tattoos": "none", "height": 165}
        upstream = {"name": "New", "aliases": ["A", "B"], "tattoos": "dragon", "height": 170}

        changes = diff_performer_fields(
            local, upstream, snapshot=None,
            enabled_fields={"name", "aliases", "tattoos", "height"},
        )
        merge_types = {c["field"]: c["merge_type"] for c in changes}
        assert merge_types["name"] == "name"
        assert merge_types["aliases"] == "alias_list"
        assert merge_types["tattoos"] == "text"
        assert merge_types["height"] == "simple"

    def test_alias_diff_is_case_insensitive(self):
        local = {"aliases": ["Jane Doe", "JD"]}
        upstream = {"aliases": ["jane doe", "jd", "Janey"]}
        snapshot = {"aliases": ["Jane Doe"]}

        changes = diff_performer_fields(local, upstream, snapshot, enabled_fields={"aliases"})
        assert len(changes) == 1
        # The upstream added "Janey" and "jd" (but "jd" matches "JD" case-insensitively)

    def test_url_diff_treats_as_alias_list(self):
        local = {"urls": ["https://a.com"]}
        upstream = {"urls": ["https://a.com", "https://b.com"]}

        changes = diff_performer_fields(local, upstream, snapshot=None, enabled_fields={"urls"})
        assert len(changes) == 1
        assert changes[0]["merge_type"] == "alias_list"


class TestFieldMergeTypes:
    """Tests for merge type assignments."""

    def test_all_default_fields_have_merge_types(self):
        for field in DEFAULT_PERFORMER_FIELDS:
            assert field in FIELD_MERGE_TYPES, f"Missing merge type for field: {field}"
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_field_mapper.py -v`
Expected: FAIL - module doesn't exist

**Step 3: Implement the field mapper**

```python
# api/upstream_field_mapper.py
"""
Upstream Field Mapper & Diff Engine

Maps fields between stash-box schema and local Stash schema,
normalizes values for comparison, and produces structured diffs.
"""

from typing import Optional


# Default performer fields to monitor (all enabled by default)
DEFAULT_PERFORMER_FIELDS = {
    "name", "disambiguation", "aliases", "gender", "birthdate",
    "death_date", "ethnicity", "country", "eye_color", "hair_color",
    "height", "cup_size", "band_size", "waist_size", "hip_size",
    "breast_type", "tattoos", "piercings", "career_start_year",
    "career_end_year", "urls", "details", "favorite",
}

# Merge type for each field (determines UI controls)
FIELD_MERGE_TYPES = {
    "name": "name",
    "disambiguation": "simple",
    "aliases": "alias_list",
    "gender": "simple",
    "birthdate": "simple",
    "death_date": "simple",
    "ethnicity": "simple",
    "country": "simple",
    "eye_color": "simple",
    "hair_color": "simple",
    "height": "simple",
    "cup_size": "simple",
    "band_size": "simple",
    "waist_size": "simple",
    "hip_size": "simple",
    "breast_type": "simple",
    "tattoos": "text",
    "piercings": "text",
    "career_start_year": "simple",
    "career_end_year": "simple",
    "urls": "alias_list",
    "details": "text",
    "favorite": "simple",
}

# Human-readable labels for fields
FIELD_LABELS = {
    "name": "Name",
    "disambiguation": "Disambiguation",
    "aliases": "Aliases",
    "gender": "Gender",
    "birthdate": "Birthdate",
    "death_date": "Death Date",
    "ethnicity": "Ethnicity",
    "country": "Country",
    "eye_color": "Eye Color",
    "hair_color": "Hair Color",
    "height": "Height",
    "cup_size": "Cup Size",
    "band_size": "Band Size",
    "waist_size": "Waist Size",
    "hip_size": "Hip Size",
    "breast_type": "Breast Type",
    "tattoos": "Tattoos",
    "piercings": "Piercings",
    "career_start_year": "Career Start",
    "career_end_year": "Career End",
    "urls": "URLs",
    "details": "Details",
    "favorite": "Favorite",
}


def _format_body_modifications(mods: list[dict] | None) -> str:
    """Format stash-box BodyModification list to readable text."""
    if not mods:
        return ""
    parts = []
    for mod in mods:
        loc = mod.get("location", "")
        desc = mod.get("description", "")
        if loc and desc:
            parts.append(f"{loc}: {desc}")
        elif loc:
            parts.append(loc)
        elif desc:
            parts.append(desc)
    return "; ".join(parts)


def normalize_upstream_performer(upstream: dict) -> dict:
    """
    Convert stash-box performer data to a normalized dict
    using local Stash field names for comparison.
    """
    result = {}

    # Direct mappings
    for field in ["name", "disambiguation", "gender", "country",
                  "eye_color", "hair_color", "height", "cup_size",
                  "band_size", "waist_size", "hip_size", "breast_type",
                  "career_start_year", "career_end_year", "death_date",
                  "details"]:
        result[field] = upstream.get(field)

    # Field name mappings
    result["birthdate"] = upstream.get("birth_date")
    result["favorite"] = upstream.get("is_favorite", False)

    # Aliases - normalize None to empty list
    result["aliases"] = upstream.get("aliases") or []

    # Body modifications -> text
    result["tattoos"] = _format_body_modifications(upstream.get("tattoos"))
    result["piercings"] = _format_body_modifications(upstream.get("piercings"))

    # URLs - extract url strings from objects
    urls = upstream.get("urls") or []
    result["urls"] = [u["url"] for u in urls if isinstance(u, dict) and u.get("url")]

    return result


def _values_equal(local_val, upstream_val, merge_type: str) -> bool:
    """Compare two field values, handling type-specific comparison."""
    if merge_type == "alias_list":
        # Case-insensitive set comparison for lists
        local_set = {v.lower() for v in (local_val or [])}
        upstream_set = {v.lower() for v in (upstream_val or [])}
        return local_set == upstream_set

    # For everything else, direct comparison
    # Normalize None vs empty string
    if local_val in (None, "") and upstream_val in (None, ""):
        return True
    return local_val == upstream_val


def diff_performer_fields(
    local: dict,
    upstream: dict,
    snapshot: Optional[dict],
    enabled_fields: set[str],
) -> list[dict]:
    """
    Compute 3-way diff between local Stash data, upstream stash-box data,
    and our previous snapshot of upstream data.

    Returns list of change dicts with: field, field_label, local_value,
    upstream_value, previous_upstream_value, merge_type.
    """
    changes = []

    for field in enabled_fields:
        if field not in FIELD_MERGE_TYPES:
            continue

        merge_type = FIELD_MERGE_TYPES[field]
        local_val = local.get(field)
        upstream_val = upstream.get(field)

        # Already in sync - skip
        if _values_equal(local_val, upstream_val, merge_type):
            continue

        if snapshot is not None:
            previous_val = snapshot.get(field)
            # If upstream hasn't changed from snapshot, user intentionally set locally - skip
            if _values_equal(upstream_val, previous_val, merge_type):
                continue
        else:
            previous_val = None

        changes.append({
            "field": field,
            "field_label": FIELD_LABELS.get(field, field),
            "local_value": local_val,
            "upstream_value": upstream_val,
            "previous_upstream_value": previous_val,
            "merge_type": merge_type,
        })

    return changes
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_field_mapper.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/upstream_field_mapper.py api/tests/test_upstream_field_mapper.py
git commit -m "feat: add field mapping and 3-way diff engine for upstream sync"
```

---

### Task 5: Stash Client - Add Performer Query with Stash-Box Filter

Add a method to query local Stash for performers linked to a specific stash-box endpoint, with full fields needed for comparison.

**Files:**
- Modify: `api/stash_client_unified.py`
- Create: `api/tests/test_upstream_stash_queries.py`

**Step 1: Write the failing test**

```python
# api/tests/test_upstream_stash_queries.py
"""Tests for Stash client upstream sync queries."""

import pytest
from unittest.mock import AsyncMock, patch


class TestGetPerformersForEndpoint:
    """Tests for querying performers linked to a stash-box endpoint."""

    @pytest.mark.asyncio
    async def test_returns_performers_with_stash_ids(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        mock_response = {
            "findPerformers": {
                "count": 1,
                "performers": [
                    {
                        "id": "42",
                        "name": "Jane Doe",
                        "disambiguation": "",
                        "alias_list": ["JD"],
                        "gender": "FEMALE",
                        "birthdate": "1990-01-15",
                        "death_date": None,
                        "ethnicity": "Caucasian",
                        "country": "US",
                        "eye_color": "Brown",
                        "hair_color": "Brown",
                        "height_cm": 165,
                        "measurements": "32B-24-34",
                        "fake_tits": "Natural",
                        "career_length": "2015-",
                        "tattoos": "Left arm: Dragon",
                        "piercings": "",
                        "details": "",
                        "urls": ["https://twitter.com/jane"],
                        "favorite": False,
                        "image_path": "/performer/42/image",
                        "stash_ids": [
                            {"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}
                        ],
                    }
                ],
            }
        }

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_response
            performers = await client.get_performers_for_endpoint(
                "https://stashdb.org/graphql"
            )

            assert len(performers) == 1
            assert performers[0]["id"] == "42"
            assert performers[0]["name"] == "Jane Doe"

    @pytest.mark.asyncio
    async def test_uses_stash_ids_endpoint_filter(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"findPerformers": {"count": 0, "performers": []}}
            await client.get_performers_for_endpoint("https://stashdb.org/graphql")

            call_args = mock_exec.call_args
            variables = call_args[1].get("variables") if call_args[1] else call_args[0][1]
            pf = variables["performer_filter"]
            assert pf["stash_id_endpoint"]["endpoint"] == "https://stashdb.org/graphql"
            assert pf["stash_id_endpoint"]["modifier"] == "NOT_NULL"


class TestUpdatePerformerFull:
    """Tests for the full performer update mutation."""

    @pytest.mark.asyncio
    async def test_sends_performer_update_mutation(self):
        from stash_client_unified import StashClientUnified

        client = StashClientUnified("http://localhost:9999")

        with patch.object(client, "_execute", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = {"performerUpdate": {"id": "42"}}
            await client.update_performer("42", name="Jane Smith", height_cm=168)

            call_args = mock_exec.call_args
            query = call_args[0][0]
            assert "performerUpdate" in query
            variables = call_args[0][1] if len(call_args[0]) > 1 else call_args[1].get("variables")
            assert variables["input"]["id"] == "42"
            assert variables["input"]["name"] == "Jane Smith"
            assert variables["input"]["height_cm"] == 168
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_stash_queries.py -v`
Expected: FAIL - methods don't exist

**Step 3: Implement the new Stash client methods**

Add to `api/stash_client_unified.py` in the Performers section:

```python
    async def get_performers_for_endpoint(self, endpoint: str) -> list[dict]:
        """Get all local performers linked to a specific stash-box endpoint."""
        query = """
        query PerformersForEndpoint($performer_filter: PerformerFilterType) {
          findPerformers(performer_filter: $performer_filter, filter: { per_page: -1 }) {
            count
            performers {
              id
              name
              disambiguation
              alias_list
              gender
              birthdate
              death_date
              ethnicity
              country
              eye_color
              hair_color
              height_cm
              measurements
              fake_tits
              career_length
              tattoos
              piercings
              details
              urls
              favorite
              image_path
              stash_ids {
                endpoint
                stash_id
              }
            }
          }
        }
        """
        variables = {
            "performer_filter": {
                "stash_id_endpoint": {
                    "endpoint": endpoint,
                    "modifier": "NOT_NULL",
                }
            }
        }
        data = await self._execute(query, variables=variables)
        return data["findPerformers"]["performers"]

    async def update_performer(self, performer_id: str, **fields) -> dict:
        """Update a performer with arbitrary fields via PerformerUpdateInput."""
        query = """
        mutation UpdatePerformer($input: PerformerUpdateInput!) {
          performerUpdate(input: $input) {
            id
          }
        }
        """
        input_data = {"id": performer_id, **fields}
        data = await self._execute(
            query, {"input": input_data}, priority=Priority.CRITICAL
        )
        return data["performerUpdate"]
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_stash_queries.py -v`
Expected: PASS

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/stash_client_unified.py api/tests/test_upstream_stash_queries.py
git commit -m "feat: add performer endpoint query and update methods to Stash client"
```

---

### Task 6: UpstreamPerformerAnalyzer

The core analyzer that ties everything together.

**Files:**
- Create: `api/analyzers/upstream_performer.py`
- Modify: `api/analyzers/__init__.py`
- Create: `api/tests/test_upstream_performer_analyzer.py`

**Step 1: Write the failing tests**

```python
# api/tests/test_upstream_performer_analyzer.py
"""Tests for UpstreamPerformerAnalyzer."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


class TestUpstreamPerformerAnalyzer:
    """Tests for the upstream performer changes analyzer."""

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "test-key", "name": "stashdb"},
        ])
        stash.get_performers_for_endpoint = AsyncMock(return_value=[])
        return stash

    @pytest.fixture
    def rec_db(self, tmp_path):
        from recommendations_db import RecommendationsDB
        return RecommendationsDB(tmp_path / "test.db")

    def test_analyzer_type(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        assert analyzer.type == "upstream_performer_changes"

    @pytest.mark.asyncio
    async def test_no_performers_no_recommendations(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[])

        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(return_value=([], 0))
            MockSBC.return_value = mock_sbc

            result = await analyzer.run()

        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_creates_recommendation_for_changed_performer(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        # Local performer
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42",
                "name": "Jane Doe",
                "disambiguation": "",
                "alias_list": ["JD"],
                "gender": "FEMALE",
                "birthdate": "1990-01-15",
                "death_date": None,
                "ethnicity": "Caucasian",
                "country": "US",
                "eye_color": "Brown",
                "hair_color": "Brown",
                "height_cm": 165,
                "measurements": "",
                "fake_tits": "",
                "career_length": "",
                "tattoos": "",
                "piercings": "",
                "details": "",
                "urls": [],
                "favorite": False,
                "image_path": "/performer/42/image",
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}
                ],
            }
        ])

        # Upstream performer with changed height
        upstream_performer = {
            "id": "abc-123",
            "name": "Jane Doe",
            "disambiguation": "",
            "aliases": ["JD"],
            "gender": "FEMALE",
            "birth_date": "1990-01-15",
            "death_date": None,
            "ethnicity": "CAUCASIAN",
            "country": "US",
            "eye_color": "BROWN",
            "hair_color": "BROWN",
            "height": 168,
            "cup_size": None,
            "band_size": None,
            "waist_size": None,
            "hip_size": None,
            "breast_type": None,
            "career_start_year": None,
            "career_end_year": None,
            "tattoos": [],
            "piercings": [],
            "urls": [],
            "is_favorite": False,
            "deleted": False,
            "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-15T10:00:00Z",
        }

        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(return_value=([upstream_performer], 1))
            MockSBC.return_value = mock_sbc

            result = await analyzer.run()

        assert result.recommendations_created == 1
        recs = rec_db.get_recommendations(type="upstream_performer_changes")
        assert len(recs) == 1
        assert recs[0].details["performer_id"] == "42"
        assert any(c["field"] == "height" for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_updates_existing_pending_recommendation(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        # Create an existing pending recommendation
        rec_id = rec_db.create_recommendation(
            type="upstream_performer_changes",
            target_type="performer",
            target_id="42",
            details={"changes": [{"field": "height", "upstream_value": 167}]},
            confidence=1.0,
        )

        # Local performer
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42",
                "name": "Jane Doe",
                "disambiguation": "",
                "alias_list": [],
                "gender": None,
                "birthdate": None,
                "death_date": None,
                "ethnicity": None,
                "country": None,
                "eye_color": None,
                "hair_color": None,
                "height_cm": 165,
                "measurements": "",
                "fake_tits": "",
                "career_length": "",
                "tattoos": "",
                "piercings": "",
                "details": "",
                "urls": [],
                "favorite": False,
                "image_path": None,
                "stash_ids": [
                    {"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}
                ],
            }
        ])

        upstream_performer = {
            "id": "abc-123",
            "name": "Jane Doe",
            "disambiguation": "",
            "aliases": [],
            "gender": None,
            "birth_date": None,
            "death_date": None,
            "ethnicity": None,
            "country": None,
            "eye_color": None,
            "hair_color": None,
            "height": 170,  # changed again
            "cup_size": None, "band_size": None, "waist_size": None,
            "hip_size": None, "breast_type": None,
            "career_start_year": None, "career_end_year": None,
            "tattoos": [], "piercings": [], "urls": [],
            "is_favorite": False, "deleted": False, "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z",
            "updated": "2026-01-16T10:00:00Z",
        }

        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(return_value=([upstream_performer], 1))
            MockSBC.return_value = mock_sbc

            result = await analyzer.run()

        # Should update existing, not create new
        assert result.recommendations_created == 0  # updated, not created
        recs = rec_db.get_recommendations(type="upstream_performer_changes")
        assert len(recs) == 1
        assert any(c["upstream_value"] == 170 for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_skips_permanently_dismissed(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        # Permanently dismiss performer 42
        with rec_db._connection() as conn:
            conn.execute(
                "INSERT INTO dismissed_targets (type, target_type, target_id, permanent) VALUES (?, ?, ?, ?)",
                ("upstream_performer_changes", "performer", "42", 1),
            )

        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42", "name": "Jane Doe", "disambiguation": "",
                "alias_list": [], "gender": None, "birthdate": None,
                "death_date": None, "ethnicity": None, "country": None,
                "eye_color": None, "hair_color": None, "height_cm": 165,
                "measurements": "", "fake_tits": "", "career_length": "",
                "tattoos": "", "piercings": "", "details": "", "urls": [],
                "favorite": False, "image_path": None,
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
            }
        ])

        upstream = {
            "id": "abc-123", "name": "Jane Smith", "disambiguation": "",
            "aliases": [], "gender": None, "birth_date": None, "death_date": None,
            "ethnicity": None, "country": None, "eye_color": None, "hair_color": None,
            "height": 168, "cup_size": None, "band_size": None, "waist_size": None,
            "hip_size": None, "breast_type": None, "career_start_year": None,
            "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
            "is_favorite": False, "deleted": False, "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }

        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(return_value=([upstream], 1))
            MockSBC.return_value = mock_sbc

            result = await analyzer.run()

        assert result.recommendations_created == 0

    @pytest.mark.asyncio
    async def test_skips_deleted_upstream_performers(self, mock_stash, rec_db):
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[
            {
                "id": "42", "name": "Jane Doe", "disambiguation": "",
                "alias_list": [], "gender": None, "birthdate": None,
                "death_date": None, "ethnicity": None, "country": None,
                "eye_color": None, "hair_color": None, "height_cm": 165,
                "measurements": "", "fake_tits": "", "career_length": "",
                "tattoos": "", "piercings": "", "details": "", "urls": [],
                "favorite": False, "image_path": None,
                "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
            }
        ])

        upstream = {
            "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
            "aliases": [], "gender": None, "birth_date": None, "death_date": None,
            "ethnicity": None, "country": None, "eye_color": None, "hair_color": None,
            "height": 165, "cup_size": None, "band_size": None, "waist_size": None,
            "hip_size": None, "breast_type": None, "career_start_year": None,
            "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
            "is_favorite": False, "deleted": True, "merged_into_id": None,
            "created": "2024-01-01T00:00:00Z", "updated": "2026-01-15T10:00:00Z",
        }

        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(return_value=([upstream], 1))
            MockSBC.return_value = mock_sbc

            result = await analyzer.run()

        assert result.recommendations_created == 0
```

**Step 2: Run tests to verify they fail**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_performer_analyzer.py -v`
Expected: FAIL - module doesn't exist

**Step 3: Implement the analyzer**

```python
# api/analyzers/upstream_performer.py
"""
Upstream Performer Changes Analyzer

Detects changes in stash-box linked performers by comparing upstream
data against local Stash data using 3-way diffing with stored snapshots.
"""

import logging

from .base import BaseAnalyzer, AnalysisResult
from stashbox_client import StashBoxClient
from upstream_field_mapper import (
    normalize_upstream_performer,
    diff_performer_fields,
    DEFAULT_PERFORMER_FIELDS,
)
from config import get_stashbox_shortname

logger = logging.getLogger(__name__)


def _build_local_performer_data(performer: dict) -> dict:
    """Extract comparable field values from a local Stash performer."""
    return {
        "name": performer.get("name"),
        "disambiguation": performer.get("disambiguation") or "",
        "aliases": performer.get("alias_list") or [],
        "gender": performer.get("gender"),
        "birthdate": performer.get("birthdate"),
        "death_date": performer.get("death_date"),
        "ethnicity": performer.get("ethnicity"),
        "country": performer.get("country"),
        "eye_color": performer.get("eye_color"),
        "hair_color": performer.get("hair_color"),
        "height": performer.get("height_cm"),
        "cup_size": None,  # Not directly available from Stash query
        "band_size": None,
        "waist_size": None,
        "hip_size": None,
        "breast_type": performer.get("fake_tits"),
        "tattoos": performer.get("tattoos") or "",
        "piercings": performer.get("piercings") or "",
        "career_start_year": None,  # Parsed from career_length if needed
        "career_end_year": None,
        "urls": performer.get("urls") or [],
        "details": performer.get("details") or "",
        "favorite": performer.get("favorite", False),
    }


class UpstreamPerformerAnalyzer(BaseAnalyzer):
    """
    Detects upstream changes in stash-box linked performers.

    Compares performer data from configured stash-box endpoints against
    local Stash data, using stored snapshots for 3-way diffing.
    """

    type = "upstream_performer_changes"

    async def run(self, incremental: bool = True) -> AnalysisResult:
        connections = await self.stash.get_stashbox_connections()

        # Load endpoint settings from recommendation_settings config
        settings = self.rec_db.get_settings(self.type)
        endpoint_config = {}
        if settings and settings.config:
            endpoint_config = settings.config.get("endpoints", {})

        total_processed = 0
        total_created = 0
        errors = []

        for conn in connections:
            endpoint = conn["endpoint"]
            api_key = conn.get("api_key", "")

            # Check if endpoint is enabled (default: enabled)
            ep_cfg = endpoint_config.get(endpoint, {})
            if not ep_cfg.get("enabled", True):
                continue

            try:
                created, processed = await self._process_endpoint(
                    endpoint, api_key, incremental
                )
                total_created += created
                total_processed += processed
            except Exception as e:
                error_msg = f"Error processing {endpoint}: {e}"
                logger.error(error_msg)
                errors.append(error_msg)

        return AnalysisResult(
            items_processed=total_processed,
            recommendations_created=total_created,
            errors=errors if errors else None,
        )

    async def _process_endpoint(
        self, endpoint: str, api_key: str, incremental: bool
    ) -> tuple[int, int]:
        """Process a single stash-box endpoint. Returns (created, processed)."""
        # Get local performers linked to this endpoint
        local_performers = await self.stash.get_performers_for_endpoint(endpoint)

        if not local_performers:
            return 0, 0

        # Build lookup: stash_box_id -> local performer data
        local_lookup = {}
        for p in local_performers:
            for sid in p.get("stash_ids", []):
                if sid["endpoint"] == endpoint:
                    local_lookup[sid["stash_id"]] = p

        # Load watermark for incremental fetching
        watermark = None
        if incremental:
            wm = self.rec_db.get_watermark(f"{self.type}:{endpoint}")
            if wm and wm.get("last_cursor"):
                watermark = wm["last_cursor"]

        # Get enabled fields
        enabled_fields = self.rec_db.get_enabled_fields(endpoint, "performer")
        if enabled_fields is None:
            enabled_fields = DEFAULT_PERFORMER_FIELDS

        # Fetch upstream performers
        sbc = StashBoxClient(endpoint, api_key)
        latest_updated_at = None
        created = 0
        processed = 0
        page = 1

        while True:
            upstream_performers, total_count = await sbc.query_performers(
                page=page, per_page=25
            )

            if not upstream_performers:
                break

            for up in upstream_performers:
                updated_at = up.get("updated")

                # Track latest for watermark
                if latest_updated_at is None or (updated_at and updated_at > latest_updated_at):
                    latest_updated_at = updated_at

                # Incremental: stop if we've passed the watermark
                if watermark and updated_at and updated_at <= watermark:
                    # All remaining performers are older than watermark
                    upstream_performers = []  # Signal to break outer loop
                    break

                stash_box_id = up.get("id")
                if not stash_box_id or stash_box_id not in local_lookup:
                    continue

                # Skip deleted performers
                if up.get("deleted"):
                    continue

                # Skip merged performers
                if up.get("merged_into_id"):
                    continue

                local_performer = local_lookup[stash_box_id]
                local_id = local_performer["id"]

                # Skip permanently dismissed
                if self.rec_db.is_permanently_dismissed(self.type, "performer", local_id):
                    continue

                # Normalize upstream data
                normalized = normalize_upstream_performer(up)

                # Get previous snapshot for 3-way diff
                snapshot_row = self.rec_db.get_upstream_snapshot(
                    entity_type="performer",
                    endpoint=endpoint,
                    stash_box_id=stash_box_id,
                )
                snapshot = snapshot_row["upstream_data"] if snapshot_row else None

                # Build local comparable data
                local_data = _build_local_performer_data(local_performer)

                # Compute diff
                changes = diff_performer_fields(
                    local_data, normalized, snapshot, enabled_fields
                )

                # Update snapshot
                self.rec_db.upsert_upstream_snapshot(
                    entity_type="performer",
                    local_entity_id=local_id,
                    endpoint=endpoint,
                    stash_box_id=stash_box_id,
                    upstream_data=normalized,
                    upstream_updated_at=updated_at or "",
                )

                processed += 1

                if not changes:
                    continue

                # Build recommendation details
                endpoint_name = get_stashbox_shortname(endpoint)
                details = {
                    "endpoint": endpoint,
                    "endpoint_name": endpoint_name,
                    "stash_box_id": stash_box_id,
                    "performer_id": local_id,
                    "performer_name": local_performer.get("name", ""),
                    "performer_image_path": local_performer.get("image_path"),
                    "upstream_updated_at": updated_at,
                    "changes": changes,
                }

                # Check for existing pending recommendation
                existing_recs = self.rec_db.get_recommendations(
                    type=self.type, target_type="performer",
                    status="pending", limit=1,
                )
                existing = None
                for r in self.rec_db.get_recommendations(type=self.type, status="pending"):
                    if r.target_id == local_id:
                        existing = r
                        break

                if existing:
                    # Update existing recommendation
                    self.rec_db.update_recommendation_details(existing.id, details)
                else:
                    # Check soft dismissal - undismiss if upstream changed
                    if self.rec_db.is_dismissed(self.type, "performer", local_id):
                        self.rec_db.undismiss(self.type, "performer", local_id)

                    rec_id = self.create_recommendation(
                        target_type="performer",
                        target_id=local_id,
                        details=details,
                        confidence=1.0,
                    )
                    if rec_id:
                        created += 1

            if not upstream_performers:
                break
            page += 1

        # Update watermark
        if latest_updated_at:
            self.rec_db.set_watermark(
                f"{self.type}:{endpoint}",
                last_cursor=latest_updated_at,
            )

        return created, processed
```

**Step 4: Update `api/analyzers/__init__.py`**

```python
from .upstream_performer import UpstreamPerformerAnalyzer
```

And add it to `__all__`.

**Step 5: Run tests to verify they pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_performer_analyzer.py -v`
Expected: PASS

**Step 6: Run full test suite**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 7: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/analyzers/upstream_performer.py api/analyzers/__init__.py api/tests/test_upstream_performer_analyzer.py
git commit -m "feat: add UpstreamPerformerAnalyzer for detecting stash-box changes"
```

---

### Task 7: Register Analyzer & Add API Endpoints

Wire the analyzer into the router and add endpoints for upstream sync operations.

**Files:**
- Modify: `api/recommendations_router.py`
- Modify: `plugin/stash_sense_backend.py`

**Step 1: Register the analyzer**

In `api/recommendations_router.py`:

1. Add import:
```python
from analyzers import UpstreamPerformerAnalyzer
```

2. Add to `ANALYZERS` dict:
```python
ANALYZERS = {
    "duplicate_performer": DuplicatePerformerAnalyzer,
    "duplicate_scene_files": DuplicateSceneFilesAnalyzer,
    "duplicate_scenes": DuplicateScenesAnalyzer,
    "upstream_performer_changes": UpstreamPerformerAnalyzer,
}
```

3. Add new endpoints for upstream sync actions:

```python
# ==================== Upstream Sync Actions ====================

@router.post("/actions/update-performer")
async def update_performer_fields(performer_id: str, fields: dict):
    """Apply selected upstream changes to a performer."""
    stash = get_stash_client()
    try:
        result = await stash.update_performer(performer_id, **fields)
        return {"success": True, "performer": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UpstreamDismissRequest(BaseModel):
    """Request to dismiss an upstream recommendation."""
    reason: Optional[str] = Field(None)
    permanent: bool = Field(False, description="If true, never show updates for this entity again")


@router.post("/{rec_id}/dismiss-upstream")
async def dismiss_upstream_recommendation(rec_id: int, request: UpstreamDismissRequest = None):
    """Dismiss an upstream recommendation with permanent option."""
    db = get_rec_db()
    permanent = request.permanent if request else False
    reason = request.reason if request else None
    success = db.dismiss_recommendation(rec_id, reason=reason, permanent=permanent)
    if not success:
        raise HTTPException(status_code=404, detail="Recommendation not found")
    return {"success": True, "permanent": permanent}


@router.get("/upstream/field-config/{endpoint_b64}")
async def get_field_config(endpoint_b64: str):
    """Get field monitoring config for an endpoint. Endpoint is base64-encoded."""
    import base64
    endpoint = base64.b64decode(endpoint_b64).decode()
    db = get_rec_db()
    fields = db.get_enabled_fields(endpoint, "performer")
    from upstream_field_mapper import DEFAULT_PERFORMER_FIELDS, FIELD_LABELS
    if fields is None:
        # Return defaults
        return {
            "endpoint": endpoint,
            "fields": {f: {"enabled": True, "label": FIELD_LABELS.get(f, f)} for f in DEFAULT_PERFORMER_FIELDS},
        }
    return {
        "endpoint": endpoint,
        "fields": {f: {"enabled": f in fields, "label": FIELD_LABELS.get(f, f)} for f in DEFAULT_PERFORMER_FIELDS},
    }


@router.post("/upstream/field-config/{endpoint_b64}")
async def set_field_config(endpoint_b64: str, field_configs: dict[str, bool]):
    """Set field monitoring config for an endpoint."""
    import base64
    endpoint = base64.b64decode(endpoint_b64).decode()
    db = get_rec_db()
    db.set_field_config(endpoint, "performer", field_configs)
    return {"success": True}
```

**Step 2: Add backend proxy operations**

In `plugin/stash_sense_backend.py`, add to `handle_recommendations()`:

```python
    elif mode == "rec_update_performer":
        performer_id = args.get("performer_id")
        fields = args.get("fields", {})
        if not performer_id:
            return {"error": "performer_id required"}
        return sidecar_post(
            sidecar_url,
            "/recommendations/actions/update-performer",
            {"performer_id": performer_id, "fields": fields},
            timeout=60,
        )

    elif mode == "rec_dismiss_upstream":
        rec_id = args.get("rec_id")
        if not rec_id:
            return {"error": "No rec_id provided"}
        return sidecar_post(
            sidecar_url,
            f"/recommendations/{rec_id}/dismiss-upstream",
            {"reason": args.get("reason"), "permanent": args.get("permanent", False)},
        )

    elif mode == "rec_get_field_config":
        import base64
        endpoint = args.get("endpoint", "")
        endpoint_b64 = base64.b64encode(endpoint.encode()).decode()
        return sidecar_get(sidecar_url, f"/recommendations/upstream/field-config/{endpoint_b64}")

    elif mode == "rec_set_field_config":
        import base64
        endpoint = args.get("endpoint", "")
        endpoint_b64 = base64.b64encode(endpoint.encode()).decode()
        return sidecar_post(
            sidecar_url,
            f"/recommendations/upstream/field-config/{endpoint_b64}",
            args.get("field_configs", {}),
        )
```

**Step 3: Verify existing tests still pass**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/recommendations_router.py plugin/stash_sense_backend.py
git commit -m "feat: register upstream analyzer and add sync API endpoints"
```

---

### Task 8: Plugin UI - Upstream Recommendation Card & Detail View

Add rendering for the upstream performer changes recommendation type in the plugin JS.

**Files:**
- Modify: `plugin/stash-sense-recommendations.js`
- Modify: `plugin/stash-sense.css`

**Step 1: Read existing UI code for patterns**

Review `plugin/stash-sense-recommendations.js` to understand:
- How `renderRecommendationCard()` dispatches by rec type (around line 622)
- How `renderDetail()` dispatches to type-specific detail renderers (around line 685)
- The `RecommendationsAPI` pattern for new operations

**Step 2: Add API methods for upstream operations**

In the `RecommendationsAPI` section (around line 29), add:

```javascript
async updatePerformer(performerId, fields) {
    return apiCall('rec_update_performer', { performer_id: performerId, fields });
},
async dismissUpstream(recId, reason, permanent) {
    return apiCall('rec_dismiss_upstream', { rec_id: recId, reason, permanent: !!permanent });
},
async getFieldConfig(endpoint) {
    return apiCall('rec_get_field_config', { endpoint });
},
async setFieldConfig(endpoint, fieldConfigs) {
    return apiCall('rec_set_field_config', { endpoint, field_configs: fieldConfigs });
},
```

**Step 3: Add card renderer**

In `renderRecommendationCard()`, add a case for `upstream_performer_changes`:

```javascript
} else if (rec.type === 'upstream_performer_changes') {
    const details = rec.details;
    const changeCount = (details.changes || []).length;
    const changedFields = (details.changes || []).map(c => c.field_label).join(', ');

    card.innerHTML = `
        <div class="ss-rec-card-header">
            <img src="${details.performer_image_path || ''}" class="ss-rec-thumb" onerror="this.style.display='none'"/>
            <div class="ss-rec-card-info">
                <div class="ss-rec-card-title">Upstream Changes: ${details.performer_name || 'Unknown'}</div>
                <div class="ss-rec-card-subtitle">
                    ${changeCount} field${changeCount !== 1 ? 's' : ''} changed · ${details.endpoint_name || ''}
                </div>
                <div class="ss-rec-card-fields">${changedFields}</div>
            </div>
        </div>
    `;
}
```

**Step 4: Add detail renderer**

Create function `renderUpstreamPerformerDetail(container, rec)` that:

1. Renders a header with performer image, name, endpoint badge
2. For each change in `rec.details.changes`, renders a field row based on `merge_type`:
   - `simple`: Radio buttons (Keep local / Accept upstream / Custom)
   - `name`: 5-option radio with validation
   - `alias_list`: Union checkbox list
   - `text`: Radio + editable textarea
3. Action bar with "Apply Selected Changes" and "Dismiss" dropdown
4. Validation logic before submit
5. Calls `RecommendationsAPI.updatePerformer()` on apply

Wire it into `renderDetail()`:
```javascript
} else if (rec.type === 'upstream_performer_changes') {
    renderUpstreamPerformerDetail(detailContainer, rec);
}
```

**Step 5: Add CSS for upstream diff UI**

Add to `plugin/stash-sense.css`:

```css
/* Upstream sync styles */
.ss-upstream-field-row { ... }
.ss-upstream-local-value { ... }
.ss-upstream-upstream-value { ... }
.ss-upstream-radio-group { ... }
.ss-upstream-alias-list { ... }
.ss-upstream-alias-item { ... }
.ss-upstream-alias-local-only { ... }
.ss-upstream-alias-upstream-only { ... }
.ss-upstream-validation-error { ... }
.ss-upstream-action-bar { ... }
.ss-upstream-dismiss-dropdown { ... }
```

**Step 6: Add dashboard card for upstream sync**

In the dashboard rendering section, add a card for `upstream_performer_changes` that shows:
- Count of pending upstream recommendations
- "Check for Updates" button that calls `RecommendationsAPI.runAnalysis('upstream_performer_changes')`

**Step 7: Manual test**

Load the plugin in Stash, navigate to the recommendations dashboard, and verify:
- The "Check for Updates" button appears
- Running the analysis works (shows progress)
- Recommendation cards render with the diff UI
- Field merge controls work

**Step 8: Commit**

```bash
cd /home/carrot/code/stash-sense
git add plugin/stash-sense-recommendations.js plugin/stash-sense.css
git commit -m "feat: add upstream performer sync UI with field-level merge controls"
```

---

### Task 9: Validation & Error Handling in UI

Add client-side validation and server-side error handling to the merge UI.

**Files:**
- Modify: `plugin/stash-sense-recommendations.js`

**Step 1: Add validation functions**

```javascript
async function validatePerformerMerge(performerId, proposedName, proposedDisambig, proposedAliases) {
    const errors = [];

    // 1. Name uniqueness - query all performers
    // Use stashQuery from stash-sense-core.js to check
    const nameCheck = await SS.stashQuery(`
        query { findPerformers(performer_filter: { name: { value: "${proposedName}", modifier: EQUALS } }) {
            performers { id name disambiguation }
        }}
    `);
    const conflicts = (nameCheck?.findPerformers?.performers || [])
        .filter(p => p.id !== performerId);
    if (conflicts.length > 0) {
        const c = conflicts[0];
        const disambigNote = proposedDisambig ? '' : ' (add a disambiguation to make it unique)';
        errors.push(`Name "${proposedName}" already used by performer "${c.name}"${disambigNote}`);
    }

    // 2. Alias can't match performer's own name
    const nameLower = proposedName.toLowerCase();
    for (const alias of proposedAliases) {
        if (alias.toLowerCase() === nameLower) {
            errors.push(`Alias "${alias}" matches the performer's name`);
        }
    }

    // 3. No duplicate aliases
    const seen = new Set();
    for (const alias of proposedAliases) {
        const lower = alias.toLowerCase();
        if (seen.has(lower)) {
            errors.push(`Duplicate alias: "${alias}"`);
        }
        seen.add(lower);
    }

    return errors;
}
```

**Step 2: Wire validation into the Apply button**

Before calling `updatePerformer()`, run validation and display errors inline. Block submission if errors exist.

**Step 3: Add server-side error handling**

Wrap the `updatePerformer()` call in try/catch. Parse GraphQL error responses:
- If error contains "already exists" → show name conflict message
- If error contains "duplicate alias" → highlight the alias
- Otherwise → show raw error with retry option

**Step 4: Manual test**

Test validation:
- Try setting a performer name that conflicts with another performer
- Try adding an alias that matches the performer's own name
- Try adding duplicate aliases
- Verify errors show inline and block submission

**Step 5: Commit**

```bash
cd /home/carrot/code/stash-sense
git add plugin/stash-sense-recommendations.js
git commit -m "feat: add client-side validation and error handling for upstream sync"
```

---

### Task 10: Settings UI for Upstream Sync

Add settings section to the recommendations dashboard for configuring endpoints and monitored fields.

**Files:**
- Modify: `plugin/stash-sense-recommendations.js`

**Step 1: Add settings section to dashboard**

After the existing dashboard cards, add an "Upstream Sync Settings" section:

1. Query `getStashBoxConnections()` via the existing Stash GraphQL client to get available endpoints
2. For each endpoint, show:
   - Enable/disable toggle
   - Rate limit delay input
3. "Monitored Fields" expandable section with checkboxes for each performer field
4. "Save Settings" button that calls the field config and recommendation settings APIs

**Step 2: Wire up save/load**

- On page load, fetch current settings via `getFieldConfig(endpoint)` API
- On save, call `setFieldConfig(endpoint, configs)` and update recommendation_settings config JSON

**Step 3: Manual test**

- Verify endpoints are listed
- Toggle fields on/off, save, reload, verify persistence
- Run a scan with some fields disabled, verify those fields are excluded from diffs

**Step 4: Commit**

```bash
cd /home/carrot/code/stash-sense
git add plugin/stash-sense-recommendations.js
git commit -m "feat: add upstream sync settings UI for endpoint and field configuration"
```

---

### Task 11: Integration Testing & Polish

End-to-end verification and edge case handling.

**Files:**
- Modify: various files for bug fixes discovered during testing

**Step 1: Write integration test**

```python
# api/tests/test_upstream_sync_integration.py
"""Integration tests for the full upstream sync flow."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from recommendations_db import RecommendationsDB


class TestUpstreamSyncIntegration:

    @pytest.fixture
    def rec_db(self, tmp_path):
        return RecommendationsDB(tmp_path / "test.db")

    @pytest.fixture
    def mock_stash(self):
        stash = MagicMock()
        stash.get_stashbox_connections = AsyncMock(return_value=[
            {"endpoint": "https://stashdb.org/graphql", "api_key": "key", "name": "stashdb"},
        ])
        return stash

    @pytest.mark.asyncio
    async def test_full_flow_first_scan_then_rescan(self, mock_stash, rec_db):
        """Test: first scan creates recs, rescan updates existing."""
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        local_performer = {
            "id": "42", "name": "Jane Doe", "disambiguation": "",
            "alias_list": ["JD"], "gender": "FEMALE", "birthdate": "1990-01-15",
            "death_date": None, "ethnicity": "Caucasian", "country": "US",
            "eye_color": "Brown", "hair_color": "Brown", "height_cm": 165,
            "measurements": "", "fake_tits": "", "career_length": "",
            "tattoos": "", "piercings": "", "details": "", "urls": [],
            "favorite": False, "image_path": None,
            "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
        }
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[local_performer])

        def make_upstream(height, updated):
            return {
                "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
                "aliases": ["JD"], "gender": "FEMALE", "birth_date": "1990-01-15",
                "death_date": None, "ethnicity": "CAUCASIAN", "country": "US",
                "eye_color": "BROWN", "hair_color": "BROWN",
                "height": height,
                "cup_size": None, "band_size": None, "waist_size": None,
                "hip_size": None, "breast_type": None,
                "career_start_year": None, "career_end_year": None,
                "tattoos": [], "piercings": [], "urls": [],
                "is_favorite": False, "deleted": False, "merged_into_id": None,
                "created": "2024-01-01T00:00:00Z", "updated": updated,
            }

        # First scan: height changed from 165 -> 168
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(
                return_value=([make_upstream(168, "2026-01-15T10:00:00Z")], 1)
            )
            MockSBC.return_value = mock_sbc
            result1 = await analyzer.run(incremental=False)

        assert result1.recommendations_created == 1

        # Second scan: height changed again 168 -> 170
        analyzer2 = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(
                return_value=([make_upstream(170, "2026-01-16T10:00:00Z")], 1)
            )
            MockSBC.return_value = mock_sbc
            result2 = await analyzer2.run(incremental=False)

        # Should update existing, not create new
        assert result2.recommendations_created == 0
        recs = rec_db.get_recommendations(type="upstream_performer_changes")
        assert len(recs) == 1
        assert any(c["upstream_value"] == 170 for c in recs[0].details["changes"])

    @pytest.mark.asyncio
    async def test_soft_dismiss_then_new_changes_creates_new_rec(self, mock_stash, rec_db):
        """Test: soft dismiss, then new upstream changes create new recommendation."""
        from analyzers.upstream_performer import UpstreamPerformerAnalyzer

        local_performer = {
            "id": "42", "name": "Jane Doe", "disambiguation": "",
            "alias_list": [], "gender": None, "birthdate": None,
            "death_date": None, "ethnicity": None, "country": None,
            "eye_color": None, "hair_color": None, "height_cm": 165,
            "measurements": "", "fake_tits": "", "career_length": "",
            "tattoos": "", "piercings": "", "details": "", "urls": [],
            "favorite": False, "image_path": None,
            "stash_ids": [{"endpoint": "https://stashdb.org/graphql", "stash_id": "abc-123"}],
        }
        mock_stash.get_performers_for_endpoint = AsyncMock(return_value=[local_performer])

        def make_upstream(height, updated):
            return {
                "id": "abc-123", "name": "Jane Doe", "disambiguation": "",
                "aliases": [], "gender": None, "birth_date": None, "death_date": None,
                "ethnicity": None, "country": None, "eye_color": None, "hair_color": None,
                "height": height, "cup_size": None, "band_size": None, "waist_size": None,
                "hip_size": None, "breast_type": None, "career_start_year": None,
                "career_end_year": None, "tattoos": [], "piercings": [], "urls": [],
                "is_favorite": False, "deleted": False, "merged_into_id": None,
                "created": "2024-01-01T00:00:00Z", "updated": updated,
            }

        # First scan creates recommendation
        analyzer = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(
                return_value=([make_upstream(168, "2026-01-15T10:00:00Z")], 1)
            )
            MockSBC.return_value = mock_sbc
            await analyzer.run(incremental=False)

        recs = rec_db.get_recommendations(type="upstream_performer_changes", status="pending")
        assert len(recs) == 1

        # Soft dismiss
        rec_db.dismiss_recommendation(recs[0].id, permanent=False)

        # New scan with newer upstream changes
        analyzer2 = UpstreamPerformerAnalyzer(mock_stash, rec_db)
        with patch("analyzers.upstream_performer.StashBoxClient") as MockSBC:
            mock_sbc = MagicMock()
            mock_sbc.query_performers = AsyncMock(
                return_value=([make_upstream(170, "2026-01-16T10:00:00Z")], 1)
            )
            MockSBC.return_value = mock_sbc
            result = await analyzer2.run(incremental=False)

        assert result.recommendations_created == 1
```

**Step 2: Run integration tests**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/test_upstream_sync_integration.py -v`
Expected: PASS

**Step 3: Run full test suite**

Run: `cd /home/carrot/code/stash-sense/api && python -m pytest tests/ -v --tb=short`
Expected: All PASS

**Step 4: Commit**

```bash
cd /home/carrot/code/stash-sense
git add api/tests/test_upstream_sync_integration.py
git commit -m "test: add integration tests for upstream performer sync flow"
```

---

## Task Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | DB schema migration V4 | `api/recommendations_db.py` |
| 2 | Snapshot & field config DB methods | `api/recommendations_db.py` |
| 3 | StashBox GraphQL client | `api/stashbox_client.py` |
| 4 | Field mapping & diff engine | `api/upstream_field_mapper.py` |
| 5 | Stash client query additions | `api/stash_client_unified.py` |
| 6 | UpstreamPerformerAnalyzer | `api/analyzers/upstream_performer.py` |
| 7 | Router registration & API endpoints | `api/recommendations_router.py`, `plugin/stash_sense_backend.py` |
| 8 | Plugin UI - card & detail view | `plugin/stash-sense-recommendations.js`, `plugin/stash-sense.css` |
| 9 | Validation & error handling | `plugin/stash-sense-recommendations.js` |
| 10 | Settings UI | `plugin/stash-sense-recommendations.js` |
| 11 | Integration testing | `api/tests/test_upstream_sync_integration.py` |

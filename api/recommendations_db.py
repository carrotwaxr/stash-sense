"""
SQLite Database Layer for Stash Sense Recommendations

Stores user-local recommendations, analysis state, and settings.
Separate from the distributed performers.db to allow independent updates.

See: docs/plans/2026-01-28-recommendations-engine-design.md
"""

import sqlite3
import json
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator, Any


SCHEMA_VERSION = 7


@dataclass
class Recommendation:
    """A recommendation for user action."""
    id: int
    type: str
    status: str  # 'pending', 'dismissed', 'resolved'
    target_type: str  # 'scene', 'performer', 'studio', 'file'
    target_id: str
    details: dict
    resolution_action: Optional[str]
    resolution_details: Optional[dict]
    resolved_at: Optional[str]
    confidence: Optional[float]
    source_analysis_id: Optional[int]
    created_at: str
    updated_at: str


@dataclass
class AnalysisRun:
    """Record of an analysis run."""
    id: int
    type: str
    status: str  # 'running', 'completed', 'failed'
    started_at: str
    completed_at: Optional[str]
    items_total: Optional[int]
    items_processed: Optional[int]
    recommendations_created: int
    cursor: Optional[str]
    error_message: Optional[str]


@dataclass
class RecommendationSettings:
    """Settings for a recommendation type."""
    type: str
    enabled: bool
    auto_dismiss_threshold: Optional[float]
    notify: bool
    interval_hours: Optional[int]
    last_run_at: Optional[str]
    next_run_at: Optional[str]
    config: Optional[dict]


class RecommendationsDB:
    """
    SQLite database for recommendations and analysis state.

    Usage:
        db = RecommendationsDB("stash_sense.db")

        # Create a recommendation
        rec_id = db.create_recommendation(
            type="duplicate_performer",
            target_type="performer",
            target_id="123",
            details={"duplicate_ids": ["123", "456"], "suggested_keeper": "123"}
        )

        # Get pending recommendations
        recs = db.get_recommendations(status="pending", type="duplicate_performer")

        # Resolve a recommendation
        db.resolve_recommendation(rec_id, action="merged", details={"kept_id": "123"})
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._init_database()

    def _init_database(self):
        """Initialize database schema if needed."""
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                self._create_schema(conn)
            else:
                version = conn.execute("SELECT version FROM schema_version").fetchone()[0]
                if version < SCHEMA_VERSION:
                    self._migrate_schema(conn, version)

    def _create_schema(self, conn: sqlite3.Connection):
        """Create the database schema."""
        conn.executescript(f"""
            -- Schema version tracking
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            );
            INSERT INTO schema_version (version) VALUES ({SCHEMA_VERSION});

            -- Core recommendations table
            CREATE TABLE recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                details JSON NOT NULL,
                resolution_action TEXT,
                resolution_details JSON,
                resolved_at TEXT,
                confidence REAL,
                source_analysis_id INTEGER REFERENCES analysis_runs(id),
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                UNIQUE(type, target_type, target_id)
            );
            CREATE INDEX idx_rec_status ON recommendations(status);
            CREATE INDEX idx_rec_type ON recommendations(type);
            CREATE INDEX idx_rec_target ON recommendations(target_type, target_id);

            -- Track analysis runs
            CREATE TABLE analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                type TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                items_total INTEGER,
                items_processed INTEGER,
                recommendations_created INTEGER DEFAULT 0,
                cursor TEXT,
                error_message TEXT
            );
            CREATE INDEX idx_analysis_type_status ON analysis_runs(type, status);

            -- User preferences per recommendation type
            CREATE TABLE recommendation_settings (
                type TEXT PRIMARY KEY,
                enabled INTEGER DEFAULT 1,
                auto_dismiss_threshold REAL,
                notify INTEGER DEFAULT 1,
                interval_hours INTEGER,
                last_run_at TEXT,
                next_run_at TEXT,
                config JSON
            );

            -- Dismissed targets (don't re-recommend)
            CREATE TABLE dismissed_targets (
                type TEXT NOT NULL,
                target_type TEXT NOT NULL,
                target_id TEXT NOT NULL,
                dismissed_at TEXT DEFAULT (datetime('now')),
                reason TEXT,
                permanent INTEGER DEFAULT 0,
                PRIMARY KEY (type, target_type, target_id)
            );

            -- Track analysis watermarks for incremental runs
            CREATE TABLE analysis_watermarks (
                type TEXT PRIMARY KEY,
                last_completed_at TEXT,
                last_cursor TEXT,
                last_stash_updated_at TEXT
            );

            -- Scene fingerprints for duplicate detection
            CREATE TABLE scene_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                stash_scene_id INTEGER NOT NULL UNIQUE,
                total_faces INTEGER NOT NULL DEFAULT 0,
                frames_analyzed INTEGER NOT NULL DEFAULT 0,
                fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                db_version TEXT,  -- Face recognition DB version used to generate this fingerprint
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_scene_fp_stash_id ON scene_fingerprints(stash_scene_id);
            CREATE INDEX idx_scene_fp_status ON scene_fingerprints(fingerprint_status);
            CREATE INDEX idx_scene_fp_db_version ON scene_fingerprints(db_version);

            -- Face entries within scene fingerprints
            CREATE TABLE scene_fingerprint_faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                performer_id TEXT NOT NULL,
                face_count INTEGER NOT NULL DEFAULT 0,
                avg_confidence REAL,
                proportion REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(fingerprint_id, performer_id)
            );
            CREATE INDEX idx_scene_fp_faces_fingerprint ON scene_fingerprint_faces(fingerprint_id);
            CREATE INDEX idx_scene_fp_faces_performer ON scene_fingerprint_faces(performer_id);

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
                bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(stash_image_id, performer_id)
            );
            CREATE INDEX idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
            CREATE INDEX idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);

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

            -- User settings (key-value store)
            CREATE TABLE user_settings (
                key TEXT PRIMARY KEY,
                value JSON NOT NULL,
                updated_at TEXT DEFAULT (datetime('now'))
            );

            -- Seed default settings
            INSERT INTO user_settings (key, value) VALUES ('normalize_enum_display', 'true');

            -- Duplicate scene candidates (work queue for scoring)
            CREATE TABLE duplicate_candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scene_a_id INTEGER NOT NULL,
                scene_b_id INTEGER NOT NULL,
                source TEXT NOT NULL,
                run_id INTEGER REFERENCES analysis_runs(id),
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(scene_a_id, scene_b_id)
            );
            CREATE INDEX idx_dup_candidates_run ON duplicate_candidates(run_id);
        """)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Migrate schema from older version."""
        if from_version < 2:
            # Add scene fingerprint tables
            conn.executescript("""
                -- Scene fingerprints for duplicate detection
                CREATE TABLE IF NOT EXISTS scene_fingerprints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    stash_scene_id INTEGER NOT NULL UNIQUE,
                    total_faces INTEGER NOT NULL DEFAULT 0,
                    frames_analyzed INTEGER NOT NULL DEFAULT 0,
                    fingerprint_status TEXT NOT NULL DEFAULT 'pending',
                    db_version TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    updated_at TEXT DEFAULT (datetime('now'))
                );
                CREATE INDEX IF NOT EXISTS idx_scene_fp_stash_id ON scene_fingerprints(stash_scene_id);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_status ON scene_fingerprints(fingerprint_status);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_db_version ON scene_fingerprints(db_version);

                -- Face entries within scene fingerprints
                CREATE TABLE IF NOT EXISTS scene_fingerprint_faces (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    fingerprint_id INTEGER NOT NULL REFERENCES scene_fingerprints(id) ON DELETE CASCADE,
                    performer_id TEXT NOT NULL,
                    face_count INTEGER NOT NULL DEFAULT 0,
                    avg_confidence REAL,
                    proportion REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(fingerprint_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_scene_fp_faces_fingerprint ON scene_fingerprint_faces(fingerprint_id);
                CREATE INDEX IF NOT EXISTS idx_scene_fp_faces_performer ON scene_fingerprint_faces(performer_id);

                -- Update schema version
                UPDATE schema_version SET version = 3;
            """)

        if from_version == 2:
            # Add db_version column to scene_fingerprints
            conn.executescript("""
                ALTER TABLE scene_fingerprints ADD COLUMN db_version TEXT;
                CREATE INDEX IF NOT EXISTS idx_scene_fp_db_version ON scene_fingerprints(db_version);
                UPDATE schema_version SET version = 3;
            """)

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

        if from_version < 5:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS user_settings (
                    key TEXT PRIMARY KEY,
                    value JSON NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                INSERT OR IGNORE INTO user_settings (key, value) VALUES ('normalize_enum_display', 'true');

                UPDATE schema_version SET version = 5;
            """)

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
                    bbox_x REAL, bbox_y REAL, bbox_w REAL, bbox_h REAL,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE(stash_image_id, performer_id)
                );
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_image ON image_fingerprint_faces(stash_image_id);
                CREATE INDEX IF NOT EXISTS idx_image_fp_faces_performer ON image_fingerprint_faces(performer_id);

                UPDATE schema_version SET version = 6;
            """)

        if from_version < 7:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS duplicate_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scene_a_id INTEGER NOT NULL,
                    scene_b_id INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    run_id INTEGER REFERENCES analysis_runs(id),
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(scene_a_id, scene_b_id)
                );
                CREATE INDEX IF NOT EXISTS idx_dup_candidates_run ON duplicate_candidates(run_id);

                UPDATE schema_version SET version = 7;
            """)

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ==================== Recommendations ====================

    def create_recommendation(
        self,
        type: str,
        target_type: str,
        target_id: str,
        details: dict,
        confidence: Optional[float] = None,
        source_analysis_id: Optional[int] = None,
    ) -> Optional[int]:
        """
        Create a recommendation. Returns ID if created, None if duplicate.
        """
        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO recommendations (
                        type, target_type, target_id, details, confidence, source_analysis_id
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (type, target_type, target_id, json.dumps(details), confidence, source_analysis_id)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # Already exists
                return None

    def get_recommendation(self, rec_id: int) -> Optional[Recommendation]:
        """Get a recommendation by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM recommendations WHERE id = ?", (rec_id,)
            ).fetchone()
            if row:
                return self._row_to_recommendation(row)
        return None

    def get_recommendations(
        self,
        status: Optional[str] = None,
        type: Optional[str] = None,
        target_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Recommendation]:
        """Get recommendations with optional filtering."""
        query = "SELECT * FROM recommendations WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        if target_type:
            query += " AND target_type = ?"
            params.append(target_type)

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [self._row_to_recommendation(row) for row in rows]

    def count_recommendations(self, status=None, type=None, target_type=None) -> int:
        """Count recommendations with optional filtering (for pagination totals)."""
        query = "SELECT COUNT(*) FROM recommendations WHERE 1=1"
        params = []
        if status:
            query += " AND status = ?"
            params.append(status)
        if type:
            query += " AND type = ?"
            params.append(type)
        if target_type:
            query += " AND target_type = ?"
            params.append(target_type)
        with self._connection() as conn:
            return conn.execute(query, params).fetchone()[0]

    def get_recommendation_by_target(
        self,
        type: str,
        target_type: str,
        target_id: str,
        status: Optional[str] = None,
    ) -> Optional[Recommendation]:
        """Get a recommendation by target (uses idx_rec_target index). Returns first match or None."""
        query = "SELECT * FROM recommendations WHERE type = ? AND target_type = ? AND target_id = ?"
        params: list = [type, target_type, target_id]
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " LIMIT 1"

        with self._connection() as conn:
            row = conn.execute(query, params).fetchone()
            if row:
                return self._row_to_recommendation(row)
        return None

    def get_recommendation_counts(self) -> dict[str, dict[str, int]]:
        """Get counts by type and status."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT type, status, COUNT(*) as count FROM recommendations GROUP BY type, status"
            ).fetchall()

            counts = {}
            for row in rows:
                if row['type'] not in counts:
                    counts[row['type']] = {}
                counts[row['type']][row['status']] = row['count']
            return counts

    def resolve_recommendation(
        self,
        rec_id: int,
        action: str,
        details: Optional[dict] = None,
    ) -> bool:
        """Mark a recommendation as resolved. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET status = 'resolved',
                    resolution_action = ?,
                    resolution_details = ?,
                    resolved_at = datetime('now'),
                    updated_at = datetime('now')
                WHERE id = ?
                """,
                (action, json.dumps(details) if details else None, rec_id)
            )
            return cursor.rowcount > 0

    def dismiss_recommendation(self, rec_id: int, reason: Optional[str] = None, permanent: bool = False) -> bool:
        """Dismiss a recommendation and add to dismissed_targets."""
        with self._connection() as conn:
            # Get the recommendation first
            row = conn.execute(
                "SELECT type, target_type, target_id FROM recommendations WHERE id = ?",
                (rec_id,)
            ).fetchone()

            if not row:
                return False

            # Mark as dismissed
            conn.execute(
                """
                UPDATE recommendations
                SET status = 'dismissed', updated_at = datetime('now')
                WHERE id = ?
                """,
                (rec_id,)
            )

            # Add to dismissed_targets to prevent re-recommendation
            try:
                conn.execute(
                    """
                    INSERT INTO dismissed_targets (type, target_type, target_id, reason, permanent)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (row['type'], row['target_type'], row['target_id'], reason, int(permanent))
                )
            except sqlite3.IntegrityError:
                pass  # Already dismissed

            return True

    def is_dismissed(self, type: str, target_type: str, target_id: str) -> bool:
        """Check if a target has been dismissed."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ?
                """,
                (type, target_type, target_id)
            ).fetchone()
            return row is not None

    def is_permanently_dismissed(self, type: str, target_type: str, target_id: str) -> bool:
        """Check if a target has been permanently dismissed."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ? AND permanent = 1
                """,
                (type, target_type, target_id)
            ).fetchone()
            return row is not None

    def undismiss(self, type: str, target_type: str, target_id: str):
        """Remove soft dismissals for a target (does not remove permanent dismissals)."""
        with self._connection() as conn:
            conn.execute(
                """
                DELETE FROM dismissed_targets
                WHERE type = ? AND target_type = ? AND target_id = ? AND permanent = 0
                """,
                (type, target_type, target_id)
            )

    def update_recommendation_details(self, rec_id: int, details: dict) -> bool:
        """Update details on a pending recommendation. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET details = ?, updated_at = datetime('now')
                WHERE id = ? AND status = 'pending'
                """,
                (json.dumps(details), rec_id)
            )
            return cursor.rowcount > 0

    def reopen_recommendation(self, rec_id: int, details: dict) -> bool:
        """Reopen a dismissed recommendation with new details. Returns True if updated."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE recommendations
                SET status = 'pending', details = ?,
                    resolution_action = NULL, resolution_details = NULL,
                    resolved_at = NULL, updated_at = datetime('now')
                WHERE id = ? AND status = 'dismissed'
                """,
                (json.dumps(details), rec_id)
            )
            return cursor.rowcount > 0

    def _row_to_recommendation(self, row: sqlite3.Row) -> Recommendation:
        """Convert a database row to a Recommendation object."""
        return Recommendation(
            id=row['id'],
            type=row['type'],
            status=row['status'],
            target_type=row['target_type'],
            target_id=row['target_id'],
            details=json.loads(row['details']),
            resolution_action=row['resolution_action'],
            resolution_details=json.loads(row['resolution_details']) if row['resolution_details'] else None,
            resolved_at=row['resolved_at'],
            confidence=row['confidence'],
            source_analysis_id=row['source_analysis_id'],
            created_at=row['created_at'],
            updated_at=row['updated_at'],
        )

    # ==================== Analysis Runs ====================

    def start_analysis_run(self, type: str, items_total: Optional[int] = None) -> int:
        """Start a new analysis run. Returns run ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_runs (type, status, started_at, items_total)
                VALUES (?, 'running', datetime('now'), ?)
                """,
                (type, items_total)
            )
            return cursor.lastrowid

    def update_analysis_progress(
        self,
        run_id: int,
        items_processed: int,
        recommendations_created: int,
        cursor: Optional[str] = None,
    ):
        """Update analysis run progress."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET items_processed = ?, recommendations_created = ?, cursor = ?
                WHERE id = ?
                """,
                (items_processed, recommendations_created, cursor, run_id)
            )

    def fail_stale_analysis_runs(self) -> int:
        """Mark any 'running' analysis runs as failed (e.g. after sidecar restart). Returns count."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'failed', completed_at = datetime('now'),
                    error_message = 'Sidecar restarted while analysis was running'
                WHERE status = 'running'
                """
            )
            return cursor.rowcount

    def update_analysis_items_total(self, run_id: int, items_total: int):
        """Update the total items count for an analysis run."""
        with self._connection() as conn:
            conn.execute(
                "UPDATE analysis_runs SET items_total = ? WHERE id = ?",
                (items_total, run_id)
            )

    def complete_analysis_run(self, run_id: int, recommendations_created: int):
        """Mark an analysis run as completed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'completed', completed_at = datetime('now'), recommendations_created = ?
                WHERE id = ?
                """,
                (recommendations_created, run_id)
            )

    def fail_analysis_run(self, run_id: int, error_message: str):
        """Mark an analysis run as failed."""
        with self._connection() as conn:
            conn.execute(
                """
                UPDATE analysis_runs
                SET status = 'failed', completed_at = datetime('now'), error_message = ?
                WHERE id = ?
                """,
                (error_message, run_id)
            )

    def get_analysis_run(self, run_id: int) -> Optional[AnalysisRun]:
        """Get an analysis run by ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row:
                return AnalysisRun(**dict(row))
        return None

    def get_recent_analysis_runs(self, type: Optional[str] = None, limit: int = 20) -> list[AnalysisRun]:
        """Get recent analysis runs."""
        query = "SELECT * FROM analysis_runs"
        params = []

        if type:
            query += " WHERE type = ?"
            params.append(type)

        query += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)

        with self._connection() as conn:
            rows = conn.execute(query, params).fetchall()
            return [AnalysisRun(**dict(row)) for row in rows]

    # ==================== Settings ====================

    def get_settings(self, type: str) -> Optional[RecommendationSettings]:
        """Get settings for a recommendation type."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM recommendation_settings WHERE type = ?", (type,)
            ).fetchone()
            if row:
                return RecommendationSettings(
                    type=row['type'],
                    enabled=bool(row['enabled']),
                    auto_dismiss_threshold=row['auto_dismiss_threshold'],
                    notify=bool(row['notify']),
                    interval_hours=row['interval_hours'],
                    last_run_at=row['last_run_at'],
                    next_run_at=row['next_run_at'],
                    config=json.loads(row['config']) if row['config'] else None,
                )
        return None

    def get_all_settings(self) -> list[RecommendationSettings]:
        """Get all recommendation settings."""
        with self._connection() as conn:
            rows = conn.execute("SELECT * FROM recommendation_settings").fetchall()
            return [
                RecommendationSettings(
                    type=row['type'],
                    enabled=bool(row['enabled']),
                    auto_dismiss_threshold=row['auto_dismiss_threshold'],
                    notify=bool(row['notify']),
                    interval_hours=row['interval_hours'],
                    last_run_at=row['last_run_at'],
                    next_run_at=row['next_run_at'],
                    config=json.loads(row['config']) if row['config'] else None,
                )
                for row in rows
            ]

    def upsert_settings(
        self,
        type: str,
        enabled: Optional[bool] = None,
        auto_dismiss_threshold: Optional[float] = None,
        notify: Optional[bool] = None,
        interval_hours: Optional[int] = None,
        config: Optional[dict] = None,
    ):
        """Create or update settings for a recommendation type."""
        with self._connection() as conn:
            # Check if exists
            existing = conn.execute(
                "SELECT 1 FROM recommendation_settings WHERE type = ?", (type,)
            ).fetchone()

            if existing:
                updates = []
                params = []
                if enabled is not None:
                    updates.append("enabled = ?")
                    params.append(int(enabled))
                if auto_dismiss_threshold is not None:
                    updates.append("auto_dismiss_threshold = ?")
                    params.append(auto_dismiss_threshold)
                if notify is not None:
                    updates.append("notify = ?")
                    params.append(int(notify))
                if interval_hours is not None:
                    updates.append("interval_hours = ?")
                    params.append(interval_hours)
                if config is not None:
                    updates.append("config = ?")
                    params.append(json.dumps(config))

                if updates:
                    params.append(type)
                    conn.execute(
                        f"UPDATE recommendation_settings SET {', '.join(updates)} WHERE type = ?",
                        params
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO recommendation_settings (type, enabled, auto_dismiss_threshold, notify, interval_hours, config)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (type, int(enabled) if enabled is not None else 1,
                     auto_dismiss_threshold, int(notify) if notify is not None else 1,
                     interval_hours, json.dumps(config) if config else None)
                )

    # ==================== Watermarks ====================

    def get_watermark(self, type: str) -> Optional[dict]:
        """Get analysis watermark for incremental runs."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM analysis_watermarks WHERE type = ?", (type,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def set_watermark(
        self,
        type: str,
        last_cursor: Optional[str] = None,
        last_stash_updated_at: Optional[str] = None,
    ):
        """Update analysis watermark."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO analysis_watermarks (type, last_completed_at, last_cursor, last_stash_updated_at)
                VALUES (?, datetime('now'), ?, ?)
                ON CONFLICT(type) DO UPDATE SET
                    last_completed_at = datetime('now'),
                    last_cursor = COALESCE(?, last_cursor),
                    last_stash_updated_at = COALESCE(?, last_stash_updated_at)
                """,
                (type, last_cursor, last_stash_updated_at, last_cursor, last_stash_updated_at)
            )

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
        """
        Create or update an upstream snapshot. Returns the snapshot ID.
        Uses upsert on the unique constraint (entity_type, endpoint, stash_box_id).
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO upstream_snapshots (
                    entity_type, local_entity_id, endpoint, stash_box_id,
                    upstream_data, upstream_updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(entity_type, endpoint, stash_box_id) DO UPDATE SET
                    local_entity_id = excluded.local_entity_id,
                    upstream_data = excluded.upstream_data,
                    upstream_updated_at = excluded.upstream_updated_at,
                    fetched_at = datetime('now')
                RETURNING id
                """,
                (entity_type, local_entity_id, endpoint, stash_box_id,
                 json.dumps(upstream_data), upstream_updated_at)
            )
            return cursor.fetchone()[0]

    def get_upstream_snapshot(
        self,
        entity_type: str,
        endpoint: str,
        stash_box_id: str,
    ) -> Optional[dict]:
        """Get an upstream snapshot by its unique key. Returns dict with parsed upstream_data, or None."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT * FROM upstream_snapshots
                WHERE entity_type = ? AND endpoint = ? AND stash_box_id = ?
                """,
                (entity_type, endpoint, stash_box_id)
            ).fetchone()
            if row:
                result = dict(row)
                result["upstream_data"] = json.loads(result["upstream_data"])
                return result
        return None

    # ==================== Upstream Field Config ====================

    def get_enabled_fields(self, endpoint: str, entity_type: str) -> Optional[set[str]]:
        """
        Get the set of enabled field names for an endpoint/entity_type.
        Returns None if no config exists (caller should use defaults).
        """
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT field_name, enabled FROM upstream_field_config
                WHERE endpoint = ? AND entity_type = ?
                """,
                (endpoint, entity_type)
            ).fetchall()
            if not rows:
                return None
            return {row["field_name"] for row in rows if row["enabled"]}

    def set_field_config(self, endpoint: str, entity_type: str, field_configs: dict[str, bool]):
        """
        Set field monitoring configuration for an endpoint/entity_type.
        Replaces all existing config for this endpoint/entity_type.
        field_configs maps field_name -> enabled bool.
        """
        with self._connection() as conn:
            # Delete existing config for this endpoint/entity_type
            conn.execute(
                """
                DELETE FROM upstream_field_config
                WHERE endpoint = ? AND entity_type = ?
                """,
                (endpoint, entity_type)
            )
            # Insert new config rows
            for field_name, enabled in field_configs.items():
                conn.execute(
                    """
                    INSERT INTO upstream_field_config (endpoint, entity_type, field_name, enabled)
                    VALUES (?, ?, ?, ?)
                    """,
                    (endpoint, entity_type, field_name, int(enabled))
                )

    # ==================== User Settings ====================

    def get_user_setting(self, key: str) -> Optional[Any]:
        """Get a user setting by key. Returns the parsed JSON value, or None if not found."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT value FROM user_settings WHERE key = ?", (key,)
            ).fetchone()
            if row:
                return json.loads(row["value"])
        return None

    def set_user_setting(self, key: str, value: Any):
        """Set a user setting. Creates or updates."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO user_settings (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value = excluded.value,
                    updated_at = datetime('now')
                """,
                (key, json.dumps(value))
            )

    def get_all_user_settings(self) -> dict[str, Any]:
        """Get all user settings as a dict."""
        with self._connection() as conn:
            rows = conn.execute("SELECT key, value FROM user_settings").fetchall()
            return {row["key"]: json.loads(row["value"]) for row in rows}

    # ==================== Scene Fingerprints ====================

    def create_scene_fingerprint(
        self,
        stash_scene_id: int,
        total_faces: int,
        frames_analyzed: int,
        fingerprint_status: str = "pending",
        db_version: Optional[str] = None,
    ) -> int:
        """
        Create or update a scene fingerprint. Returns the fingerprint ID.
        Uses upsert - if fingerprint exists for scene, updates it.

        Args:
            stash_scene_id: The Stash scene ID
            total_faces: Total faces detected in the scene
            frames_analyzed: Number of frames analyzed
            fingerprint_status: Status ('pending', 'complete', 'error')
            db_version: Face recognition DB version used for this fingerprint
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scene_fingerprints (stash_scene_id, total_faces, frames_analyzed, fingerprint_status, db_version)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(stash_scene_id) DO UPDATE SET
                    total_faces = excluded.total_faces,
                    frames_analyzed = excluded.frames_analyzed,
                    fingerprint_status = excluded.fingerprint_status,
                    db_version = excluded.db_version,
                    updated_at = datetime('now')
                RETURNING id
                """,
                (stash_scene_id, total_faces, frames_analyzed, fingerprint_status, db_version)
            )
            return cursor.fetchone()[0]

    def get_scene_fingerprint(self, stash_scene_id: int) -> Optional[dict]:
        """Get a scene fingerprint by stash scene ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM scene_fingerprints WHERE stash_scene_id = ?",
                (stash_scene_id,)
            ).fetchone()
            if row:
                return dict(row)
        return None

    def get_all_scene_fingerprints(self, status: Optional[str] = None) -> list[dict]:
        """Get all scene fingerprints, optionally filtered by status."""
        with self._connection() as conn:
            if status is not None:
                rows = conn.execute(
                    "SELECT * FROM scene_fingerprints WHERE fingerprint_status = ?",
                    (status,)
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM scene_fingerprints").fetchall()
            return [dict(row) for row in rows]

    def add_fingerprint_face(
        self,
        fingerprint_id: int,
        performer_id: str,
        face_count: int,
        avg_confidence: Optional[float] = None,
        proportion: Optional[float] = None,
    ) -> int:
        """Add or update a face entry in a scene fingerprint. Returns the face entry ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO scene_fingerprint_faces (fingerprint_id, performer_id, face_count, avg_confidence, proportion)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint_id, performer_id) DO UPDATE SET
                    face_count = excluded.face_count,
                    avg_confidence = excluded.avg_confidence,
                    proportion = excluded.proportion
                RETURNING id
                """,
                (fingerprint_id, performer_id, face_count, avg_confidence, proportion)
            )
            return cursor.fetchone()[0]

    def get_fingerprint_faces(self, fingerprint_id: int) -> list[dict]:
        """Get all face entries for a scene fingerprint."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM scene_fingerprint_faces WHERE fingerprint_id = ?",
                (fingerprint_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def delete_fingerprint_faces(self, fingerprint_id: int) -> int:
        """Delete all face entries for a scene fingerprint. Returns count deleted."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM scene_fingerprint_faces WHERE fingerprint_id = ?",
                (fingerprint_id,)
            )
            return cursor.rowcount

    def get_fingerprints_needing_refresh(self, current_db_version: str) -> list[dict]:
        """Get fingerprints that were generated with an older DB version."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM scene_fingerprints
                WHERE db_version IS NULL OR db_version != ?
                ORDER BY stash_scene_id
                """,
                (current_db_version,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_scene_ids_without_fingerprints(self, scene_ids: list[int]) -> list[int]:
        """Given a list of scene IDs, return those without fingerprints."""
        if not scene_ids:
            return []
        with self._connection() as conn:
            placeholders = ",".join("?" * len(scene_ids))
            rows = conn.execute(
                f"""
                SELECT stash_scene_id FROM scene_fingerprints
                WHERE stash_scene_id IN ({placeholders})
                """,
                scene_ids
            ).fetchall()
            existing = {row[0] for row in rows}
            return [sid for sid in scene_ids if sid not in existing]

    def get_fingerprint_stats(self, current_db_version: Optional[str] = None) -> dict:
        """Get fingerprint coverage statistics."""
        with self._connection() as conn:
            stats = {}
            stats['total_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints"
            ).fetchone()[0]
            stats['complete_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'complete'"
            ).fetchone()[0]
            stats['pending_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'pending'"
            ).fetchone()[0]
            stats['error_fingerprints'] = conn.execute(
                "SELECT COUNT(*) FROM scene_fingerprints WHERE fingerprint_status = 'error'"
            ).fetchone()[0]

            if current_db_version:
                stats['current_version_count'] = conn.execute(
                    "SELECT COUNT(*) FROM scene_fingerprints WHERE db_version = ?",
                    (current_db_version,)
                ).fetchone()[0]
                stats['needs_refresh_count'] = conn.execute(
                    "SELECT COUNT(*) FROM scene_fingerprints WHERE db_version IS NULL OR db_version != ?",
                    (current_db_version,)
                ).fetchone()[0]

            return stats

    def mark_fingerprints_for_refresh(self, scene_ids: Optional[list[int]] = None) -> int:
        """
        Mark fingerprints for refresh by clearing their db_version.
        If scene_ids is None, marks all fingerprints.
        Returns count of fingerprints marked.
        """
        with self._connection() as conn:
            if scene_ids is None:
                cursor = conn.execute(
                    "UPDATE scene_fingerprints SET db_version = NULL, updated_at = datetime('now')"
                )
            else:
                placeholders = ",".join("?" * len(scene_ids))
                cursor = conn.execute(
                    f"""
                    UPDATE scene_fingerprints
                    SET db_version = NULL, updated_at = datetime('now')
                    WHERE stash_scene_id IN ({placeholders})
                    """,
                    scene_ids
                )
            return cursor.rowcount

    # ==================== Image Fingerprints ====================

    def create_image_fingerprint(
        self,
        stash_image_id: str,
        gallery_id: Optional[str] = None,
        faces_detected: int = 0,
        db_version: Optional[str] = None,
    ) -> int:
        """
        Create or update an image fingerprint. Returns the fingerprint ID.
        Uses upsert - if fingerprint exists for image, updates it.

        Args:
            stash_image_id: The Stash image ID
            gallery_id: The gallery this image belongs to (optional)
            faces_detected: Number of faces detected in the image
            db_version: Face recognition DB version used for this fingerprint
        """
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

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}
            stats['total_recommendations'] = conn.execute(
                "SELECT COUNT(*) FROM recommendations"
            ).fetchone()[0]
            stats['pending_recommendations'] = conn.execute(
                "SELECT COUNT(*) FROM recommendations WHERE status = 'pending'"
            ).fetchone()[0]
            stats['dismissed_count'] = conn.execute(
                "SELECT COUNT(*) FROM dismissed_targets"
            ).fetchone()[0]
            stats['analysis_runs_today'] = conn.execute(
                "SELECT COUNT(*) FROM analysis_runs WHERE date(started_at) = date('now')"
            ).fetchone()[0]
            return stats


# Convenience function
def open_recommendations_db(path: str | Path) -> RecommendationsDB:
    """Open or create a recommendations database."""
    return RecommendationsDB(path)

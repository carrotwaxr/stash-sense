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


SCHEMA_VERSION = 1


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
        conn.executescript("""
            -- Schema version tracking
            CREATE TABLE schema_version (
                version INTEGER PRIMARY KEY
            );
            INSERT INTO schema_version (version) VALUES (1);

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
                PRIMARY KEY (type, target_type, target_id)
            );

            -- Track analysis watermarks for incremental runs
            CREATE TABLE analysis_watermarks (
                type TEXT PRIMARY KEY,
                last_completed_at TEXT,
                last_cursor TEXT,
                last_stash_updated_at TEXT
            );
        """)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Migrate schema from older version."""
        # No migrations yet - version 1
        pass

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

    def dismiss_recommendation(self, rec_id: int, reason: Optional[str] = None) -> bool:
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
                    INSERT INTO dismissed_targets (type, target_type, target_id, reason)
                    VALUES (?, ?, ?, ?)
                    """,
                    (row['type'], row['target_type'], row['target_id'], reason)
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

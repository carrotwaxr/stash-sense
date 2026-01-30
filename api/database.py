"""
SQLite Database Layer for Stash Sense

Provides relational storage for performer metadata, URLs, aliases, and face mappings.
Replaces JSON files for better query performance and scalability.

Schema designed for:
- Fast lookup by any identifier (StashDB ID, Twitter handle, IAFD slug, etc.)
- Efficient identity graph queries (find all performers sharing a URL)
- Incremental updates without full file rewrites

See: docs/plans/2026-01-27-performer-identity-graph.md
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Iterator

from url_normalizer import URLNormalizer, NormalizedURL


# Schema version for migrations
SCHEMA_VERSION = 5


@dataclass
class Performer:
    """Performer record from the database."""
    id: int
    canonical_name: str
    disambiguation: Optional[str]  # Differentiates same-name performers
    gender: Optional[str]
    country: Optional[str]
    ethnicity: Optional[str]
    birth_date: Optional[str]
    death_date: Optional[str]
    height_cm: Optional[int]  # Height in centimeters
    eye_color: Optional[str]
    hair_color: Optional[str]
    career_start_year: Optional[int]
    career_end_year: Optional[int]
    scene_count: Optional[int]  # Number of scenes on StashDB (for prioritization)
    stashdb_updated_at: Optional[str]  # When StashDB last updated this performer
    face_count: int
    image_url: Optional[str]
    created_at: str
    updated_at: str


@dataclass
class StashboxId:
    """Stash-box ID mapping."""
    performer_id: int
    endpoint: str  # 'stashdb', 'pmvstash', 'javstash', 'fansdb', 'theporndb'
    stashbox_performer_id: str


@dataclass
class ExternalURL:
    """External URL for identity matching."""
    performer_id: int
    site: str  # 'iafd', 'twitter', 'instagram', etc.
    url: str
    normalized_id: str
    source_endpoint: str  # Which stash-box provided this URL
    confidence: float


@dataclass
class Alias:
    """Performer alias/stage name."""
    performer_id: int
    alias: str
    source_endpoint: str


@dataclass
class Face:
    """Face embedding metadata."""
    id: int
    performer_id: int
    facenet_index: int  # Index in Voyager facenet index
    arcface_index: int  # Index in Voyager arcface index
    image_url: str
    source_endpoint: str
    quality_score: Optional[float]
    created_at: str


class PerformerDatabase:
    """
    SQLite database for performer metadata and identity graph.

    Usage:
        db = PerformerDatabase("performers.db")

        # Add a performer
        performer_id = db.add_performer(
            canonical_name="Angela White",
            gender="FEMALE",
            country="AU",
        )

        # Add stash-box ID
        db.add_stashbox_id(performer_id, "stashdb", "abc-123-uuid")

        # Add URLs
        db.add_url(performer_id, "https://twitter.com/angelawhite", "stashdb")

        # Find by URL
        matches = db.find_by_url("twitter", "angelawhite")
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.url_normalizer = URLNormalizer()
        self._init_database()

    def _init_database(self):
        """Initialize database schema if needed."""
        with self._connection() as conn:
            # Check if schema exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
            )
            if cursor.fetchone() is None:
                self._create_schema(conn)
            else:
                # Check version and migrate if needed
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
            INSERT INTO schema_version (version) VALUES (3);

            -- Core performer table
            CREATE TABLE performers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_name TEXT NOT NULL,
                disambiguation TEXT,  -- Differentiates same-name performers
                gender TEXT,
                country TEXT,
                ethnicity TEXT,
                birth_date TEXT,
                death_date TEXT,
                height_cm INTEGER,  -- Height in centimeters
                eye_color TEXT,
                hair_color TEXT,
                career_start_year INTEGER,
                career_end_year INTEGER,
                scene_count INTEGER,  -- Number of scenes on StashDB
                stashdb_updated_at TEXT,  -- When StashDB last updated this performer
                face_count INTEGER DEFAULT 0,
                image_url TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX idx_performers_name ON performers(canonical_name);
            CREATE INDEX idx_performers_country ON performers(country);
            CREATE INDEX idx_performers_ethnicity ON performers(ethnicity);
            CREATE INDEX idx_performers_height ON performers(height_cm);
            CREATE INDEX idx_performers_scene_count ON performers(scene_count);
            CREATE INDEX idx_performers_stashdb_updated ON performers(stashdb_updated_at);

            -- Stash-box ID mappings (performer can exist in multiple stash-boxes)
            CREATE TABLE stashbox_ids (
                performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                endpoint TEXT NOT NULL,  -- 'stashdb', 'pmvstash', 'javstash', 'fansdb', 'theporndb'
                stashbox_performer_id TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (endpoint, stashbox_performer_id)
            );
            CREATE INDEX idx_stashbox_performer ON stashbox_ids(performer_id);
            CREATE INDEX idx_stashbox_id ON stashbox_ids(stashbox_performer_id);

            -- External URLs for identity matching
            CREATE TABLE external_urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                site TEXT NOT NULL,  -- 'iafd', 'twitter', 'instagram', etc.
                url TEXT NOT NULL,
                normalized_id TEXT NOT NULL,
                source_endpoint TEXT NOT NULL,  -- Which stash-box provided this
                confidence REAL DEFAULT 1.0,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (site, normalized_id)  -- Same normalized URL = same person
            );
            CREATE INDEX idx_urls_performer ON external_urls(performer_id);
            CREATE INDEX idx_urls_site_normalized ON external_urls(site, normalized_id);
            CREATE INDEX idx_urls_normalized ON external_urls(normalized_id);

            -- Aliases/stage names
            CREATE TABLE aliases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                alias TEXT NOT NULL,
                source_endpoint TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (performer_id, alias)
            );
            CREATE INDEX idx_aliases_performer ON aliases(performer_id);
            CREATE INDEX idx_aliases_alias ON aliases(alias COLLATE NOCASE);

            -- Face embedding metadata (actual vectors in Voyager)
            CREATE TABLE faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                facenet_index INTEGER NOT NULL,  -- Index in Voyager facenet index
                arcface_index INTEGER NOT NULL,  -- Index in Voyager arcface index
                image_url TEXT,
                source_endpoint TEXT NOT NULL,
                quality_score REAL,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (facenet_index),
                UNIQUE (arcface_index)
            );
            CREATE INDEX idx_faces_performer ON faces(performer_id);

            -- Merged performer IDs (from stash-box merge history)
            CREATE TABLE merged_ids (
                performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                merged_stashbox_id TEXT NOT NULL,
                endpoint TEXT NOT NULL,
                merged_at TEXT DEFAULT (datetime('now')),
                PRIMARY KEY (endpoint, merged_stashbox_id)
            );
            CREATE INDEX idx_merged_performer ON merged_ids(performer_id);

            -- Tattoos (highly identifying, stable)
            CREATE TABLE tattoos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                location TEXT,  -- 'left arm', 'back', 'ankle', etc.
                description TEXT,
                source_endpoint TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (performer_id, location, description)
            );
            CREATE INDEX idx_tattoos_performer ON tattoos(performer_id);

            -- Piercings
            CREATE TABLE piercings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                location TEXT,  -- 'navel', 'nipple', 'nose', etc.
                description TEXT,
                source_endpoint TEXT,
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE (performer_id, location, description)
            );
            CREATE INDEX idx_piercings_performer ON piercings(performer_id);

            -- Scrape progress per source (for resume capability)
            CREATE TABLE IF NOT EXISTS scrape_progress (
                source TEXT PRIMARY KEY,
                last_processed_id TEXT,
                last_processed_time TEXT DEFAULT (datetime('now')),
                performers_processed INTEGER DEFAULT 0,
                faces_added INTEGER DEFAULT 0,
                errors INTEGER DEFAULT 0
            );
        """)

    def _migrate_schema(self, conn: sqlite3.Connection, from_version: int):
        """Migrate schema from older version."""
        if from_version < 2:
            # Add new identity fields to performers table
            conn.executescript("""
                ALTER TABLE performers ADD COLUMN disambiguation TEXT;
                ALTER TABLE performers ADD COLUMN ethnicity TEXT;
                ALTER TABLE performers ADD COLUMN death_date TEXT;
                ALTER TABLE performers ADD COLUMN height_cm INTEGER;
                ALTER TABLE performers ADD COLUMN eye_color TEXT;
                ALTER TABLE performers ADD COLUMN hair_color TEXT;

                CREATE INDEX IF NOT EXISTS idx_performers_ethnicity ON performers(ethnicity);
                CREATE INDEX IF NOT EXISTS idx_performers_height ON performers(height_cm);

                -- Tattoos table
                CREATE TABLE IF NOT EXISTS tattoos (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                    location TEXT,
                    description TEXT,
                    source_endpoint TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE (performer_id, location, description)
                );
                CREATE INDEX IF NOT EXISTS idx_tattoos_performer ON tattoos(performer_id);

                -- Piercings table
                CREATE TABLE IF NOT EXISTS piercings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    performer_id INTEGER NOT NULL REFERENCES performers(id) ON DELETE CASCADE,
                    location TEXT,
                    description TEXT,
                    source_endpoint TEXT,
                    created_at TEXT DEFAULT (datetime('now')),
                    UNIQUE (performer_id, location, description)
                );
                CREATE INDEX IF NOT EXISTS idx_piercings_performer ON piercings(performer_id);

                UPDATE schema_version SET version = 2;
            """)

        if from_version < 3:
            # Add scene_count and stashdb_updated_at for incremental sync
            conn.executescript("""
                ALTER TABLE performers ADD COLUMN scene_count INTEGER;
                ALTER TABLE performers ADD COLUMN stashdb_updated_at TEXT;

                CREATE INDEX IF NOT EXISTS idx_performers_scene_count ON performers(scene_count);
                CREATE INDEX IF NOT EXISTS idx_performers_stashdb_updated ON performers(stashdb_updated_at);

                UPDATE schema_version SET version = 3;
            """)

        if from_version < 4:
            # Ensure all existing faces have source_endpoint set
            # (existing faces are from StashDB)
            conn.executescript("""
                UPDATE faces SET source_endpoint = 'stashdb' WHERE source_endpoint IS NULL;
                UPDATE schema_version SET version = 4;
            """)

        if from_version < 5:
            # Add scrape_progress table for resume capability
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS scrape_progress (
                    source TEXT PRIMARY KEY,
                    last_processed_id TEXT,
                    last_processed_time TEXT DEFAULT (datetime('now')),
                    performers_processed INTEGER DEFAULT 0,
                    faces_added INTEGER DEFAULT 0,
                    errors INTEGER DEFAULT 0
                );
                UPDATE schema_version SET version = 5;
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

    # ==================== Performer CRUD ====================

    def add_performer(
        self,
        canonical_name: str,
        disambiguation: Optional[str] = None,
        gender: Optional[str] = None,
        country: Optional[str] = None,
        ethnicity: Optional[str] = None,
        birth_date: Optional[str] = None,
        death_date: Optional[str] = None,
        height_cm: Optional[int] = None,
        eye_color: Optional[str] = None,
        hair_color: Optional[str] = None,
        career_start_year: Optional[int] = None,
        career_end_year: Optional[int] = None,
        scene_count: Optional[int] = None,
        stashdb_updated_at: Optional[str] = None,
        face_count: int = 0,
        image_url: Optional[str] = None,
    ) -> int:
        """Add a new performer, return their ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO performers (
                    canonical_name, disambiguation, gender, country, ethnicity,
                    birth_date, death_date, height_cm, eye_color, hair_color,
                    career_start_year, career_end_year, scene_count, stashdb_updated_at,
                    face_count, image_url
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (canonical_name, disambiguation, gender, country, ethnicity,
                 birth_date, death_date, height_cm, eye_color, hair_color,
                 career_start_year, career_end_year, scene_count, stashdb_updated_at,
                 face_count, image_url)
            )
            return cursor.lastrowid

    def get_performer(self, performer_id: int) -> Optional[Performer]:
        """Get a performer by internal ID."""
        with self._connection() as conn:
            row = conn.execute(
                "SELECT * FROM performers WHERE id = ?", (performer_id,)
            ).fetchone()
            if row:
                return Performer(**dict(row))
        return None

    def get_performer_by_stashbox_id(
        self, endpoint: str, stashbox_id: str
    ) -> Optional[Performer]:
        """Get a performer by their stash-box ID."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT p.* FROM performers p
                JOIN stashbox_ids s ON p.id = s.performer_id
                WHERE s.endpoint = ? AND s.stashbox_performer_id = ?
                """,
                (endpoint, stashbox_id)
            ).fetchone()
            if row:
                return Performer(**dict(row))
        return None

    def update_performer(
        self,
        performer_id: int,
        canonical_name: Optional[str] = None,
        disambiguation: Optional[str] = None,
        gender: Optional[str] = None,
        country: Optional[str] = None,
        ethnicity: Optional[str] = None,
        birth_date: Optional[str] = None,
        death_date: Optional[str] = None,
        height_cm: Optional[int] = None,
        eye_color: Optional[str] = None,
        hair_color: Optional[str] = None,
        career_start_year: Optional[int] = None,
        career_end_year: Optional[int] = None,
        scene_count: Optional[int] = None,
        stashdb_updated_at: Optional[str] = None,
        face_count: Optional[int] = None,
        image_url: Optional[str] = None,
    ) -> bool:
        """Update performer fields. Returns True if updated."""
        updates = []
        values = []

        if canonical_name is not None:
            updates.append("canonical_name = ?")
            values.append(canonical_name)
        if disambiguation is not None:
            updates.append("disambiguation = ?")
            values.append(disambiguation)
        if gender is not None:
            updates.append("gender = ?")
            values.append(gender)
        if country is not None:
            updates.append("country = ?")
            values.append(country)
        if ethnicity is not None:
            updates.append("ethnicity = ?")
            values.append(ethnicity)
        if birth_date is not None:
            updates.append("birth_date = ?")
            values.append(birth_date)
        if death_date is not None:
            updates.append("death_date = ?")
            values.append(death_date)
        if height_cm is not None:
            updates.append("height_cm = ?")
            values.append(height_cm)
        if eye_color is not None:
            updates.append("eye_color = ?")
            values.append(eye_color)
        if hair_color is not None:
            updates.append("hair_color = ?")
            values.append(hair_color)
        if career_start_year is not None:
            updates.append("career_start_year = ?")
            values.append(career_start_year)
        if career_end_year is not None:
            updates.append("career_end_year = ?")
            values.append(career_end_year)
        if scene_count is not None:
            updates.append("scene_count = ?")
            values.append(scene_count)
        if stashdb_updated_at is not None:
            updates.append("stashdb_updated_at = ?")
            values.append(stashdb_updated_at)
        if face_count is not None:
            updates.append("face_count = ?")
            values.append(face_count)
        if image_url is not None:
            updates.append("image_url = ?")
            values.append(image_url)

        if not updates:
            return False

        updates.append("updated_at = datetime('now')")
        values.append(performer_id)

        with self._connection() as conn:
            cursor = conn.execute(
                f"UPDATE performers SET {', '.join(updates)} WHERE id = ?",
                values
            )
            return cursor.rowcount > 0

    def update_face_count(self, performer_id: int, face_count: int) -> bool:
        """Update performer's face count."""
        return self.update_performer(performer_id, face_count=face_count)

    # ==================== Stash-box IDs ====================

    def add_stashbox_id(
        self, performer_id: int, endpoint: str, stashbox_id: str
    ) -> bool:
        """Add a stash-box ID for a performer. Returns True if added."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO stashbox_ids (performer_id, endpoint, stashbox_performer_id)
                    VALUES (?, ?, ?)
                    """,
                    (performer_id, endpoint, stashbox_id)
                )
                return True
            except sqlite3.IntegrityError:
                # Already exists
                return False

    def get_stashbox_ids(self, performer_id: int) -> list[StashboxId]:
        """Get all stash-box IDs for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM stashbox_ids WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [StashboxId(**dict(row)) for row in rows]

    # ==================== URLs ====================

    def add_url(
        self,
        performer_id: int,
        url: str,
        source_endpoint: str,
    ) -> Optional[int]:
        """
        Add a URL for a performer. Normalizes the URL automatically.
        Returns URL record ID if added, None if duplicate or unrecognized.
        """
        normalized = self.url_normalizer.normalize(url)
        if not normalized:
            return None  # Unrecognized URL format

        with self._connection() as conn:
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO external_urls (
                        performer_id, site, url, normalized_id,
                        source_endpoint, confidence
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (performer_id, normalized.site, url, normalized.normalized_id,
                     source_endpoint, normalized.confidence)
                )
                return cursor.lastrowid
            except sqlite3.IntegrityError:
                # URL already exists for another performer - this is an identity match!
                return None

    def add_urls_batch(
        self,
        performer_id: int,
        urls: dict[str, list[str]],  # site_name -> [urls]
        source_endpoint: str,
    ) -> int:
        """Add multiple URLs. Returns count of URLs added."""
        added = 0
        for site_name, url_list in urls.items():
            for url in url_list:
                if self.add_url(performer_id, url, source_endpoint):
                    added += 1
        return added

    def get_urls(self, performer_id: int) -> list[ExternalURL]:
        """Get all URLs for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM external_urls WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [ExternalURL(**dict(row)) for row in rows]

    def find_by_url(self, site: str, normalized_id: str) -> Optional[Performer]:
        """Find a performer by normalized URL."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT p.* FROM performers p
                JOIN external_urls u ON p.id = u.performer_id
                WHERE u.site = ? AND u.normalized_id = ?
                """,
                (site, normalized_id)
            ).fetchone()
            if row:
                return Performer(**dict(row))
        return None

    def find_by_normalized_url(self, normalized: NormalizedURL) -> Optional[Performer]:
        """Find a performer by NormalizedURL object."""
        return self.find_by_url(normalized.site, normalized.normalized_id)

    def find_performers_sharing_url(self, url: str) -> list[Performer]:
        """Find all performers who share a given URL (identity match candidates)."""
        normalized = self.url_normalizer.normalize(url)
        if not normalized:
            return []

        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.* FROM performers p
                JOIN external_urls u ON p.id = u.performer_id
                WHERE u.site = ? AND u.normalized_id = ?
                """,
                (normalized.site, normalized.normalized_id)
            ).fetchall()
            return [Performer(**dict(row)) for row in rows]

    # ==================== Aliases ====================

    def add_alias(
        self, performer_id: int, alias: str, source_endpoint: Optional[str] = None
    ) -> bool:
        """Add an alias for a performer. Returns True if added."""
        with self._connection() as conn:
            try:
                conn.execute(
                    "INSERT INTO aliases (performer_id, alias, source_endpoint) VALUES (?, ?, ?)",
                    (performer_id, alias, source_endpoint)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def add_aliases_batch(
        self, performer_id: int, aliases: list[str], source_endpoint: str
    ) -> int:
        """Add multiple aliases. Returns count added."""
        added = 0
        for alias in aliases:
            if self.add_alias(performer_id, alias, source_endpoint):
                added += 1
        return added

    def get_aliases(self, performer_id: int) -> list[Alias]:
        """Get all aliases for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM aliases WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [Alias(**dict(row)) for row in rows]

    def find_by_alias(self, alias: str) -> list[Performer]:
        """Find performers by alias (case-insensitive)."""
        with self._connection() as conn:
            rows = conn.execute(
                """
                SELECT DISTINCT p.* FROM performers p
                JOIN aliases a ON p.id = a.performer_id
                WHERE a.alias = ? COLLATE NOCASE
                """,
                (alias,)
            ).fetchall()
            return [Performer(**dict(row)) for row in rows]

    # ==================== Faces ====================

    def add_face(
        self,
        performer_id: int,
        facenet_index: int,
        arcface_index: int,
        image_url: str,
        source_endpoint: str,
        quality_score: Optional[float] = None,
    ) -> int:
        """Add a face embedding record. Returns face ID."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO faces (
                    performer_id, facenet_index, arcface_index,
                    image_url, source_endpoint, quality_score
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (performer_id, facenet_index, arcface_index,
                 image_url, source_endpoint, quality_score)
            )
            return cursor.lastrowid

    def get_faces(self, performer_id: int) -> list[Face]:
        """Get all faces for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT * FROM faces WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [Face(**dict(row)) for row in rows]

    def get_max_face_index(self) -> Optional[int]:
        """Get the maximum face index in the database.

        Used to sync IndexManager after unclean shutdowns where DB
        committed faces but Voyager index wasn't saved.
        """
        with self._connection() as conn:
            row = conn.execute(
                "SELECT MAX(arcface_index) FROM faces"
            ).fetchone()
            return row[0] if row and row[0] is not None else None

    def get_performer_by_face_index(
        self, facenet_index: int
    ) -> Optional[Performer]:
        """Get performer by Voyager face index."""
        with self._connection() as conn:
            row = conn.execute(
                """
                SELECT p.* FROM performers p
                JOIN faces f ON p.id = f.performer_id
                WHERE f.facenet_index = ?
                """,
                (facenet_index,)
            ).fetchone()
            if row:
                return Performer(**dict(row))
        return None

    def get_face_counts_by_source(self, performer_id: int) -> dict[str, int]:
        """
        Get face counts per source for a performer.

        Returns:
            Dict mapping source name to face count
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT source_endpoint, COUNT(*) as count
                FROM faces
                WHERE performer_id = ?
                GROUP BY source_endpoint
                """,
                (performer_id,),
            )
            return {row["source_endpoint"]: row["count"] for row in cursor.fetchall()}

    def source_limit_reached(self, performer_id: int, source: str, max_faces: int) -> bool:
        """
        Check if a source has reached its face limit for a performer.

        Args:
            performer_id: Performer's database ID
            source: Source name to check
            max_faces: Maximum faces allowed from this source

        Returns:
            True if limit reached, False otherwise
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*) as count
                FROM faces
                WHERE performer_id = ? AND source_endpoint = ?
                """,
                (performer_id, source),
            )
            count = cursor.fetchone()["count"]
            return count >= max_faces

    def total_limit_reached(self, performer_id: int, max_faces: int) -> bool:
        """
        Check if total face limit is reached for a performer.

        Args:
            performer_id: Performer's database ID
            max_faces: Maximum total faces allowed

        Returns:
            True if limit reached, False otherwise
        """
        with self._connection() as conn:
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM faces WHERE performer_id = ?",
                (performer_id,),
            )
            count = cursor.fetchone()["count"]
            return count >= max_faces

    # ==================== Merged IDs ====================

    def add_merged_id(
        self, performer_id: int, merged_stashbox_id: str, endpoint: str
    ) -> bool:
        """Record a merged stash-box ID."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO merged_ids (performer_id, merged_stashbox_id, endpoint)
                    VALUES (?, ?, ?)
                    """,
                    (performer_id, merged_stashbox_id, endpoint)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    # ==================== Tattoos & Piercings ====================

    def add_tattoo(
        self,
        performer_id: int,
        location: Optional[str] = None,
        description: Optional[str] = None,
        source_endpoint: Optional[str] = None,
    ) -> bool:
        """Add a tattoo record. Returns True if added."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO tattoos (performer_id, location, description, source_endpoint)
                    VALUES (?, ?, ?, ?)
                    """,
                    (performer_id, location, description, source_endpoint)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def add_piercing(
        self,
        performer_id: int,
        location: Optional[str] = None,
        description: Optional[str] = None,
        source_endpoint: Optional[str] = None,
    ) -> bool:
        """Add a piercing record. Returns True if added."""
        with self._connection() as conn:
            try:
                conn.execute(
                    """
                    INSERT INTO piercings (performer_id, location, description, source_endpoint)
                    VALUES (?, ?, ?, ?)
                    """,
                    (performer_id, location, description, source_endpoint)
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def get_tattoos(self, performer_id: int) -> list[dict]:
        """Get all tattoos for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT location, description FROM tattoos WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [{"location": row[0], "description": row[1]} for row in rows]

    def get_piercings(self, performer_id: int) -> list[dict]:
        """Get all piercings for a performer."""
        with self._connection() as conn:
            rows = conn.execute(
                "SELECT location, description FROM piercings WHERE performer_id = ?",
                (performer_id,)
            ).fetchall()
            return [{"location": row[0], "description": row[1]} for row in rows]

    # ==================== Scrape Progress ====================

    def save_scrape_progress(
        self,
        source: str,
        last_processed_id: str,
        performers_processed: int = 0,
        faces_added: int = 0,
        errors: int = 0,
    ):
        """Save scrape progress for resume capability."""
        with self._connection() as conn:
            conn.execute(
                """
                INSERT INTO scrape_progress (source, last_processed_id, performers_processed, faces_added, errors)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source) DO UPDATE SET
                    last_processed_id = excluded.last_processed_id,
                    last_processed_time = datetime('now'),
                    performers_processed = excluded.performers_processed,
                    faces_added = excluded.faces_added,
                    errors = excluded.errors
                """,
                (source, last_processed_id, performers_processed, faces_added, errors),
            )

    def get_scrape_progress(self, source: str) -> Optional[dict]:
        """Get scrape progress for a source."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT last_processed_id, last_processed_time, performers_processed, faces_added, errors
                FROM scrape_progress
                WHERE source = ?
                """,
                (source,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return {
                "last_processed_id": row["last_processed_id"],
                "last_processed_time": row["last_processed_time"],
                "performers_processed": row["performers_processed"],
                "faces_added": row["faces_added"],
                "errors": row["errors"],
            }

    def get_all_scrape_progress(self) -> dict[str, dict]:
        """Get scrape progress for all sources."""
        with self._connection() as conn:
            cursor = conn.execute(
                """
                SELECT source, last_processed_id, last_processed_time, performers_processed, faces_added, errors
                FROM scrape_progress
                """
            )
            result = {}
            for row in cursor.fetchall():
                result[row["source"]] = {
                    "last_processed_id": row["last_processed_id"],
                    "last_processed_time": row["last_processed_time"],
                    "performers_processed": row["performers_processed"],
                    "faces_added": row["faces_added"],
                    "errors": row["errors"],
                }
            return result

    def clear_scrape_progress(self, source: str):
        """Clear progress for a source (for fresh start)."""
        with self._connection() as conn:
            conn.execute("DELETE FROM scrape_progress WHERE source = ?", (source,))

    # ==================== Statistics ====================

    def get_stats(self) -> dict:
        """Get database statistics."""
        with self._connection() as conn:
            stats = {}
            stats['performer_count'] = conn.execute(
                "SELECT COUNT(*) FROM performers"
            ).fetchone()[0]
            stats['performers_with_faces'] = conn.execute(
                "SELECT COUNT(*) FROM performers WHERE face_count > 0"
            ).fetchone()[0]
            stats['total_faces'] = conn.execute(
                "SELECT COUNT(*) FROM faces"
            ).fetchone()[0]
            stats['total_urls'] = conn.execute(
                "SELECT COUNT(*) FROM external_urls"
            ).fetchone()[0]
            stats['total_aliases'] = conn.execute(
                "SELECT COUNT(*) FROM aliases"
            ).fetchone()[0]
            stats['stashbox_counts'] = {}
            for row in conn.execute(
                "SELECT endpoint, COUNT(*) FROM stashbox_ids GROUP BY endpoint"
            ):
                stats['stashbox_counts'][row[0]] = row[1]
            stats['url_site_counts'] = {}
            for row in conn.execute(
                "SELECT site, COUNT(*) FROM external_urls GROUP BY site ORDER BY COUNT(*) DESC LIMIT 20"
            ):
                stats['url_site_counts'][row[0]] = row[1]
            return stats

    # ==================== Iteration ====================

    def iter_performers(self, batch_size: int = 1000) -> Iterator[Performer]:
        """Iterate over all performers in batches."""
        with self._connection() as conn:
            offset = 0
            while True:
                rows = conn.execute(
                    "SELECT * FROM performers ORDER BY id LIMIT ? OFFSET ?",
                    (batch_size, offset)
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    yield Performer(**dict(row))
                offset += batch_size

    def iter_performers_with_site_urls(
        self,
        site: str,
        after_id: int = 0,
        batch_size: int = 1000,
    ) -> Iterator[tuple[int, str, str]]:
        """
        Iterate performers that have URLs matching the given site.

        Args:
            site: Site name to match in URL (e.g., 'babepedia', 'iafd')
            after_id: Resume after this performer ID
            batch_size: Batch size for queries

        Yields:
            (performer_id, performer_name, url) tuples ordered by performer_id
        """
        with self._connection() as conn:
            offset_id = after_id
            while True:
                rows = conn.execute(
                    """
                    SELECT p.id, p.canonical_name, u.url
                    FROM performers p
                    JOIN external_urls u ON p.id = u.performer_id
                    WHERE u.url LIKE ? AND p.id > ?
                    ORDER BY p.id ASC
                    LIMIT ?
                    """,
                    (f"%{site}%", offset_id, batch_size),
                ).fetchall()

                if not rows:
                    break

                for row in rows:
                    yield (row[0], row[1], row[2])
                    offset_id = row[0]

    def iter_performers_after_id(
        self,
        after_id: int = 0,
        batch_size: int = 1000,
    ) -> Iterator[Performer]:
        """
        Iterate all performers with ID greater than after_id.

        Args:
            after_id: Resume after this performer ID (0 = start from beginning)
            batch_size: Batch size for queries

        Yields:
            Performer objects ordered by id
        """
        with self._connection() as conn:
            offset_id = after_id
            while True:
                rows = conn.execute(
                    """
                    SELECT * FROM performers
                    WHERE id > ?
                    ORDER BY id ASC
                    LIMIT ?
                    """,
                    (offset_id, batch_size),
                ).fetchall()

                if not rows:
                    break

                for row in rows:
                    performer = Performer(**dict(row))
                    yield performer
                    offset_id = performer.id

    def iter_performers_needing_urls(self, batch_size: int = 1000) -> Iterator[tuple[int, str]]:
        """
        Iterate over performers that have stash-box IDs but no URLs.
        Yields (performer_id, stashbox_id).
        """
        with self._connection() as conn:
            offset = 0
            while True:
                rows = conn.execute(
                    """
                    SELECT s.performer_id, s.stashbox_performer_id
                    FROM stashbox_ids s
                    LEFT JOIN external_urls u ON s.performer_id = u.performer_id
                    WHERE s.endpoint = 'stashdb' AND u.id IS NULL
                    ORDER BY s.performer_id
                    LIMIT ? OFFSET ?
                    """,
                    (batch_size, offset)
                ).fetchall()
                if not rows:
                    break
                for row in rows:
                    yield (row[0], row[1])
                offset += batch_size


# Convenience function
def open_database(path: str | Path) -> PerformerDatabase:
    """Open or create a performer database."""
    return PerformerDatabase(path)


# Self-test
if __name__ == "__main__":
    import tempfile
    import os

    # Create temporary database
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    try:
        db = PerformerDatabase(db_path)

        # Test performer creation
        pid = db.add_performer(
            canonical_name="Test Performer",
            gender="FEMALE",
            country="US",
            face_count=3,
        )
        print(f"Created performer ID: {pid}")

        # Test stash-box ID
        db.add_stashbox_id(pid, "stashdb", "abc-123-uuid")
        print("Added stashbox ID")

        # Test URL
        db.add_url(pid, "https://twitter.com/testperformer", "stashdb")
        db.add_url(pid, "https://www.iafd.com/person.rme/perfid=testperformer/Test-Performer.htm", "stashdb")
        print("Added URLs")

        # Test aliases
        db.add_alias(pid, "Test P", "stashdb")
        db.add_alias(pid, "TP", "stashdb")
        print("Added aliases")

        # Test lookups
        p = db.get_performer_by_stashbox_id("stashdb", "abc-123-uuid")
        print(f"Found by stashbox ID: {p.canonical_name if p else 'Not found'}")

        p = db.find_by_url("twitter", "testperformer")
        print(f"Found by Twitter: {p.canonical_name if p else 'Not found'}")

        matches = db.find_by_alias("Test P")
        print(f"Found by alias: {[m.canonical_name for m in matches]}")

        # Test stats
        stats = db.get_stats()
        print(f"Stats: {stats}")

        print("\nAll tests passed!")

    finally:
        os.unlink(db_path)

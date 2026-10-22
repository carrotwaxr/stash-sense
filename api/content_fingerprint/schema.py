"""The content fingerprint store's schema: one SQLite file (``content_fp.db``) with its own schema version.

It is deliberately not part of ``stash_sense.db``: the fingerprints run to gigabytes, and a worker filling it for days
must not contend with the sidecar's database. stash-sense ATTACHes this file when it needs it.
"""

import sqlite3

CONTENT_FP_SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS content_fp_schema_version (version INTEGER NOT NULL);

-- One row per distinct file content. Paths are advisory: re-resolve them from Stash before use.
CREATE TABLE IF NOT EXISTS content_files (
    id           INTEGER PRIMARY KEY,
    oshash       TEXT NOT NULL,
    size_bytes   INTEGER NOT NULL,
    md5          TEXT,
    duration_s   REAL,
    last_path    TEXT,
    last_seen_at TEXT DEFAULT (datetime('now')),
    UNIQUE (oshash, size_bytes)
);
CREATE INDEX IF NOT EXISTS idx_content_files_md5 ON content_files (md5);

-- Chromaprint raw fingerprint of the whole audio track: little-endian uint32, ~8 items a second.
CREATE TABLE IF NOT EXISTS audio_fingerprints (
    file_id      INTEGER NOT NULL REFERENCES content_files (id) ON DELETE CASCADE,
    algo         TEXT    NOT NULL,
    algo_version INTEGER NOT NULL,
    params       TEXT    NOT NULL,            -- JSON: ffmpeg version, chromaprint algorithm
    status       TEXT    NOT NULL,            -- ok | no_audio | failed
    n_items      INTEGER NOT NULL DEFAULT 0,
    fp           BLOB,
    error        TEXT,
    seconds      REAL,                        -- time spent computing it
    created_at   TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (file_id, algo, algo_version)
);

-- v2: verified overlaps between two files' audio (file_a < file_b). Offsets and ranges in seconds; a_start..a_end in
-- file a lines up with b_start..b_end in file b. Several rows per pair when the shared audio is in several pieces.
CREATE TABLE IF NOT EXISTS audio_overlaps (
    file_a       INTEGER NOT NULL REFERENCES content_files (id) ON DELETE CASCADE,
    file_b       INTEGER NOT NULL REFERENCES content_files (id) ON DELETE CASCADE,
    algo_version INTEGER NOT NULL,
    a_start      REAL NOT NULL,
    a_end        REAL NOT NULL,
    b_start      REAL NOT NULL,
    b_end        REAL NOT NULL,
    ber          REAL NOT NULL,
    found_at     TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (file_a, file_b, algo_version, a_start)
);
CREATE INDEX IF NOT EXISTS idx_audio_overlaps_b ON audio_overlaps (file_b);
"""


def connect(path, check_same_thread=True):
    """A connection to the store at ``path`` with the schema in place; WAL, so readers don't block the worker."""
    conn = sqlite3.connect(path, timeout=30, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    ensure_schema(conn)
    return conn


def ensure_schema(conn):
    conn.executescript(_SCHEMA)
    row = conn.execute("SELECT version FROM content_fp_schema_version").fetchone()
    if row is None:
        conn.execute("INSERT INTO content_fp_schema_version (version) VALUES (?)", (CONTENT_FP_SCHEMA_VERSION,))
    elif row[0] > CONTENT_FP_SCHEMA_VERSION:
        raise RuntimeError(f"content_fp.db is schema v{row[0]}, newer than this code (v{CONTENT_FP_SCHEMA_VERSION})")
    elif row[0] < CONTENT_FP_SCHEMA_VERSION:   # every change so far only adds tables, which the script above created
        conn.execute("UPDATE content_fp_schema_version SET version = ?", (CONTENT_FP_SCHEMA_VERSION,))
    conn.commit()

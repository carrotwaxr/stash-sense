"""Cached per-file quality checks, keyed by content (oshash, size)."""

import json
import sqlite3
import threading
from datetime import datetime, timezone

_TABLES = ("cheap", "sharpness")

_SCHEMA = "".join(
    f"""CREATE TABLE IF NOT EXISTS {t} (
        oshash TEXT, size_bytes INTEGER, version INTEGER, result TEXT, checked_at TEXT,
        PRIMARY KEY (oshash, size_bytes));
    """
    for t in _TABLES
)


class QualityStore:
    """The store at ``path``. Safe to share between threads: one connection, every statement under a lock."""

    def __init__(self, path):
        self.path = path
        self._conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._conn.close()

    def _get(self, table, oshash, size, min_version):
        with self._lock:
            row = self._conn.execute(
                f"SELECT version, result FROM {table} WHERE oshash = ? AND size_bytes = ?",
                (oshash, size)).fetchone()
        if row is None or row[0] < min_version:
            return None
        return json.loads(row[1])

    def _put(self, table, oshash, size, version, result):
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                f"""INSERT INTO {table} (oshash, size_bytes, version, result, checked_at) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT (oshash, size_bytes) DO UPDATE SET
                      version = excluded.version, result = excluded.result, checked_at = excluded.checked_at""",
                (oshash, size, version, json.dumps(result), now))
            self._conn.commit()

    def get_cheap(self, oshash, size, min_version):
        """The cached cheap-pass result for this content at ``min_version`` or newer, else None."""
        return self._get("cheap", oshash, size, min_version)

    def put_cheap(self, oshash, size, version, result):
        self._put("cheap", oshash, size, version, result)

    def get_sharpness(self, oshash, size, min_version):
        """The cached sharpness result for this content at ``min_version`` or newer, else None."""
        return self._get("sharpness", oshash, size, min_version)

    def put_sharpness(self, oshash, size, version, result):
        self._put("sharpness", oshash, size, version, result)

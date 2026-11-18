"""The splits store (#131): cached signals per file, series labels for the profiles, and the alias list.

One SQLite file at DATA_DIR/splits.db. Aliases and the seed labels name real titles, so they only ever come
from `DATA_DIR/split_seed.json`, imported once.
"""

import json
import sqlite3
import threading
from pathlib import Path

from . import profiles as profiles_mod
from . import signals as sg

SEED_NAME = "split_seed.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS signals (
    oshash TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    version INTEGER NOT NULL,
    blob BLOB NOT NULL,
    extracted_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (oshash, size_bytes)
);
CREATE TABLE IF NOT EXISTS series_labels (
    series TEXT NOT NULL,
    key TEXT NOT NULL,
    total REAL NOT NULL,
    cuts_json TEXT NOT NULL,
    feats_json TEXT,
    source TEXT NOT NULL,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (series, key)
);
CREATE TABLE IF NOT EXISTS aliases (
    pattern TEXT NOT NULL,
    name TEXT NOT NULL,
    PRIMARY KEY (pattern, name)
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);
"""


class SplitStore:
    """The store at ``path``. Safe to share between threads: one connection, every statement under a lock."""

    def __init__(self, path):
        self.path = path
        self._conn = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._conn.close()

    # signals

    def put_signals(self, oshash, size_bytes, version, blob):
        """Cache the encoded signals of a file, replacing any earlier row."""
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO signals (oshash, size_bytes, version, blob) VALUES (?, ?, ?, ?)",
                (oshash, size_bytes, version, blob),
            )
            self._conn.commit()

    def get_signals(self, oshash, size_bytes):
        """The decoded `Signals`, or None when missing or extracted by an older version."""
        with self._lock:
            row = self._conn.execute(
                "SELECT version, blob FROM signals WHERE oshash = ? AND size_bytes = ?",
                (oshash, size_bytes),
            ).fetchone()
        if row is None or row["version"] < sg.SIGNALS_VERSION:
            return None
        return sg.decode(row["blob"])

    # labels

    def _add_label(self, series, key, total, cuts, feats, source, replace=True):
        verb = "INSERT OR REPLACE" if replace else "INSERT OR IGNORE"
        cur = self._conn.execute(
            f"""{verb} INTO series_labels (series, key, total, cuts_json, feats_json, source)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (
                series,
                str(key),
                total,
                json.dumps(list(cuts)),
                None if feats is None else json.dumps(list(feats)),
                source,
            ),
        )
        return cur.rowcount

    def add_label(self, series, key, total, cuts, feats, source):
        """Record a known title: its cuts (seconds) and the `features_at` dict at each cut (or None). The newer
        label for a (series, key) wins."""
        with self._lock:
            self._add_label(series, key, total, cuts, feats, source)
            self._conn.commit()

    def profiles(self):
        """{series: profile} over every label with at least one cut."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT series, key, total, cuts_json, feats_json FROM series_labels ORDER BY series, key"
            ).fetchall()
        titles = {}
        for r in rows:
            cuts = json.loads(r["cuts_json"])
            if not cuts:
                continue
            feats = json.loads(r["feats_json"]) if r["feats_json"] else None
            if not feats or len(feats) != len(cuts):
                feats = [None] * len(cuts)
            titles.setdefault(r["series"], []).append(
                {"key": r["key"], "total": r["total"], "cuts": cuts, "feat": feats}
            )
        return {s: profiles_mod.build_profile(t) for s, t in titles.items()}

    # aliases and the seed

    def aliases(self):
        """[(regex, name)] in insertion order, for `profiles.series_key`."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT pattern, name FROM aliases ORDER BY rowid"
            ).fetchall()
        return [(r["pattern"], r["name"]) for r in rows]

    def import_seed(self, data_dir):
        """Import `split_seed.json` from the data dir once and rename it `.imported`. Returns the number of
        labels added (0 when there is no file). Labels already in the store are kept."""
        src = Path(data_dir) / SEED_NAME
        if not src.exists():
            return 0
        seed = json.loads(src.read_text())
        added = 0
        with self._lock:
            for pattern, name in seed.get("aliases", []):
                self._conn.execute(
                    "INSERT OR IGNORE INTO aliases (pattern, name) VALUES (?, ?)",
                    (pattern, name),
                )
            for lb in seed.get("labels", []):
                added += self._add_label(
                    lb["series"],
                    lb["key"],
                    lb["total"],
                    lb.get("cuts", []),
                    lb.get("feats"),
                    lb.get("source", "owner"),
                    replace=False,
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('seed_imported', datetime('now'))"
            )
            self._conn.commit()
        src.rename(src.with_name(SEED_NAME + ".imported"))
        return added

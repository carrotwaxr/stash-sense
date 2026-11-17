"""Reads and writes of the content fingerprint store."""

import json
import threading

import numpy as np

from . import schema


class ContentFpStore:
    """The store at ``path``. Safe to share between threads: one connection, every statement under a lock."""

    def __init__(self, path):
        self.path = path
        self._conn = schema.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._lock = threading.Lock()

    def close(self):
        with self._lock:
            self._conn.close()

    def _one(self, sql, args):
        with self._lock:
            return self._conn.execute(sql, args).fetchone()

    def upsert_file(self, oshash, size_bytes, md5=None, duration_s=None, path=None):
        """The id of the file with this content, created or refreshed (md5 is only filled in, never cleared)."""
        with self._lock:
            self._conn.execute(
                """INSERT INTO content_files (oshash, size_bytes, md5, duration_s, last_path) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (oshash, size_bytes) DO UPDATE SET
                     md5 = COALESCE(excluded.md5, content_files.md5),
                     duration_s = COALESCE(excluded.duration_s, content_files.duration_s),
                     last_path = COALESCE(excluded.last_path, content_files.last_path),
                     last_seen_at = datetime('now')""",
                (oshash, size_bytes, md5, duration_s, path))
            row = self._conn.execute("SELECT id FROM content_files WHERE oshash = ? AND size_bytes = ?",
                                     (oshash, size_bytes)).fetchone()
            self._conn.commit()
            return row[0]

    def has_audio(self, file_id, algo, algo_version, retry_failed=False):
        """True when this file already has a fingerprint of this version (a failed one counts unless retrying)."""
        row = self._one("SELECT status FROM audio_fingerprints WHERE file_id = ? AND algo = ? AND algo_version = ?",
                        (file_id, algo, algo_version))
        return row is not None and not (retry_failed and row[0] == "failed")

    def put_audio(self, file_id, algo, algo_version, params, status, fp=None, error=None, seconds=None):
        """Store a fingerprint result. An ok row is never replaced by a failure."""
        blob = None if fp is None else np.asarray(fp, dtype="<u4").tobytes()
        n = 0 if fp is None else len(fp)
        with self._lock:
            cur = self._conn.execute(
                "SELECT status FROM audio_fingerprints WHERE file_id = ? AND algo = ? AND algo_version = ?",
                (file_id, algo, algo_version)).fetchone()
            if cur is not None and cur[0] == "ok" and status != "ok":
                return
            self._conn.execute(
                """INSERT OR REPLACE INTO audio_fingerprints
                   (file_id, algo, algo_version, params, status, n_items, fp, error, seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (file_id, algo, algo_version, json.dumps(params, sort_keys=True), status, n, blob, error, seconds))
            self._conn.commit()

    def get_audio(self, file_id, algo, algo_version):
        """The fingerprint as a uint32 array, or None."""
        row = self._one(
            "SELECT fp FROM audio_fingerprints WHERE file_id = ? AND algo = ? AND algo_version = ? AND status = 'ok'",
            (file_id, algo, algo_version))
        return None if row is None else np.frombuffer(row[0], dtype="<u4")

    def all_audio(self, algo, algo_version):
        """[(file_id, uint32 fingerprint)] for every ok fingerprint of this version."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT file_id, fp FROM audio_fingerprints WHERE algo = ? AND algo_version = ? AND status = 'ok'",
                (algo, algo_version)).fetchall()
        return [(r[0], np.frombuffer(r[1], dtype="<u4")) for r in rows]

    def files(self, ids=None):
        """{file_id: {oshash, size_bytes, md5, duration_s, last_path}}."""
        with self._lock:
            rows = self._conn.execute("SELECT id, oshash, size_bytes, md5, duration_s, last_path FROM content_files").fetchall()
        return {r[0]: dict(r) for r in rows if ids is None or r[0] in ids}

    def put_overlaps(self, algo_version, rows):
        """Replace the stored overlaps of each pair in ``rows`` [(file_a, file_b, a_start, a_end, b_start, b_end, ber)]
        (seconds); pairs are normalized to file_a < file_b."""
        norm = []
        for a, b, a0, a1, b0, b1, ber in rows:
            norm.append((a, b, a0, a1, b0, b1, ber) if a < b else (b, a, b0, b1, a0, a1, ber))
        pairs = {(r[0], r[1]) for r in norm}
        with self._lock:
            for a, b in pairs:
                self._conn.execute("DELETE FROM audio_overlaps WHERE file_a = ? AND file_b = ? AND algo_version = ?",
                                   (a, b, algo_version))
            self._conn.executemany(
                """INSERT OR REPLACE INTO audio_overlaps (file_a, file_b, algo_version, a_start, a_end, b_start, b_end, ber)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                [(a, b, algo_version, a0, a1, b0, b1, ber) for a, b, a0, a1, b0, b1, ber in norm])
            self._conn.commit()

    def overlaps(self, algo_version, file_id=None):
        """Stored overlaps (as dicts), all of them or those involving ``file_id``."""
        sql = "SELECT * FROM audio_overlaps WHERE algo_version = ?"
        args = [algo_version]
        if file_id is not None:
            sql += " AND (file_a = ? OR file_b = ?)"
            args += [file_id, file_id]
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, args).fetchall()]

    def counts(self, algo, algo_version):
        """{status: n} for this algorithm version."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM audio_fingerprints WHERE algo = ? AND algo_version = ? GROUP BY status",
                (algo, algo_version)).fetchall()
        return {r[0]: r[1] for r in rows}

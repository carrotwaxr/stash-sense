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

    def counts(self, algo, algo_version, table="audio_fingerprints"):
        """{status: n} for this algorithm version."""
        if table not in ("audio_fingerprints", "video_signatures"):
            raise ValueError(table)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT status, COUNT(*) FROM {table} WHERE algo = ? AND algo_version = ? GROUP BY status",
                (algo, algo_version)).fetchall()
        return {r[0]: r[1] for r in rows}

    # -- bulk reads and writes for the job ------------------------------------------------------------------------

    def upsert_files(self, records):
        """{(oshash, size): file_id} for ``records`` (objects with oshash, size, md5, duration, path), in one
        transaction; same rules as upsert_file."""
        rows = [(r.oshash, r.size, r.md5, r.duration, r.path) for r in records]
        with self._lock:
            self._conn.executemany(
                """INSERT INTO content_files (oshash, size_bytes, md5, duration_s, last_path) VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT (oshash, size_bytes) DO UPDATE SET
                     md5 = COALESCE(excluded.md5, content_files.md5),
                     duration_s = COALESCE(excluded.duration_s, content_files.duration_s),
                     last_path = COALESCE(excluded.last_path, content_files.last_path),
                     last_seen_at = datetime('now')""", rows)
            self._conn.commit()
            ids = self._conn.execute("SELECT oshash, size_bytes, id FROM content_files").fetchall()
        wanted = {(r.oshash, r.size) for r in records}
        return {(o, s): i for o, s, i in ids if (o, s) in wanted}

    def done_ids(self, table, algo, algo_version, retry_failed=False):
        """File ids that already have a row of this version in ``table`` (failed rows excluded when retrying)."""
        if table not in ("audio_fingerprints", "video_signatures"):
            raise ValueError(table)
        sql = f"SELECT file_id FROM {table} WHERE algo = ? AND algo_version = ?"
        if retry_failed:
            sql += " AND status != 'failed'"
        with self._lock:
            return {r[0] for r in self._conn.execute(sql, (algo, algo_version)).fetchall()}

    def ok_audio_ids(self, algo, algo_version):
        with self._lock:
            return {r[0] for r in self._conn.execute(
                "SELECT file_id FROM audio_fingerprints WHERE algo = ? AND algo_version = ? AND status = 'ok'",
                (algo, algo_version)).fetchall()}

    def iter_audio(self, algo, algo_version):
        """(file_id, uint32 fingerprint) for every ok fingerprint, read through a separate connection so a full pass
        neither holds every fingerprint in memory nor blocks the job's writes."""
        conn = schema.connect(self.path)
        try:
            for fid, blob in conn.execute(
                    "SELECT file_id, fp FROM audio_fingerprints WHERE algo = ? AND algo_version = ? AND status = 'ok'",
                    (algo, algo_version)):
                yield fid, np.frombuffer(blob, dtype="<u4")
        finally:
            conn.close()

    def put_video(self, file_id, algo, algo_version, params, status, times=None, hashes=None, error=None, seconds=None):
        """Store a keyframe signature (times in ms, 64-bit hashes). An ok row is never replaced by a failure."""
        n = 0 if hashes is None else len(hashes)
        tb = None if times is None else np.asarray(times, dtype="<u4").tobytes()
        hb = None if hashes is None else np.asarray(hashes, dtype="<u8").tobytes()
        with self._lock:
            cur = self._conn.execute(
                "SELECT status FROM video_signatures WHERE file_id = ? AND algo = ? AND algo_version = ?",
                (file_id, algo, algo_version)).fetchone()
            if cur is not None and cur[0] == "ok" and status != "ok":
                return
            self._conn.execute(
                """INSERT OR REPLACE INTO video_signatures
                   (file_id, algo, algo_version, params, status, n_items, times, hashes, error, seconds)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (file_id, algo, algo_version, json.dumps(params, sort_keys=True), status, n, tb, hb, error, seconds))
            self._conn.commit()

    def get_video(self, file_id, algo, algo_version):
        """(times in ms as uint32, hashes as uint64), or None."""
        row = self._one(
            "SELECT times, hashes FROM video_signatures WHERE file_id = ? AND algo = ? AND algo_version = ? "
            "AND status = 'ok'", (file_id, algo, algo_version))
        if row is None:
            return None
        return np.frombuffer(row[0], dtype="<u4"), np.frombuffer(row[1], dtype="<u8")

    def put_health(self, file_id, status, reasons, params):
        """Record (replacing any earlier) the health check result of a file."""
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO file_health (file_id, status, reasons, params) VALUES (?, ?, ?, ?)""",
                (file_id, status, json.dumps(list(reasons)), json.dumps(params, sort_keys=True)))
            self._conn.commit()

    def get_health(self, file_id):
        """{status, reasons, params, checked_at} of the file's last health check, or None."""
        row = self._one("SELECT status, reasons, params, checked_at FROM file_health WHERE file_id = ?", (file_id,))
        if row is None:
            return None
        return {"status": row[0], "reasons": json.loads(row[1]), "params": json.loads(row[2]), "checked_at": row[3]}

    def unqueried(self, algo, algo_version):
        """Ok audio fingerprints not yet looked up in the index at this version."""
        with self._lock:
            return [r[0] for r in self._conn.execute(
                """SELECT a.file_id FROM audio_fingerprints a
                   LEFT JOIN audio_match_state m ON m.file_id = a.file_id AND m.algo_version = a.algo_version
                   WHERE a.algo = ? AND a.algo_version = ? AND a.status = 'ok' AND m.file_id IS NULL
                   ORDER BY a.file_id DESC""", (algo, algo_version)).fetchall()]

    def mark_queried(self, file_ids, algo_version):
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO audio_match_state (file_id, algo_version) VALUES (?, ?)",
                [(f, algo_version) for f in file_ids])
            self._conn.commit()

    def overlap_count(self, algo_version):
        return self._one("SELECT COUNT(*) FROM audio_overlaps WHERE algo_version = ?", (algo_version,))[0]

    def copy_audio_from(self, src_conn, keys, algo, algo_version):
        """Copy the ok fingerprints of ``keys`` [(oshash, size)] from another store's connection. Rows this store
        already has (any status) are kept. Returns the number of fingerprints copied."""
        copied = 0
        with self._lock:
            for oshash, size in keys:
                src = src_conn.execute(
                    """SELECT f.md5, f.duration_s, f.last_path, a.params, a.n_items, a.fp, a.seconds
                       FROM content_files f JOIN audio_fingerprints a ON a.file_id = f.id
                       WHERE f.oshash = ? AND f.size_bytes = ? AND a.algo = ? AND a.algo_version = ?
                         AND a.status = 'ok'""", (oshash, size, algo, algo_version)).fetchone()
                if src is None:
                    continue
                md5, duration, path, params, n_items, fp, seconds = src
                self._conn.execute(
                    """INSERT INTO content_files (oshash, size_bytes, md5, duration_s, last_path) VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT (oshash, size_bytes) DO UPDATE SET
                         md5 = COALESCE(content_files.md5, excluded.md5),
                         duration_s = COALESCE(content_files.duration_s, excluded.duration_s)""",
                    (oshash, size, md5, duration, path))
                fid = self._conn.execute("SELECT id FROM content_files WHERE oshash = ? AND size_bytes = ?",
                                         (oshash, size)).fetchone()[0]
                cur = self._conn.execute(
                    """INSERT OR IGNORE INTO audio_fingerprints
                       (file_id, algo, algo_version, params, status, n_items, fp, seconds)
                       VALUES (?, ?, ?, ?, 'ok', ?, ?, ?)""", (fid, algo, algo_version, params, n_items, fp, seconds))
                copied += cur.rowcount
            self._conn.commit()
        return copied

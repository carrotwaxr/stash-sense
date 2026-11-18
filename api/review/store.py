"""Apply log and labeled corrections, in stash_sense.db (tables from schema v10)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

_JSON_COLUMNS = ("option", "plan", "prior_state", "result",
                 "suggested", "chosen", "evidence")


def _row(row: sqlite3.Row | None) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    for k in _JSON_COLUMNS:
        if k in d and d[k] is not None:
            d[k] = json.loads(d[k])
    return d


def new_batch_id() -> str:
    return uuid.uuid4().hex[:12]


class ReviewStore:
    def __init__(self, db_path: str | Path):
        # The tables are created by RecommendationsDB's schema migration.
        self.db_path = Path(db_path)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---------------- apply log ----------------

    def begin(
        self,
        *,
        action: str,
        rec_id: int | None,
        rec_type: str,
        target_type: str | None,
        target_id: str | None,
        option: dict | None,
        plan: dict,
        plan_token: str | None,
        prior_state: dict,
        trash_path: str = "",
        batch_id: str | None = None,
        reverts_id: int | None = None,
    ) -> int:
        """Record an apply before its first write, so a crash still leaves the prior state."""
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO apply_log (action, status, rec_id, rec_type, target_type, target_id,
                    batch_id, reverts_id, option, plan, plan_token, prior_state, trash_path)
                VALUES (?, 'applying', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (action, rec_id, rec_type, target_type, target_id, batch_id, reverts_id,
                 json.dumps(option), json.dumps(plan), plan_token, json.dumps(prior_state),
                 trash_path or None),
            )
            return cur.lastrowid

    def finish(self, entry_id: int, *, status: str, result: dict, error: str | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE apply_log SET status = ?, result = ?, error = ?, completed_at = datetime('now') WHERE id = ?",
                (status, json.dumps(result), error, entry_id),
            )

    def mark_reverted(self, entry_id: int, revert_id: int) -> None:
        with self._conn() as conn:
            conn.execute("UPDATE apply_log SET reverted_by = ? WHERE id = ?", (revert_id, entry_id))

    def get(self, entry_id: int) -> dict | None:
        with self._conn() as conn:
            return _row(conn.execute("SELECT * FROM apply_log WHERE id = ?", (entry_id,)).fetchone())

    def latest_applied_for_rec(self, rec_id: int) -> dict | None:
        with self._conn() as conn:
            return _row(conn.execute(
                "SELECT * FROM apply_log WHERE rec_id = ? AND action = 'apply' AND status = 'applied' "
                "ORDER BY id DESC LIMIT 1",
                (rec_id,),
            ).fetchone())

    def list(
        self,
        *,
        rec_type: str | None = None,
        since_id: int | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        q = "SELECT * FROM apply_log WHERE 1=1"
        params: list[Any] = []
        if rec_type:
            q += " AND rec_type = ?"
            params.append(rec_type)
        if since_id is not None:
            q += " AND id > ?"
            params.append(since_id)
        q += " ORDER BY id DESC LIMIT ? OFFSET ?"
        params += [limit, offset]
        with self._conn() as conn:
            return [_row(r) for r in conn.execute(q, params).fetchall()]

    def count(self, *, rec_type: str | None = None) -> int:
        q, params = "SELECT COUNT(*) FROM apply_log", []
        if rec_type:
            q += " WHERE rec_type = ?"
            params.append(rec_type)
        with self._conn() as conn:
            return conn.execute(q, params).fetchone()[0]

    # ---------------- labels ----------------

    def add_label(
        self,
        *,
        rec_type: str,
        rec_id: int | None,
        target_type: str | None,
        target_id: str | None,
        suggested: dict,
        chosen: dict,
        evidence: dict,
        apply_log_id: int | None,
    ) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO review_labels (rec_type, rec_id, target_type, target_id,
                    suggested, chosen, evidence, apply_log_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (rec_type, rec_id, target_type, target_id, json.dumps(suggested),
                 json.dumps(chosen), json.dumps(evidence), apply_log_id),
            )
            return cur.lastrowid

    def labels(self, *, rec_type: str | None = None, limit: int = 1000) -> list[dict]:
        q, params = "SELECT * FROM review_labels", []
        if rec_type:
            q += " WHERE rec_type = ?"
            params.append(rec_type)
        q += " ORDER BY id LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            return [_row(r) for r in conn.execute(q, params).fetchall()]

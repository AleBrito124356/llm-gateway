"""Thin synchronous SQLite wrapper shared by the cache and accounting stores.

A single connection is reused with a lock. This is deliberately simple: local
SQLite writes are fast and a gateway process is I/O-bound on the upstream call,
not the ledger. For a multi-replica deployment swap this for Postgres and Redis
(the store interfaces are small on purpose).

Schema changes are applied by ``_migrate`` so a ``gateway.db`` created by an
older version keeps working after an upgrade.
"""

from __future__ import annotations

import sqlite3
import threading
from typing import Any, Iterable, Optional


SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    exact_key     TEXT NOT NULL,
    route         TEXT NOT NULL,
    model         TEXT NOT NULL,
    embedding     BLOB,
    request_json  TEXT NOT NULL,
    response_json TEXT NOT NULL,
    created_at    REAL NOT NULL,
    expires_at    REAL NOT NULL,
    context_key   TEXT
);
CREATE INDEX IF NOT EXISTS idx_cache_exact ON cache(exact_key);
CREATE INDEX IF NOT EXISTS idx_cache_route ON cache(route);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at);

CREATE TABLE IF NOT EXISTS usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                REAL NOT NULL,
    virtual_key       TEXT NOT NULL,
    model             TEXT NOT NULL,
    provider          TEXT NOT NULL,
    endpoint          TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens      INTEGER NOT NULL,
    cost_usd          REAL NOT NULL,
    cached            INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_key ON usage(virtual_key);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage(ts);
"""

# Columns added after the first release: (table, column, declaration).
_ADDED_COLUMNS = (("cache", "context_key", "TEXT"),)

# Indexes that depend on migrated columns, created after ``_migrate``.
_POST_MIGRATION = """
CREATE INDEX IF NOT EXISTS idx_cache_context ON cache(route, context_key);
"""


class Database:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()
            self._conn.executescript(_POST_MIGRATION)
            self._conn.commit()

    def _migrate(self) -> None:
        """Idempotently add columns that older databases are missing."""
        for table, column, decl in _ADDED_COLUMNS:
            existing = {row["name"] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")

    def execute(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Run a write and return the last inserted row id."""
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur.lastrowid or 0

    def execute_count(self, sql: str, params: Iterable[Any] = ()) -> int:
        """Run a write and return the number of affected rows."""
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return max(cur.rowcount, 0)

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def query_one(self, sql: str, params: Iterable[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

"""계정을 SQLite(storage/accounts.db)에 영속화한다 — 파이프라인의 출발점(identity).

단일 사용자 전제지만 다중 사용자를 대비해 계정을 1급 엔티티로 둔다. 계정 1건:
  id           고유 식별자(기본 계정은 DEFAULT_ACCOUNT, 예: "local")
  name         표시 이름
  attributes   자유 형식 메타/선호(JSON) — Memory Profile이 읽어 계획에 반영
  created_at   생성 ISO8601(UTC)
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

from config.config import ACCOUNT_STORE_PATH
from core.logger import get_logger

log = get_logger("account_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    attributes TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(ACCOUNT_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(ACCOUNT_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {"id": row[0], "name": row[1], "attributes": json.loads(row[2]), "created_at": row[3]}


def upsert(account: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO accounts (id, name, attributes, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                account["id"], account["name"],
                json.dumps(account.get("attributes", {}), ensure_ascii=False),
                account["created_at"],
            ),
        )


def get(account_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM accounts WHERE id = ?", (account_id,)).fetchone()
    return _row(row) if row else None


def load_all() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM accounts ORDER BY created_at ASC").fetchall()
    return [_row(r) for r in rows]


def delete(account_id: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM accounts WHERE id = ?", (account_id,))
        return cur.rowcount > 0

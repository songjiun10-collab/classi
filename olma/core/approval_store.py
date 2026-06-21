"""승인 요청을 SQLite(storage/approvals.db)에 영속화한다 — 재시작해도 대기 중 결정이 남는다.

Human Approval Gate가 위험·비가역 액션을 실행하기 전에 만드는 '대기 카드' 1건:
  id           고유 식별자
  action       대상 액션(예: browser_click)
  preview      무엇을 할지 사람에게 보여줄 입력 미리보기(잘린 input)
  status       pending | approved | rejected | expired
  reason       결정 사유(승인/거절 시, 없으면 빈 문자열)
  created_at   요청 생성 epoch
  decided_at   결정 epoch(없으면 None)
"""
from __future__ import annotations

import os
import sqlite3
import uuid
from contextlib import contextmanager

from config.config import APPROVAL_STORE_PATH
from core.logger import get_logger

log = get_logger("approval_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id TEXT PRIMARY KEY,
    action TEXT NOT NULL,
    preview TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL DEFAULT '',
    created_at REAL NOT NULL,
    decided_at REAL
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(APPROVAL_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(APPROVAL_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {
        "id": row[0], "action": row[1], "preview": row[2], "status": row[3],
        "reason": row[4], "created_at": row[5], "decided_at": row[6],
    }


def create(action: str, preview: str, now: float) -> str:
    aid = uuid.uuid4().hex
    with _connect() as conn:
        conn.execute(
            "INSERT INTO approvals (id, action, preview, status, reason, created_at, decided_at) "
            "VALUES (?, ?, ?, 'pending', '', ?, NULL)",
            (aid, action, preview, now),
        )
    return aid


def get(aid: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM approvals WHERE id = ?", (aid,)).fetchone()
    return _row(row) if row else None


def decide(aid: str, status: str, reason: str, now: float) -> bool:
    """pending 상태일 때만 결정으로 전이한다. 이미 결정/만료된 건은 건드리지 않는다(False)."""
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE approvals SET status = ?, reason = ?, decided_at = ? "
            "WHERE id = ? AND status = 'pending'",
            (status, reason, now, aid),
        )
        return cur.rowcount > 0


def list_pending() -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at ASC"
        ).fetchall()
    return [_row(r) for r in rows]


def list_all(limit: int = 100) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM approvals ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [_row(r) for r in rows]

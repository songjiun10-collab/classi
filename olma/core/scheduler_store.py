"""스케줄을 SQLite(storage/schedules.db)에 영속화한다 — 재시작해도 주기 작업이 살아남는다.

memory/task_store와 동일한 stdlib sqlite3 패턴. 스케줄 1건:
  id              고유 식별자
  kind            "dynamic"(자연어 요청) | "template"(저장된 워크플로우)
  payload         실행 정보(dynamic: {"request": ...}, template: {"name":..., "params":{...}})
  interval_seconds 반복 주기(초)
  next_run        다음 실행 예정 epoch(초)
  enabled         활성 여부
  last_run        마지막 실행 epoch(없으면 None)
  last_status     마지막 실행 결과 문자열(없으면 None)
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

from config.config import SCHEDULER_STORE_PATH
from core.logger import get_logger

log = get_logger("scheduler_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schedules (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    interval_seconds INTEGER NOT NULL,
    next_run REAL NOT NULL,
    enabled INTEGER NOT NULL,
    last_run REAL,
    last_status TEXT
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(SCHEDULER_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(SCHEDULER_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {
        "id": row[0],
        "kind": row[1],
        "payload": json.loads(row[2]),
        "interval_seconds": row[3],
        "next_run": row[4],
        "enabled": bool(row[5]),
        "last_run": row[6],
        "last_status": row[7],
    }


def upsert(s: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO schedules "
            "(id, kind, payload, interval_seconds, next_run, enabled, last_run, last_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                s["id"], s["kind"], json.dumps(s["payload"], ensure_ascii=False),
                int(s["interval_seconds"]), float(s["next_run"]), 1 if s["enabled"] else 0,
                s.get("last_run"), s.get("last_status"),
            ),
        )


def get(sid: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM schedules WHERE id = ?", (sid,)).fetchone()
    return _row(row) if row else None


def load_all() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM schedules ORDER BY next_run ASC").fetchall()
    return [_row(r) for r in rows]


def delete(sid: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM schedules WHERE id = ?", (sid,))
        return cur.rowcount > 0

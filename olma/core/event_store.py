"""이벤트 트리거를 SQLite(storage/events.db)에 영속화한다 — 재시작해도 살아남는다.

scheduler_store와 동일한 stdlib sqlite3 패턴. 트리거 1건:
  id              고유 식별자
  source          감지기 이름("file_exists" | "file_changed" 등, event_triggers.CHECKERS 키)
  source_config   감지기 입력(JSON, 예: {"path": "/tmp/x"})
  kind            "dynamic"(자연어 요청) | "template"(저장된 워크플로우)
  payload         실행 정보(dynamic: {"request": ...}, template: {"name":..., "params":{...}})
  enabled         활성 여부
  state           마지막 관측 상태(JSON) — 엣지 트리거(변화 시점) 판정에 쓴다. 최초 None.
  last_fired      마지막 발화 epoch(없으면 None)
  last_status     마지막 실행 결과 문자열(없으면 None)
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

from config.config import EVENT_STORE_PATH
from core.logger import get_logger

log = get_logger("event_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS event_triggers (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    source_config TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    state TEXT,
    last_fired REAL,
    last_status TEXT
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(EVENT_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(EVENT_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {
        "id": row[0],
        "source": row[1],
        "source_config": json.loads(row[2]),
        "kind": row[3],
        "payload": json.loads(row[4]),
        "enabled": bool(row[5]),
        "state": json.loads(row[6]) if row[6] is not None else None,
        "last_fired": row[7],
        "last_status": row[8],
    }


def upsert(t: dict) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO event_triggers "
            "(id, source, source_config, kind, payload, enabled, state, last_fired, last_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                t["id"], t["source"],
                json.dumps(t["source_config"], ensure_ascii=False),
                t["kind"], json.dumps(t["payload"], ensure_ascii=False),
                1 if t["enabled"] else 0,
                None if t.get("state") is None else json.dumps(t["state"], ensure_ascii=False),
                t.get("last_fired"), t.get("last_status"),
            ),
        )


def get(tid: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM event_triggers WHERE id = ?", (tid,)).fetchone()
    return _row(row) if row else None


def load_all() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM event_triggers ORDER BY rowid ASC").fetchall()
    return [_row(r) for r in rows]


def delete(tid: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM event_triggers WHERE id = ?", (tid,))
        return cur.rowcount > 0

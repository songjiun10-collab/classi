"""재사용 가능한 Workflow 템플릿을 SQLite(storage/workflows.db)에 영속화한다.

memory/task_store와 동일한 stdlib sqlite3 패턴(트랜잭션 원자성, 새 의존성 없음).
템플릿은 {name, description, steps(JSON)}이며 name이 기본키다 — 같은 이름으로 다시
저장하면 덮어쓴다(=갱신). 정렬용 seq는 DB의 MAX(seq)+1로 매겨 재시작에도 안전하다
(Date.now류를 쓰지 않아 결정론/테스트 안정성 유지)."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager

from config.config import WORKFLOW_STORE_PATH
from core.logger import get_logger

log = get_logger("workflow_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS workflows (
    name TEXT PRIMARY KEY,
    description TEXT NOT NULL,
    steps TEXT NOT NULL,
    seq INTEGER NOT NULL
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(WORKFLOW_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(WORKFLOW_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {"name": row[0], "description": row[1], "steps": json.loads(row[2])}


def save(name: str, description: str, steps: list) -> dict:
    """템플릿을 저장(또는 갱신)한다. 갱신 시 정렬 순번을 새로 받아 목록 최상단으로 올라온다."""
    with _connect() as conn:
        nxt = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM workflows").fetchone()[0]
        conn.execute(
            "INSERT OR REPLACE INTO workflows (name, description, steps, seq) VALUES (?, ?, ?, ?)",
            (name, description or "", json.dumps(steps, ensure_ascii=False), nxt),
        )
    return {"name": name, "description": description or "", "steps": steps}


def get(name: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT name, description, steps FROM workflows WHERE name = ?", (name,)
        ).fetchone()
    return _row(row) if row else None


def list_all() -> list:
    """최근 저장/갱신 순(seq 내림차순)으로 모든 템플릿을 돌려준다."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT name, description, steps FROM workflows ORDER BY seq DESC"
        ).fetchall()
    return [_row(r) for r in rows]


def delete(name: str) -> bool:
    """템플릿을 삭제한다. 실제로 지워졌으면 True."""
    with _connect() as conn:
        cur = conn.execute("DELETE FROM workflows WHERE name = ?", (name,))
        return cur.rowcount > 0

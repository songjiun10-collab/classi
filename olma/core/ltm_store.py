"""장기 기억(사실/선호)을 SQLite(storage/ltm.db)에 영속화한다.

작업 이력(memory.py)이 "무엇을 했나"라면, 이쪽은 "무엇을 아는가"다 — 세션을 넘어 유지돼야
하는 사용자 사실·선호·참조다(예: "보고서는 한국어로", "주 거래 은행은 X"). memory_store류와
동일한 stdlib sqlite3 패턴. 사실 1건:
  id          자동 증가 식별자
  key         안정 키(있으면 같은 키를 덮어쓴다 — 변하는 단일 사실용). 없으면 누적.
  kind        "fact" | "preference" | "reference" (분류용, 자유 문자열 허용)
  content     본문(사실 텍스트)
  tags        쉼표 구분 태그(검색 보조). 없으면 빈 문자열.
  created_at  최초 생성 ISO8601(UTC)
  updated_at  마지막 갱신 ISO8601(UTC)
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from config.config import LTM_PATH
from core.logger import get_logger

log = get_logger("ltm_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ltm (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,
    kind TEXT NOT NULL,
    content TEXT NOT NULL,
    tags TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""
# key는 UNIQUE지만 SQLite는 여러 NULL을 서로 다르게 취급하므로, 키 없는 누적 사실은
# 제약 없이 여러 건 쌓이고 키 있는 사실만 유일성이 강제된다(ON CONFLICT(key) 타겟이 됨).


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(LTM_PATH), exist_ok=True)
    conn = sqlite3.connect(LTM_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {
        "id": row[0], "key": row[1], "kind": row[2], "content": row[3],
        "tags": row[4], "created_at": row[5], "updated_at": row[6],
    }


def upsert(key, kind: str, content: str, tags: str, now: str) -> dict:
    """key가 있으면 기존 사실을 덮어쓰고(없으면 생성), 없으면 새 사실을 누적한다.
    저장된 레코드를 반환한다."""
    with _connect() as conn:
        if key:
            existing = conn.execute("SELECT created_at FROM ltm WHERE key = ?", (key,)).fetchone()
            created = existing[0] if existing else now
            conn.execute(
                "INSERT INTO ltm (key, kind, content, tags, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET "
                "kind=excluded.kind, content=excluded.content, tags=excluded.tags, "
                "updated_at=excluded.updated_at",
                (key, kind, content, tags, created, now),
            )
            row = conn.execute("SELECT * FROM ltm WHERE key = ?", (key,)).fetchone()
        else:
            cur = conn.execute(
                "INSERT INTO ltm (key, kind, content, tags, created_at, updated_at) "
                "VALUES (NULL, ?, ?, ?, ?, ?)",
                (kind, content, tags, now, now),
            )
            row = conn.execute("SELECT * FROM ltm WHERE id = ?", (cur.lastrowid,)).fetchone()
    return _row(row)


def get(fact_id: int) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ltm WHERE id = ?", (fact_id,)).fetchone()
    return _row(row) if row else None


def get_by_key(key: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM ltm WHERE key = ?", (key,)).fetchone()
    return _row(row) if row else None


def load_all() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM ltm ORDER BY updated_at DESC, id DESC").fetchall()
    return [_row(r) for r in rows]


def search(keyword: str, n: int = 10) -> list:
    """content/key/tags에 keyword가 포함된 사실을 최근 갱신순으로 반환한다.
    '%','_','\\'는 LIKE 와일드카드라 이스케이프한다(memory.find와 동일 규칙)."""
    if not keyword:
        return []
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{escaped}%"
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM ltm WHERE content LIKE ? ESCAPE '\\' OR key LIKE ? ESCAPE '\\' "
            "OR tags LIKE ? ESCAPE '\\' ORDER BY updated_at DESC, id DESC LIMIT ?",
            (like, like, like, n),
        ).fetchall()
    return [_row(r) for r in rows]


def delete(fact_id: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM ltm WHERE id = ?", (fact_id,))
        return cur.rowcount > 0

"""사용자 정의 스킬을 SQLite(storage/skills.db)에 영속화한다.

스킬 1건: 자주 쓰는 작업을 이름 붙여 저장해 둔 '자연어 작업 단축'이다(예: name="아침브리핑",
body="메일함 열어서 안 읽은 중요 메일 요약해줘"). recipes(코드로 고정된 파라미터 워크플로)와
달리, 사용자가 런타임에 만들고 지우는 가벼운 단축이다. stdlib sqlite3, 기존 *_store 패턴.
  id          자동 증가 식별자
  name        스킬 이름(UNIQUE — 같은 이름은 덮어쓴다)
  body        실행할 자연어 작업 내용
  description 설명(선택)
  created_at  최초 생성 ISO8601(UTC)
  updated_at  마지막 갱신 ISO8601(UTC)
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from config.config import SKILL_STORE_PATH
from core.logger import get_logger

log = get_logger("skill_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    body TEXT NOT NULL,
    description TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(SKILL_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(SKILL_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {
        "id": row[0], "name": row[1], "body": row[2],
        "description": row[3], "created_at": row[4], "updated_at": row[5],
    }


def upsert(name: str, body: str, description: str, now: str) -> dict:
    """같은 name이 있으면 덮어쓰고(created_at 보존), 없으면 새로 만든다. 저장 레코드 반환."""
    with _connect() as conn:
        existing = conn.execute("SELECT created_at FROM skills WHERE name = ?", (name,)).fetchone()
        created = existing[0] if existing else now
        conn.execute(
            "INSERT INTO skills (name, body, description, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET "
            "body=excluded.body, description=excluded.description, updated_at=excluded.updated_at",
            (name, body, description, created, now),
        )
        row = conn.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
    return _row(row)


def get(name: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM skills WHERE name = ?", (name,)).fetchone()
    return _row(row) if row else None


def load_all() -> list:
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM skills ORDER BY name ASC").fetchall()
    return [_row(r) for r in rows]


def delete(name: str) -> bool:
    with _connect() as conn:
        cur = conn.execute("DELETE FROM skills WHERE name = ?", (name,))
        return cur.rowcount > 0

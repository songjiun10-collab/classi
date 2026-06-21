"""모델 사용량을 SQLite(storage/usage.db)에 일·모델 단위로 누적한다.

호출 1건마다 (날짜, 모델)별 호출 수와 토큰(프롬프트/생성)을 더한다. 일일 한도(quota)는
이 누적값을 읽어 판정한다. 행 1건:
  day                 UTC 날짜(YYYY-MM-DD)
  model               모델명
  calls               그 날 호출 횟수
  prompt_tokens       누적 프롬프트 토큰
  completion_tokens   누적 생성 토큰
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager

from config.config import USAGE_STORE_PATH
from core.logger import get_logger

log = get_logger("usage_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage (
    day TEXT NOT NULL,
    model TEXT NOT NULL,
    calls INTEGER NOT NULL DEFAULT 0,
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (day, model)
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(USAGE_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(USAGE_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row(row: tuple) -> dict:
    return {
        "day": row[0], "model": row[1], "calls": row[2],
        "prompt_tokens": row[3], "completion_tokens": row[4],
        "total_tokens": row[3] + row[4],
    }


def add(day: str, model: str, calls: int, prompt_tokens: int, completion_tokens: int) -> None:
    """(day, model)에 사용량을 더한다(없으면 생성)."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO usage (day, model, calls, prompt_tokens, completion_tokens) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(day, model) DO UPDATE SET "
            "calls = calls + excluded.calls, "
            "prompt_tokens = prompt_tokens + excluded.prompt_tokens, "
            "completion_tokens = completion_tokens + excluded.completion_tokens",
            (day, model, calls, prompt_tokens, completion_tokens),
        )


def get(day: str, model: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM usage WHERE day = ? AND model = ?", (day, model)
        ).fetchone()
    return _row(row) if row else None


def for_day(day: str) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM usage WHERE day = ? ORDER BY (prompt_tokens + completion_tokens) DESC",
            (day,),
        ).fetchall()
    return [_row(r) for r in rows]


def recent(days: list) -> list:
    """주어진 날짜 목록에 해당하는 모든 사용량 행."""
    if not days:
        return []
    placeholders = ",".join("?" * len(days))
    with _connect() as conn:
        rows = conn.execute(
            f"SELECT * FROM usage WHERE day IN ({placeholders}) ORDER BY day DESC", tuple(days)
        ).fetchall()
    return [_row(r) for r in rows]

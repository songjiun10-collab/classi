"""task 기록을 SQLite(storage/tasks.db)에 영속화한다.

목적은 재시작 후에도 task 히스토리를 잃지 않는 것이다 — 재개(resume)가 아니다.
재시작 시점에 "queued"/"processing" 상태로 남아있던 task는 브라우저 세션과
워커 스레드가 이미 사라졌으므로 안전하게 이어서 실행할 수 없다. 그래서
mark_interrupted_as_failed()로 그런 task들을 "failed"로 정리한다(core/task_queue.py
가 기동 시 호출). 분산 큐가 아니므로 stdlib sqlite3로 충분하다(새 의존성 없음)."""
import json
import os
import sqlite3
from contextlib import contextmanager

from config.config import TASK_STORE_PATH
from core.logger import get_logger

log = get_logger("task_store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    task_id TEXT PRIMARY KEY,
    input TEXT NOT NULL,
    status TEXT NOT NULL,
    outcome TEXT,
    results TEXT,
    error TEXT,
    created_at TEXT NOT NULL
)
"""

_INTERRUPTED_ERROR = "재시작으로 작업이 중단되어 안전하게 재개할 수 없음(failed로 표시됨)"


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(TASK_STORE_PATH), exist_ok=True)
    conn = sqlite3.connect(TASK_STORE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row_to_task(row: tuple) -> dict:
    task_id, input_, status, outcome, results, error, created_at = row
    return {
        "task_id": task_id,
        "input": input_,
        "status": status,
        "outcome": outcome,
        "results": json.loads(results) if results is not None else None,
        "error": error,
        "created_at": created_at,
    }


def upsert(task: dict) -> None:
    results = task.get("results")
    with _connect() as conn:
        conn.execute(
            """INSERT INTO tasks (task_id, input, status, outcome, results, error, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(task_id) DO UPDATE SET
                 status = excluded.status,
                 outcome = excluded.outcome,
                 results = excluded.results,
                 error = excluded.error""",
            (
                task["task_id"],
                task["input"],
                task["status"],
                task.get("outcome"),
                json.dumps(results, ensure_ascii=False) if results is not None else None,
                task.get("error"),
                task["created_at"],
            ),
        )


def load_all() -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT task_id, input, status, outcome, results, error, created_at "
            "FROM tasks ORDER BY created_at ASC"
        ).fetchall()
    return [_row_to_task(r) for r in rows]


def delete(task_id: str) -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM tasks WHERE task_id = ?", (task_id,))


def mark_interrupted_as_failed() -> None:
    """재시작 시 한 번 호출: 직전에 처리 중이던 task를 failed로 정리한다."""
    with _connect() as conn:
        conn.execute(
            "UPDATE tasks SET status = 'failed', error = ? WHERE status IN ('queued', 'processing')",
            (_INTERRUPTED_ERROR,),
        )

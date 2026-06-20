"""작업 기록을 SQLite(storage/memory.db)에 누적 저장한다 (state 시스템).

V0.2: 단순 {task, result} 평면 기록 대신 task/step 단위 상태를 저장한다.
- task 단위: status(done/failed/partial), step별 요약, timestamp, schema_version
- 조회: recent(), get_context()(플래너에 줄 최근 맥락), find()(키워드 검색)

SQLite는 트랜잭션으로 원자성을 보장하므로(이전 JSON 구현의 임시파일+os.replace
원자적 쓰기나 손상 파일 백업 로직이 더 필요 없다 — task_store.py와 동일 패턴),
새 의존성 없이 stdlib sqlite3만 쓴다."""
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone

from config.config import MEMORY_MAX_RECORDS, MEMORY_PATH
from core.logger import get_logger
from core.schema import MAX_INPUT_CHARS, SCHEMA_VERSION

log = get_logger("memory")

_RESULT_PREVIEW = 500

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task TEXT NOT NULL,
    status TEXT NOT NULL,
    steps TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    schema_version INTEGER NOT NULL
)
"""


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    conn = sqlite3.connect(MEMORY_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _row_to_record(row: tuple) -> dict:
    _id, task, status, steps, timestamp, schema_version = row
    return {
        "task": task,
        "status": status,
        "steps": json.loads(steps),
        "timestamp": timestamp,
        "schema_version": schema_version,
    }


def _derive_status(step_results: list) -> str:
    if not step_results:
        return "failed"
    statuses = [r.get("status") for r in step_results]
    ok = [s for s in statuses if s in ("ok", "fallback")]
    if len(ok) == len(statuses):
        return "done"
    if not ok:
        return "failed"
    return "partial"


def _summarize_step(r: dict) -> dict:
    result = r.get("result")
    if isinstance(result, str) and len(result) > _RESULT_PREVIEW:
        result = result[:_RESULT_PREVIEW] + "…"
    return {
        "action": r.get("action") or r.get("step", {}).get("action"),
        "target": r.get("target"),
        "status": r.get("status"),
        "attempts": r.get("attempts"),
        "duration": r.get("duration"),
        "result": result,
        "error": r.get("error"),
    }


def save(task: str, step_results: list, status: str = None) -> dict:
    """task 1건을 step 단위 상태와 함께 저장하고 저장된 레코드를 돌려준다."""
    record = {
        "task": task[:MAX_INPUT_CHARS],
        "status": status or _derive_status(step_results),
        "steps": [_summarize_step(r) for r in step_results],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    with _connect() as conn:
        conn.execute(
            "INSERT INTO memory (task, status, steps, timestamp, schema_version) VALUES (?, ?, ?, ?, ?)",
            (
                record["task"],
                record["status"],
                json.dumps(record["steps"], ensure_ascii=False),
                record["timestamp"],
                record["schema_version"],
            ),
        )
        # 무한정 누적 방지: 초과분은 오래된 레코드부터 버린다(가장 단순한 회전 정책).
        conn.execute(
            "DELETE FROM memory WHERE id NOT IN (SELECT id FROM memory ORDER BY id DESC LIMIT ?)",
            (MEMORY_MAX_RECORDS,),
        )
    return record


def load_all() -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, task, status, steps, timestamp, schema_version FROM memory ORDER BY id ASC"
        ).fetchall()
    return [_row_to_record(r) for r in rows]


def recent(n: int = 5) -> list:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, task, status, steps, timestamp, schema_version "
            "FROM memory ORDER BY id DESC LIMIT ?",
            (n,),
        ).fetchall()
    return [_row_to_record(r) for r in rows[::-1]]


def get_context(n: int = 3) -> str:
    """플래너에 참고로 줄 최근 작업 맥락 요약 문자열. 없으면 빈 문자열."""
    items = recent(n)
    lines = []
    for rec in items:
        task = rec.get("task", "")
        status = rec.get("status", "unknown")
        step_count = len(rec.get("steps", [])) if isinstance(rec.get("steps"), list) else 0
        lines.append(f"- [{status}] {task} ({step_count} steps)")
    return "\n".join(lines)


def find(keyword: str, n: int = 10) -> list:
    """task 텍스트 또는 step의 action/result에 keyword가 포함된 최근 레코드를
    대소문자 구분 없이 검색해 최신순으로 돌려준다."""
    if not keyword:
        return []
    needle = keyword.lower()
    matched = []
    for rec in load_all():
        haystacks = [str(rec.get("task", ""))]
        steps = rec.get("steps")
        if isinstance(steps, list):
            for step in steps:
                haystacks.append(str(step.get("action", "")))
                haystacks.append(str(step.get("result", "")))
        if any(needle in h.lower() for h in haystacks):
            matched.append(rec)
    return matched[-n:][::-1]

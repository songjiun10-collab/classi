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
        "agent": r.get("agent"),      # 어느 에이전트가 실행했나(Agent Pool) — 경험 집계용(#46)
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


def successful_examples(keyword: str, n: int = 2) -> list:
    """과거 'done' 작업 중 keyword와 관련된 것을 골라 플래너 few-shot 재료로 돌려준다.

    실패/부분 성공 작업은 나쁜 예시가 되므로 제외한다. 반환: [{"task", "actions": [..]}].
    플라이휠: 성공한 계획의 action 흐름을 다음 계획에 다시 주입해 일관성을 높인다."""
    out = []
    for rec in find(keyword, n=n * 5):  # 여유 있게 받아 done만 추린다
        if rec.get("status") != "done":
            continue
        steps = rec.get("steps")
        actions = [s.get("action") for s in steps if isinstance(s, dict)] if isinstance(steps, list) else []
        actions = [a for a in actions if a]
        if actions:
            out.append({"task": rec.get("task", ""), "actions": actions})
        if len(out) >= n:
            break
    return out


def find(keyword: str, n: int = 10) -> list:
    """task 텍스트 또는 step(JSON으로 직렬화된 action/result 포함)에 keyword가 포함된
    레코드를 최신순으로 돌려준다.

    이전엔 load_all()로 전체 레코드(최대 MEMORY_MAX_RECORDS=1000건)를 로드 후 파이썬에서
    필터링했다 — successful_examples()가 매 plan마다 이를 호출해 비용이 컸다. 이제 SQL
    LIKE로 DB에서 직접 걸러 필요한 n건만 파싱한다(steps는 같은 행의 JSON 텍스트라 한 컬럼
    검색으로 action/result까지 함께 매칭된다). SQLite LIKE는 ASCII 대소문자 무시이고
    한국어는 대소문자가 없어 기존 동작과 동치다. '%','_','\\'는 와일드카드라 이스케이프한다."""
    if not keyword:
        return []
    escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    like = f"%{escaped}%"
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, task, status, steps, timestamp, schema_version FROM memory "
            "WHERE task LIKE ? ESCAPE '\\' OR steps LIKE ? ESCAPE '\\' "
            "ORDER BY id DESC LIMIT ?",
            (like, like, n),
        ).fetchall()
    return [_row_to_record(r) for r in rows]

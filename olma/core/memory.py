"""작업 기록을 storage/memory.json 에 누적 저장한다 (state 시스템).

V0.2: 단순 {task, result} 평면 기록 대신 task/step 단위 상태를 저장한다.
- task 단위: status(done/failed/partial), step별 요약, timestamp, schema_version
- 조회: recent(), get_context()(플래너에 줄 최근 맥락), find()(키워드 검색)
- 구버전(v1) 평면 레코드도 그대로 읽을 수 있다(마이그레이션 프레임워크는 만들지 않음)."""
import json
import os
from datetime import datetime, timezone

from config.config import MEMORY_PATH
from core.schema import MAX_INPUT_CHARS, SCHEMA_VERSION

_RESULT_PREVIEW = 500


def _load_raw() -> list:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        return json.loads(content) if content else []


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
    records = _load_raw()
    record = {
        "task": task[:MAX_INPUT_CHARS],
        "status": status or _derive_status(step_results),
        "steps": [_summarize_step(r) for r in step_results],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
    }
    records.append(record)
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    return record


def load_all() -> list:
    return _load_raw()


def recent(n: int = 5) -> list:
    return _load_raw()[-n:]


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
    """task 문자열에 keyword가 포함된 최근 레코드를 최신순으로 돌려준다."""
    if not keyword:
        return []
    matched = [rec for rec in _load_raw() if keyword in str(rec.get("task", ""))]
    return matched[-n:][::-1]

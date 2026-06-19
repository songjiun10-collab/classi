"""작업 기록을 storage/memory.json 에 누적 저장한다 (state 시스템).

V0.2: 단순 {task, result} 평면 기록 대신 task/step 단위 상태를 저장한다.
- task 단위: status(done/failed/partial), step별 요약, timestamp, schema_version
- 조회: recent(), get_context()(플래너에 줄 최근 맥락), find()(키워드 검색)
- 구버전(v1) 평면 레코드도 그대로 읽을 수 있다(마이그레이션 프레임워크는 만들지 않음).

내구성: save()는 임시 파일에 쓰고 os.replace로 교체한다(쓰기 중 프로세스가 죽어도
기존 파일은 그대로 남아 손상되지 않음). _load_raw()는 그래도 파일이 손상돼 있으면
(예: 과거 비-원자적 쓰기, 디스크 오류) 예외를 밖으로 던지지 않고 손상 파일을
백업한 뒤 빈 기록으로 취급한다 — 메모리 파일 손상 한 번이 이후 모든 요청의
계획 생성을 영구히 막는 사고(plan() 호출 전 get_context()가 매번 죽음)를 방지한다."""
import json
import os
from datetime import datetime, timezone

from config.config import MEMORY_MAX_RECORDS, MEMORY_PATH
from core.logger import get_logger
from core.schema import MAX_INPUT_CHARS, SCHEMA_VERSION

log = get_logger("memory")

_RESULT_PREVIEW = 500


def _load_raw() -> list:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return []
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.error("memory 파일 손상, 백업 후 빈 기록으로 시작: %s (%s)", MEMORY_PATH, exc)
        try:
            os.replace(MEMORY_PATH, f"{MEMORY_PATH}.corrupt-{int(datetime.now().timestamp())}")
        except OSError:
            pass
        return []
    return data if isinstance(data, list) else []


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
    if len(records) > MEMORY_MAX_RECORDS:
        records = records[-MEMORY_MAX_RECORDS:]

    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    # 원자적 쓰기: 같은 디렉터리의 임시 파일에 먼저 쓰고 os.replace로 교체한다.
    # 직접 덮어쓰다 중간에 죽으면(프로세스 강제종료·디스크 풀 등) 파일이 반쪽짜리
    # JSON으로 남아 다음 _load_raw() 호출부터 전부 깨진다 — os.replace는 같은
    # 파일시스템 내에서 원자적이라 이 중간 상태가 생기지 않는다.
    tmp_path = f"{MEMORY_PATH}.tmp-{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, MEMORY_PATH)
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

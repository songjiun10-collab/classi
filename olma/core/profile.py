"""Memory Profile — 계정의 '누적 행동'을 한 프로파일로 합쳐 Planner에 공급한다.

파이프라인의 두 번째 레이어. 흩어진 기억(작업 이력 memory.py, 장기 기억 long_term_memory)과
계정 속성(accounts.attributes)을 하나로 모아 (1) 사람이 보는 프로파일(build)과 (2) 플래너에
주입할 맥락/사실 문자열(planner_context)을 만든다. 누적 행동에서 통계를 뽑는 일은 metrics
집계를 재사용하며 LLM 없이 결정론적이다.

단일 사용자 전제라 memory/ltm은 아직 계정 단위로 분리돼 있지 않다(전역). account_id는
프로파일의 귀속·속성 주입에 쓰이고, 데이터 격리는 다중 사용자(#88)에서 확장한다."""
from __future__ import annotations

from core import accounts, long_term_memory, memory, metrics
from core.logger import get_logger

log = get_logger("profile")

_RECENT_TASKS = 5
_TOP_ACTIONS = 5


def _stats() -> dict:
    records = memory.load_all()
    agg = metrics.aggregate(records)
    by_action = agg.get("by_action", {})
    top = sorted(by_action.items(), key=lambda kv: kv[1]["count"], reverse=True)[:_TOP_ACTIONS]
    recent = [r.get("task", "") for r in records[-_RECENT_TASKS:]]
    return {
        "total_tasks": agg.get("total_tasks", 0),
        "task_success_rate": agg.get("task_success_rate", 0.0),
        "top_actions": [{"action": a, "count": v["count"]} for a, v in top],
        "recent_tasks": recent,
    }


def build(account_id: str = None) -> dict:
    """계정의 누적 행동 프로파일. account_id 미지정이면 현재 활성 계정."""
    account_id = account_id or accounts.current_id()
    acc = accounts.get(account_id) or accounts.ensure_default()
    return {
        "account_id": acc["id"],
        "name": acc["name"],
        "attributes": acc.get("attributes", {}),
        "stats": _stats(),
        "facts": long_term_memory.all_facts(),
    }


def _attributes_lines(attributes: dict) -> list:
    return [f"- {k}: {v}" for k, v in attributes.items()]


def planner_context(account_id: str, request: str) -> tuple:
    """플래너에 줄 (context, facts) 문자열 쌍을 만든다 — 두 블록은 planner._build_prompt의
    '최근 작업 맥락'과 '장기 기억(+계정 선호)' 자리로 그대로 들어간다.

    context = 최근 작업 맥락(memory.get_context) + 누적 행동 힌트(자주 쓰는 action).
    facts   = 계정 선호(attributes) + 요청 관련 장기 기억."""
    account_id = account_id or accounts.current_id()
    acc = accounts.get(account_id) or accounts.ensure_default()

    parts = []
    recent = memory.get_context()
    if recent:
        parts.append(recent)
    stats = _stats()
    if stats["top_actions"]:
        hint = ", ".join(f"{a['action']}({a['count']})" for a in stats["top_actions"])
        parts.append(f"자주 쓰는 작업 유형: {hint}")
    context = "\n".join(parts)

    fact_parts = []
    attr_lines = _attributes_lines(acc.get("attributes", {}))
    if attr_lines:
        fact_parts.append("계정 선호:\n" + "\n".join(attr_lines))
    relevant = long_term_memory.context(request)
    if relevant:
        fact_parts.append(relevant)
    facts = "\n".join(fact_parts)

    return context, facts

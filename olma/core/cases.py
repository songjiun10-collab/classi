"""Cases — 성공/실패 사례와 에이전트 경험을 작업 이력(memory)에서 끌어낸다.

#47 실패 사례 저장 · #48 성공 사례 저장 · #46 Agent 경험 저장. 별도 저장소를 새로 두지
않고 이미 쌓이는 memory(작업 이력, status·steps·agent 포함)를 '사례 관점'으로 조회한다 —
중복 저장을 피하고(SSOT) 항상 최신 이력과 일치한다.

사례는 다음 계획의 few-shot(성공 흐름 재주입)과 회피(실패 흐름 경고)에 쓰인다."""
from __future__ import annotations

from collections import defaultdict

from core import memory
from core.logger import get_logger

log = get_logger("cases")

_SUCCESS = ("done",)
_FAILURE = ("failed", "partial")


def _actions(rec: dict) -> list:
    steps = rec.get("steps")
    if not isinstance(steps, list):
        return []
    return [s.get("action") for s in steps if isinstance(s, dict) and s.get("action")]


def _cases(statuses: tuple, keyword: str, n: int) -> list:
    records = memory.find(keyword, n=n * 5) if keyword else memory.load_all()[::-1]
    out = []
    for rec in records:
        if rec.get("status") not in statuses:
            continue
        out.append({
            "task": rec.get("task", ""),
            "status": rec.get("status"),
            "actions": _actions(rec),
            "timestamp": rec.get("timestamp"),
        })
        if len(out) >= n:
            break
    return out


def success_cases(keyword: str = "", n: int = 5) -> list:
    """성공('done')한 작업 사례(#48). keyword를 주면 관련된 것만."""
    return _cases(_SUCCESS, keyword, n)


def failure_cases(keyword: str = "", n: int = 5) -> list:
    """실패/부분성공 작업 사례(#47). 같은 실수를 피하는 데 쓴다."""
    return _cases(_FAILURE, keyword, n)


def agent_experience() -> list:
    """에이전트별 누적 성공/실패 경험(#46). step의 agent×status를 집계해 성공률을 낸다."""
    stats: dict = defaultdict(lambda: {"ok": 0, "failed": 0, "total": 0})
    for rec in memory.load_all():
        for step in rec.get("steps", []):
            if not isinstance(step, dict):
                continue
            agent = step.get("agent")
            if not agent:
                continue
            s = stats[agent]
            s["total"] += 1
            if step.get("status") in ("ok", "fallback"):
                s["ok"] += 1
            elif step.get("status") in ("failed", "rejected", "skipped"):
                s["failed"] += 1
    out = []
    for agent, s in stats.items():
        out.append({
            "agent": agent, "total": s["total"], "ok": s["ok"], "failed": s["failed"],
            "success_rate": round(s["ok"] / s["total"], 3) if s["total"] else 0.0,
        })
    return sorted(out, key=lambda a: a["total"], reverse=True)

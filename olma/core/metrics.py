"""관측가능성: 메모리에 쌓인 작업 기록에서 운영 지표를 집계한다.

순수 함수(records -> dict)로, 외부 의존 없이 테스트 가능하다. API의 /api/metrics가
이 집계를 큐 깊이·가동시간과 합쳐 노출한다. 어떤 action이 자주 실패하는지/느린지 같은
운영 신호를 한눈에 보기 위한 것."""
from __future__ import annotations

_STEP_STATUSES = ("ok", "fallback", "failed", "skipped")


def aggregate(records: list) -> dict:
    """task 기록 리스트 -> 집계 지표.

    반환:
      total_tasks, by_status(작업 상태별 수), task_success_rate(done 비율),
      by_action(action별 {count, ok, fallback, failed, skipped, avg_duration, success_rate}).
    """
    by_status: dict = {}
    by_action: dict = {}

    for rec in records:
        status = rec.get("status", "unknown")
        by_status[status] = by_status.get(status, 0) + 1

        steps = rec.get("steps") if isinstance(rec.get("steps"), list) else []
        for step in steps:
            if not isinstance(step, dict):
                continue
            action = step.get("action") or "?"
            agg = by_action.setdefault(
                action,
                {k: 0 for k in _STEP_STATUSES} | {"count": 0, "_total_duration": 0.0},
            )
            agg["count"] += 1
            sstatus = step.get("status")
            if sstatus in _STEP_STATUSES:
                agg[sstatus] += 1
            duration = step.get("duration")
            if isinstance(duration, (int, float)):
                agg["_total_duration"] += duration

    for agg in by_action.values():
        count = agg["count"]
        succeeded = agg["ok"] + agg["fallback"]
        agg["avg_duration"] = round(agg.pop("_total_duration") / count, 3) if count else 0.0
        agg["success_rate"] = round(succeeded / count, 3) if count else 0.0

    total = len(records)
    return {
        "total_tasks": total,
        "by_status": by_status,
        "task_success_rate": round(by_status.get("done", 0) / total, 3) if total else 0.0,
        "by_action": by_action,
    }

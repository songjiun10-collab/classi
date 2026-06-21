"""관측가능성 지표 집계(core.metrics.aggregate)와 JSON 로그 포매터 테스트."""
import json
import logging

from core import metrics
from core.logger import JsonFormatter


def _rec(status, steps):
    return {"task": "t", "status": status, "steps": steps,
            "timestamp": "2026-06-20T00:00:00+00:00", "schema_version": 2}


def _step(action, status, duration):
    return {"action": action, "target": "ollama", "status": status,
            "attempts": 1, "duration": duration, "result": "r", "error": None}


def test_aggregate_empty():
    agg = metrics.aggregate([])
    assert agg["total_tasks"] == 0
    assert agg["task_success_rate"] == 0.0
    assert agg["by_action"] == {}


def test_aggregate_counts_status_and_actions():
    records = [
        _rec("done", [_step("llm", "ok", 1.0)]),
        _rec("done", [_step("llm", "ok", 3.0), _step("summarize", "ok", 2.0)]),
        _rec("failed", [_step("browser_open", "failed", 0.5)]),
    ]
    agg = metrics.aggregate(records)

    assert agg["total_tasks"] == 3
    assert agg["by_status"] == {"done": 2, "failed": 1}
    assert agg["task_success_rate"] == round(2 / 3, 3)

    llm = agg["by_action"]["llm"]
    assert llm["count"] == 2
    assert llm["ok"] == 2
    assert llm["avg_duration"] == 2.0          # (1.0 + 3.0) / 2
    assert llm["success_rate"] == 1.0

    browser = agg["by_action"]["browser_open"]
    assert browser["failed"] == 1
    assert browser["success_rate"] == 0.0


def test_aggregate_counts_fallback_as_success():
    agg = metrics.aggregate([_rec("done", [_step("browser_search", "fallback", 1.0)])])
    bs = agg["by_action"]["browser_search"]
    assert bs["fallback"] == 1
    assert bs["success_rate"] == 1.0           # fallback도 성공으로 집계


def test_json_formatter_emits_parseable_line():
    fmt = JsonFormatter()
    record = logging.LogRecord(
        name="olma.test", level=logging.INFO, pathname=__file__, lineno=1,
        msg="작업 %d 완료", args=(7,), exc_info=None,
    )
    line = fmt.format(record)
    parsed = json.loads(line)
    assert parsed["level"] == "INFO"
    assert parsed["name"] == "olma.test"
    assert parsed["message"] == "작업 7 완료"      # %-포매팅이 적용돼야 한다
    assert "time" in parsed

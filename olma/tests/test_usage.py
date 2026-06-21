"""Usage/Quota Manager(core.usage + core.usage_store) 테스트 — 기록·집계·한도 집행."""
import importlib

import pytest


def _fresh(tmp_path, monkeypatch, call_limit="0", token_limit="0"):
    monkeypatch.setenv("USAGE_STORE_PATH", str(tmp_path / "usage.db"))
    monkeypatch.setenv("USAGE_DAILY_CALL_LIMIT", call_limit)
    monkeypatch.setenv("USAGE_DAILY_TOKEN_LIMIT", token_limit)
    import config.config as cfg
    importlib.reload(cfg)
    import core.usage_store as store
    importlib.reload(store)
    import core.usage as usage
    importlib.reload(usage)
    return usage, store


def test_record_accumulates(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch)
    usage.record("m1", 10, 20, now_day="2026-06-21")
    usage.record("m1", 5, 5, now_day="2026-06-21")
    rec = store.get("2026-06-21", "m1")
    assert rec["calls"] == 2
    assert rec["prompt_tokens"] == 15
    assert rec["completion_tokens"] == 25
    assert rec["total_tokens"] == 40


def test_record_separates_by_model_and_day(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch)
    usage.record("m1", 1, 1, now_day="2026-06-21")
    usage.record("m2", 2, 2, now_day="2026-06-21")
    usage.record("m1", 3, 3, now_day="2026-06-22")
    assert len(store.for_day("2026-06-21")) == 2
    assert store.get("2026-06-22", "m1")["calls"] == 1


def test_enforce_noop_when_no_limits(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch)
    for _ in range(100):
        usage.record("m1", 100, 100, now_day="2026-06-21")
    usage.enforce("m1", now_day="2026-06-21")   # 한도 0 → 통과(예외 없음)


def test_enforce_blocks_over_call_limit(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch, call_limit="3")
    for _ in range(3):
        usage.record("m1", 0, 0, now_day="2026-06-21")
    with pytest.raises(usage.QuotaExceeded):
        usage.enforce("m1", now_day="2026-06-21")
    # 다른 모델은 영향 없음
    usage.enforce("m2", now_day="2026-06-21")


def test_enforce_blocks_over_token_limit(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch, token_limit="50")
    usage.record("m1", 30, 25, now_day="2026-06-21")   # 55 > 50
    with pytest.raises(usage.QuotaExceeded):
        usage.enforce("m1", now_day="2026-06-21")


def test_enforce_passes_under_limit(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch, call_limit="5", token_limit="100")
    usage.record("m1", 10, 10, now_day="2026-06-21")
    usage.enforce("m1", now_day="2026-06-21")   # 1콜/20토큰 < 한도


def test_quota_exceeded_is_runtime_error(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch, call_limit="1")
    usage.record("m1", 0, 0, now_day="2026-06-21")
    # ollama_client의 페일오버가 RuntimeError를 받으므로 하위 타입이어야 한다.
    assert issubclass(usage.QuotaExceeded, RuntimeError)
    with pytest.raises(RuntimeError):
        usage.enforce("m1", now_day="2026-06-21")


def test_stats_summary(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch, call_limit="10")
    usage.record("m1", 10, 5, now_day="2026-06-21")
    usage.record("m2", 2, 3, now_day="2026-06-21")
    s = usage.stats(now_day="2026-06-21")
    assert s["totals"]["calls"] == 2
    assert s["totals"]["total_tokens"] == 20
    assert s["limits"]["daily_call_limit"] == 10


def test_recent_spans_days(tmp_path, monkeypatch):
    usage, store = _fresh(tmp_path, monkeypatch)
    usage.record("m1", 1, 1, now_day="2026-06-21")
    usage.record("m1", 1, 1, now_day="2026-06-20")
    usage.record("m1", 1, 1, now_day="2026-06-10")   # 7일 밖
    rows = usage.recent(days=7, now_day="2026-06-21")
    days = {r["day"] for r in rows}
    assert "2026-06-21" in days and "2026-06-20" in days
    assert "2026-06-10" not in days

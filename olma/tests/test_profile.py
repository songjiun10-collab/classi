"""Memory Profile(core.profile) 테스트 — 누적 행동·선호·장기기억을 한 프로파일로 합치기."""
import importlib

import pytest


@pytest.fixture
def prof_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.db"))
    monkeypatch.setenv("LTM_PATH", str(tmp_path / "ltm.db"))
    monkeypatch.setenv("ACCOUNT_STORE_PATH", str(tmp_path / "accounts.db"))
    import config.config as cfg
    importlib.reload(cfg)
    for mod in ("core.memory", "core.ltm_store", "core.long_term_memory",
                "core.account_store", "core.accounts", "core.metrics", "core.profile"):
        importlib.reload(importlib.import_module(mod))
    import core.profile as profile
    return profile


def test_build_includes_account_and_stats(prof_env):
    profile = prof_env
    import core.memory as memory
    memory.save("뉴스 검색", [{"action": "browser_search", "status": "ok"}])
    p = profile.build()
    assert p["account_id"] == "local"
    assert p["stats"]["total_tasks"] == 1
    assert p["stats"]["top_actions"][0]["action"] == "browser_search"


def test_build_includes_facts(prof_env):
    profile = prof_env
    import core.long_term_memory as ltm
    ltm.remember("보고서는 한국어로", kind="preference")
    p = profile.build()
    assert any("한국어" in f["content"] for f in p["facts"])


def test_planner_context_merges_behavior_and_recent(prof_env):
    profile = prof_env
    import core.memory as memory
    memory.save("작업 A", [{"action": "llm", "status": "ok"}])
    memory.save("작업 B", [{"action": "llm", "status": "ok"}])
    context, facts = profile.planner_context("local", "새 요청")
    assert "자주 쓰는 작업 유형" in context
    assert "llm" in context


def test_planner_context_includes_account_prefs_and_facts(prof_env):
    profile = prof_env
    import core.accounts as accounts
    import core.long_term_memory as ltm
    accounts.ensure_default()
    accounts.set_attribute("local", "언어", "한국어")
    ltm.remember("배포는 금요일 피함", kind="preference", tags=["배포"])
    context, facts = profile.planner_context("local", "배포 일정 잡아줘")
    assert "계정 선호" in facts
    assert "언어" in facts
    assert "배포는 금요일 피함" in facts


def test_planner_context_empty_when_no_history(prof_env):
    profile = prof_env
    context, facts = profile.planner_context("local", "아무 요청")
    assert context == ""
    assert facts == ""

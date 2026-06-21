"""Agent Pool(core.agent_pool) 테스트 — 내장 매핑·동적 등록·해석."""
import importlib

import pytest


@pytest.fixture
def pool():
    import core.agent_pool as ap
    importlib.reload(ap)        # _registered를 깨끗이 비운 상태로 시작
    return ap


def test_builtin_for_action(pool):
    assert pool.for_action("llm")["agent"] == "local_llm"
    assert pool.for_action("llm")["target"] == "ollama"
    assert pool.for_action("browser_click")["agent"] == "browser"
    assert pool.for_action("web_ai_ask")["agent"] == "web_ai"
    assert pool.for_action("vision_describe")["agent"] == "vision"
    assert pool.for_action("notification_check")["agent"] == "notifier"


def test_unknown_action_returns_none(pool):
    assert pool.for_action("없는액션") is None


def test_resolve_uses_step_action(pool):
    assert pool.resolve({"action": "summarize"})["agent"] == "local_llm"


def test_agents_lists_builtin_with_handles(pool):
    by_name = {a["agent"]: a for a in pool.agents()}
    assert "local_llm" in by_name
    assert set(by_name["local_llm"]["handles"]) == {"llm", "summarize"}
    assert "browser_click" in by_name["browser"]["handles"]


def test_register_overrides_builtin(pool):
    pool.register("vision_pro", ["vision_describe"], "browser", "고급 비전")
    d = pool.for_action("vision_describe")
    assert d["agent"] == "vision_pro"      # 등록 에이전트가 내장보다 우선
    assert d["description"] == "고급 비전"


def test_register_adds_new_action_handler(pool):
    pool.register("research", ["browser_search"], "browser", "리서치 에이전트")
    assert pool.for_action("browser_search")["agent"] == "research"


def test_register_rejects_empty(pool):
    with pytest.raises(ValueError):
        pool.register("", ["llm"], "ollama")
    with pytest.raises(ValueError):
        pool.register("x", [], "ollama")


def test_unregister_restores_builtin(pool):
    pool.register("vision_pro", ["vision_describe"], "browser")
    assert pool.unregister("vision_pro") is True
    assert pool.for_action("vision_describe")["agent"] == "vision"   # 내장으로 복귀
    assert pool.unregister("vision_pro") is False

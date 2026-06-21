"""모델 선택 기능 테스트 — ai_roles 런타임 오버라이드 + ollama_client.list_models.

오버라이드는 모든 텍스트 역할의 모델 결정 단일 지점(model_for)을 거치므로, 여기서만
검증하면 planner/executor 전반에 반영됨이 보장된다. 비전(VLM)은 오버라이드에서 제외된다."""
from unittest.mock import MagicMock, patch

import pytest
import requests

from core import ai_roles


@pytest.fixture(autouse=True)
def _reset_override():
    from core import web_ai_providers
    ai_roles.clear_override()
    ai_roles.set_backend_mode("auto")
    web_ai_providers.clear_active()
    yield
    ai_roles.clear_override()
    ai_roles.set_backend_mode("auto")
    web_ai_providers.clear_active()


def test_override_applies_to_text_roles_not_vision():
    base_vision = ai_roles.model_for(ai_roles.ROLE_VISION)
    ai_roles.set_override("mymodel:latest")
    assert ai_roles.model_for(ai_roles.ROLE_CHAT) == "mymodel:latest"
    assert ai_roles.model_for(ai_roles.ROLE_PLAN) == "mymodel:latest"
    assert ai_roles.model_for(ai_roles.ROLE_SUMMARIZE) == "mymodel:latest"
    assert ai_roles.model_for(ai_roles.ROLE_VISION) == base_vision  # 비전은 그대로
    assert ai_roles.current_override() == "mymodel:latest"


def test_clear_override_restores_default():
    base = ai_roles.model_for(ai_roles.ROLE_CHAT)
    ai_roles.set_override("x:1")
    ai_roles.clear_override()
    assert ai_roles.model_for(ai_roles.ROLE_CHAT) == base
    assert ai_roles.current_override() is None


def test_set_override_empty_clears():
    ai_roles.set_override("x:1")
    assert ai_roles.set_override("") is None
    assert ai_roles.current_override() is None


def test_models_for_prepends_override_but_keeps_base_fallback():
    base = ai_roles._base_model(ai_roles.ROLE_CHAT)
    ai_roles.set_override("over:1")
    chain = ai_roles.models_for(ai_roles.ROLE_CHAT)
    assert chain[0] == "over:1"
    assert base in chain  # 잘못된 모델명을 골라도 기본 모델로 폴백 가능


def test_list_models_parses_and_sorts():
    fake = MagicMock()
    fake.json.return_value = {"models": [{"name": "b:1"}, {"name": "a:2"}, {"nokey": 1}]}
    fake.raise_for_status.return_value = None
    with patch("llm.ollama_client.requests.get", return_value=fake):
        from llm import ollama_client
        names = ollama_client.list_models()
    assert names == ["a:2", "b:1"]  # 정렬 + 이름 없는 항목 제외


def test_list_models_empty_on_connection_error():
    with patch("llm.ollama_client.requests.get", side_effect=requests.ConnectionError("down")):
        from llm import ollama_client
        assert ollama_client.list_models() == []


# ── 백엔드 토글(자동/로컬/웹) ─────────────────────────────────────────────────

def test_backend_mode_default_auto():
    assert ai_roles.backend_mode() == "auto"


def test_set_backend_mode_normalizes_unknown_to_auto():
    assert ai_roles.set_backend_mode("local") == "local"
    assert ai_roles.set_backend_mode("web") == "web"
    assert ai_roles.set_backend_mode("nonsense") == "auto"
    assert ai_roles.set_backend_mode("") == "auto"


def test_effective_action_web_routes_llm_to_web_ai():
    ai_roles.set_backend_mode("web")
    assert ai_roles.effective_action("llm") == "web_ai_ask"
    assert ai_roles.effective_action("web_ai_ask") == "web_ai_ask"
    assert ai_roles.effective_action("browser_open") == "browser_open"  # 그대로


def test_effective_action_local_routes_web_ai_to_llm():
    ai_roles.set_backend_mode("local")
    assert ai_roles.effective_action("web_ai_ask") == "llm"
    assert ai_roles.effective_action("llm") == "llm"


def test_effective_action_auto_keeps_planner_choice():
    ai_roles.set_backend_mode("auto")
    assert ai_roles.effective_action("llm") == "llm"
    assert ai_roles.effective_action("web_ai_ask") == "web_ai_ask"


def test_executor_apply_backend_mode_rewrites_action():
    from executor import executor
    ai_roles.set_backend_mode("web")
    out = executor._apply_backend_mode({"action": "llm", "input": "안녕", "depends_on": None})
    assert out["action"] == "web_ai_ask"
    assert out["input"] == "안녕"
    # auto면 원본 객체를 그대로 돌려준다(불필요한 복사 없음).
    ai_roles.set_backend_mode("auto")
    step = {"action": "llm", "input": "x"}
    assert executor._apply_backend_mode(step) is step

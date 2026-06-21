"""AI 역할 레지스트리(core.ai_roles) + 역할별 모델 분담 테스트.

역할→모델/백엔드 매핑, action→역할 매핑, 그리고 planner가 PLANNER 모델을, executor의
ollama 폴백이 REASONING 모델을 쓰는지(역할 분담이 실제 호출에 반영되는지) 검증한다.
외부 의존 없이 환경변수 reload + 모킹으로 돈다.
"""
import importlib
from unittest.mock import patch


def _fresh_roles(monkeypatch, **env):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    import config.config as cfg
    importlib.reload(cfg)
    import core.ai_roles as ar
    importlib.reload(ar)
    return ar


def test_model_for_falls_back_to_default_when_unset(monkeypatch):
    ar = _fresh_roles(monkeypatch, OLLAMA_MODEL="base:7b")
    # 역할별 모델 미지정 → 모두 기본 모델.
    assert ar.model_for(ar.ROLE_PLAN) == "base:7b"
    assert ar.model_for(ar.ROLE_CHAT) == "base:7b"
    assert ar.model_for(ar.ROLE_REASON) == "base:7b"


def test_model_for_uses_role_specific_models(monkeypatch):
    ar = _fresh_roles(
        monkeypatch,
        OLLAMA_MODEL="base:7b",
        OLLAMA_PLANNER_MODEL="planner:3b",
        OLLAMA_REASONING_MODEL="big:32b",
        OLLAMA_VLM_MODEL="vlm:7b",
    )
    assert ar.model_for(ar.ROLE_PLAN) == "planner:3b"
    assert ar.model_for(ar.ROLE_REASON) == "big:32b"
    assert ar.model_for(ar.ROLE_VISION) == "vlm:7b"
    assert ar.model_for(ar.ROLE_CHAT) == "base:7b"     # 미지정 역할은 기본


def test_models_for_single_when_no_backup(monkeypatch):
    ar = _fresh_roles(monkeypatch, OLLAMA_MODEL="base:7b")
    # 백업 미설정 → 길이 1 체인(기존 단일 모델 동작과 동일).
    assert ar.models_for(ar.ROLE_CHAT) == ["base:7b"]


def test_models_for_text_roles_chain_to_backup(monkeypatch):
    ar = _fresh_roles(monkeypatch, OLLAMA_MODEL="base:7b", OLLAMA_BACKUP_MODEL="backup:7b")
    assert ar.models_for(ar.ROLE_CHAT) == ["base:7b", "backup:7b"]
    assert ar.models_for(ar.ROLE_PLAN) == ["base:7b", "backup:7b"]


def test_models_for_vision_uses_vision_backup_not_text(monkeypatch):
    ar = _fresh_roles(
        monkeypatch,
        OLLAMA_MODEL="base:7b",
        OLLAMA_BACKUP_MODEL="textbackup:7b",
        OLLAMA_VLM_MODEL="vlm:7b",
        OLLAMA_VLM_BACKUP_MODEL="vlm-backup:7b",
    )
    # 비전은 텍스트 백업으로 내려가지 않고 비전 전용 백업만 쓴다.
    assert ar.models_for(ar.ROLE_VISION) == ["vlm:7b", "vlm-backup:7b"]


def test_models_for_dedupes_when_primary_equals_backup(monkeypatch):
    ar = _fresh_roles(monkeypatch, OLLAMA_MODEL="same:7b", OLLAMA_BACKUP_MODEL="same:7b")
    assert ar.models_for(ar.ROLE_CHAT) == ["same:7b"]   # 중복 제거


def test_backend_and_action_role_mapping(monkeypatch):
    ar = _fresh_roles(monkeypatch)
    assert ar.backend_for(ar.ROLE_REASON) == "external"
    assert ar.backend_for(ar.ROLE_VISION) == "vision"
    assert ar.backend_for(ar.ROLE_CHAT) == "local"
    assert ar.role_for_action("llm") == ar.ROLE_CHAT
    assert ar.role_for_action("summarize") == ar.ROLE_SUMMARIZE
    assert ar.role_for_action("web_ai_ask") == ar.ROLE_REASON
    assert ar.role_for_action("vision_describe") == ar.ROLE_VISION


def test_describe_lists_all_roles(monkeypatch):
    ar = _fresh_roles(monkeypatch)
    roles = {r["role"] for r in ar.describe()}
    assert roles == {ar.ROLE_PLAN, ar.ROLE_CHAT, ar.ROLE_SUMMARIZE, ar.ROLE_REASON, ar.ROLE_VISION}
    for r in ar.describe():
        assert r["backend"] and r["model"] and r["description"]


def test_planner_uses_planner_role_model():
    from core import ai_roles, planner

    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["model"] = kwargs.get("model")
        return '{"schema_version": 2, "steps": [{"action": "llm", "input": "x", "depends_on": null}]}'

    with patch.object(ai_roles, "model_for", return_value="planner-model") as mf, \
         patch("llm.ollama_client.generate", side_effect=fake_generate):
        planner.plan("뭐 좀 해줘")

    mf.assert_any_call(ai_roles.ROLE_PLAN)
    assert captured["model"] == "planner-model"


def test_llm_step_uses_chat_role_model():
    from executor import executor

    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["model"] = kwargs.get("model")
        return "답변"

    steps = [{"action": "llm", "input": "안녕", "depends_on": None}]
    with patch("executor.executor.ai_roles.model_for", return_value="chat-model") as mf, \
         patch("llm.ollama_client.generate", side_effect=fake_generate):
        executor.execute_steps(steps)

    mf.assert_called()        # role_for_action("llm")=chat 으로 모델 조회
    assert captured["model"] == "chat-model"


def test_browser_fallback_uses_reasoning_role_model():
    """브라우저 실패 → ollama 폴백은 reason(가장 강한 로컬) 역할 모델로 받는다."""
    from unittest.mock import MagicMock

    from executor import executor

    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("브라우저 죽음")
    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["model"] = kwargs.get("model")
        return "로컬 폴백 답변"

    def fake_model_for(role):
        return "reason-model" if role == executor.ai_roles.ROLE_REASON else "other"

    steps = [{"action": "browser_search", "input": "뉴스", "depends_on": None}]
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.ai_roles.model_for", side_effect=fake_model_for), \
         patch("llm.ollama_client.generate", side_effect=fake_generate):
        results = executor.execute_steps(steps)

    assert results[0]["status"] == "fallback"
    assert captured["model"] == "reason-model"

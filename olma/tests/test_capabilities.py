"""Capability Layer(core.capabilities) 테스트 — 액션 메타·위험도·승인 게이트 유도."""
import importlib


def _fresh(monkeypatch, required=""):
    monkeypatch.setenv("APPROVAL_REQUIRED_ACTIONS", required)
    import config.config as cfg
    importlib.reload(cfg)
    import core.capabilities as cap
    importlib.reload(cap)
    return cap


def test_get_known_action_has_full_meta(monkeypatch):
    cap = _fresh(monkeypatch)
    c = cap.get("browser_type")
    assert c["target"] == "browser"
    assert c["risk"] == "dangerous"
    assert c["reversible"] is False
    assert c["requires_approval"] is True
    assert c["description"]


def test_get_unknown_action_returns_none(monkeypatch):
    cap = _fresh(monkeypatch)
    assert cap.get("존재하지않음") is None


def test_safe_actions_not_gated(monkeypatch):
    cap = _fresh(monkeypatch)
    for a in ("llm", "summarize", "browser_get_text", "browser_screenshot",
              "browser_open", "browser_search", "vision_describe", "notification_check"):
        assert cap.get(a)["risk"] == "safe"
        assert cap.requires_approval(a) is False


def test_caution_actions_not_gated_by_default(monkeypatch):
    cap = _fresh(monkeypatch)
    assert cap.get("web_ai_ask")["risk"] == "caution"
    assert cap.requires_approval("web_ai_ask") is False
    assert cap.requires_approval("login") is False


def test_dangerous_actions_gated_by_default(monkeypatch):
    cap = _fresh(monkeypatch)
    assert cap.requires_approval("browser_click") is True
    assert cap.requires_approval("browser_type") is True


def test_required_override_gates_extra_actions(monkeypatch):
    cap = _fresh(monkeypatch, required="web_ai_ask, login")
    assert cap.requires_approval("web_ai_ask") is True   # override로 추가 게이트
    assert cap.requires_approval("login") is True
    assert cap.requires_approval("llm") is False         # 목록에 없으면 그대로


def test_unknown_action_requires_approval_conservatively(monkeypatch):
    cap = _fresh(monkeypatch)
    assert cap.requires_approval("미지의액션") is True


def test_describe_covers_all_actions(monkeypatch):
    cap = _fresh(monkeypatch)
    from core.schema import ActionType
    described = {c["action"] for c in cap.describe()}
    assert described == {a.value for a in ActionType}

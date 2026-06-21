import pytest

from core.router import _BROWSER_ACTIONS, _NOTIFIER_ACTIONS, _OLLAMA_ACTIONS, route, route_policy
from core.schema import ActionType


def test_every_action_type_maps_to_exactly_one_target():
    mapped = _OLLAMA_ACTIONS | _BROWSER_ACTIONS | _NOTIFIER_ACTIONS
    assert mapped == set(ActionType)
    assert len(_OLLAMA_ACTIONS) + len(_BROWSER_ACTIONS) + len(_NOTIFIER_ACTIONS) == len(ActionType)


@pytest.mark.parametrize(
    "action,target",
    [
        ("llm", "ollama"),
        ("summarize", "ollama"),
        ("browser_open", "browser"),
        ("browser_search", "browser"),
        ("browser_click", "browser"),
        ("browser_type", "browser"),
        ("browser_get_text", "browser"),
        ("browser_screenshot", "browser"),
        ("web_ai_ask", "browser"),
        ("notification_check", "notifier"),
    ],
)
def test_route_dispatches_to_correct_target(action, target):
    assert route({"action": action}) == target


def test_route_rejects_unknown_action():
    with pytest.raises(ValueError):
        route({"action": "delete_everything"})


def test_route_policy_browser_falls_back_to_ollama():
    policy = route_policy({"action": "browser_search", "input": "뉴스"})
    assert policy["target"] == "browser"
    assert policy["fallback"] == "ollama"
    assert policy["confidence"] == "high"


def test_route_policy_notifier_has_no_fallback():
    policy = route_policy({"action": "notification_check", "input": ""})
    assert policy["target"] == "notifier"
    assert policy["fallback"] is None


def test_route_policy_low_confidence_when_required_input_missing():
    policy = route_policy({"action": "browser_open", "input": ""})
    assert policy["confidence"] == "low"


def test_route_policy_web_ai_latest_info_has_no_local_fallback():
    # 최신·실시간 정보(search 분류)는 로컬로 폴백하면 옛 정보가 되므로 폴백을 끈다.
    policy = route_policy({"action": "web_ai_ask", "input": "오늘 환율 알려줘"})
    assert policy["target"] == "browser"
    assert policy["fallback"] is None


def test_route_policy_web_ai_non_search_keeps_local_fallback():
    # 코딩/일반 web_ai_ask는 웹 AI 전멸 시 로컬 best-effort가 의미 있어 폴백 유지.
    policy = route_policy({"action": "web_ai_ask", "input": "이 코드 리뷰 해줘"})
    assert policy["fallback"] == "ollama"


def test_route_policy_login_and_vision_have_no_fallback():
    assert route_policy({"action": "login", "input": ""})["fallback"] is None
    assert route_policy({"action": "vision_describe", "input": ""})["fallback"] is None

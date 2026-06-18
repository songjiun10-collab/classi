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

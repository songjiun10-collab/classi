import pytest

from core.router import _BROWSER_ACTIONS, _NOTIFIER_ACTIONS, _OLLAMA_ACTIONS, route
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
        ("notification_check", "notifier"),
    ],
)
def test_route_dispatches_to_correct_target(action, target):
    assert route({"action": action}) == target


def test_route_rejects_unknown_action():
    with pytest.raises(ValueError):
        route({"action": "delete_everything"})

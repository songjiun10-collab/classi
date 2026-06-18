"""Planner가 만든 step을 보고 어디서 실행할지 결정한다."""
from core.schema import ActionType

VALID_TARGETS = {"ollama", "browser", "notifier"}

_OLLAMA_ACTIONS = {ActionType.LLM, ActionType.SUMMARIZE}
_BROWSER_ACTIONS = {
    ActionType.BROWSER_OPEN,
    ActionType.BROWSER_SEARCH,
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_TYPE,
    ActionType.BROWSER_GET_TEXT,
    ActionType.BROWSER_SCREENSHOT,
}
_NOTIFIER_ACTIONS = {ActionType.NOTIFICATION_CHECK}


def route(step: dict) -> str:
    raw_action = step.get("action", "")
    try:
        action = ActionType(raw_action)
    except ValueError:
        raise ValueError(f"알 수 없는 action: {raw_action!r}")

    if action in _OLLAMA_ACTIONS:
        return "ollama"
    if action in _BROWSER_ACTIONS:
        return "browser"
    if action in _NOTIFIER_ACTIONS:
        return "notifier"

    # ActionType에 새 멤버가 추가됐는데 위 세 집합 어디에도 매핑되지 않은 경우를
    # 즉시 드러내기 위한 안전장치(정상 동작 시 도달 불가능).
    raise AssertionError(f"ActionType.{action.name}이 라우팅 집합 중 어디에도 속하지 않음")

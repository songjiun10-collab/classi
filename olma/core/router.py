"""Planner가 만든 step을 보고 어디서 실행할지 결정하는 정책 엔진.

기본 라우팅(action -> 실행 타겟)은 결정론적이지만, 그 위에 (1) 타겟 실패 시
대체 타겟(fallback chain)과 (2) 입력 완결성 기반의 단순 confidence를 얹는다.
confidence는 ML이 아니라 "필수 입력이 채워졌는가" 휴리스틱이다(과도한 추상화 회피)."""
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

# 타겟이 죽었을 때 대신 시도할 타겟. browser 실패는 "아는 선에서 답해" 식으로 ollama가 받는다.
# notifier는 읽기 전용/외부 의존이라 가짜 대체를 만들지 않는다(None).
_FALLBACK_TARGET = {
    "browser": "ollama",
    "ollama": None,
    "notifier": None,
}

# 입력이 비어 있으면 안 되는 action들(비면 confidence를 낮춘다).
_INPUT_REQUIRED = {
    ActionType.BROWSER_OPEN,
    ActionType.BROWSER_SEARCH,
    ActionType.BROWSER_CLICK,
    ActionType.BROWSER_TYPE,
    ActionType.LLM,
}


def _parse_action(step: dict) -> ActionType:
    raw_action = step.get("action", "")
    try:
        return ActionType(raw_action)
    except ValueError:
        raise ValueError(f"알 수 없는 action: {raw_action!r}")


def route(step: dict) -> str:
    """action -> 실행 타겟 문자열. 미등록 action이면 ValueError."""
    action = _parse_action(step)

    if action in _OLLAMA_ACTIONS:
        return "ollama"
    if action in _BROWSER_ACTIONS:
        return "browser"
    if action in _NOTIFIER_ACTIONS:
        return "notifier"

    # ActionType에 새 멤버가 추가됐는데 위 세 집합 어디에도 매핑되지 않은 경우를
    # 즉시 드러내기 위한 안전장치(정상 동작 시 도달 불가능).
    raise AssertionError(f"ActionType.{action.name}이 라우팅 집합 중 어디에도 속하지 않음")


def route_policy(step: dict) -> dict:
    """타겟 + 대체 타겟 + confidence를 함께 돌려준다."""
    action = _parse_action(step)
    target = route(step)

    has_input = bool(str(step.get("input", "")).strip())
    confidence = "low" if (action in _INPUT_REQUIRED and not has_input) else "high"

    return {
        "target": target,
        "fallback": _FALLBACK_TARGET.get(target),
        "confidence": confidence,
    }

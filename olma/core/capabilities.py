"""Capability Layer — Olma가 '무엇을, 어떤 위험으로' 할 수 있는지의 단일 정의(SSOT).

schema.ActionType가 "어떤 액션이 있나"라면, 여기선 각 액션의 실행 메타를 더한다:
  target          실행 백엔드(ollama/browser/notifier) — router와 일치
  risk            safe(읽기/로컬) | caution(외부 전송·인증) | dangerous(상태 변경·비가역)
  reversible      되돌릴 수 있는 동작인가(되돌리기 어려우면 승인 가치↑)
  requires_approval  Human Approval Gate가 기본으로 막을지(위험도에서 유도, config로 가감)

이 레이어는 두 곳에 쓰인다: (1) Human Approval Gate가 어떤 step을 사람 승인 앞에 세울지
판단하고, (2) /api/capabilities로 '내가 할 수 있는 일'을 외부에 노출한다. 위험도/타겟은
임의 추측이 아니라 실제 동작 성격에서 정한다(예: browser_type/click은 페이지 상태를 바꾸므로
dangerous, web_ai_ask·login은 외부 전송/인증이라 caution, 나머지 읽기/로컬은 safe)."""
from __future__ import annotations

from config.config import APPROVAL_REQUIRED_ACTIONS
from core.schema import ACTION_DESCRIPTIONS, ActionType

# 액션별 (target, risk, reversible). requires_approval은 risk에서 유도한다.
_RISK = {
    ActionType.LLM: ("ollama", "safe", True),
    ActionType.SUMMARIZE: ("ollama", "safe", True),
    ActionType.VISION_DESCRIBE: ("browser", "safe", True),       # 화면 읽기(VLM), 상태 불변
    ActionType.BROWSER_OPEN: ("browser", "safe", True),
    ActionType.BROWSER_SEARCH: ("browser", "safe", True),
    ActionType.BROWSER_GET_TEXT: ("browser", "safe", True),
    ActionType.BROWSER_SCREENSHOT: ("browser", "safe", True),
    ActionType.NOTIFICATION_CHECK: ("notifier", "safe", True),
    ActionType.WEB_AI_ASK: ("browser", "caution", True),         # 외부 웹 AI로 데이터 전송
    ActionType.LOGIN: ("browser", "caution", True),              # 인증 흐름
    ActionType.BROWSER_CLICK: ("browser", "dangerous", False),   # 페이지 상태 변경(전송/구매 등)
    ActionType.BROWSER_TYPE: ("browser", "dangerous", False),    # 입력 → 비가역 부작용 가능
}

# risk → 승인 기본 요구 여부. dangerous만 기본 게이트 대상.
_RISK_GATES = {"safe": False, "caution": False, "dangerous": True}


def _required_overrides() -> set:
    return {a.strip() for a in APPROVAL_REQUIRED_ACTIONS.split(",") if a.strip()}


def get(action) -> dict | None:
    """액션 1건의 capability 메타. 알 수 없는 액션이면 None."""
    try:
        at = ActionType(action)
    except ValueError:
        return None
    target, risk, reversible = _RISK[at]
    name = at.value
    return {
        "action": name,
        "target": target,
        "risk": risk,
        "reversible": reversible,
        "requires_approval": _RISK_GATES[risk] or name in _required_overrides(),
        "description": ACTION_DESCRIPTIONS.get(at, ""),
    }


def describe() -> list:
    """모든 액션의 capability 목록(/api/capabilities용)."""
    return [get(at.value) for at in ActionType]


def requires_approval(action) -> bool:
    """이 액션이 사람 승인 게이트 대상인가. 알 수 없는 액션은 보수적으로 True(승인 요구)."""
    cap = get(action)
    return True if cap is None else cap["requires_approval"]

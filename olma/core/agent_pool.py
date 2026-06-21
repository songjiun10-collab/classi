"""Agent Pool — 파이프라인의 마지막 레이어. step을 '누가 실행하나'를 정하는 에이전트 풀.

Workflow Builder가 만든 step들은 결국 어떤 실행 주체(agent)에게 넘어간다. 지금까지는
executor가 router의 target(ollama/browser/notifier)으로 곧장 디스패치했는데, 이 모듈은 그
디스패치를 '명명된 에이전트'의 풀로 형식화한다 — 각 에이전트는 자기가 처리하는 action들과
실제 실행 target을 선언한다. 덕분에 (1) "지금 어떤 에이전트들이 있나"를 조회·노출할 수
있고, (2) register()로 전문 에이전트(Vision/Research/Reviewer …, #21~30)를 런타임에 끼워
넣을 수 있다(dynamic). 실행 의미는 그대로다 — agent는 여전히 target으로 실행된다.

내장 에이전트는 capabilities(액션별 target)에서 유도해 SSOT 중복을 피한다."""
from __future__ import annotations

from core import capabilities
from core.logger import get_logger
from core.schema import ActionType

log = get_logger("agent_pool")

# 내장 에이전트: 실행 의미(target)는 같지만, 미래의 전문화(#21~30)를 위해 논리적으로 나눈다.
# action → 어느 에이전트가 맡나. (capabilities가 각 action의 target을 들고 있으므로 여기선
# '묶음'만 정의한다.)
_BUILTIN_GROUPS = {
    "local_llm": (["llm", "summarize"], "로컬 LLM 추론/요약"),
    "web_ai": (["web_ai_ask"], "외부 웹 AI 호출(브라우저 경유)"),
    "vision": (["vision_describe"], "화면 이해(VLM)"),
    "notifier": (["notification_check"], "알림 읽기/분류"),
    "browser": (
        ["browser_open", "browser_search", "browser_click", "browser_type",
         "browser_get_text", "browser_screenshot", "login"],
        "브라우저 조작(열기/검색/클릭/입력/추출/로그인)",
    ),
}

# 런타임 등록 에이전트(dynamic). action → agent descriptor. 내장보다 우선한다.
_registered: dict = {}


def _builtin_for(action: str):
    for name, (actions, desc) in _BUILTIN_GROUPS.items():
        if action in actions:
            cap = capabilities.get(action)
            return {"agent": name, "target": cap["target"] if cap else None, "description": desc}
    return None


def register(name: str, handles: list, target: str, description: str = "") -> None:
    """전문 에이전트를 런타임에 추가한다. handles의 각 action을 이 에이전트가 맡게 된다
    (내장보다 우선). target은 실제 실행 백엔드(ollama/browser/notifier)."""
    if not name or not handles:
        raise ValueError("agent 이름과 handles는 비어 있을 수 없습니다")
    for action in handles:
        _registered[action] = {"agent": name, "target": target, "description": description}
    log.info("에이전트 등록: %s (handles=%s, target=%s)", name, handles, target)


def unregister(name: str) -> bool:
    """이름으로 등록 에이전트를 제거한다(내장은 영향 없음)."""
    before = len(_registered)
    for action in [a for a, d in _registered.items() if d["agent"] == name]:
        del _registered[action]
    return len(_registered) < before


def for_action(action: str) -> dict | None:
    """이 action을 맡는 에이전트 descriptor. 등록 에이전트 우선, 없으면 내장, 둘 다 없으면 None."""
    return _registered.get(action) or _builtin_for(action)


def resolve(step: dict) -> dict | None:
    return for_action(step.get("action", ""))


def agents() -> list:
    """현재 풀의 모든 에이전트(내장 + 등록)를 묶어서 반환. 같은 이름은 handles를 합친다."""
    merged: dict = {}
    for at in ActionType:
        action = at.value
        d = for_action(action)
        if d is None:
            continue
        entry = merged.setdefault(
            d["agent"], {"agent": d["agent"], "target": d["target"],
                         "description": d["description"], "handles": []}
        )
        entry["handles"].append(action)
    return list(merged.values())

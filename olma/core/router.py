"""Planner가 만든 step을 보고 어디서 실행할지 결정한다."""

VALID_TARGETS = {"ollama", "browser", "notifier"}


def route(step: dict) -> str:
    action = step.get("action", "")

    if action in ("llm", "summarize"):
        return "ollama"
    if action.startswith("browser_"):
        return "browser"
    if action == "notification_check":
        return "notifier"

    raise ValueError(f"알 수 없는 action: {action!r}")

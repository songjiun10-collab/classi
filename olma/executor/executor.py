"""step list를 순차 실행하고 결과를 모은다. 실패 시 step당 1회 재시도."""
from config.config import RETRY_COUNT
from core import notifier, router
from llm import ollama_client
from tools.browser import Browser


def _run_ollama(step: dict) -> str:
    action = step["action"]
    text = step.get("input", "")
    if action == "summarize":
        prompt = f"다음 내용을 한국어로 간결하게 요약해라:\n\n{text}"
    else:
        prompt = text
    return ollama_client.generate(prompt)


def _run_browser(step: dict, browser: Browser) -> str:
    action = step["action"]
    arg = step.get("input", "")

    if action == "browser_open":
        browser.open(arg)
        return f"opened {arg}"
    if action == "browser_search":
        return browser.search(arg)
    if action == "browser_click":
        browser.click(arg)
        return f"clicked {arg}"
    if action == "browser_type":
        selector, _, text = arg.partition("|||")
        browser.type(selector, text)
        return f"typed into {selector}"
    if action == "browser_get_text":
        return browser.get_text()
    if action == "browser_screenshot":
        return browser.screenshot()

    raise ValueError(f"알 수 없는 browser action: {action}")


def execute_steps(steps: list) -> list:
    results = []
    browser = None
    needs_browser = any(router.route(s) in ("browser", "notifier") for s in steps)

    try:
        if needs_browser:
            browser = Browser()
            browser.__enter__()

        for step in steps:
            target = router.route(step)
            attempts = RETRY_COUNT + 1
            last_error = None
            status = "failed"
            result = None

            for _ in range(attempts):
                try:
                    if target == "ollama":
                        result = _run_ollama(step)
                    elif target == "browser":
                        result = _run_browser(step, browser)
                    elif target == "notifier":
                        result = notifier.analyze_notifications(browser)
                    status = "ok"
                    break
                except Exception as exc:
                    last_error = exc

            results.append(
                {
                    "step": step,
                    "result": result if status == "ok" else f"error: {last_error}",
                    "status": status,
                }
            )
    finally:
        if browser:
            browser.__exit__(None, None, None)

    return results

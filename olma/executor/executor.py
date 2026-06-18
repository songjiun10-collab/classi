"""step list를 순차 실행하고 결과를 모은다. 실패 시 step당 1회 재시도."""
from config.config import OLLAMA_TEMPERATURE_DEFAULT, RETRY_COUNT
from core import notifier, router
from core.schema import EXTERNAL_DATA_BEGIN, EXTERNAL_DATA_END, MAX_INPUT_CHARS
from llm import ollama_client
from tools.browser import Browser

_RESULT_TOKEN = "{{result}}"


def _wrap_external(text: str) -> str:
    text = text[:MAX_INPUT_CHARS]
    return f"{EXTERNAL_DATA_BEGIN}\n{text}\n{EXTERNAL_DATA_END}"


def _substitute_dependency(step: dict, results_by_index: dict) -> dict:
    depends_on = step.get("depends_on")
    text = step.get("input", "")
    if depends_on is None or _RESULT_TOKEN not in text:
        return step

    prior = results_by_index.get(depends_on)
    if prior is None:
        return step

    step = dict(step)
    step["input"] = text.replace(_RESULT_TOKEN, _wrap_external(str(prior["result"])))
    return step


def _run_ollama(step: dict) -> str:
    action = step["action"]
    text = step.get("input", "")
    if action == "summarize":
        prompt = f"다음 내용을 한국어로 간결하게 요약해라:\n\n{text}"
    else:
        prompt = text
    return ollama_client.generate(prompt, temperature=OLLAMA_TEMPERATURE_DEFAULT)


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
    results_by_index = {}
    browser = None
    needs_browser = any(router.route(s) in ("browser", "notifier") for s in steps)

    try:
        if needs_browser:
            browser = Browser()
            browser.__enter__()

        for i, raw_step in enumerate(steps):
            step = _substitute_dependency(raw_step, results_by_index)
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

            record = {
                "step": step,
                "result": result if status == "ok" else f"error: {last_error}",
                "status": status,
            }
            results.append(record)
            results_by_index[i] = record
    finally:
        if browser:
            browser.__exit__(None, None, None)

    return results

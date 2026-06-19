"""step list를 순차 실행하고 결과를 모은다 (fault-tolerant 시스템).

각 step은 재시도(backoff 포함) → 그래도 실패하면 대체 타겟으로 폴백 → 그래도 안 되면
실패로 기록하고 다음 step을 계속 진행한다. 모든 단계가 구조적으로 로깅된다.

timeout은 신호(signal)로 강제 종료하지 않고 I/O 계층(Ollama 요청 timeout,
Playwright 동작 timeout)에서 적용한다 — Playwright sync 객체를 스레드/신호로
강제 중단하면 브라우저 상태가 깨지기 때문(과도한 추상화/취약성 회피)."""
import time

from config.config import (
    OLLAMA_TEMPERATURE_DEFAULT,
    RETRY_BACKOFF,
    RETRY_COUNT,
    STEP_TIMEOUT,
    WEB_AI_INPUT_SELECTOR,
    WEB_AI_RESPONSE_SELECTOR,
    WEB_AI_SUBMIT_SELECTOR,
    WEB_AI_URL,
    WEB_AI_WAIT_MS,
)
from core import notifier, router
from core.logger import get_logger
from core.schema import EXTERNAL_DATA_BEGIN, EXTERNAL_DATA_END, MAX_INPUT_CHARS
from llm import ollama_client
from tools.browser import Browser

log = get_logger("executor")

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
    return ollama_client.generate(
        prompt, timeout=STEP_TIMEOUT, temperature=OLLAMA_TEMPERATURE_DEFAULT
    )


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
    if action == "web_ai_ask":
        return browser.ask_web_ai(
            arg,
            WEB_AI_URL,
            WEB_AI_INPUT_SELECTOR,
            WEB_AI_SUBMIT_SELECTOR,
            WEB_AI_RESPONSE_SELECTOR,
            WEB_AI_WAIT_MS,
        )

    raise ValueError(f"알 수 없는 browser action: {action}")


def _run_target(target: str, step: dict, browser: Browser) -> str:
    if target == "ollama":
        return _run_ollama(step)
    if target == "browser":
        return _run_browser(step, browser)
    if target == "notifier":
        return notifier.analyze_notifications(browser)
    raise ValueError(f"알 수 없는 target: {target}")


def _run_with_retries(target: str, step: dict, browser: Browser):
    """(result, status, attempts, error) 반환. status는 'ok' 또는 'failed'."""
    attempts = RETRY_COUNT + 1
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            result = _run_target(target, step, browser)
            return result, "ok", attempt, None
        except Exception as exc:
            last_error = exc
            log.warning("step 실패 (%s, 시도 %d/%d): %s", step.get("action"), attempt, attempts, exc)
            if attempt < attempts:
                time.sleep(RETRY_BACKOFF * (2 ** (attempt - 1)))
            elif browser is not None and target in ("browser", "notifier"):
                browser.debug_screenshot(step.get("action", "step"))
    return None, "failed", attempts, last_error


def _run_fallback(fallback_target: str, step: dict, browser: Browser):
    """대체 타겟으로 1회 시도. browser 실패를 ollama가 '아는 선에서' 받는 식."""
    action = step.get("action", "")
    user_input = step.get("input", "")
    if fallback_target == "ollama":
        prompt = (
            f"브라우저 동작('{action}', 입력: '{user_input[:MAX_INPUT_CHARS]}')이 실패했다. "
            "브라우저 없이 네가 아는 지식 범위에서 사용자를 최대한 도와라. "
            "모르면 모른다고 솔직히 답해라."
        )
        try:
            result = ollama_client.generate(
                prompt, timeout=STEP_TIMEOUT, temperature=OLLAMA_TEMPERATURE_DEFAULT
            )
            return result, "fallback"
        except Exception as exc:
            log.warning("폴백(%s) 실패: %s", fallback_target, exc)
    return None, "failed"


_SUCCESS_STATUSES = ("ok", "fallback")


def _step_needs_browser(step: dict) -> bool:
    try:
        return router.route(step) in ("browser", "notifier")
    except Exception:
        # 미등록 action 등 잘못된 step은 실행 시점에 _run_step에서 그 step만 실패로
        # 기록된다 — 여기서는 browser를 미리 띄울지 판단만 하므로 보수적으로 False.
        return False


def _failed_dependency_record(raw_step: dict, depends_on: int) -> dict:
    return {
        "step": raw_step,
        "action": raw_step.get("action"),
        "target": None,
        "result": f"skipped: 의존 step {depends_on}이 실패해 실행하지 않음",
        "status": "skipped",
        "attempts": 0,
        "duration": 0.0,
        "error": None,
        "confidence": None,
    }


def _exception_record(raw_step, exc: Exception) -> dict:
    return {
        "step": raw_step,
        "action": raw_step.get("action") if isinstance(raw_step, dict) else None,
        "target": None,
        "result": f"error: {exc}",
        "status": "failed",
        "attempts": 0,
        "duration": 0.0,
        "error": str(exc),
        "confidence": None,
    }


def _run_step(i: int, raw_step: dict, results_by_index: dict, browser: Browser) -> dict:
    depends_on = raw_step.get("depends_on")
    if depends_on is not None:
        prior = results_by_index.get(depends_on)
        if prior is not None and prior["status"] not in _SUCCESS_STATUSES:
            # 실패/스킵된 step의 결과를 에러 문자열로 받아 실행을 강행하는 대신,
            # 의존 step이 죽었다는 사실 자체를 그대로 전파한다(연쇄 스킵 포함).
            log.info("step %d: 의존 step %d 실패로 건너뜀", i, depends_on)
            return _failed_dependency_record(raw_step, depends_on)

    step = _substitute_dependency(raw_step, results_by_index)
    policy = router.route_policy(step)
    target = policy["target"]
    log.info(
        "step %d 실행: action=%s target=%s confidence=%s",
        i, step.get("action"), target, policy["confidence"],
    )

    started = time.monotonic()
    result, status, attempts, error = _run_with_retries(target, step, browser)

    if status == "failed" and policy["fallback"]:
        log.info("step %d 대체 타겟(%s)으로 폴백 시도", i, policy["fallback"])
        fb_result, fb_status = _run_fallback(policy["fallback"], step, browser)
        if fb_status != "failed":
            result, status = fb_result, fb_status

    duration = round(time.monotonic() - started, 3)
    return {
        "step": step,
        "action": step.get("action"),
        "target": target,
        "result": result if status in _SUCCESS_STATUSES else f"error: {error}",
        "status": status,
        "attempts": attempts,
        "duration": duration,
        "error": None if status in _SUCCESS_STATUSES else str(error),
        "confidence": policy["confidence"],
    }


def execute_steps(steps: list) -> list:
    results = []
    results_by_index = {}
    browser = None
    needs_browser = any(_step_needs_browser(s) for s in steps)

    log.info("실행 시작: step %d개 (browser 필요=%s)", len(steps), needs_browser)

    try:
        if needs_browser:
            browser = Browser()
            browser.__enter__()

        for i, raw_step in enumerate(steps):
            try:
                record = _run_step(i, raw_step, results_by_index, browser)
            except Exception as exc:
                # route_policy/조립 단계의 예상치 못한 예외가 전체 배치를 날리지
                # 않도록, 이 step만 실패로 기록하고 나머지 step은 계속 진행한다.
                log.error("step %d 처리 중 예외, 해당 step만 실패로 기록: %s", i, exc)
                record = _exception_record(raw_step, exc)
            results.append(record)
            results_by_index[i] = record
            log.info("step %d 완료: status=%s (%.3fs)", i, record["status"], record["duration"])
    finally:
        if browser:
            browser.__exit__(None, None, None)

    return results

"""step list를 순차 실행하고 결과를 모은다 (fault-tolerant 시스템).

각 step은 재시도(backoff 포함) → 그래도 실패하면 대체 타겟으로 폴백 → 그래도 안 되면
실패로 기록하고 다음 step을 계속 진행한다. 모든 단계가 구조적으로 로깅된다.

timeout은 신호(signal)로 강제 종료하지 않고 I/O 계층(Ollama 요청 timeout,
Playwright 동작 timeout)에서 적용한다 — Playwright sync 객체를 스레드/신호로
강제 중단하면 브라우저 상태가 깨지기 때문(과도한 추상화/취약성 회피)."""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from config.config import (
    AUTO_RELOGIN,
    BROWSER_USER_DATA_DIR,
    LOGIN_CHECK_MS,
    OLLAMA_PARALLEL,
    OLLAMA_PARALLEL_MAX,
    OLLAMA_TEMPERATURE_DEFAULT,
    RETRY_BACKOFF,
    RETRY_COUNT,
    STEP_TIMEOUT,
)
from core import agent_pool, ai_roles, approval, login_providers, notifier, router, web_ai_providers
from core.logger import get_logger
from core.schema import EXTERNAL_DATA_BEGIN, EXTERNAL_DATA_END, MAX_INPUT_CHARS
from llm import ollama_client
from tools.browser import Browser, resolve_profile_dir

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


def _run_ollama(step: dict, on_token=None) -> str:
    action = step["action"]
    text = step.get("input", "")
    if action == "summarize":
        prompt = f"다음 내용을 한국어로 간결하게 요약해라:\n\n{text}"
    else:
        prompt = text
    # 역할별 모델 분담: llm→chat, summarize→summarize 역할 모델(기본은 둘 다 OLLAMA_MODEL).
    models = ai_roles.models_for(ai_roles.role_for_action(action))
    if on_token is not None:
        # 스트리밍: 청크를 콜백으로 흘려보내면서 최종 결과로 누적한다(메모리/의존성용).
        # 스트리밍은 부분 출력 후 모델 전환 시 출력이 중복되므로 1순위 모델만 쓴다(백업 미적용).
        parts = []
        for piece in ollama_client.generate_stream(
            prompt, model=models[0], timeout=STEP_TIMEOUT, temperature=OLLAMA_TEMPERATURE_DEFAULT
        ):
            parts.append(piece)
            on_token(piece)
        return "".join(parts)
    return ollama_client.generate(
        prompt, model=models[0], fallback_models=models[1:],
        timeout=STEP_TIMEOUT, temperature=OLLAMA_TEMPERATURE_DEFAULT,
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
        providers, prompt = web_ai_providers.resolve_chain(arg)
        return _ask_web_ai_chain(browser, providers, prompt)
    if action == "login":
        provider = login_providers.resolve(arg)
        return browser.ensure_logged_in(
            provider["url"],
            provider["logged_in_selector"],
            provider["google_button_selector"],
            provider["wait_ms"],
            check_ms=LOGIN_CHECK_MS,
        )
    if action == "vision_describe":
        # 'vision' 역할: 현재 화면을 스크린샷해 로컬 VLM이 이해/설명한다. input은 보는 관점
        # (질문/지시)이며 비우면 화면 전반을 설명한다. URL을 먼저 열려면 browser_open과 함께 쓴다.
        # 비전 모델도 1순위 실패 시 백업 비전 모델로 페일오버한다(ai_roles vision 체인).
        vision_models = ai_roles.models_for(ai_roles.ROLE_VISION)
        return browser.describe_screen(
            arg, model=vision_models[0], fallback_models=vision_models[1:],
        )

    raise ValueError(f"알 수 없는 browser action: {action}")


def _ask_web_ai_chain(browser: Browser, providers: list, prompt: str) -> str:
    """웹 AI 제공자 체인을 순서대로 시도해 첫 성공을 돌려준다(예: claude 실패 → zai 백업).

    모두 실패하면 마지막 오류를 올려 보낸다 — 그러면 상위 라우터 폴백(로컬 reason 모델)이
    받는다. 체인이 비면(웹 AI 미설정) ask_web_ai의 명확한 설정 오류가 그대로 난다."""
    if not providers:
        # 미설정 시 ask_web_ai가 던지는 친절한 설정 오류를 그대로 유도한다.
        return browser.ask_web_ai(prompt, "", "")
    last_error = None
    for idx, provider in enumerate(providers):
        try:
            return browser.ask_web_ai(
                prompt,
                provider["url"],
                provider["input_selector"],
                provider["submit_selector"],
                provider["response_selector"],
                provider["wait_ms"],
            )
        except Exception as exc:
            last_error = exc
            if idx + 1 < len(providers):
                log.warning("웹 AI 제공자 실패(%s), 백업으로 페일오버: %s", provider.get("url"), exc)
    raise last_error


def _run_target(target: str, step: dict, browser: Browser, on_token=None) -> str:
    if target == "ollama":
        return _run_ollama(step, on_token=on_token)
    if target == "browser":
        return _run_browser(step, browser)
    if target == "notifier":
        return notifier.analyze_notifications(browser)
    raise ValueError(f"알 수 없는 target: {target}")


def _run_with_retries(target: str, step: dict, browser: Browser, on_token=None):
    """(result, status, attempts, error) 반환. status는 'ok' 또는 'failed'.

    스트리밍(on_token) 중 ollama가 일부 청크를 낸 뒤 실패하면 재시도가 그 일부를 다시
    내보낼 수 있다 — 최종 result는 마지막 성공 시도 기준으로 정확하다(드문 표시상 중복)."""
    attempts = RETRY_COUNT + 1
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            result = _run_target(target, step, browser, on_token=on_token)
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
            # 외부 웹 AI/브라우저가 받던 고난도 작업의 로컬 폴백이므로 'reason' 역할 모델
            # 체인(가장 강한 로컬 모델 + 백업)으로 받는다 — 없으면 기본 모델로 폴백된다(ai_roles).
            reason_models = ai_roles.models_for(ai_roles.ROLE_REASON)
            result = ollama_client.generate(
                prompt, model=reason_models[0], fallback_models=reason_models[1:],
                timeout=STEP_TIMEOUT, temperature=OLLAMA_TEMPERATURE_DEFAULT,
            )
            return result, "fallback"
        except Exception as exc:
            log.warning("폴백(%s) 실패: %s", fallback_target, exc)
    return None, "failed"


_SUCCESS_STATUSES = ("ok", "fallback")


def _maybe_relogin_and_retry(i: int, step: dict, target: str, browser: Browser, on_token=None):
    """브라우저 step이 실패했을 때, 현재 페이지에 로그인 벽이 감지되면 default 제공자로
    자동 재로그인한 뒤 그 step을 1회 재시도한다(AUTO_RELOGIN=true일 때만).

    반환: 성공 시 (result, status, attempts, error) 튜플, 그 외 None. login_wall_selector가
    설정된 default 제공자가 있을 때만 동작해 오탐으로 멀쩡한 실패를 재로그인으로 덮지 않는다.
    login 액션 자신은 재귀 방지를 위해 대상에서 제외한다."""
    if not AUTO_RELOGIN or browser is None or target not in ("browser", "notifier"):
        return None
    if step.get("action") == "login":
        return None
    provider = login_providers.default_for_auto()
    if provider is None:
        return None
    try:
        if not browser.is_present(provider["login_wall_selector"], LOGIN_CHECK_MS):
            return None  # 로그인 벽이 아니면(다른 이유의 실패) 건드리지 않는다
        log.info("step %d: 로그인 벽 감지 → 자동 재로그인 후 재시도", i)
        browser.ensure_logged_in(
            provider["url"], provider["logged_in_selector"],
            provider["google_button_selector"], provider["wait_ms"], check_ms=LOGIN_CHECK_MS,
        )
    except Exception as exc:
        log.warning("자동 재로그인 실패(원래 실패 유지): %s", exc)
        return None
    return _run_with_retries(target, step, browser, on_token=on_token)


def _step_needs_browser(step: dict) -> bool:
    try:
        return router.route(step) in ("browser", "notifier")
    except Exception:
        # 미등록 action 등 잘못된 step은 실행 시점에 _run_step에서 그 step만 실패로
        # 기록된다 — 여기서는 browser를 미리 띄울지 판단만 하므로 보수적으로 False.
        return False


def _step_is_ollama(step: dict) -> bool:
    try:
        return router.route(step) == "ollama"
    except Exception:
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


def _rejected_record(step: dict, reason: str) -> dict:
    return {
        "step": step,
        "action": step.get("action"),
        "target": None,
        "result": f"rejected: {reason}",
        "status": "rejected",
        "attempts": 0,
        "duration": 0.0,
        "error": reason,
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


def _run_step(i: int, raw_step: dict, results_by_index: dict, browser: Browser,
              on_token=None) -> dict:
    depends_on = raw_step.get("depends_on")
    if depends_on is not None:
        prior = results_by_index.get(depends_on)
        if prior is not None and prior["status"] not in _SUCCESS_STATUSES:
            # 실패/스킵된 step의 결과를 에러 문자열로 받아 실행을 강행하는 대신,
            # 의존 step이 죽었다는 사실 자체를 그대로 전파한다(연쇄 스킵 포함).
            log.info("step %d: 의존 step %d 실패로 건너뜀", i, depends_on)
            return _failed_dependency_record(raw_step, depends_on)

    step = _substitute_dependency(raw_step, results_by_index)

    # Human Approval Gate: 위험·비가역 액션이면 사람 승인 전까지 실행하지 않는다(게이트 off면
    # 즉시 통과). 거절/만료 시 그 step만 'rejected'로 기록하고 다음 step은 계속 진행한다.
    allowed, reason = approval.guard(step)
    if not allowed:
        log.info("step %d: 승인 거절/만료로 실행 안 함 (%s)", i, reason)
        return _rejected_record(step, reason)

    policy = router.route_policy(step)
    target = policy["target"]
    log.info(
        "step %d 실행: action=%s target=%s confidence=%s",
        i, step.get("action"), target, policy["confidence"],
    )

    started = time.monotonic()
    result, status, attempts, error = _run_with_retries(target, step, browser, on_token=on_token)

    if status == "failed":
        relogin = _maybe_relogin_and_retry(i, step, target, browser, on_token=on_token)
        if relogin is not None:
            r2, s2, a2, e2 = relogin
            if s2 != "failed":
                log.info("step %d: 자동 재로그인 후 재시도 성공", i)
                result, status, attempts, error = r2, s2, a2, e2

    if status == "failed" and policy["fallback"]:
        log.info("step %d 대체 타겟(%s)으로 폴백 시도", i, policy["fallback"])
        fb_result, fb_status = _run_fallback(policy["fallback"], step, browser)
        if fb_status != "failed":
            result, status = fb_result, fb_status

    duration = round(time.monotonic() - started, 3)
    agent = agent_pool.resolve(step)
    return {
        "step": step,
        "action": step.get("action"),
        "target": target,
        "agent": agent["agent"] if agent else None,   # 이 step을 맡은 에이전트(Agent Pool)
        "result": result if status in _SUCCESS_STATUSES else f"error: {error}",
        "status": status,
        "attempts": attempts,
        "duration": duration,
        "error": None if status in _SUCCESS_STATUSES else str(error),
        "confidence": policy["confidence"],
    }


def _safe_run_step(i: int, raw_step: dict, results_by_index: dict, browser: Browser,
                   on_token=None) -> dict:
    """_run_step을 감싸 예외가 나도 그 step만 실패로 기록한다(배치 전체를 보호)."""
    try:
        return _run_step(i, raw_step, results_by_index, browser, on_token=on_token)
    except Exception as exc:
        log.error("step %d 처리 중 예외, 해당 step만 실패로 기록: %s", i, exc)
        return _exception_record(raw_step, exc)


def _parallel_candidates(steps: list, on_token) -> list:
    """동시에 돌려도 안전한 ollama step 인덱스(의존성 없는 1차 웨이브). 조건: depends_on=null
    (다른 step 결과를 읽지 않음) + ollama 타겟(브라우저 같은 공유 상태 없음). 스트리밍
    (on_token) 시엔 출력 순서 보존을 위해 비활성. 2개 미만이면 병렬 이득이 없어 빈 리스트.

    웨이브 실행(_precompute_waves)의 깊이 0 부분집합이며, 후방 호환·진단용으로 남겨둔다."""
    if not OLLAMA_PARALLEL or on_token is not None:
        return []
    idxs = [
        i for i, s in enumerate(steps)
        if s.get("depends_on") is None and _step_is_ollama(s)
    ]
    return idxs if len(idxs) >= 2 else []


def _precompute_waves(steps: list, on_token) -> list:
    """병렬 선실행 가능한 ollama step들을 의존 깊이(wave)별로 묶어 반환한다.

    한 step이 'precomputable'하려면 ollama 타겟이고, 의존이 없거나(None) 의존 대상도
    precomputable해야 한다 — 즉 조상이 전부 ollama여야 한다. 브라우저 결과에 의존하는
    ollama는 제외된다(그 step은 본 실행 walk에서 브라우저 step 뒤에 순차로 돌아야 하므로).

    깊이 d 웨이브의 step은 깊이 <d 웨이브에만 의존하므로(같은 웨이브 내 상호 의존 없음)
    웨이브 단위로 안전하게 동시 실행할 수 있다. depends_on은 schema에서 항상 유효한 이전
    인덱스로 정규화되지만, 방어적으로 int·범위를 직접 확인한다.

    스트리밍/병렬 비활성이면 빈 리스트. 반환: [[idx,...](wave0), [idx,...](wave1), ...]."""
    if not OLLAMA_PARALLEL or on_token is not None:
        return []
    depth: dict = {}
    for i, s in enumerate(steps):
        if not _step_is_ollama(s):
            continue
        dep = s.get("depends_on")
        if dep is None:
            depth[i] = 0
        elif isinstance(dep, int) and dep in depth:
            depth[i] = depth[dep] + 1
        # 의존은 있으나 precomputable 아님(브라우저 의존 등) → 선실행 대상에서 제외
    waves_map: dict = {}
    for i, d in depth.items():       # i는 증가 순서라 각 웨이브도 인덱스 순서로 쌓인다
        waves_map.setdefault(d, []).append(i)
    return [waves_map[d] for d in sorted(waves_map)]


def _run_wave_parallel(wave: list, steps: list, results_by_index: dict) -> dict:
    """한 웨이브의 ollama step들을 동시 실행해 {idx: record}를 반환. 의존성 치환·실패 스킵은
    이미 채워진 results_by_index(이전 웨이브 결과)를 기준으로 처리된다. 웨이브 내 step끼리는
    상호 의존이 없어 같은 스냅샷을 공유해도 안전하다. 브라우저를 쓰지 않으므로 browser=None."""
    out: dict = {}
    workers = min(OLLAMA_PARALLEL_MAX, len(wave))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_safe_run_step, i, steps[i], results_by_index, None): i
            for i in wave
        }
        for fut in as_completed(futures):
            out[futures[fut]] = fut.result()
    return out


def _apply_backend_mode(step: dict) -> dict:
    """백엔드 토글(자동/로컬/웹)에 맞춰 step의 action을 재라우팅한다(필요할 때만 복사).

    웹 모드면 llm→web_ai_ask, 로컬 모드면 web_ai_ask→llm. 라우팅·웨이브 병렬 판단보다
    먼저 적용돼야 web_ai_ask가 브라우저(순차)로 올바르게 분류된다."""
    action = step.get("action")
    new_action = ai_roles.effective_action(action)
    return step if new_action == action else {**step, "action": new_action}


def execute_steps(steps: list, profile_slot: int = 0, on_token=None) -> list:
    """profile_slot은 작업 큐의 워커 인덱스다(core/task_queue.py). 0(기본값)이면 기존
    브라우저 프로필을 그대로 쓰고, 그 외에는 워커 전용 프로필로 격리해 여러 워커가
    동시에 Playwright 영구 프로필을 열어도 충돌하지 않게 한다.

    on_token(콜백)을 주면 ollama step의 생성 토큰을 청크 단위로 흘려보낸다(REPL 스트리밍).
    OLLAMA_PARALLEL=true면 의존성 그래프를 깊이별 웨이브로 나눠, 각 웨이브의 독립 ollama
    step들을 동시에 선실행한다(DAG 병렬). 브라우저 step과 브라우저 결과에 의존하는 step은
    상태·순서를 보존하기 위해 본 walk에서 순차 실행된다."""
    # 백엔드 토글(자동/로컬/웹)을 라우팅·병렬 판단 전에 먼저 반영한다.
    steps = [_apply_backend_mode(s) for s in steps]
    results_by_index: dict = {}
    browser = None
    needs_browser = any(_step_needs_browser(s) for s in steps)
    waves = _precompute_waves(steps, on_token)
    # 실제 병렬 이득이 있는 웨이브(>=2)가 하나도 없으면 선실행을 건너뛰고 순수 순차로 간다.
    use_waves = any(len(w) >= 2 for w in waves)

    log.info(
        "실행 시작: step %d개 (browser 필요=%s, profile_slot=%d, 병렬 웨이브=%s)",
        len(steps), needs_browser, profile_slot,
        [len(w) for w in waves] if use_waves else "off",
    )

    try:
        if needs_browser:
            browser = Browser(user_data_dir=resolve_profile_dir(BROWSER_USER_DATA_DIR, profile_slot))
            browser.__enter__()

        # 1) 선실행: precomputable ollama step을 웨이브 순서대로 처리한다. 웨이브 간에는
        #    순서를 지켜(깊은 step은 얕은 결과에 의존) 진행하고, 웨이브 내 2개 이상은 동시
        #    실행한다. 결과를 results_by_index에 채워 다음 웨이브·본 walk가 참조하게 한다.
        precomputed: dict = {}
        if use_waves:
            for wave in waves:
                if len(wave) >= 2:
                    done = _run_wave_parallel(wave, steps, results_by_index)
                else:
                    i = wave[0]
                    done = {i: _safe_run_step(i, steps[i], results_by_index, None)}
                precomputed.update(done)
                results_by_index.update(done)

        # 2) 순서대로 조립 — 선실행된 step은 결과를 재사용하고, 나머지(브라우저·브라우저
        #    의존 step)는 이전 결과를 보고 원래 인덱스 순서대로 실행한다.
        results = []
        for i, raw_step in enumerate(steps):
            if i in precomputed:
                record = precomputed[i]
            else:
                record = _safe_run_step(i, raw_step, results_by_index, browser, on_token=on_token)
            results.append(record)
            results_by_index[i] = record
            log.info("step %d 완료: status=%s (%.3fs)", i, record["status"], record["duration"])
    finally:
        if browser:
            browser.__exit__(None, None, None)

    return results

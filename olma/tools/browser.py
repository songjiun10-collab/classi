"""Playwright 기반 브라우저 엔진 (sync API). 카톡 등 로그인 세션 유지를 위해 영구 프로필 사용.

V0.2 안정화: 명시적 wait 전략, selector fallback(여러 후보 시도),
DOM 텍스트가 비면 OCR로 폴백, 동작 실패 시 디버그 스크린샷 저장,
web AI 응답은 고정 대기 대신 텍스트 길이 안정화 폴링으로 완료 감지."""
from __future__ import annotations

import base64
import os
import shutil
import time
from urllib.parse import quote_plus

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from config.config import (
    BROWSER_CHANNEL,
    BROWSER_HEADLESS,
    BROWSER_LOCALE,
    BROWSER_STEALTH,
    BROWSER_TIMEOUT,
    BROWSER_USER_AGENT,
    BROWSER_USER_DATA_DIR,
    CLOUDFLARE_WAIT_MS,
    SCREENSHOT_DIR,
    WEB_AI_POLL_INTERVAL_MS,
    WEB_AI_STABLE_POLLS,
    WEB_AI_WAIT_MS,
)
from core.logger import get_logger
from llm import ollama_client
from tools import ocr

log = get_logger("browser")

_VISION_DEFAULT_PROMPT = "이 화면을 한국어로 간결하게 설명해라. 무엇이 보이는지, 핵심 내용/상태를 알려줘."

# 봇 탐지/캡차 완화용 stealth. 자동화 플래그를 끄고, 페이지마다 흔한 탐지 신호
# (navigator.webdriver 등)를 사람 브라우저처럼 위장한다. 완전 회피는 불가능하다 —
# 가장 확실한 건 실제 Chrome 채널 + headed + 로그인된 영구 프로필. add_init_script는 모든
# 새 문서에 먼저 실행된다.
_STEALTH_ARGS = ["--disable-blink-features=AutomationControlled"]
_STEALTH_INIT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['ko-KR','ko','en-US','en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
window.chrome = window.chrome || { runtime: {} };
const _q = window.navigator.permissions && window.navigator.permissions.query;
if (_q) {
  window.navigator.permissions.query = (p) => (
    p && p.name === 'notifications'
      ? Promise.resolve({ state: Notification.permission })
      : _q(p)
  );
}
"""

# 응답 영역에서 이런 신호가 보이면 캡차/로그인 차단으로 보고 명확히 실패시킨다(빈칸 대신).
_CAPTCHA_SIGNS = (
    "recaptcha", "captcha", "verify you are human", "i'm not a robot", "cloudflare",
    "unusual traffic", "are you a robot", "로봇이 아닙니다", "사람인지 확인", "보안문자",
    "자동화된", "비정상적인 트래픽",
)

# Cloudflare 등 '관리형 챌린지' 인터스티셜 신호(제목/본문). 이게 보이면 곧장 실패시키지
# 않고 스스로 통과할 때까지 잠시 기다린다(headed + 영구 프로필이면 보통 수 초 내 통과).
_CHALLENGE_TITLE_SIGNS = ("just a moment", "attention required", "잠시만 기다")
_CHALLENGE_BODY_SIGNS = (
    "verifying you are human", "checking your browser", "needs to review the security",
    "just a moment", "enable javascript and cookies to continue",
    "사람인지 확인하고 있습니다", "브라우저를 확인", "잠시만 기다",
)
_CHALLENGE_POLL_MS = 700


def _as_selector_list(selector) -> list:
    if isinstance(selector, (list, tuple)):
        return [s for s in selector if s]
    return [selector]


def resolve_profile_dir(base_dir: str, slot: int) -> str:
    """slot==0이면 기존 프로필 그대로 쓴다(기존 동작과 완전히 동일, 복사 없음).

    slot>0이면 워커 전용 디렉터리를 쓰되, 처음 만들 때 기존 프로필을 복사해
    로그인 세션(카톡 등)을 물려받는다 — 빈 프로필로 시작하면 다시 로그인해야 하므로.
    이후로는 워커별로 독립적으로 갈라지는 것을 허용한다(매번 재복사하지 않음)."""
    if slot == 0:
        return base_dir
    worker_dir = f"{base_dir}_worker{slot}"
    if not os.path.exists(worker_dir) and os.path.exists(base_dir):
        shutil.copytree(base_dir, worker_dir, ignore=shutil.ignore_patterns("Singleton*"))
    return worker_dir


class Browser:
    def __init__(
        self,
        headless: bool = BROWSER_HEADLESS,
        user_data_dir: str = BROWSER_USER_DATA_DIR,
        timeout: int = BROWSER_TIMEOUT,
    ):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self.timeout = timeout
        self._playwright = None
        self._context = None
        self._page = None

    def __enter__(self):
        self._launch()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._teardown()

    def _launch(self) -> None:
        os.makedirs(self.user_data_dir, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._launch_context()
        if BROWSER_STEALTH:
            self._context.add_init_script(_STEALTH_INIT)
        self._context.set_default_timeout(self.timeout)
        self._page = self._context.new_page()

    def _launch_context(self):
        """영구 컨텍스트를 띄운다. 실제 설치된 Chrome 채널을 1순위로 시도해 Cloudflare의
        패시브 핑거프린트 검사를 통과할 확률을 높이고, 채널이 없으면 번들 Chromium으로 폴백한다."""
        base = {"headless": self.headless}
        if BROWSER_STEALTH:
            base["args"] = _STEALTH_ARGS
            base["ignore_default_args"] = ["--enable-automation"]
            if BROWSER_LOCALE:
                base["locale"] = BROWSER_LOCALE
        # 1순위: 실제 Chrome 채널. 채널을 쓰면 UA를 강제하지 않는다 — 실제 Chrome UA와
        # client-hints가 일치해야 하기 때문(억지 UA는 오히려 Cloudflare 플래그가 된다).
        if BROWSER_CHANNEL:
            try:
                return self._playwright.chromium.launch_persistent_context(
                    self.user_data_dir, channel=BROWSER_CHANNEL, **base
                )
            except Exception as exc:
                log.warning("Chrome 채널(%s) 실행 실패 → 번들 Chromium으로 폴백: %s", BROWSER_CHANNEL, exc)
        # 폴백(번들 Chromium): 헤드리스/번들 UA 표식을 가리려고 실제 크롬 UA를 덮어쓴다.
        if BROWSER_STEALTH and BROWSER_USER_AGENT:
            base["user_agent"] = BROWSER_USER_AGENT
        return self._playwright.chromium.launch_persistent_context(self.user_data_dir, **base)

    def _teardown(self) -> None:
        try:
            if self._context:
                self._context.close()
        except Exception as exc:
            log.debug("context 종료 중 오류(무시): %s", exc)
        try:
            if self._playwright:
                self._playwright.stop()
        except Exception as exc:
            log.debug("playwright 종료 중 오류(무시): %s", exc)

    def _ensure_page(self) -> None:
        """배치 중간에 페이지/탭이 죽었을 때(사용자가 닫음, 탭 크래시 등) 복구한다.

        context 자체는 살아있는데 page만 닫혔으면 새 page만 새로 열고,
        context까지 죽어 있으면 전체를 재launch한다 — 한 step의 장애가
        이후 모든 step을 영구히 실패시키지 않도록 하는 것이 목적이다."""
        if self._context is None or self._page is None:
            self._restart()
            return
        try:
            _ = self._context.pages  # 접근만으로 context 생존 확인(닫혀 있으면 예외)
        except Exception:
            self._restart()
            return
        if self._page.is_closed():
            log.warning("페이지가 닫혀 있어 새 페이지를 생성")
            try:
                self._page = self._context.new_page()
            except Exception:
                self._restart()

    def _restart(self) -> None:
        log.warning("브라우저 컨텍스트가 죽어 있어 재시작")
        self._teardown()
        self._launch()

    def open(self, url: str) -> None:
        self._ensure_page()
        # 네트워크가 끝나길 무한정 기다리지 않도록 DOM 로드 기준으로 대기한다.
        self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)
        self._pass_challenge(CLOUDFLARE_WAIT_MS)

    def _on_challenge_page(self) -> bool:
        """현재 페이지가 Cloudflare 등 관리형 챌린지 인터스티셜인지(제목/본문 신호로) 판단한다.

        selector가 아니라 텍스트 신호만 본다 — 사이트별 selector를 추측하지 않으려는
        기존 원칙과 같고, 챌린지 페이지는 제목/본문 문구가 일관돼 가장 안정적이기 때문."""
        try:
            title = str(self._page.title()).lower()
        except Exception:
            title = ""
        if any(s in title for s in _CHALLENGE_TITLE_SIGNS):
            return True
        try:
            body = str(self._page.inner_text("body", timeout=2000)).lower()
        except Exception:
            return False
        return any(s in body for s in _CHALLENGE_BODY_SIGNS)

    def _pass_challenge(self, max_wait_ms: int) -> None:
        """관리형 챌린지가 떠 있으면 스스로 통과할 때까지 폴링하며 기다린다(상한 max_wait_ms).

        챌린지가 아니면 즉시 반환하므로 일반 페이지에는 비용이 거의 없다. 끝내 통과하지
        못하면 그냥 반환하고, 이후 응답 읽기 단계의 캡차 신호 검사가 명확히 실패시킨다."""
        if max_wait_ms <= 0:
            return
        elapsed = 0
        while elapsed < max_wait_ms:
            if not self._on_challenge_page():
                return
            log.info("Cloudflare 챌린지 감지 — 자동 통과 대기 중(%dms 경과)", elapsed)
            self._page.wait_for_timeout(_CHALLENGE_POLL_MS)
            elapsed += _CHALLENGE_POLL_MS
        log.warning("Cloudflare 챌린지가 %dms 안에 통과되지 않음 — 다음 단계로 진행", max_wait_ms)

    def _first_visible(self, selectors: list):
        """후보 selector들을 순서대로 시도해 먼저 나타나는 요소의 selector를 돌려준다."""
        last_error = None
        for sel in selectors:
            try:
                self._page.wait_for_selector(sel, state="visible", timeout=self.timeout)
                return sel
            except PlaywrightTimeoutError as exc:
                last_error = exc
                log.debug("selector 미발견, 다음 후보 시도: %s", sel)
        raise PlaywrightTimeoutError(
            f"후보 selector 중 표시되는 요소를 찾지 못함: {selectors} ({last_error})"
        )

    def click(self, selector) -> None:
        self._ensure_page()
        sel = self._first_visible(_as_selector_list(selector))
        self._page.click(sel, timeout=self.timeout)

    def type(self, selector, text: str) -> None:
        self._ensure_page()
        sel = self._first_visible(_as_selector_list(selector))
        self._page.fill(sel, text, timeout=self.timeout)

    def get_text(self, selector="body") -> str:
        """DOM 텍스트를 우선 추출하고, 비어 있거나 실패하면 스크린샷+OCR로 폴백한다.

        selector는 단일 문자열 또는 후보 목록(list)일 수 있다 — 후보가 여러 개면
        순서대로 시도해 첫 번째로 비어있지 않은 결과를 쓴다."""
        self._ensure_page()
        for sel in _as_selector_list(selector):
            try:
                text = self._page.inner_text(sel, timeout=self.timeout).strip()
                if text:
                    return text
            except PlaywrightTimeoutError:
                log.debug("get_text: selector 타임아웃, 다음 후보 시도: %s", sel)
        log.info("DOM 텍스트가 비어 OCR 폴백 수행")

        path = self.screenshot()
        return ocr.extract_text(path)

    def screenshot(self, path: str = None) -> str:
        self._ensure_page()
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        path = path or os.path.join(SCREENSHOT_DIR, "capture.png")
        self._page.screenshot(path=path)
        return path

    def debug_screenshot(self, label: str = "error") -> str | None:
        """동작 실패 시 디버깅용 스크린샷. 실패해도 본 흐름을 막지 않는다."""
        try:
            os.makedirs(SCREENSHOT_DIR, exist_ok=True)
            safe = "".join(c if c.isalnum() else "_" for c in label)[:40]
            path = os.path.join(SCREENSHOT_DIR, f"debug_{safe}_{int(time.time())}.png")
            self._page.screenshot(path=path)
            log.info("디버그 스크린샷 저장: %s", path)
            return path
        except Exception as exc:
            log.debug("디버그 스크린샷 실패: %s", exc)
            return None

    def search(self, query: str) -> str:
        self.open(f"https://www.google.com/search?q={quote_plus(query)}")
        return self.get_text()

    def describe_screen(self, prompt: str = "", model: str = "", fallback_models=None) -> str:
        """현재 화면을 스크린샷해 로컬 VLM이 '이해/설명'하게 한다(vision 역할).

        get_text의 OCR(글자 전사)과 다르다 — 여기서는 레이아웃·상태·의미를 묻는다. prompt가
        비면 화면 전반 설명을 요청한다. model(비전 모델)이 설정돼 있지 않으면 명확히 막는다
        (OLLAMA_VLM_MODEL/OCR_VLM_MODEL 미설정 시 임의 추측 대신 오류). fallback_models를 주면
        1순위 VLM이 실패할 때 백업 비전 모델로 페일오버한다(ai_roles.models_for(vision))."""
        if not model:
            raise ValueError(
                "비전 모델이 설정되지 않았습니다. OLLAMA_VLM_MODEL(또는 OCR_VLM_MODEL)에 "
                "로컬 Ollama 비전 모델명(예: qwen2.5vl:7b)을 지정하세요."
            )
        self._ensure_page()
        path = self.screenshot()
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        question = prompt.strip() or _VISION_DEFAULT_PROMPT
        return ollama_client.generate(
            question, model=model, images=[encoded], fallback_models=fallback_models
        )

    def is_present(self, selector, timeout_ms: int) -> bool:
        """후보 selector 중 하나라도 timeout_ms 안에 보이면 True. 로그인 여부/로그인 벽
        감지에 쓰는 '존재 확인'(예외 없이 불리언 반환)."""
        self._ensure_page()
        for sel in _as_selector_list(selector):
            try:
                self._page.wait_for_selector(sel, state="visible", timeout=timeout_ms)
                return True
            except PlaywrightTimeoutError:
                continue
        return False

    def ensure_logged_in(
        self,
        url: str,
        logged_in_selector,
        google_button_selector,
        wait_ms: int = WEB_AI_WAIT_MS,
        check_ms: int = 3000,
    ) -> str:
        """로그인 페이지를 열어 이미 로그인돼 있으면 그대로 두고, 풀렸으면 '구글로 로그인'
        버튼을 눌러 재인증한다. 영구 프로필의 구글 세션을 타므로 비번 입력이 없다(2FA·추가
        인증이 뜨는 사이트는 자동화로 넘기지 않고 명확한 오류를 던진다).

        url/logged_in_selector는 사이트마다 달라 호출자(login_providers)가 반드시 넘긴다.
        반환: 사람이 읽을 상태 문자열('이미 로그인됨' / '구글로 재로그인 성공')."""
        if not url or not logged_in_selector:
            raise ValueError(
                "LOGIN_URL / LOGIN_LOGGED_IN_SELECTOR가 설정되지 않았습니다. "
                "로그인할 사이트의 페이지 URL과 '로그인됐을 때만 보이는 요소' selector를 "
                "환경변수나 LOGIN_PROVIDERS_PATH JSON으로 지정하세요."
            )
        self.open(url)
        if self.is_present(logged_in_selector, check_ms):
            log.info("이미 로그인된 상태 — 재로그인 생략")
            return "이미 로그인됨"

        log.info("로그인되지 않음 → '구글로 로그인' 시도")
        sel = self._first_visible(_as_selector_list(google_button_selector))
        self._page.click(sel, timeout=self.timeout)

        if self.is_present(logged_in_selector, wait_ms):
            log.info("구글 세션으로 재로그인 성공")
            return "구글로 재로그인 성공"
        raise RuntimeError(
            "'구글로 로그인' 후에도 로그인 상태를 확인하지 못함 "
            "(2FA·기기 인증 등 추가 단계가 필요할 수 있음 — 수동 로그인 필요)"
        )

    def _wait_for_response_stable(
        self,
        selector,
        max_wait_ms: int,
        poll_interval_ms: int = WEB_AI_POLL_INTERVAL_MS,
        stable_polls_required: int = WEB_AI_STABLE_POLLS,
    ) -> None:
        """응답 영역 텍스트 길이가 더 늘지 않을 때까지 폴링한다(스트리밍 종료 감지).

        Claude(수 초)처럼 빠른 응답과 ChatGPT(수십 초)처럼 느린 응답을 사이트별
        튜닝 없이 같은 로직으로 다루기 위해, 고정 대기 대신 길이 변화 추세를 본다.
        max_wait_ms는 안정화가 끝내 감지되지 않을 때의 상한(타임아웃)이다.

        selector는 단일 문자열 또는 후보 목록(list)일 수 있다. 시작 시점에 미리 하나를
        확정하지 않는 이유는, 응답 영역이 즉시 보이지 않아도(스트리밍 UI가 나중에 DOM에
        넣는 경우) max_wait_ms까지 patient하게 기다리는 기존 동작을 깨지 않기 위함이다 —
        그래서 매 폴링마다 후보를 순서대로 다시 시도한다."""
        candidates = _as_selector_list(selector)
        elapsed_ms = 0
        last_len = -1
        stable_count = 0
        while elapsed_ms < max_wait_ms:
            self._page.wait_for_timeout(poll_interval_ms)
            elapsed_ms += poll_interval_ms
            text = ""
            for sel in candidates:
                try:
                    text = self._page.inner_text(sel, timeout=self.timeout)
                    if text:
                        break
                except PlaywrightTimeoutError:
                    continue
            length = len(text)
            if length > 0 and length == last_len:
                stable_count += 1
                if stable_count >= stable_polls_required:
                    return
            else:
                stable_count = 0
            last_len = length

    def ask_web_ai(
        self,
        prompt: str,
        url: str,
        input_selector,
        submit_selector: str = "",
        response_selector="body",
        wait_ms: int = WEB_AI_WAIT_MS,
    ) -> str:
        """웹 AI 채팅 페이지를 열어 프롬프트를 보내고 응답 텍스트를 읽어 온다.

        URL/selector는 사이트마다 달라 호출자가 반드시 넘겨야 한다(코드에 박지 않음).
        input_selector/response_selector는 단일 문자열 또는 후보 목록(list)일 수 있다.
        응답이 스트리밍이라 고정 대기 대신 _wait_for_response_stable로 완료를 감지하고
        (wait_ms는 그 상한), DOM이 비면 get_text의 OCR 폴백이 받는다."""
        if not url or not input_selector:
            raise ValueError(
                "WEB_AI_URL / WEB_AI_INPUT_SELECTOR가 설정되지 않았습니다. "
                "사용할 웹 AI의 페이지 URL과 입력창/응답 selector를 환경변수로 지정하세요."
            )
        self.open(url)
        sel = self._first_visible(_as_selector_list(input_selector))
        self._page.fill(sel, prompt, timeout=self.timeout)
        if submit_selector:
            self.click(submit_selector)
        else:
            self._page.press(sel, "Enter")
        self._wait_for_response_stable(response_selector, wait_ms)
        text = self.get_text(response_selector)

        # 캡차/로그인 차단 감지 — 빈칸을 그냥 돌려주면(성공 처리) 화면에 blank로 뜬다.
        # 명확히 실패시켜 상위 라우터가 로컬로 폴백하거나, 사용자가 원인을 알게 한다.
        low = text.lower()
        if any(sign in low for sign in _CAPTCHA_SIGNS):
            self.debug_screenshot("web_ai_captcha")
            raise RuntimeError(
                "웹 AI 페이지에서 캡차/로그인 확인이 감지됐습니다. headed 모드"
                "(BROWSER_HEADLESS=false)로 직접 한 번 로그인/캡차를 풀어 영구 프로필에 "
                "세션을 남기거나, 다른 제공자를 사용하세요."
            )
        if not text.strip():
            raise RuntimeError(
                "웹 AI 응답이 비어 있습니다. 응답 selector(WEB_AI_RESPONSE_SELECTOR)가 "
                "페이지와 맞지 않거나 캡차/로그인으로 차단됐을 수 있습니다."
            )
        return text

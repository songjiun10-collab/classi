"""Playwright 기반 브라우저 엔진 (sync API). 카톡 등 로그인 세션 유지를 위해 영구 프로필 사용.

V0.2 안정화: 명시적 wait 전략, selector fallback(여러 후보 시도),
DOM 텍스트가 비면 OCR로 폴백, 동작 실패 시 디버그 스크린샷 저장,
web AI 응답은 고정 대기 대신 텍스트 길이 안정화 폴링으로 완료 감지."""
import os
import time
from urllib.parse import quote_plus

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

from config.config import (
    BROWSER_HEADLESS,
    BROWSER_TIMEOUT,
    BROWSER_USER_DATA_DIR,
    SCREENSHOT_DIR,
    WEB_AI_POLL_INTERVAL_MS,
    WEB_AI_STABLE_POLLS,
    WEB_AI_WAIT_MS,
)
from core.logger import get_logger
from tools import ocr

log = get_logger("browser")


def _as_selector_list(selector) -> list:
    if isinstance(selector, (list, tuple)):
        return [s for s in selector if s]
    return [selector]


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
        os.makedirs(self.user_data_dir, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            self.user_data_dir, headless=self.headless
        )
        self._context.set_default_timeout(self.timeout)
        self._page = self._context.new_page()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            self._context.close()
        if self._playwright:
            self._playwright.stop()

    def open(self, url: str) -> None:
        # 네트워크가 끝나길 무한정 기다리지 않도록 DOM 로드 기준으로 대기한다.
        self._page.goto(url, wait_until="domcontentloaded", timeout=self.timeout)

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
        sel = self._first_visible(_as_selector_list(selector))
        self._page.click(sel, timeout=self.timeout)

    def type(self, selector, text: str) -> None:
        sel = self._first_visible(_as_selector_list(selector))
        self._page.fill(sel, text, timeout=self.timeout)

    def get_text(self, selector: str = "body") -> str:
        """DOM 텍스트를 우선 추출하고, 비어 있거나 실패하면 스크린샷+OCR로 폴백한다."""
        try:
            text = self._page.inner_text(selector, timeout=self.timeout).strip()
            if text:
                return text
            log.info("DOM 텍스트가 비어 OCR 폴백 수행")
        except PlaywrightTimeoutError:
            log.warning("inner_text 타임아웃, OCR 폴백 수행")

        path = self.screenshot()
        return ocr.image_to_text(path)

    def screenshot(self, path: str = None) -> str:
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

    def _wait_for_response_stable(
        self,
        selector: str,
        max_wait_ms: int,
        poll_interval_ms: int = WEB_AI_POLL_INTERVAL_MS,
        stable_polls_required: int = WEB_AI_STABLE_POLLS,
    ) -> None:
        """응답 영역 텍스트 길이가 더 늘지 않을 때까지 폴링한다(스트리밍 종료 감지).

        Claude(수 초)처럼 빠른 응답과 ChatGPT(수십 초)처럼 느린 응답을 사이트별
        튜닝 없이 같은 로직으로 다루기 위해, 고정 대기 대신 길이 변화 추세를 본다.
        max_wait_ms는 안정화가 끝내 감지되지 않을 때의 상한(타임아웃)이다."""
        elapsed_ms = 0
        last_len = -1
        stable_count = 0
        while elapsed_ms < max_wait_ms:
            self._page.wait_for_timeout(poll_interval_ms)
            elapsed_ms += poll_interval_ms
            try:
                text = self._page.inner_text(selector, timeout=self.timeout)
            except PlaywrightTimeoutError:
                text = ""
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
        input_selector: str,
        submit_selector: str = "",
        response_selector: str = "body",
        wait_ms: int = WEB_AI_WAIT_MS,
    ) -> str:
        """웹 AI 채팅 페이지를 열어 프롬프트를 보내고 응답 텍스트를 읽어 온다.

        URL/selector는 사이트마다 달라 호출자가 반드시 넘겨야 한다(코드에 박지 않음).
        응답이 스트리밍이라 고정 대기 대신 _wait_for_response_stable로 완료를 감지하고
        (wait_ms는 그 상한), DOM이 비면 get_text의 OCR 폴백이 받는다."""
        if not url or not input_selector:
            raise ValueError(
                "WEB_AI_URL / WEB_AI_INPUT_SELECTOR가 설정되지 않았습니다. "
                "사용할 웹 AI의 페이지 URL과 입력창/응답 selector를 환경변수로 지정하세요."
            )
        self.open(url)
        sel = self._first_visible([input_selector])
        self._page.fill(sel, prompt, timeout=self.timeout)
        if submit_selector:
            self.click(submit_selector)
        else:
            self._page.press(sel, "Enter")
        self._wait_for_response_stable(response_selector, wait_ms)
        return self.get_text(response_selector)

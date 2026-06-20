"""Playwright 기반 브라우저 엔진 (sync API). 카톡 등 로그인 세션 유지를 위해 영구 프로필 사용.

V0.2 안정화: 명시적 wait 전략, selector fallback(여러 후보 시도),
DOM 텍스트가 비면 OCR로 폴백, 동작 실패 시 디버그 스크린샷 저장,
web AI 응답은 고정 대기 대신 텍스트 길이 안정화 폴링으로 완료 감지."""
import os
import shutil
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
        self._context = self._playwright.chromium.launch_persistent_context(
            self.user_data_dir, headless=self.headless
        )
        self._context.set_default_timeout(self.timeout)
        self._page = self._context.new_page()

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
            self._context.pages  # 접근만으로 context 생존 확인(닫혀 있으면 예외)
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
        return self.get_text(response_selector)

"""Playwright 기반 브라우저 엔진 (sync API). 카톡 등 로그인 세션 유지를 위해 영구 프로필 사용."""
import os
from urllib.parse import quote_plus

from playwright.sync_api import sync_playwright

from config.config import BROWSER_HEADLESS, BROWSER_USER_DATA_DIR, SCREENSHOT_DIR


class Browser:
    def __init__(self, headless: bool = BROWSER_HEADLESS, user_data_dir: str = BROWSER_USER_DATA_DIR):
        self.headless = headless
        self.user_data_dir = user_data_dir
        self._playwright = None
        self._context = None
        self._page = None

    def __enter__(self):
        os.makedirs(self.user_data_dir, exist_ok=True)
        self._playwright = sync_playwright().start()
        self._context = self._playwright.chromium.launch_persistent_context(
            self.user_data_dir, headless=self.headless
        )
        self._page = self._context.new_page()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._context:
            self._context.close()
        if self._playwright:
            self._playwright.stop()

    def open(self, url: str) -> None:
        self._page.goto(url)

    def click(self, selector: str) -> None:
        self._page.click(selector)

    def type(self, selector: str, text: str) -> None:
        self._page.fill(selector, text)

    def get_text(self, selector: str = "body") -> str:
        return self._page.inner_text(selector)

    def screenshot(self, path: str = None) -> str:
        os.makedirs(SCREENSHOT_DIR, exist_ok=True)
        path = path or os.path.join(SCREENSHOT_DIR, "capture.png")
        self._page.screenshot(path=path)
        return path

    def search(self, query: str) -> str:
        self.open(f"https://www.google.com/search?q={quote_plus(query)}")
        return self.get_text()

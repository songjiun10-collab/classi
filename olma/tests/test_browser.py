from unittest.mock import MagicMock, patch

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from tools.browser import Browser, _as_selector_list


def _browser_with_mock_page():
    b = Browser()
    b._page = MagicMock()
    return b


def test_as_selector_list_normalizes_input():
    assert _as_selector_list("a") == ["a"]
    assert _as_selector_list(["a", "b"]) == ["a", "b"]
    assert _as_selector_list(["a", None]) == ["a"]


def test_open_uses_domcontentloaded_wait():
    b = _browser_with_mock_page()
    b.open("https://example.com")
    _, kwargs = b._page.goto.call_args
    assert kwargs["wait_until"] == "domcontentloaded"


def test_click_tries_selector_fallback_in_order():
    b = _browser_with_mock_page()
    # 첫 selector는 타임아웃, 둘째는 성공
    b._page.wait_for_selector.side_effect = [PlaywrightTimeoutError("no"), None]

    b.click(["#first", "#second"])

    assert b._page.wait_for_selector.call_count == 2
    clicked_selector = b._page.click.call_args.args[0]
    assert clicked_selector == "#second"


def test_get_text_returns_dom_text_when_present():
    b = _browser_with_mock_page()
    b._page.inner_text.return_value = "  실제 본문  "
    assert b.get_text() == "실제 본문"


def test_get_text_falls_back_to_ocr_when_dom_empty():
    b = _browser_with_mock_page()
    b._page.inner_text.return_value = "   "

    with patch.object(b, "screenshot", return_value="/tmp/x.png"), patch(
        "tools.browser.ocr.image_to_text", return_value="OCR로 읽은 텍스트"
    ) as ocr_fn:
        text = b.get_text()

    assert text == "OCR로 읽은 텍스트"
    ocr_fn.assert_called_once_with("/tmp/x.png")


def test_get_text_falls_back_to_ocr_on_timeout():
    b = _browser_with_mock_page()
    b._page.inner_text.side_effect = PlaywrightTimeoutError("timeout")

    with patch.object(b, "screenshot", return_value="/tmp/x.png"), patch(
        "tools.browser.ocr.image_to_text", return_value="OCR 폴백"
    ):
        assert b.get_text() == "OCR 폴백"


def test_debug_screenshot_never_raises():
    b = _browser_with_mock_page()
    b._page.screenshot.side_effect = RuntimeError("스크린샷 실패")
    # 디버그 스크린샷은 실패해도 None을 돌려줄 뿐 예외를 던지지 않아야 한다.
    assert b.debug_screenshot("test") is None

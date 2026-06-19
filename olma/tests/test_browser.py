from unittest.mock import MagicMock, patch

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from tools.browser import Browser, _as_selector_list


def _browser_with_mock_page():
    b = Browser()
    b._context = MagicMock()
    b._page = MagicMock()
    b._page.is_closed.return_value = False
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


def test_ask_web_ai_requires_url_and_selector():
    b = _browser_with_mock_page()
    with pytest.raises(ValueError):
        b.ask_web_ai("안녕", url="", input_selector="")


def test_ask_web_ai_fills_submits_and_reads_response():
    b = _browser_with_mock_page()
    b._page.wait_for_selector.return_value = None
    b._page.inner_text.return_value = "AI의 답변입니다"

    text = b.ask_web_ai(
        "파이썬이 뭐야?",
        url="https://example-ai.test/chat",
        input_selector="#prompt",
        submit_selector="#send",
        response_selector="#answer",
    )

    assert text == "AI의 답변입니다"
    b._page.goto.assert_called_once()
    # 프롬프트가 입력창에 채워졌는지
    fill_args = b._page.fill.call_args.args
    assert fill_args[0] == "#prompt"
    assert fill_args[1] == "파이썬이 뭐야?"
    # 전송 버튼 클릭
    b._page.click.assert_called_once()


def test_ask_web_ai_presses_enter_when_no_submit_selector():
    b = _browser_with_mock_page()
    b._page.wait_for_selector.return_value = None
    b._page.inner_text.return_value = "답변"

    b.ask_web_ai("질문", url="https://x.test", input_selector="#p")

    b._page.press.assert_called_once()
    assert b._page.press.call_args.args[1] == "Enter"


def test_wait_for_response_stable_returns_once_length_stops_growing():
    b = _browser_with_mock_page()
    # 같은 길이의 응답이 연속 2번(stable_polls_required) 더 나오면 안정으로 보고 멈춘다.
    b._page.inner_text.side_effect = ["답변", "답변", "답변", "이 값까지 소비되면 안됨"]

    b._wait_for_response_stable("#answer", max_wait_ms=10000, poll_interval_ms=100, stable_polls_required=2)

    # 안정 판정 직후 멈췄으므로 마지막 side_effect 값까지는 소비하지 않는다.
    assert b._page.inner_text.call_count == 3


def test_wait_for_response_stable_gives_up_at_max_wait_when_never_stable():
    b = _browser_with_mock_page()
    # 매번 다른 길이를 반환 -> 절대 안정화되지 않음 -> max_wait_ms에서 포기.
    counter = iter(range(1000))
    b._page.inner_text.side_effect = lambda *a, **k: "x" * next(counter)

    b._wait_for_response_stable("#answer", max_wait_ms=300, poll_interval_ms=100, stable_polls_required=2)

    assert b._page.wait_for_timeout.call_count == 3


def test_wait_for_response_stable_treats_timeout_as_not_yet_stable():
    b = _browser_with_mock_page()
    b._page.inner_text.side_effect = [
        PlaywrightTimeoutError("아직 응답 없음"),
        "답변",
        "답변",
        "답변",
    ]

    b._wait_for_response_stable("#answer", max_wait_ms=10000, poll_interval_ms=100, stable_polls_required=2)

    assert b._page.inner_text.call_count == 4


def test_ensure_page_recreates_page_when_closed():
    b = _browser_with_mock_page()
    old_page = b._page
    old_page.is_closed.return_value = True
    new_page = MagicMock()
    b._context.new_page.return_value = new_page

    b._ensure_page()

    assert b._page is new_page
    b._context.new_page.assert_called_once()


def test_ensure_page_does_nothing_when_page_alive():
    b = _browser_with_mock_page()
    old_page = b._page

    b._ensure_page()

    assert b._page is old_page
    b._context.new_page.assert_not_called()


class _DeadContext:
    """pages 접근만으로 예외를 던지는, 죽은 context를 흉내내는 더미.

    MagicMock 클래스 자체에 property를 얹으면 다른 테스트의 MagicMock까지
    오염되므로, 이 테스트만을 위한 독립된 더미 클래스를 쓴다."""

    @property
    def pages(self):
        raise RuntimeError("context closed")


def test_ensure_page_restarts_when_context_dead():
    b = _browser_with_mock_page()
    b._context = _DeadContext()

    with patch.object(b, "_restart") as restart:
        b._ensure_page()

    restart.assert_called_once()


def test_ensure_page_restarts_when_context_missing():
    b = Browser()
    assert b._context is None
    assert b._page is None

    with patch.object(b, "_restart") as restart:
        b._ensure_page()

    restart.assert_called_once()


def test_restart_tears_down_and_relaunches():
    b = _browser_with_mock_page()

    with patch.object(b, "_teardown") as teardown, patch.object(b, "_launch") as launch:
        b._restart()

    teardown.assert_called_once()
    launch.assert_called_once()


def test_open_recovers_when_page_was_closed():
    b = _browser_with_mock_page()
    b._page.is_closed.return_value = True
    new_page = MagicMock()
    b._context.new_page.return_value = new_page

    b.open("https://example.com")

    new_page.goto.assert_called_once()

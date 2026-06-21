import os
from unittest.mock import MagicMock, patch

import pytest
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from tools.browser import Browser, _as_selector_list, resolve_profile_dir


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
        "tools.browser.ocr.extract_text", return_value="OCR로 읽은 텍스트"
    ) as ocr_fn:
        text = b.get_text()

    assert text == "OCR로 읽은 텍스트"
    ocr_fn.assert_called_once_with("/tmp/x.png")


def test_get_text_falls_back_to_ocr_on_timeout():
    b = _browser_with_mock_page()
    b._page.inner_text.side_effect = PlaywrightTimeoutError("timeout")

    with patch.object(b, "screenshot", return_value="/tmp/x.png"), patch(
        "tools.browser.ocr.extract_text", return_value="OCR 폴백"
    ):
        assert b.get_text() == "OCR 폴백"


def test_get_text_tries_selector_candidates_in_order():
    b = _browser_with_mock_page()
    # 첫 후보는 빈 문자열, 둘째 후보는 실제 텍스트.
    b._page.inner_text.side_effect = ["   ", "둘째 후보 텍스트"]

    text = b.get_text(["#first", "#second"])

    assert text == "둘째 후보 텍스트"
    assert b._page.inner_text.call_count == 2


def test_get_text_falls_back_to_ocr_when_all_candidates_empty():
    b = _browser_with_mock_page()
    b._page.inner_text.side_effect = ["", "   "]

    with patch.object(b, "screenshot", return_value="/tmp/x.png"), patch(
        "tools.browser.ocr.extract_text", return_value="OCR 폴백"
    ):
        assert b.get_text(["#first", "#second"]) == "OCR 폴백"


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


def test_ask_web_ai_tries_input_selector_candidates_in_order():
    b = _browser_with_mock_page()
    # 첫 input selector는 타임아웃, 둘째는 성공. 이후 submit_selector 클릭은 정상 성공.
    b._page.wait_for_selector.side_effect = [PlaywrightTimeoutError("no"), None, None]
    b._page.inner_text.return_value = "답변"

    b.ask_web_ai(
        "질문",
        url="https://x.test",
        input_selector=["#first", "#second"],
        submit_selector="#send",
    )

    fill_args = b._page.fill.call_args.args
    assert fill_args[0] == "#second"


def test_ask_web_ai_raises_on_empty_response():
    """응답이 비면 그냥 빈칸을 돌려주지 않고(=화면 blank 방지) 명확히 실패시킨다."""
    b = _browser_with_mock_page()
    b._page.wait_for_selector.return_value = None
    b.get_text = lambda *a, **k: "   "   # 빈/공백 응답
    with pytest.raises(RuntimeError, match="비어"):
        b.ask_web_ai("질문", url="https://x.test", input_selector="#p")


def test_ask_web_ai_raises_on_captcha():
    """응답 영역에 캡차/로그인 신호가 보이면 blank 대신 명확한 안내로 실패시킨다."""
    b = _browser_with_mock_page()
    b._page.wait_for_selector.return_value = None
    b.get_text = lambda *a, **k: "Please verify you are human — reCAPTCHA"
    b.debug_screenshot = lambda *a, **k: None
    with pytest.raises(RuntimeError, match="캡차"):
        b.ask_web_ai("질문", url="https://x.test", input_selector="#p")


def test_launch_applies_stealth(monkeypatch):
    """stealth가 켜지면 자동화 표식 제거 args와 init script가 적용된다."""
    import tools.browser as browser_mod
    b = Browser()
    fake_ctx = MagicMock()
    fake_chromium = MagicMock()
    fake_chromium.launch_persistent_context.return_value = fake_ctx
    fake_pw = MagicMock()
    fake_pw.chromium = fake_chromium
    monkeypatch.setattr(browser_mod, "sync_playwright", lambda: MagicMock(start=lambda: fake_pw))
    monkeypatch.setattr(browser_mod, "BROWSER_STEALTH", True)
    b._launch()
    kwargs = fake_chromium.launch_persistent_context.call_args.kwargs
    assert "--disable-blink-features=AutomationControlled" in kwargs["args"]
    assert kwargs["ignore_default_args"] == ["--enable-automation"]
    fake_ctx.add_init_script.assert_called_once()


def test_launch_uses_chrome_channel_when_available(monkeypatch):
    """BROWSER_CHANNEL이 설정되면 실제 Chrome 채널로 먼저 시도한다."""
    import tools.browser as browser_mod
    b = Browser()
    fake_ctx = MagicMock()
    fake_chromium = MagicMock()
    fake_chromium.launch_persistent_context.return_value = fake_ctx
    fake_pw = MagicMock()
    fake_pw.chromium = fake_chromium
    monkeypatch.setattr(browser_mod, "sync_playwright", lambda: MagicMock(start=lambda: fake_pw))
    monkeypatch.setattr(browser_mod, "BROWSER_CHANNEL", "chrome")
    monkeypatch.setattr(browser_mod, "BROWSER_STEALTH", False)
    b._launch()
    kwargs = fake_chromium.launch_persistent_context.call_args.kwargs
    assert kwargs.get("channel") == "chrome"


def test_launch_fallback_to_chromium_when_channel_fails(monkeypatch):
    """Chrome 채널 실행이 실패하면 번들 Chromium으로 폴백한다."""
    import tools.browser as browser_mod
    b = Browser()
    fake_ctx = MagicMock()
    call_count = {"n": 0}

    def launch_side_effect(path, **kwargs):
        call_count["n"] += 1
        if kwargs.get("channel"):
            raise RuntimeError("Chrome not found")
        return fake_ctx

    fake_chromium = MagicMock()
    fake_chromium.launch_persistent_context.side_effect = launch_side_effect
    fake_pw = MagicMock()
    fake_pw.chromium = fake_chromium
    monkeypatch.setattr(browser_mod, "sync_playwright", lambda: MagicMock(start=lambda: fake_pw))
    monkeypatch.setattr(browser_mod, "BROWSER_CHANNEL", "chrome")
    monkeypatch.setattr(browser_mod, "BROWSER_STEALTH", False)
    b._launch()
    assert call_count["n"] == 2  # 첫 번째(채널 시도) + 두 번째(폴백)
    assert b._context is fake_ctx


def test_on_challenge_page_detects_by_title(monkeypatch):
    """제목에 챌린지 신호가 있으면 True를 돌려준다."""
    b = _browser_with_mock_page()
    b._page.title.return_value = "Just a moment..."
    assert b._on_challenge_page() is True


def test_on_challenge_page_detects_by_body(monkeypatch):
    """제목은 정상인데 본문에 챌린지 신호가 있으면 True."""
    b = _browser_with_mock_page()
    b._page.title.return_value = "Normal Site"
    b._page.inner_text.return_value = "Verifying you are human. This may take a few seconds."
    assert b._on_challenge_page() is True


def test_on_challenge_page_returns_false_for_normal_page():
    b = _browser_with_mock_page()
    b._page.title.return_value = "ChatGPT"
    b._page.inner_text.return_value = "Ask anything..."
    assert b._on_challenge_page() is False


def test_pass_challenge_waits_until_cleared():
    """챌린지가 감지되면 폴링하며 기다리다 없어지면 반환한다."""
    b = _browser_with_mock_page()
    # 처음 두 번은 챌린지, 세 번째부터 정상
    call_count = {"n": 0}

    def on_challenge():
        call_count["n"] += 1
        return call_count["n"] <= 2

    with patch.object(b, "_on_challenge_page", side_effect=on_challenge):
        b._pass_challenge(max_wait_ms=10000)

    assert call_count["n"] == 3  # 2번 챌린지 감지 + 1번 통과 확인
    assert b._page.wait_for_timeout.call_count == 2  # 두 번 대기


def test_pass_challenge_skips_when_no_challenge():
    """챌린지가 없으면 폴링 없이 즉시 반환한다."""
    b = _browser_with_mock_page()
    with patch.object(b, "_on_challenge_page", return_value=False) as chk:
        b._pass_challenge(max_wait_ms=10000)
    b._page.wait_for_timeout.assert_not_called()


def test_pass_challenge_gives_up_at_max_wait():
    """max_wait_ms 안에 챌린지가 풀리지 않으면 조용히 포기한다."""
    import tools.browser as browser_mod
    b = _browser_with_mock_page()
    with patch.object(b, "_on_challenge_page", return_value=True), \
         patch.object(browser_mod, "_CHALLENGE_POLL_MS", 100):
        b._pass_challenge(max_wait_ms=250)
    # 최대 3회(0→100→200→250 초과) 폴링
    assert b._page.wait_for_timeout.call_count <= 3


def test_describe_screen_requires_model():
    b = _browser_with_mock_page()
    with pytest.raises(ValueError):
        b.describe_screen("이 화면 설명", model="")


def test_describe_screen_screenshots_and_calls_vlm(tmp_path):
    b = _browser_with_mock_page()
    img = tmp_path / "cap.png"
    img.write_bytes(b"\x89PNG fake bytes")
    with patch.object(b, "screenshot", return_value=str(img)), \
         patch("tools.browser.ollama_client.generate", return_value="화면 설명 결과") as gen:
        result = b.describe_screen("뭐가 보여?", model="vlm:7b")

    assert result == "화면 설명 결과"
    _, kwargs = gen.call_args
    assert kwargs["model"] == "vlm:7b"
    assert kwargs["images"] and isinstance(kwargs["images"][0], str)  # base64 인코딩 전달
    assert gen.call_args.args[0] == "뭐가 보여?"


def test_describe_screen_uses_default_prompt_when_empty(tmp_path):
    b = _browser_with_mock_page()
    img = tmp_path / "cap.png"
    img.write_bytes(b"x")
    with patch.object(b, "screenshot", return_value=str(img)), \
         patch("tools.browser.ollama_client.generate", return_value="설명") as gen:
        b.describe_screen("", model="vlm:7b")

    assert gen.call_args.args[0]  # 빈 입력이면 기본 프롬프트가 채워짐


def test_is_present_true_when_selector_visible():
    b = _browser_with_mock_page()
    b._page.wait_for_selector.return_value = None
    assert b.is_present("#avatar", timeout_ms=1000) is True


def test_is_present_false_when_all_candidates_timeout():
    b = _browser_with_mock_page()
    b._page.wait_for_selector.side_effect = PlaywrightTimeoutError("no")
    assert b.is_present(["#a", "#b"], timeout_ms=500) is False
    assert b._page.wait_for_selector.call_count == 2   # 후보를 순서대로 다 시도


def test_ensure_logged_in_requires_url_and_selector():
    b = _browser_with_mock_page()
    with pytest.raises(ValueError):
        b.ensure_logged_in(url="", logged_in_selector="", google_button_selector="#g")


def test_ensure_logged_in_skips_when_already_logged_in():
    b = _browser_with_mock_page()
    # logged_in_selector가 바로 보임 → 재로그인 생략.
    with patch.object(b, "is_present", return_value=True):
        result = b.ensure_logged_in(
            url="https://s.test/login", logged_in_selector="#avatar",
            google_button_selector='button:has-text("Google")',
        )
    assert result == "이미 로그인됨"
    b._page.click.assert_not_called()        # 구글 버튼을 누르지 않음


def test_ensure_logged_in_clicks_google_and_verifies():
    b = _browser_with_mock_page()
    b._page.wait_for_selector.return_value = None   # _first_visible(구글 버튼) 성공
    # 첫 확인: 미로그인(False), 클릭 후 확인: 성공(True).
    with patch.object(b, "is_present", side_effect=[False, True]):
        result = b.ensure_logged_in(
            url="https://s.test/login", logged_in_selector="#avatar",
            google_button_selector='button:has-text("Google")',
        )
    assert result == "구글로 재로그인 성공"
    b._page.click.assert_called_once()       # 구글 버튼 클릭됨


def test_ensure_logged_in_raises_when_still_not_logged_in():
    b = _browser_with_mock_page()
    b._page.wait_for_selector.return_value = None
    # 클릭 후에도 로그인 확인 실패(2FA 등) → 명확한 오류.
    with patch.object(b, "is_present", side_effect=[False, False]):
        with pytest.raises(RuntimeError):
            b.ensure_logged_in(
                url="https://s.test/login", logged_in_selector="#avatar",
                google_button_selector='button:has-text("Google")',
            )


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


def test_wait_for_response_stable_tries_selector_candidates_each_poll():
    b = _browser_with_mock_page()
    # 매 폴링마다 첫 후보는 타임아웃, 둘째 후보가 안정된 텍스트를 돌려준다.
    b._page.inner_text.side_effect = [
        PlaywrightTimeoutError("없음"), "답변",
        PlaywrightTimeoutError("없음"), "답변",
        PlaywrightTimeoutError("없음"), "답변",
    ]

    b._wait_for_response_stable(
        ["#missing", "#answer"], max_wait_ms=10000, poll_interval_ms=100, stable_polls_required=2
    )

    assert b._page.inner_text.call_count == 6


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


def test_resolve_profile_dir_slot_zero_returns_base_without_copy(tmp_path):
    base = tmp_path / "profile"
    base.mkdir()

    with patch("tools.browser.shutil.copytree") as copytree:
        result = resolve_profile_dir(str(base), 0)

    assert result == str(base)
    copytree.assert_not_called()


def test_resolve_profile_dir_copies_base_on_first_use(tmp_path):
    base = tmp_path / "profile"
    base.mkdir()
    (base / "marker.txt").write_text("로그인 세션")

    worker_dir = resolve_profile_dir(str(base), 1)

    assert worker_dir == f"{base}_worker1"
    assert os.path.exists(worker_dir)
    assert (tmp_path / "profile_worker1" / "marker.txt").read_text() == "로그인 세션"


def test_resolve_profile_dir_does_not_recopy_when_worker_dir_exists(tmp_path):
    base = tmp_path / "profile"
    base.mkdir()
    worker_dir = tmp_path / "profile_worker1"
    worker_dir.mkdir()

    with patch("tools.browser.shutil.copytree") as copytree:
        result = resolve_profile_dir(str(base), 1)

    assert result == str(worker_dir)
    copytree.assert_not_called()


def test_resolve_profile_dir_skips_copy_when_base_missing(tmp_path):
    base = tmp_path / "missing_profile"

    with patch("tools.browser.shutil.copytree") as copytree:
        result = resolve_profile_dir(str(base), 1)

    assert result == f"{base}_worker1"
    copytree.assert_not_called()
    assert not os.path.exists(result)

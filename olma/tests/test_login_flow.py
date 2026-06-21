"""기능: 명시적 login 액션 + 자동 재로그인 폴백.

login step이 browser.ensure_logged_in으로 분배되는지, 그리고 AUTO_RELOGIN=true일 때
브라우저 step이 로그인 벽에 막혀 실패하면 자동 재로그인 후 재시도하는지 검증한다.
외부 의존 없이 Browser와 login_providers를 모킹해 돈다.
"""
from unittest.mock import MagicMock, patch

from executor import executor
from executor.executor import execute_steps


def test_login_action_dispatches_to_ensure_logged_in():
    mock_browser = MagicMock()
    mock_browser.ensure_logged_in.return_value = "구글로 재로그인 성공"
    provider = {
        "url": "https://s.test/login", "logged_in_selector": "#avatar",
        "google_button_selector": 'button:has-text("Google")', "wait_ms": 30000,
    }

    steps = [{"action": "login", "input": "notion", "depends_on": None}]
    with patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.login_providers.resolve", return_value=provider) as resolve:
        results = execute_steps(steps)

    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "구글로 재로그인 성공"
    resolve.assert_called_once_with("notion")
    assert mock_browser.ensure_logged_in.call_args.args[0] == "https://s.test/login"


def test_login_failure_does_not_fall_back_to_ollama():
    """login은 실패해도 ollama 폴백이 무의미하므로(로컬 LLM이 로그인 불가) 폴백하지 않는다."""
    mock_browser = MagicMock()
    mock_browser.ensure_logged_in.side_effect = RuntimeError("2FA 필요")

    steps = [{"action": "login", "input": "", "depends_on": None}]
    provider = {"url": "u", "logged_in_selector": "s", "google_button_selector": "g", "wait_ms": 1}
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.login_providers.resolve", return_value=provider), \
         patch("executor.executor.login_providers.default_for_auto", return_value=None), \
         patch("executor.executor.ollama_client.generate") as gen:
        results = execute_steps(steps)

    assert results[0]["status"] == "failed"
    gen.assert_not_called()           # ollama 폴백을 시도하지 않음


def _auto_provider():
    return {
        "url": "https://s.test/login", "logged_in_selector": "#avatar",
        "google_button_selector": 'button:has-text("Google")',
        "login_wall_selector": "#login-form", "wait_ms": 30000,
    }


def test_auto_relogin_retries_after_login_wall(monkeypatch):
    monkeypatch.setattr(executor, "AUTO_RELOGIN", True)
    monkeypatch.setattr(executor, "RETRY_COUNT", 0)   # step당 1회 시도로 고정(결정론)
    mock_browser = MagicMock()
    # 첫 검색은 로그인 벽 때문에 실패, 재로그인 후 재시도는 성공.
    mock_browser.search.side_effect = [RuntimeError("로그인 필요"), "로그인 후 검색 결과"]
    mock_browser.is_present.return_value = True   # 로그인 벽 감지됨

    steps = [{"action": "browser_search", "input": "뉴스", "depends_on": None}]
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.login_providers.default_for_auto", return_value=_auto_provider()):
        results = execute_steps(steps)

    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "로그인 후 검색 결과"
    mock_browser.ensure_logged_in.assert_called_once()   # 자동 재로그인 1회


def test_auto_relogin_skipped_when_no_login_wall(monkeypatch):
    monkeypatch.setattr(executor, "AUTO_RELOGIN", True)
    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("다른 이유의 실패")
    mock_browser.is_present.return_value = False   # 로그인 벽 아님

    steps = [{"action": "browser_search", "input": "뉴스", "depends_on": None}]
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.login_providers.default_for_auto", return_value=_auto_provider()), \
         patch("executor.executor.ollama_client.generate", return_value="폴백 답변"):
        results = execute_steps(steps)

    mock_browser.ensure_logged_in.assert_not_called()    # 로그인 벽 아니면 재로그인 안 함
    assert results[0]["status"] == "fallback"            # 평소대로 ollama 폴백


def test_auto_relogin_disabled_by_default(monkeypatch):
    monkeypatch.setattr(executor, "AUTO_RELOGIN", False)
    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("로그인 필요")

    steps = [{"action": "browser_search", "input": "뉴스", "depends_on": None}]
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.login_providers.default_for_auto", return_value=_auto_provider()), \
         patch("executor.executor.ollama_client.generate", return_value="폴백 답변"):
        execute_steps(steps)

    mock_browser.ensure_logged_in.assert_not_called()    # 끄면 자동 재로그인 안 함
    mock_browser.is_present.assert_not_called()


def test_auto_relogin_none_provider_does_nothing(monkeypatch):
    """default 제공자가 자동 대상이 아니면(login_wall 미설정) 아무 일도 일어나지 않는다."""
    monkeypatch.setattr(executor, "AUTO_RELOGIN", True)
    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("로그인 필요")

    steps = [{"action": "browser_search", "input": "뉴스", "depends_on": None}]
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.login_providers.default_for_auto", return_value=None), \
         patch("executor.executor.ollama_client.generate", return_value="폴백 답변"):
        results = execute_steps(steps)

    mock_browser.ensure_logged_in.assert_not_called()
    assert results[0]["status"] == "fallback"

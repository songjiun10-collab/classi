"""터미널 UI 프리미티브(core.ui) 테스트 — 색 토글·글리프·박스·렌더가 결정론적인지.

렌더 함수는 use_color를 인자로 받는 순수 함수라 tty 없이도 정확히 검증된다."""
from core import ui


def test_paint_off_returns_plain():
    assert ui.paint("hi", "green", use_color=False) == "hi"


def test_paint_on_wraps_with_ansi():
    out = ui.paint("hi", "green", use_color=True)
    assert out.startswith("\033[")
    assert out.endswith("\033[0m")
    assert "hi" in out


def test_status_label_uses_known_glyph():
    assert ui.status_glyph("done") == "●"
    assert ui.status_glyph("rejected") == "✗"
    # 모르는 상태는 기본 글리프로 떨어진다(예외 없이).
    assert ui.status_glyph("weird") == "•"


def test_status_label_plain_contains_status():
    assert ui.status_label("failed", use_color=False) == "● failed"


def test_box_has_borders_and_content():
    out = ui.box(["줄1", "줄2"], use_color=False)
    assert "╭" in out and "╮" in out
    assert "╰" in out and "╯" in out
    assert "줄1" in out and "줄2" in out


def test_banner_mentions_version_and_help():
    out = ui.banner("0.5", "gemma4:latest", "/tmp/x", use_color=False)
    assert "0.5" in out
    assert "gemma4:latest" in out
    assert "/help" in out


def test_render_step_failed_shows_error():
    out = ui.render_step(
        {"action": "browser_open", "status": "failed", "error": "타임아웃"}, use_color=False
    )
    assert "browser_open" in out
    assert "타임아웃" in out
    assert out.startswith("●")


def test_render_step_truncates_long_result():
    out = ui.render_step(
        {"action": "llm", "status": "ok", "result": "가" * 500}, use_color=False
    )
    assert "…" in out


def test_render_step_meta_shows_attempts_and_agent():
    out = ui.render_step(
        {"action": "llm", "status": "ok", "attempts": 2, "agent": "local_llm", "result": "x"},
        use_color=False,
    )
    assert "2회 시도" in out
    assert "local_llm" in out


def test_render_help_lists_commands_sorted_columns():
    out = ui.render_help([("/help", "도움말"), ("/status", "상태")], use_color=False)
    assert "/help" in out
    assert "/status" in out
    assert "도움말" in out


def test_detect_color_off_when_no_color_env(monkeypatch):
    monkeypatch.setenv("NO_COLOR", "1")
    assert ui.detect_color() is False

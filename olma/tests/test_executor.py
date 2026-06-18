from unittest.mock import MagicMock, patch

from core.schema import EXTERNAL_DATA_BEGIN, EXTERNAL_DATA_END, MAX_INPUT_CHARS
from executor.executor import _substitute_dependency, _wrap_external, execute_steps


def test_wrap_external_wraps_with_delimiters():
    wrapped = _wrap_external("hello")
    assert wrapped.startswith(EXTERNAL_DATA_BEGIN)
    assert wrapped.endswith(EXTERNAL_DATA_END)
    assert "hello" in wrapped


def test_wrap_external_truncates_long_text():
    long_text = "a" * (MAX_INPUT_CHARS + 500)
    wrapped = _wrap_external(long_text)
    assert "a" * (MAX_INPUT_CHARS + 500) not in wrapped
    assert "a" * MAX_INPUT_CHARS in wrapped


def test_substitute_dependency_replaces_result_token():
    step = {"action": "summarize", "input": "look: {{result}}", "depends_on": 0}
    results_by_index = {0: {"status": "ok", "result": "the answer"}}
    new_step = _substitute_dependency(step, results_by_index)
    assert "{{result}}" not in new_step["input"]
    assert "the answer" in new_step["input"]
    assert step["input"] == "look: {{result}}"  # 원본은 변경되지 않음


def test_substitute_dependency_propagates_error_text_on_failed_dependency():
    step = {"action": "llm", "input": "react: {{result}}", "depends_on": 0}
    results_by_index = {0: {"status": "failed", "result": "error: boom"}}
    new_step = _substitute_dependency(step, results_by_index)
    assert "error: boom" in new_step["input"]


def test_substitute_dependency_noop_without_depends_on():
    step = {"action": "llm", "input": "no token here", "depends_on": None}
    new_step = _substitute_dependency(step, {})
    assert new_step == step


def test_execute_steps_resolves_dependency_referencing_earlier_index():
    steps = [
        {"action": "llm", "input": "first", "depends_on": None},
        {"action": "llm", "input": "second", "depends_on": None},
        {"action": "summarize", "input": "combine: {{result}}", "depends_on": 0},
    ]
    with patch(
        "executor.executor.ollama_client.generate",
        side_effect=["result-A", "result-B", "result-C"],
    ) as gen:
        results = execute_steps(steps)

    assert [r["status"] for r in results] == ["ok", "ok", "ok"]
    third_call_prompt = gen.call_args_list[2].args[0]
    assert "result-A" in third_call_prompt
    assert "result-B" not in third_call_prompt


def test_execute_steps_multistep_with_browser_and_dependency():
    mock_browser = MagicMock()
    mock_browser.get_text.return_value = "page content"

    steps = [
        {"action": "browser_open", "input": "https://example.com", "depends_on": None},
        {"action": "browser_get_text", "input": "", "depends_on": None},
        {"action": "summarize", "input": "{{result}}", "depends_on": 1},
    ]

    with patch("executor.executor.Browser", return_value=mock_browser), patch(
        "executor.executor.ollama_client.generate", return_value="요약 결과"
    ):
        results = execute_steps(steps)

    assert [r["status"] for r in results] == ["ok", "ok", "ok"]
    assert results[-1]["result"] == "요약 결과"
    mock_browser.open.assert_called_once_with("https://example.com")


def test_execute_steps_retries_then_succeeds():
    steps = [{"action": "llm", "input": "hi", "depends_on": None}]
    with patch("executor.executor.time.sleep"), patch(
        "executor.executor.ollama_client.generate",
        side_effect=[RuntimeError("일시 실패"), "두번째에 성공"],
    ) as gen:
        results = execute_steps(steps)

    assert results[0]["status"] == "ok"
    assert results[0]["attempts"] == 2
    assert results[0]["result"] == "두번째에 성공"
    assert gen.call_count == 2


def test_execute_steps_browser_failure_falls_back_to_ollama():
    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("브라우저 죽음")

    steps = [{"action": "browser_search", "input": "뉴스", "depends_on": None}]
    with patch("executor.executor.time.sleep"), patch(
        "executor.executor.Browser", return_value=mock_browser
    ), patch(
        "executor.executor.ollama_client.generate", return_value="아는 선에서의 답변"
    ) as gen:
        results = execute_steps(steps)

    assert results[0]["status"] == "fallback"
    assert results[0]["result"] == "아는 선에서의 답변"
    assert gen.call_count == 1
    mock_browser.debug_screenshot.assert_called()


def test_execute_steps_dispatches_web_ai_ask_to_browser():
    mock_browser = MagicMock()
    mock_browser.ask_web_ai.return_value = "웹 AI 답변"

    steps = [{"action": "web_ai_ask", "input": "안녕?", "depends_on": None}]
    with patch("executor.executor.Browser", return_value=mock_browser):
        results = execute_steps(steps)

    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "웹 AI 답변"
    assert mock_browser.ask_web_ai.call_args.args[0] == "안녕?"


def test_execute_steps_records_failed_when_target_and_fallback_both_fail():
    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("브라우저 죽음")

    steps = [{"action": "browser_search", "input": "뉴스", "depends_on": None}]
    with patch("executor.executor.time.sleep"), patch(
        "executor.executor.Browser", return_value=mock_browser
    ), patch(
        "executor.executor.ollama_client.generate", side_effect=RuntimeError("LLM도 죽음")
    ):
        results = execute_steps(steps)

    assert results[0]["status"] == "failed"
    assert "error:" in results[0]["result"]
    assert results[0]["error"] is not None

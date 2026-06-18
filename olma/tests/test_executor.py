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

from unittest.mock import MagicMock, patch

from config.config import BROWSER_USER_DATA_DIR
from core.schema import EXTERNAL_DATA_BEGIN, EXTERNAL_DATA_END, MAX_INPUT_CHARS
from executor import executor
from executor.executor import _substitute_dependency, _wrap_external, execute_steps


def test_approval_rejection_marks_step_rejected_and_skips_dependents():
    """승인 게이트가 거절하면 그 step은 rejected, 그에 의존하는 step은 skipped 되어야 한다."""
    steps = [
        {"action": "browser_click", "input": "#buy", "depends_on": None},
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},
    ]
    with patch.object(executor.approval, "guard", return_value=(False, "사용자가 거절함")), \
         patch("executor.executor.Browser"):
        results = execute_steps(steps)
    assert results[0]["status"] == "rejected"
    assert "사용자가 거절함" in results[0]["result"]
    assert results[1]["status"] == "skipped"   # 거절된 step에 의존 → 스킵


def test_approval_pass_runs_step_normally():
    """게이트가 통과시키면(allowed) step은 평소대로 실행된다."""
    steps = [{"action": "llm", "input": "안녕", "depends_on": None}]
    with patch.object(executor.approval, "guard", return_value=(True, "")), \
         patch("llm.ollama_client.generate", return_value="반가워"):
        results = execute_steps(steps)
    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "반가워"


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

    # 내용 기반 mock: 독립 step 0/1은 병렬로 돌 수 있어 호출 순서가 비결정적이므로,
    # 순서 의존 side_effect 대신 프롬프트 내용으로 응답을 결정해 결과 매핑을 고정한다.
    def fake_generate(prompt, **kwargs):
        if "combine" in prompt:        # summarize step (dep 0의 결과가 치환돼 들어옴)
            return "result-C"
        if "first" in prompt:
            return "result-A"
        return "result-B"              # "second"

    with patch("executor.executor.ollama_client.generate", side_effect=fake_generate) as gen:
        results = execute_steps(steps)

    assert [r["status"] for r in results] == ["ok", "ok", "ok"]
    # summarize는 의존성 때문에 항상 마지막(순차)에 실행된다.
    third_call_prompt = gen.call_args_list[2].args[0]
    assert "result-A" in third_call_prompt   # dep 0(=first)의 결과가 들어가야
    assert "result-B" not in third_call_prompt  # dep 1(=second)의 결과는 아님


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


def test_execute_steps_passes_profile_slot_to_browser_user_data_dir():
    mock_browser = MagicMock()
    mock_browser.get_text.return_value = "page content"

    steps = [{"action": "browser_get_text", "input": "", "depends_on": None}]
    with patch("executor.executor.Browser", return_value=mock_browser) as browser_cls, patch(
        "executor.executor.resolve_profile_dir", return_value="/tmp/profile_worker2"
    ) as resolve:
        results = execute_steps(steps, profile_slot=2)

    assert results[0]["status"] == "ok"
    resolve.assert_called_once_with(BROWSER_USER_DATA_DIR, 2)
    browser_cls.assert_called_once_with(user_data_dir="/tmp/profile_worker2")


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


def test_web_ai_fails_over_to_backup_provider():
    """첫 웹 AI(claude)가 실패하면 백업(zai)으로 페일오버해 성공한다."""
    mock_browser = MagicMock()
    mock_browser.ask_web_ai.side_effect = [RuntimeError("claude 막힘"), "zai 답변"]
    chain = [
        {"url": "https://claude.test", "input_selector": "#c", "submit_selector": "",
         "response_selector": "body", "wait_ms": 1},
        {"url": "https://z.ai", "input_selector": "#z", "submit_selector": "",
         "response_selector": "body", "wait_ms": 1},
    ]
    steps = [{"action": "web_ai_ask", "input": "이 코드 봐줘", "depends_on": None}]
    with patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.web_ai_providers.resolve_chain", return_value=(chain, "이 코드 봐줘")):
        results = execute_steps(steps)

    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "zai 답변"
    assert mock_browser.ask_web_ai.call_count == 2          # claude → zai
    assert mock_browser.ask_web_ai.call_args_list[1].args[1] == "https://z.ai"


def test_web_ai_all_providers_fail_then_local_fallback():
    """체인이 전부 실패하면 라우터 폴백(로컬 reason 모델)이 받는다."""
    mock_browser = MagicMock()
    mock_browser.ask_web_ai.side_effect = RuntimeError("전부 막힘")
    chain = [
        {"url": "https://claude.test", "input_selector": "#c", "submit_selector": "",
         "response_selector": "body", "wait_ms": 1},
        {"url": "https://z.ai", "input_selector": "#z", "submit_selector": "",
         "response_selector": "body", "wait_ms": 1},
    ]
    steps = [{"action": "web_ai_ask", "input": "질문", "depends_on": None}]
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.web_ai_providers.resolve_chain", return_value=(chain, "질문")), \
         patch("executor.executor.ollama_client.generate", return_value="로컬 폴백 답변"):
        results = execute_steps(steps)

    assert results[0]["status"] == "fallback"               # 로컬로 폴백
    assert results[0]["result"] == "로컬 폴백 답변"


def test_execute_steps_dispatches_vision_describe_to_browser():
    mock_browser = MagicMock()
    mock_browser.describe_screen.return_value = "화면엔 로그인 폼이 보인다"

    steps = [{"action": "vision_describe", "input": "뭐가 보여?", "depends_on": None}]
    with patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.ai_roles.models_for", return_value=["vlm:7b", "vlm-backup"]):
        results = execute_steps(steps)

    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "화면엔 로그인 폼이 보인다"
    # vision 체인의 1순위가 model로, 나머지가 fallback_models로 전달된다.
    assert mock_browser.describe_screen.call_args.args[0] == "뭐가 보여?"
    assert mock_browser.describe_screen.call_args.kwargs["model"] == "vlm:7b"
    assert mock_browser.describe_screen.call_args.kwargs["fallback_models"] == ["vlm-backup"]


def test_vision_describe_failure_does_not_fall_back_to_ollama():
    """vision_describe는 실패해도 텍스트 ollama 폴백이 무의미(이미지 이해 불가)하므로 폴백 안 함."""
    mock_browser = MagicMock()
    mock_browser.describe_screen.side_effect = ValueError("VLM 미설정")

    steps = [{"action": "vision_describe", "input": "", "depends_on": None}]
    with patch("executor.executor.time.sleep"), \
         patch("executor.executor.Browser", return_value=mock_browser), \
         patch("executor.executor.ai_roles.model_for", return_value=""), \
         patch("executor.executor.ollama_client.generate") as gen:
        results = execute_steps(steps)

    assert results[0]["status"] == "failed"
    gen.assert_not_called()


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


def test_execute_steps_skips_step_with_failed_dependency():
    steps = [
        {"action": "browser_search", "input": "뉴스", "depends_on": None},
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},
    ]
    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("브라우저 죽음")

    with patch("executor.executor.time.sleep"), patch(
        "executor.executor.Browser", return_value=mock_browser
    ), patch(
        "executor.executor.ollama_client.generate", side_effect=RuntimeError("폴백도 죽음")
    ) as gen:
        results = execute_steps(steps)

    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "skipped"
    # 두번째 step은 실행되지 않았으므로 ollama_client.generate가 두번째 step 때문에
    # 호출되지 않는다(첫 step의 폴백 시도 1회만 호출됨).
    assert gen.call_count == 1


def test_execute_steps_does_not_skip_when_dependency_succeeded_via_fallback():
    mock_browser = MagicMock()
    mock_browser.search.side_effect = RuntimeError("브라우저 죽음")

    steps = [
        {"action": "browser_search", "input": "뉴스", "depends_on": None},
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},
    ]
    with patch("executor.executor.time.sleep"), patch(
        "executor.executor.Browser", return_value=mock_browser
    ), patch(
        "executor.executor.ollama_client.generate",
        side_effect=["아는 선에서의 답변", "요약 결과"],
    ):
        results = execute_steps(steps)

    assert results[0]["status"] == "fallback"
    assert results[1]["status"] == "ok"
    assert results[1]["result"] == "요약 결과"


def test_execute_steps_continues_when_router_raises_unexpectedly():
    steps = [
        {"action": "llm", "input": "first", "depends_on": None},
        {"action": "llm", "input": "second", "depends_on": None},
    ]
    # 내용 기반 mock: 독립 step은 병렬로 돌 수 있어 호출 순서가 비결정적이므로,
    # step 입력으로 어느 step이 폭발할지 고정한다(순서 의존 side_effect 회피).
    def fake_policy(step):
        if step.get("input") == "first":
            raise RuntimeError("라우터 폭발")
        return {"target": "ollama", "fallback": None, "confidence": "high"}

    with patch("executor.executor.router.route_policy", side_effect=fake_policy), patch(
        "executor.executor.ollama_client.generate", return_value="두번째는 정상"
    ):
        results = execute_steps(steps)

    assert results[0]["status"] == "failed"
    assert "라우터 폭발" in results[0]["result"]
    assert results[1]["status"] == "ok"
    assert results[1]["result"] == "두번째는 정상"

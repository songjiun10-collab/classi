"""Dynamic Workflow Engine(core.workflow) 테스트.

run_dynamic(요청→계획→실행→기억), 파라미터 치환, 템플릿 저장 검증, run_template을
외부 의존 없이 planner/executor/memory와 store를 모킹/임시 DB로 검증한다.
"""
import importlib
from unittest.mock import patch

import pytest


@pytest.fixture
def wf_env(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOW_STORE_PATH", str(tmp_path / "workflows.db"))
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.workflow_store as ws
    importlib.reload(ws)
    import core.memory as mem
    importlib.reload(mem)
    import core.workflow as wf
    importlib.reload(wf)
    return wf


def test_run_dynamic_plans_executes_and_records(wf_env):
    wf = wf_env
    steps = [{"action": "llm", "input": "안녕", "depends_on": None}]
    results = [{"action": "llm", "status": "ok", "result": "반가워"}]

    with patch("core.workflow.plan", return_value=steps) as p, \
         patch("core.workflow.execute_steps", return_value=results) as e, \
         patch("core.workflow.memory.get_context", return_value=""), \
         patch("core.workflow.memory.successful_examples", return_value=[]):
        summary = wf.run_dynamic("인사해줘")

    p.assert_called_once()
    e.assert_called_once()
    assert summary["source"] == "dynamic"
    assert summary["status"] == "done"
    assert summary["step_count"] == 1
    assert summary["steps"] == results


def test_apply_params_substitutes_param_tokens(wf_env):
    wf = wf_env
    steps = [{"action": "browser_open", "input": "https://{{param:site}}/login", "depends_on": None},
             {"action": "llm", "input": "{{param:q}} 정리", "depends_on": None}]
    out = wf._apply_params(steps, {"site": "github.com", "q": "이슈"})
    assert out[0]["input"] == "https://github.com/login"
    assert out[1]["input"] == "이슈 정리"


def test_apply_params_keeps_unknown_token(wf_env):
    wf = wf_env
    steps = [{"action": "llm", "input": "{{param:missing}}", "depends_on": None}]
    out = wf._apply_params(steps, {})
    assert out[0]["input"] == "{{param:missing}}"   # 미정의 키는 원형 유지


def test_apply_params_does_not_touch_result_token(wf_env):
    wf = wf_env
    # executor의 {{result}}는 파라미터 치환 대상이 아니어야 한다(실행 시점 토큰).
    steps = [{"action": "summarize", "input": "{{result}}", "depends_on": 0}]
    out = wf._apply_params(steps, {"result": "X"})
    assert out[0]["input"] == "{{result}}"


def test_save_template_validates_and_normalizes(wf_env):
    wf = wf_env
    saved = wf.save_template("인사", "설명", [{"action": "llm", "input": "hi"}])
    assert saved["name"] == "인사"
    # depends_on 등 누락 필드는 스키마 기본값으로 정규화된다.
    assert saved["steps"][0]["depends_on"] is None
    import core.workflow_store as ws
    assert ws.get("인사") is not None


def test_save_template_rejects_invalid_steps(wf_env):
    wf = wf_env
    from pydantic import ValidationError
    with pytest.raises(ValidationError):
        wf.save_template("나쁨", "", [{"action": "존재하지않는액션"}])


def test_run_template_applies_params_and_executes(wf_env):
    wf = wf_env
    wf.save_template("리서치", "", [{"action": "browser_search", "input": "{{param:topic}}"}])
    results = [{"action": "browser_search", "status": "ok", "result": "검색됨"}]

    with patch("core.workflow.execute_steps", return_value=results) as e:
        summary = wf.run_template("리서치", params={"topic": "LLM 최신 동향"})

    # 치환된 step이 executor로 전달돼야 한다.
    passed_steps = e.call_args.args[0]
    assert passed_steps[0]["input"] == "LLM 최신 동향"
    assert summary["source"] == "template"
    assert summary["workflow"] == "리서치"
    assert summary["status"] == "done"


def test_run_template_unknown_raises(wf_env):
    wf = wf_env
    with pytest.raises(ValueError):
        wf.run_template("없는워크플로우")

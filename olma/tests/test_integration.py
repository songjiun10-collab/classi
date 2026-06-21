"""통합 테스트: plan→execute→memory 전체 배선을 실제로 태운다.

단위 테스트는 각 모듈을 격리해 모킹하지만, 여기서는 planner↔executor↔router↔memory를
실제로 연결한 채 외부 I/O 두 곳만 가짜로 바꿔 결정론적으로 돌린다:
  - 로컬 Ollama 호출(`llm.ollama_client.generate`)
  - Playwright 브라우저(`executor.executor.Browser`)

이렇게 하면 "스키마 검증 → 라우팅 → 재시도/폴백 → 의존성 치환 → 상태 집계 → SQLite 저장"이
하나로 이어졌을 때 깨지지 않는지를 검증한다.
"""
import json
from unittest.mock import patch

import pytest

from core import memory, planner
from core.schema import SCHEMA_VERSION
from executor import executor


@pytest.fixture
def temp_memory(tmp_path, monkeypatch):
    """memory가 임시 SQLite 파일을 쓰도록 모듈 전역을 갈아끼운다(실 DB 오염 방지)."""
    monkeypatch.setattr(memory, "MEMORY_PATH", str(tmp_path / "memory.db"))
    yield


class _FakeBrowser:
    """Playwright 없이 executor의 브라우저 경로를 태우기 위한 가짜. 동작은 모두 실패시켜
    라우터 폴백(browser→ollama)이 실제로 동작하는지 본다."""

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, _name):
        def _boom(*args, **kwargs):
            raise RuntimeError("테스트 환경에는 실제 브라우저가 없음")
        return _boom

    def debug_screenshot(self, *args, **kwargs):
        return ""


def test_planner_to_executor_to_memory_ollama_only(temp_memory):
    """LLM 한 스텝짜리 요청: 계획 생성(format 강제)→실행→저장이 끝까지 done으로 이어진다."""
    def fake_generate(prompt, **kwargs):
        if kwargs.get("format"):  # planner 호출만 format을 넘긴다
            return json.dumps({
                "schema_version": SCHEMA_VERSION,
                "steps": [{"action": "llm", "input": "안녕", "depends_on": None}],
            })
        return "안녕하세요!"  # executor의 llm 실행 응답

    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        steps = planner.plan("안녕이라고 답해줘")
        results = executor.execute_steps(steps)

    assert len(results) == 1
    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "안녕하세요!"

    record = memory.save("안녕이라고 답해줘", results)
    assert record["status"] == "done"
    # 저장된 기록이 맥락 조회로 다시 나와야 한다(SQLite 왕복).
    assert "안녕이라고 답해줘" in memory.get_context()


def test_browser_failure_falls_back_to_ollama(temp_memory):
    """브라우저 스텝이 실패하면 라우터 폴백으로 ollama가 받아 fallback 상태로 성공 처리된다."""
    steps = [{"action": "browser_open", "input": "https://example.com", "depends_on": None}]

    with patch("executor.executor.Browser", _FakeBrowser), \
            patch("llm.ollama_client.generate", return_value="브라우저 없이 아는 선에서 답함"):
        results = executor.execute_steps(steps)

    assert results[0]["status"] == "fallback"
    assert results[0]["result"] == "브라우저 없이 아는 선에서 답함"
    record = memory.save("사이트 열어줘", results)
    assert record["status"] == "done"  # fallback도 성공으로 집계


def test_dependency_result_is_substituted_into_next_step(temp_memory):
    """depends_on + {{result}} 토큰이 이전 스텝 결과로 실제 치환되어 다음 스텝에 전달된다."""
    steps = [
        {"action": "llm", "input": "첫 스텝", "depends_on": None},
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},
    ]

    seen_summarize_prompt = {}

    def fake_generate(prompt, **kwargs):
        if prompt.startswith("다음 내용을 한국어로 간결하게 요약"):
            seen_summarize_prompt["prompt"] = prompt
            return "요약 결과"
        return "첫 스텝 원문 결과"

    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        results = executor.execute_steps(steps)

    assert results[0]["result"] == "첫 스텝 원문 결과"
    assert results[1]["status"] == "ok"
    assert results[1]["result"] == "요약 결과"
    # 2번째 스텝 프롬프트에 1번째 결과가 (구분자로 감싸여) 실제로 들어가야 한다.
    assert "첫 스텝 원문 결과" in seen_summarize_prompt["prompt"]


def test_failed_dependency_skips_downstream_step(temp_memory):
    """대체 타겟이 없는 스텝(notification_check)이 실패하면 그에 의존하는 스텝은 건너뛴다."""
    steps = [
        {"action": "notification_check", "input": "", "depends_on": None},
        {"action": "llm", "input": "{{result}}", "depends_on": 0},
    ]

    # KAKAO_WEB_URL 미설정(기본 "") 이라 notifier가 ValueError로 실패 → fallback 없음 → failed.
    with patch("executor.executor.Browser", _FakeBrowser):
        results = executor.execute_steps(steps)

    assert results[0]["status"] == "failed"
    assert results[1]["status"] == "skipped"

    record = memory.save("카톡 확인하고 요약해줘", results)
    assert record["status"] == "failed"  # 성공 스텝이 하나도 없음


def test_invalid_plan_json_falls_back_to_rule_based_planner(temp_memory):
    """플래너 LLM이 쓰레기를 뱉어도 규칙 기반 폴백 플래너가 실행 가능한 스텝을 보장한다."""
    with patch("llm.ollama_client.generate", return_value="이건 JSON이 아니다"):
        steps = planner.plan("https://example.com 열고 요약해줘")

    # 폴백 플래너는 URL을 인식해 browser_open → get_text → summarize 체인을 만든다.
    actions = [s["action"] for s in steps]
    assert "browser_open" in actions
    assert actions  # 어떤 경우에도 최소 1개 스텝을 보장

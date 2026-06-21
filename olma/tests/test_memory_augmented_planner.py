"""기능: 메모리 증강 플래너 — 과거 성공 작업을 few-shot으로 주입한다.

memory.successful_examples(성공만 추출)와 planner가 그 예시를 프롬프트에 넣는지 검증한다.
외부 의존(Ollama/Playwright) 없이 모킹/임시 DB로만 돈다.
"""
from unittest.mock import patch

import pytest

from core import memory, planner
from core.schema import SCHEMA_VERSION


@pytest.fixture
def temp_memory(tmp_path, monkeypatch):
    monkeypatch.setattr(memory, "MEMORY_PATH", str(tmp_path / "memory.db"))
    yield


def _done(action):
    return [{"action": action, "target": "ollama", "status": "ok",
             "attempts": 1, "duration": 0.1, "result": "ok"}]


def test_successful_examples_returns_only_done(temp_memory):
    memory.save("날씨 알려줘", _done("llm"))
    # 실패 작업은 나쁜 예시이므로 제외돼야 한다.
    memory.save("깨진 작업", [{"action": "browser_open", "status": "failed",
                               "attempts": 3, "duration": 0.0, "result": "error"}])

    examples = memory.successful_examples("날씨")
    assert len(examples) == 1
    assert examples[0]["task"] == "날씨 알려줘"
    assert examples[0]["actions"] == ["llm"]


def test_successful_examples_excludes_failed_even_if_keyword_matches(temp_memory):
    memory.save("리포트 정리해줘", [{"action": "summarize", "status": "failed",
                                    "attempts": 3, "duration": 0.0, "result": "error"}])
    assert memory.successful_examples("리포트") == []


def test_build_prompt_includes_example_actions():
    examples = [{"task": "파이썬 사이트 요약", "actions": ["browser_open", "browser_get_text", "summarize"]}]
    prompt = planner._build_prompt("뭐 좀 요약해줘", examples=examples)
    assert "파이썬 사이트 요약" in prompt
    assert "browser_open → browser_get_text → summarize" in prompt


def test_build_prompt_without_examples_has_no_example_block():
    prompt = planner._build_prompt("그냥 질문", examples=None)
    assert "과거에 성공한 유사 작업" not in prompt


def test_plan_passes_examples_into_planner_prompt():
    captured = {}

    def fake_generate(prompt, **kwargs):
        captured["prompt"] = prompt
        return (f'{{"schema_version": {SCHEMA_VERSION}, "steps": '
                '[{"action": "llm", "input": "x", "depends_on": null}]}')

    examples = [{"task": "과거작업", "actions": ["llm"]}]
    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        steps = planner.plan("새 요청", examples=examples)

    assert steps  # 계획이 나와야 한다
    assert "과거작업" in captured["prompt"]

"""기능 D: 의존성 없는 ollama step 병렬 실행.

병렬 후보 선정 규칙, 결과 순서 보존, 의존성 체인과의 정합성을 검증한다. 추가로 Barrier로
'실제로 동시에 돌았는지'를 증명한다(순차였다면 Barrier가 타임아웃돼 실패). 외부 의존 없이
ollama_client.generate를 모킹해 돈다.
"""
import threading
from unittest.mock import patch

from executor import executor


def _ollama(i):
    return {"action": "llm", "input": f"질문{i}", "depends_on": None}


def test_parallel_candidates_selects_independent_ollama(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    steps = [
        _ollama(0),
        {"action": "browser_open", "input": "u", "depends_on": None},   # 브라우저 → 제외
        _ollama(1),
        {"action": "summarize", "input": "{{result}}", "depends_on": 2},  # 의존 있음 → 제외
    ]
    assert executor._parallel_candidates(steps, on_token=None) == [0, 2]


def test_parallel_candidates_empty_when_disabled(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", False)
    assert executor._parallel_candidates([_ollama(0), _ollama(1)], on_token=None) == []


def test_parallel_candidates_empty_when_streaming(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    # 스트리밍(on_token) 시엔 출력 순서를 위해 병렬 비활성.
    assert executor._parallel_candidates([_ollama(0), _ollama(1)], on_token=lambda x: None) == []


def test_parallel_candidates_empty_when_single(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    assert executor._parallel_candidates([_ollama(0)], on_token=None) == []


def test_parallel_preserves_result_order(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL_MAX", 4)
    steps = [_ollama(0), _ollama(1), _ollama(2)]

    def fake_generate(prompt, **kwargs):
        return f"답:{prompt}"

    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        results = executor.execute_steps(steps)

    # 완료 순서가 뒤섞여도 결과 리스트는 step 인덱스 순서를 유지해야 한다.
    assert [r["result"] for r in results] == ["답:질문0", "답:질문1", "답:질문2"]
    assert all(r["status"] == "ok" for r in results)


def test_parallel_actually_runs_concurrently(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL_MAX", 3)
    steps = [_ollama(0), _ollama(1), _ollama(2)]

    # 3개 스레드가 동시에 도달해야만 통과하는 Barrier. 순차 실행이면 타임아웃→실패.
    barrier = threading.Barrier(3, timeout=3)

    def fake_generate(prompt, **kwargs):
        barrier.wait()      # 동시성 증명: 셋이 함께 모여야 진행
        return prompt

    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        results = executor.execute_steps(steps)

    assert all(r["status"] == "ok" for r in results)


def test_precompute_waves_groups_by_depth(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    steps = [
        _ollama(0),                                                   # depth 0
        _ollama(1),                                                   # depth 0
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},  # depth 1
        {"action": "summarize", "input": "{{result}}", "depends_on": 1},  # depth 1
        {"action": "summarize", "input": "{{result}}", "depends_on": 2},  # depth 2
    ]
    assert executor._precompute_waves(steps, on_token=None) == [[0, 1], [2, 3], [4]]


def test_precompute_waves_excludes_browser_dependent_ollama(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    steps = [
        {"action": "browser_get_text", "input": "", "depends_on": None},   # 브라우저
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},   # 브라우저 의존 → 제외
        _ollama(2),                                                        # depth 0
    ]
    # 브라우저 결과에 의존하는 ollama(인덱스 1)는 선실행 대상에서 빠진다.
    assert executor._precompute_waves(steps, on_token=None) == [[2]]


def test_precompute_waves_empty_when_streaming(monkeypatch):
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    steps = [_ollama(0), _ollama(1)]
    assert executor._precompute_waves(steps, on_token=lambda x: None) == []


def test_multiwave_parallel_dependency_chain(monkeypatch):
    """깊이 1 웨이브의 summarize들이 깊이 0 결과를 받아 동시 선실행되는지(웨이브 병렬)."""
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL_MAX", 4)
    steps = [
        _ollama(0),
        _ollama(1),
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},
        {"action": "summarize", "input": "{{result}}", "depends_on": 1},
    ]

    def fake_generate(prompt, **kwargs):
        if "요약" in prompt:
            return f"요약({prompt[-20:]})"
        return f"답:{prompt}"

    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        results = executor.execute_steps(steps)

    assert [r["status"] for r in results] == ["ok", "ok", "ok", "ok"]
    # 각 summarize가 자기 의존 step의 결과를 받았는지(0→2, 1→3 매칭).
    assert "답:질문0" in results[2]["step"]["input"]
    assert "답:질문1" in results[3]["step"]["input"]


def test_second_wave_runs_concurrently(monkeypatch):
    """깊이 1 웨이브의 두 step이 실제로 동시에 돌았는지 Barrier로 증명."""
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL_MAX", 4)
    steps = [
        _ollama(0),
        _ollama(1),
        {"action": "summarize", "input": "{{result}}", "depends_on": 0},
        {"action": "summarize", "input": "{{result}}", "depends_on": 1},
    ]
    wave2 = threading.Barrier(2, timeout=3)

    def fake_generate(prompt, **kwargs):
        if "요약" in prompt:
            wave2.wait()      # 깊이 1 두 step이 함께 모여야 진행 — 순차면 타임아웃
        return prompt

    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        results = executor.execute_steps(steps)

    assert all(r["status"] == "ok" for r in results)


def test_dependency_chain_intact_with_parallel(monkeypatch):
    """병렬 선실행된 step의 결과를 뒤따르는 의존 step이 올바르게 참조하는지."""
    monkeypatch.setattr(executor, "OLLAMA_PARALLEL", True)
    steps = [
        _ollama(0),
        _ollama(1),
        {"action": "summarize", "input": "{{result}}", "depends_on": 1},
    ]

    seen = {}

    def fake_generate(prompt, **kwargs):
        # summarize 프롬프트엔 step1 결과가 치환돼 들어와야 한다.
        if "요약" in prompt:
            seen["summarize_prompt"] = prompt
            return "요약결과"
        return f"답:{prompt}"

    with patch("llm.ollama_client.generate", side_effect=fake_generate):
        results = executor.execute_steps(steps)

    assert results[2]["status"] == "ok"
    assert "답:질문1" in seen["summarize_prompt"]   # step1 결과가 의존 step으로 전달됨

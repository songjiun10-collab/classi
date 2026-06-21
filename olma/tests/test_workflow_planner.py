"""Workflow Planner(core.workflow_planner) 테스트 — 템플릿 재사용 vs 동적 생성 결정."""
import importlib
from unittest.mock import patch


def _fresh(tmp_path, monkeypatch, threshold="0.6"):
    monkeypatch.setenv("WORKFLOW_STORE_PATH", str(tmp_path / "workflows.db"))
    monkeypatch.setenv("WORKFLOW_REUSE_THRESHOLD", threshold)
    import config.config as cfg
    importlib.reload(cfg)
    import core.workflow_store as ws
    importlib.reload(ws)
    import core.workflow_planner as wp
    importlib.reload(wp)
    return wp, ws


def test_no_templates_returns_dynamic(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch)
    assert wp.select("아무 요청")["mode"] == "dynamic"


def test_strong_match_selects_template(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch)
    ws.save("뉴스 요약", "최신 뉴스 요약", [{"action": "llm", "input": "뉴스", "depends_on": None}])
    d = wp.select("뉴스 요약 해줘")
    assert d["mode"] == "template"
    assert d["name"] == "뉴스 요약"


def test_weak_match_falls_back_to_dynamic(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch)
    ws.save("주식 포트폴리오 리밸런싱", "", [{"action": "llm", "input": "x", "depends_on": None}])
    # 요청이 템플릿 토큰 중 극히 일부만 포함 → 임계값 미만
    assert wp.select("오늘 날씨")["mode"] == "dynamic"


def test_templates_with_params_are_excluded(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch)
    # 이름은 완전히 일치하지만 파라미터가 있으면 자동 재사용 대상에서 제외된다.
    ws.save("리서치", "리서치",
            [{"action": "browser_search", "input": "{{param:topic}}", "depends_on": None}])
    assert wp.select("리서치")["mode"] == "dynamic"


def test_best_of_multiple_is_chosen(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch)
    ws.save("이메일 정리", "", [{"action": "llm", "input": "a", "depends_on": None}])
    ws.save("이메일 정리 자동화 워크플로우", "", [{"action": "llm", "input": "b", "depends_on": None}])
    # "이메일 정리"는 토큰이 전부 포함돼 점수 1.0 → 더 긴 이름(부분 포함)보다 높다.
    d = wp.select("이메일 정리")
    assert d["mode"] == "template"
    assert d["name"] == "이메일 정리"


def test_threshold_respected(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch, threshold="0.99")
    ws.save("주간 보고", "주간 보고서 작성", [{"action": "llm", "input": "x", "depends_on": None}])
    # 설명 토큰까지 합치면 1.0 미만이라 매우 높은 임계값에선 dynamic
    assert wp.select("주간 보고")["mode"] == "dynamic"


def test_run_request_dispatches_to_template(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch)
    import core.workflow as wf
    importlib.reload(wf)
    ws.save("인사", "인사", [{"action": "llm", "input": "안녕", "depends_on": None}])
    with patch("core.workflow.run_template", return_value={"status": "done"}) as rt, \
         patch("core.workflow.run_dynamic") as rd:
        wf.run_request("인사 해줘")
    rt.assert_called_once()
    rd.assert_not_called()


def test_run_request_dispatches_to_dynamic(tmp_path, monkeypatch):
    wp, ws = _fresh(tmp_path, monkeypatch)
    import core.workflow as wf
    importlib.reload(wf)
    with patch("core.workflow.run_dynamic", return_value={"status": "done"}) as rd, \
         patch("core.workflow.run_template") as rt:
        wf.run_request("전혀 매칭 안 되는 요청 xyz")
    rd.assert_called_once()
    rt.assert_not_called()

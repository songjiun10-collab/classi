"""Workflow 템플릿 저장소(core.workflow_store) 테스트 — 저장/조회/목록/삭제/영속성."""
import importlib


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("WORKFLOW_STORE_PATH", str(tmp_path / "workflows.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.workflow_store as ws
    importlib.reload(ws)
    return ws


_STEPS = [{"action": "llm", "input": "안녕", "depends_on": None}]


def test_save_and_get_roundtrip(tmp_path, monkeypatch):
    ws = _fresh(tmp_path, monkeypatch)
    ws.save("인사", "인사 워크플로우", _STEPS)
    got = ws.get("인사")
    assert got["name"] == "인사"
    assert got["description"] == "인사 워크플로우"
    assert got["steps"] == _STEPS


def test_get_missing_returns_none(tmp_path, monkeypatch):
    ws = _fresh(tmp_path, monkeypatch)
    assert ws.get("없음") is None


def test_save_same_name_overwrites(tmp_path, monkeypatch):
    ws = _fresh(tmp_path, monkeypatch)
    ws.save("wf", "v1", _STEPS)
    ws.save("wf", "v2", [{"action": "summarize", "input": "x", "depends_on": None}])
    got = ws.get("wf")
    assert got["description"] == "v2"
    assert got["steps"][0]["action"] == "summarize"
    assert len(ws.list_all()) == 1   # 덮어쓰기지 중복 아님


def test_list_all_recent_first(tmp_path, monkeypatch):
    ws = _fresh(tmp_path, monkeypatch)
    ws.save("a", "", _STEPS)
    ws.save("b", "", _STEPS)
    ws.save("c", "", _STEPS)
    assert [w["name"] for w in ws.list_all()] == ["c", "b", "a"]


def test_update_moves_to_top(tmp_path, monkeypatch):
    ws = _fresh(tmp_path, monkeypatch)
    ws.save("a", "", _STEPS)
    ws.save("b", "", _STEPS)
    ws.save("a", "갱신", _STEPS)   # a를 갱신 → 최상단으로
    assert [w["name"] for w in ws.list_all()] == ["a", "b"]


def test_delete(tmp_path, monkeypatch):
    ws = _fresh(tmp_path, monkeypatch)
    ws.save("wf", "", _STEPS)
    assert ws.delete("wf") is True
    assert ws.get("wf") is None
    assert ws.delete("wf") is False   # 이미 없으면 False


def test_persists_across_reload(tmp_path, monkeypatch):
    ws = _fresh(tmp_path, monkeypatch)
    ws.save("wf", "유지", _STEPS)
    import core.workflow_store as ws2
    importlib.reload(ws2)
    assert ws2.get("wf")["description"] == "유지"

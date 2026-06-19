import importlib
import time
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolated_task_store(tmp_path, monkeypatch):
    """TaskQueue()는 생성 시 task_store에서 히스토리를 복원하므로, 서버를 새로
    띄우는 테스트마다 실제 storage/tasks.db 대신 테스트별 임시 DB를 쓰게 한다."""
    monkeypatch.setenv("TASK_STORE_PATH", str(tmp_path / "tasks.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.task_store as task_store
    importlib.reload(task_store)


def _fresh_server(monkeypatch, api_key=""):
    monkeypatch.setenv("OLMA_API_KEY", api_key)
    import config.config as cfg
    importlib.reload(cfg)
    import api.server as server
    importlib.reload(server)
    return server


def _wait_until_terminal(client, task_id, headers=None, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        body = client.get(f"/api/task/{task_id}", headers=headers).json()
        if body["status"] in ("completed", "failed"):
            return body
        time.sleep(0.02)
    raise AssertionError("작업이 제한 시간 내에 종료되지 않음")


def test_health_check_does_not_require_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_task_endpoint_rejects_missing_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.post("/api/task", json={"input": "hi"})
    assert r.status_code == 401


def test_task_endpoint_rejects_wrong_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.get("/api/tasks", headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_no_auth_required_when_api_key_unset(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    with patch("core.task_queue.plan", return_value=[]), patch(
        "core.task_queue.execute_steps", return_value=[]
    ), patch("core.task_queue.memory.save", return_value={"status": "done"}), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        r = client.post("/api/task", json={"input": "hi"})
    assert r.status_code == 200


def test_submitted_task_can_be_polled_until_completed(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    headers = {"X-API-Key": "secret"}
    fake_results = [{"action": "llm", "status": "ok", "result": "답변"}]
    with patch("core.task_queue.plan", return_value=[{"action": "llm", "input": "hi"}]), patch(
        "core.task_queue.execute_steps", return_value=fake_results
    ), patch(
        "core.task_queue.memory.save", return_value={"status": "done"}
    ), patch("core.task_queue.memory.get_context", return_value=""):
        r = client.post("/api/task", json={"input": "hi"}, headers=headers)
        assert r.status_code == 200
        task_id = r.json()["task_id"]
        body = _wait_until_terminal(client, task_id, headers=headers)

    assert body["status"] == "completed"
    assert body["outcome"] == "done"
    assert body["results"] == fake_results


def test_submit_rejects_blank_input(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.post("/api/task", json={"input": "   "})
    assert r.status_code == 400


def test_unknown_task_id_returns_404(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/task/does-not-exist")
    assert r.status_code == 404


def test_memory_search_requires_api_key(monkeypatch):
    server = _fresh_server(monkeypatch, api_key="secret")
    client = TestClient(server.app)
    r = client.get("/api/memory/search", params={"q": "날씨"})
    assert r.status_code == 401


def test_memory_search_returns_matching_records(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.json"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.memory as memory
    importlib.reload(memory)
    memory.save("날씨 알려줘", [{"action": "llm", "status": "ok"}])
    memory.save("뉴스 검색해줘", [{"action": "browser_search", "status": "ok"}])

    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/memory/search", params={"q": "날씨"})
    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert body[0]["task"] == "날씨 알려줘"


def test_metrics_reports_queue_depth_and_task_counts(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.json"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.memory as memory
    importlib.reload(memory)
    memory.save("작업", [{"action": "llm", "status": "ok"}])

    server = _fresh_server(monkeypatch, api_key="")
    client = TestClient(server.app)
    r = client.get("/api/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["total_tasks_recorded"] == 1
    assert body["by_status"] == {"done": 1}
    assert "queue_depth" in body
    assert "uptime_seconds" in body

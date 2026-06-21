"""기능: 재개 가능한 큐 — TASK_QUEUE_RESUME=true면 재시작 시 중단된 task를 재실행한다.

기본(false)은 종전대로 중단 task를 failed로 정리한다. 외부 의존 없이 임시 DB + 모킹으로 돈다.
"""
import importlib
import time
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_STORE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.task_store as task_store
    importlib.reload(task_store)
    import core.memory as memory
    importlib.reload(memory)


def _interrupted(task_id, status="processing"):
    return {"task_id": task_id, "input": "재개될 작업", "status": status,
            "outcome": None, "results": None, "error": "중단됨",
            "created_at": "2026-06-20T00:00:00+00:00"}


def _wait_terminal(tq, task_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = tq.get(task_id)
        if task and task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.02)
    raise AssertionError("작업이 제한 시간 내에 종료 상태에 도달하지 못함")


def test_requeue_interrupted_resets_only_unresolved():
    import core.task_store as task_store
    for tid, st in [("a", "queued"), ("b", "processing"), ("c", "completed"), ("d", "failed")]:
        task_store.upsert(_interrupted(tid, st))

    ids = task_store.requeue_interrupted()

    assert set(ids) == {"a", "b"}
    by = {t["task_id"]: t for t in task_store.load_all()}
    assert by["a"]["status"] == "queued"
    assert by["a"]["error"] is None        # error는 초기화돼야 한다
    assert by["b"]["status"] == "queued"
    assert by["c"]["status"] == "completed"  # 종료 상태는 건드리지 않음
    assert by["d"]["status"] == "failed"


def test_resume_enabled_reexecutes_interrupted_task(monkeypatch):
    import core.task_queue as tqmod
    import core.task_store as task_store
    task_store.upsert(_interrupted("t1"))
    monkeypatch.setattr(tqmod, "TASK_QUEUE_RESUME", True)

    with patch("core.task_queue.plan", return_value=[{"action": "llm", "input": "x"}]), \
         patch("core.task_queue.execute_steps",
               return_value=[{"action": "llm", "status": "ok", "result": "done"}]), \
         patch("core.task_queue.memory.save", return_value={"status": "done"}), \
         patch("core.task_queue.memory.get_context", return_value=""), \
         patch("core.task_queue.memory.successful_examples", return_value=[]):
        tq = tqmod.TaskQueue()
        task = _wait_terminal(tq, "t1")

    assert task["status"] == "completed"      # 재실행되어 완료됨


def test_resume_disabled_marks_interrupted_failed(monkeypatch):
    import core.task_queue as tqmod
    import core.task_store as task_store
    task_store.upsert(_interrupted("t1"))
    monkeypatch.setattr(tqmod, "TASK_QUEUE_RESUME", False)

    with patch("core.task_queue.plan", return_value=[]), \
         patch("core.task_queue.execute_steps", return_value=[]), \
         patch("core.task_queue.memory.save", return_value={"status": "done"}), \
         patch("core.task_queue.memory.get_context", return_value=""), \
         patch("core.task_queue.memory.successful_examples", return_value=[]):
        tq = tqmod.TaskQueue()
        task = tq.get("t1")

    assert task["status"] == "failed"         # 재실행 없이 failed로 정리

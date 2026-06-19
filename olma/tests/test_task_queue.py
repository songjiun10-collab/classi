import importlib
import time
from unittest.mock import patch

import pytest

from core.task_queue import TaskQueue


@pytest.fixture(autouse=True)
def _isolated_storage(tmp_path, monkeypatch):
    """모든 테스트가 실제 storage/tasks.db·memory.db 대신 테스트별 임시 DB를 쓰게 한다.

    core.task_queue는 task_store/memory를 모듈 객체로 그대로 참조하므로(`from core
    import memory, task_store`), 이 모듈들을 reload하면 importlib.reload가 같은
    모듈 객체를 제자리에서 갱신하기 때문에 core.task_queue.task_store/.memory도
    자동으로 새 경로를 보게 된다.

    일부 테스트(예: 큐 깊이를 확인하느라 작업 완료를 기다리지 않는 테스트)는
    백그라운드 워커가 끝나기 전에 `with patch(...)` 블록을 빠져나가, 패치가
    워커보다 먼저 풀려 워커가 실제 core.memory를 호출하는 경쟁 상태가 생길 수
    있다. autouse 픽스처는 테스트가 끝날 때까지 경로를 되돌리지 않으므로 patch
    타이밍과 무관하게 항상 임시 경로를 쓴다."""
    monkeypatch.setenv("TASK_STORE_PATH", str(tmp_path / "tasks.db"))
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.task_store as task_store
    importlib.reload(task_store)
    import core.memory as memory
    importlib.reload(memory)


def _wait_until_terminal(tq, task_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = tq.get(task_id)
        if task["status"] in ("completed", "failed"):
            return task
        time.sleep(0.02)
    raise AssertionError("작업이 제한 시간 내에 종료 상태에 도달하지 못함")


def test_submit_returns_unique_task_ids():
    with patch("core.task_queue.plan", return_value=[]), patch(
        "core.task_queue.execute_steps", return_value=[]
    ), patch("core.task_queue.memory.save", return_value={"status": "done"}), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        tq = TaskQueue()
        id1 = tq.submit("작업1")
        id2 = tq.submit("작업2")
        assert id1 != id2


def test_submitted_task_completes_with_results():
    fake_results = [{"action": "llm", "status": "ok", "result": "답변"}]
    with patch("core.task_queue.plan", return_value=[{"action": "llm", "input": "hi"}]), patch(
        "core.task_queue.execute_steps", return_value=fake_results
    ), patch(
        "core.task_queue.memory.save", return_value={"status": "done"}
    ), patch("core.task_queue.memory.get_context", return_value=""):
        tq = TaskQueue()
        task_id = tq.submit("작업")
        task = _wait_until_terminal(tq, task_id)

    assert task["status"] == "completed"
    assert task["outcome"] == "done"
    assert task["results"] == fake_results


def test_task_marked_failed_when_planning_raises():
    with patch("core.task_queue.plan", side_effect=RuntimeError("계획 실패")), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        tq = TaskQueue()
        task_id = tq.submit("작업")
        task = _wait_until_terminal(tq, task_id)

    assert task["status"] == "failed"
    assert "계획 실패" in task["error"]


def test_get_returns_none_for_unknown_task_id():
    with patch("core.task_queue.plan", return_value=[]), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        tq = TaskQueue()
        assert tq.get("존재하지않는id") is None


def test_list_recent_returns_newest_first():
    with patch("core.task_queue.plan", return_value=[]), patch(
        "core.task_queue.execute_steps", return_value=[]
    ), patch("core.task_queue.memory.save", return_value={"status": "done"}), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        tq = TaskQueue()
        id1 = tq.submit("첫번째")
        _wait_until_terminal(tq, id1)
        id2 = tq.submit("두번째")
        _wait_until_terminal(tq, id2)

        recent = tq.list_recent(10)

    assert [t["task_id"] for t in recent][:2] == [id2, id1]


def test_evict_old_tasks_drops_oldest_terminal_task_beyond_max():
    tq = TaskQueue.__new__(TaskQueue)
    tq._tasks = {}
    tq._order = []
    for i in range(5):
        tid = f"t{i}"
        tq._tasks[tid] = {"task_id": tid, "status": "completed"}
        tq._order.append(tid)

    with patch("core.task_queue.TASK_QUEUE_MAX_TASKS", 3):
        tq._evict_old_tasks()

    assert len(tq._order) == 3
    assert tq._order == ["t2", "t3", "t4"]


def test_evict_old_tasks_preserves_in_progress_tasks():
    tq = TaskQueue.__new__(TaskQueue)
    tq._tasks = {
        "a": {"task_id": "a", "status": "processing"},
        "b": {"task_id": "b", "status": "completed"},
        "c": {"task_id": "c", "status": "queued"},
    }
    tq._order = ["a", "b", "c"]

    with patch("core.task_queue.TASK_QUEUE_MAX_TASKS", 2):
        tq._evict_old_tasks()

    assert "b" not in tq._order
    assert set(tq._order) == {"a", "c"}


def test_completed_task_is_persisted_to_store():
    import core.task_store as task_store

    fake_results = [{"action": "llm", "status": "ok", "result": "답변"}]
    with patch("core.task_queue.plan", return_value=[{"action": "llm", "input": "hi"}]), patch(
        "core.task_queue.execute_steps", return_value=fake_results
    ), patch(
        "core.task_queue.memory.save", return_value={"status": "done"}
    ), patch("core.task_queue.memory.get_context", return_value=""):
        tq = TaskQueue()
        task_id = tq.submit("작업")
        _wait_until_terminal(tq, task_id)

    stored = {t["task_id"]: t for t in task_store.load_all()}
    assert stored[task_id]["status"] == "completed"
    assert stored[task_id]["results"] == fake_results


def test_new_taskqueue_restores_history_from_store():
    fake_results = [{"action": "llm", "status": "ok", "result": "답변"}]
    with patch("core.task_queue.plan", return_value=[{"action": "llm", "input": "hi"}]), patch(
        "core.task_queue.execute_steps", return_value=fake_results
    ), patch(
        "core.task_queue.memory.save", return_value={"status": "done"}
    ), patch("core.task_queue.memory.get_context", return_value=""):
        tq1 = TaskQueue()
        task_id = tq1.submit("이전 작업")
        _wait_until_terminal(tq1, task_id)

        tq2 = TaskQueue()

    restored = tq2.get(task_id)
    assert restored is not None
    assert restored["status"] == "completed"
    assert restored["results"] == fake_results


def test_new_taskqueue_marks_previously_interrupted_task_as_failed():
    import core.task_store as task_store

    task_store.upsert(
        {
            "task_id": "interrupted-1",
            "input": "중단된 작업",
            "status": "processing",
            "outcome": None,
            "results": None,
            "error": None,
            "created_at": "2026-01-01T00:00:00+00:00",
        }
    )

    with patch("core.task_queue.plan", return_value=[]), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        tq = TaskQueue()

    restored = tq.get("interrupted-1")
    assert restored["status"] == "failed"
    assert restored["error"]


def test_evict_old_tasks_also_removes_from_store():
    import core.task_store as task_store

    tq = TaskQueue.__new__(TaskQueue)
    tq._tasks = {}
    tq._order = []
    for i in range(5):
        tid = f"t{i}"
        task = {
            "task_id": tid,
            "input": "x",
            "status": "completed",
            "outcome": "done",
            "results": None,
            "error": None,
            "created_at": f"2026-01-01T00:00:0{i}+00:00",
        }
        tq._tasks[tid] = task
        tq._order.append(tid)
        task_store.upsert(task)

    with patch("core.task_queue.TASK_QUEUE_MAX_TASKS", 3):
        tq._evict_old_tasks()

    remaining_ids = {t["task_id"] for t in task_store.load_all()}
    assert remaining_ids == {"t2", "t3", "t4"}


def test_depth_reflects_queued_items_not_yet_processed():
    with patch("core.task_queue.plan", side_effect=lambda *a, **k: time.sleep(0.2) or []), patch(
        "core.task_queue.execute_steps", return_value=[]
    ), patch("core.task_queue.memory.save", return_value={"status": "done"}), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        tq = TaskQueue()
        tq.submit("작업1")
        tq.submit("작업2")
        tq.submit("작업3")
        time.sleep(0.05)
        assert tq.depth() >= 1

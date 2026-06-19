import time
from unittest.mock import patch

from core.task_queue import TaskQueue


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

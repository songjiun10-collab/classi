import importlib
import threading
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
    # task_queue는 이제 Memory Profile(profile)을 거치며 ltm/accounts/usage 스토어를
    # 건드리므로, 실제 storage/ 오염을 막기 위해 함께 임시 경로로 격리한다.
    monkeypatch.setenv("LTM_PATH", str(tmp_path / "ltm.db"))
    monkeypatch.setenv("ACCOUNT_STORE_PATH", str(tmp_path / "accounts.db"))
    monkeypatch.setenv("USAGE_STORE_PATH", str(tmp_path / "usage.db"))
    import config.config as cfg
    importlib.reload(cfg)
    for mod in ("core.task_store", "core.memory", "core.ltm_store", "core.long_term_memory",
                "core.account_store", "core.accounts", "core.usage_store", "core.usage",
                "core.profile"):
        importlib.reload(importlib.import_module(mod))
    # 이 모듈의 테스트들은 '계획 경로'(plan 호출·실패·복구·큐 깊이)를 검증하므로 간단질문
    # fast-path를 끈다(켜져 있으면 단순 입력이 plan을 건너뛴다). fast-chat 동작은 아래
    # test_fast_chat_* 와 test_triage.py에서 따로 검증한다.
    import core.task_queue as task_queue
    monkeypatch.setattr(task_queue, "FAST_CHAT", False)


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


def test_failed_task_auto_recovers_once_then_fails():
    """TASK_AUTO_RECOVERY가 켜지면 실패 작업을 1회 재큐잉 후 최종 실패한다."""
    calls = {"n": 0}

    def boom(*a, **k):
        calls["n"] += 1
        raise RuntimeError("plan down")

    with patch("core.task_queue.TASK_AUTO_RECOVERY", True), \
         patch("core.task_queue.TASK_RECOVERY_MAX", 1), \
         patch("core.task_queue.plan", side_effect=boom), \
         patch("core.task_queue.profile.planner_context", return_value=("", "")), \
         patch("core.task_queue.memory.successful_examples", return_value=[]):
        tq = TaskQueue()
        tid = tq.submit("작업")
        task = _wait_until_terminal(tq, tid)

    assert task["status"] == "failed"
    assert task["recovery_count"] == 1
    assert calls["n"] == 2          # 최초 + 복구 1회


def test_failed_task_no_recovery_when_disabled():
    """기본(복구 off)에선 실패 작업이 재큐잉 없이 곧장 failed."""
    with patch("core.task_queue.plan", side_effect=RuntimeError("x")), \
         patch("core.task_queue.profile.planner_context", return_value=("", "")), \
         patch("core.task_queue.memory.successful_examples", return_value=[]):
        tq = TaskQueue()
        tid = tq.submit("작업")
        task = _wait_until_terminal(tq, tid)

    assert task["status"] == "failed"
    assert task.get("recovery_count", 0) == 0


def test_submit_orders_by_priority_then_fifo():
    """우선순위 큐: priority가 낮을수록 먼저, 같은 priority면 제출 순서(FIFO)."""
    import queue as _q

    tq = TaskQueue.__new__(TaskQueue)   # 워커 없이 큐 동작만 검증
    tq._tasks, tq._order, tq._seq = {}, [], 0
    tq._lock = threading.Lock()
    tq._queue = _q.PriorityQueue()

    a = tq.submit("보통", priority=0)
    b = tq.submit("급함", priority=-5)
    c = tq.submit("나중", priority=10)
    d = tq.submit("보통2", priority=0)

    drained = []
    while not tq._queue.empty():
        drained.append(tq._queue.get_nowait()[2])
    assert drained == [b, a, d, c]           # 급함 → 보통(제출순 a,d) → 나중
    assert tq.get(a)["priority"] == 0
    assert tq.get(b)["priority"] == -5


def test_submit_defaults_priority_zero():
    import queue as _q

    tq = TaskQueue.__new__(TaskQueue)
    tq._tasks, tq._order, tq._seq = {}, [], 0
    tq._lock = threading.Lock()
    tq._queue = _q.PriorityQueue()
    tid = tq.submit("기본")
    assert tq.get(tid)["priority"] == 0


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


def test_taskqueue_runs_multiple_workers_concurrently_when_configured():
    slots_used = []
    lock = threading.Lock()

    def slow_execute(steps, profile_slot=0):
        with lock:
            slots_used.append(profile_slot)
        time.sleep(0.2)
        return []

    with patch("core.task_queue.TASK_QUEUE_WORKERS", 3), patch(
        "core.task_queue.plan", return_value=[{"action": "llm", "input": "hi"}]
    ), patch("core.task_queue.execute_steps", side_effect=slow_execute), patch(
        "core.task_queue.memory.save", return_value={"status": "done"}
    ), patch("core.task_queue.memory.get_context", return_value=""):
        tq = TaskQueue()
        start = time.monotonic()
        ids = [tq.submit(f"작업{i}") for i in range(3)]
        for tid in ids:
            _wait_until_terminal(tq, tid)
        elapsed = time.monotonic() - start

    # 3개 작업이 각각 0.2초 걸리지만 워커 3개가 동시에 처리하므로
    # 완전 직렬(0.6초)보다 훨씬 짧게 끝나야 한다.
    assert elapsed < 0.5
    assert sorted(slots_used) == [0, 1, 2]


def test_taskqueue_default_worker_count_is_serial():
    with patch("core.task_queue.plan", return_value=[]), patch(
        "core.task_queue.execute_steps", return_value=[]
    ), patch("core.task_queue.memory.save", return_value={"status": "done"}), patch(
        "core.task_queue.memory.get_context", return_value=""
    ):
        tq = TaskQueue()
        assert len(tq._workers) == 1


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


def test_fast_chat_skips_planning_for_simple_question(monkeypatch):
    """간단한 질문은 planner를 부르지 않고 단일 LLM step으로 바로 실행한다."""
    import core.task_queue as task_queue
    monkeypatch.setattr(task_queue, "FAST_CHAT", True)
    captured = {}

    def fake_execute(steps, **kwargs):
        captured["steps"] = steps
        return [{"action": "llm", "status": "ok", "result": "안녕하세요"}]

    with patch("core.task_queue.plan", side_effect=AssertionError("plan을 부르면 안 된다")), \
         patch("core.task_queue.execute_steps", side_effect=fake_execute), \
         patch("core.task_queue.memory.save", return_value={"status": "done"}), \
         patch("core.task_queue.memory.get_context", return_value=""):
        tq = TaskQueue()
        task_id = tq.submit("안녕")
        task = _wait_until_terminal(tq, task_id)

    assert task["status"] == "completed"
    assert captured["steps"] == [{"action": "llm", "input": "안녕", "depends_on": None}]


def test_fast_chat_off_uses_planner_for_simple_question(monkeypatch):
    """fast-chat을 끄면 간단한 질문도 planner를 거친다(토글 동작 확인)."""
    import core.task_queue as task_queue
    monkeypatch.setattr(task_queue, "FAST_CHAT", False)
    called = {"plan": False}

    def fake_plan(*a, **k):
        called["plan"] = True
        return [{"action": "llm", "input": "안녕", "depends_on": None}]

    with patch("core.task_queue.plan", side_effect=fake_plan), \
         patch("core.task_queue.execute_steps", return_value=[{"action": "llm", "status": "ok"}]), \
         patch("core.task_queue.memory.save", return_value={"status": "done"}), \
         patch("core.task_queue.memory.get_context", return_value=""):
        tq = TaskQueue()
        task_id = tq.submit("안녕")
        _wait_until_terminal(tq, task_id)

    assert called["plan"] is True

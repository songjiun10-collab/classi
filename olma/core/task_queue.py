"""HTTP로 들어오는 요청을 처리하는 인프로세스 작업 큐.

Playwright는 영구 프로필(로그인 세션)을 쓰는 브라우저 컨텍스트를 전제하므로, 같은
프로필 디렉터리를 두 코드가 동시에 열면 충돌한다. 그래서 Redis/Celery 같은 분산 큐
대신, 기본은 워커 스레드 1개가 큐를 순서대로 비우는 가장 단순한 형태로 직렬성을
보장한다(main.py REPL의 plan→execute→save 흐름과 동일). `TASK_QUEUE_WORKERS`를
1보다 크게 설정하면 워커마다 독립된 브라우저 프로필(tools/browser.resolve_profile_dir)을
써서 충돌 없이 병렬로 처리한다."""
import queue
import threading
import uuid
from datetime import datetime, timezone

from config.config import TASK_QUEUE_MAX_TASKS, TASK_QUEUE_WORKERS
from core import memory, task_store
from core.logger import get_logger
from core.planner import plan
from executor.executor import execute_steps

log = get_logger("task_queue")

_TERMINAL_STATUSES = ("completed", "failed")


class TaskQueue:
    def __init__(self):
        self._tasks: dict = {}
        self._order: list = []
        self._queue: "queue.Queue[str]" = queue.Queue()
        self._lock = threading.Lock()
        self._restore_from_store()
        self._workers = [
            threading.Thread(target=self._run_worker, args=(i,), daemon=True)
            for i in range(TASK_QUEUE_WORKERS)
        ]
        for worker in self._workers:
            worker.start()

    def _restore_from_store(self):
        """재시작 시 히스토리를 복원한다. 재개가 아니라 기록 보존이 목적이므로,
        직전에 처리 중이던 task는 안전하게 이어서 실행할 수 없어 failed로 정리한다."""
        try:
            task_store.mark_interrupted_as_failed()
            for task in task_store.load_all():
                self._tasks[task["task_id"]] = task
                self._order.append(task["task_id"])
        except Exception as exc:
            log.error("task store 복원 실패, 빈 상태로 시작: %s", exc)

    def submit(self, user_input: str) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            task = {
                "task_id": task_id,
                "input": user_input,
                "status": "queued",
                "outcome": None,
                "results": None,
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._tasks[task_id] = task
            self._order.append(task_id)
            self._evict_old_tasks()
        self._safe_upsert(task)
        self._queue.put(task_id)
        return task_id

    def _safe_upsert(self, task: dict):
        try:
            task_store.upsert(task)
        except Exception as exc:
            log.error("task store 저장 실패(%s): %s", task.get("task_id"), exc)

    def get(self, task_id: str) -> dict | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return dict(task) if task is not None else None

    def list_recent(self, n: int = 20) -> list:
        with self._lock:
            ids = self._order[-n:][::-1]
            return [dict(self._tasks[tid]) for tid in ids if tid in self._tasks]

    def depth(self) -> int:
        return self._queue.qsize()

    def _evict_old_tasks(self):
        # 락은 호출자(submit)가 이미 들고 있다는 전제.
        while len(self._order) > TASK_QUEUE_MAX_TASKS:
            for idx, tid in enumerate(self._order):
                task = self._tasks.get(tid)
                if task is not None and task["status"] in _TERMINAL_STATUSES:
                    del self._order[idx]
                    self._tasks.pop(tid, None)
                    try:
                        task_store.delete(tid)
                    except Exception as exc:
                        log.error("task store에서 삭제 실패(%s): %s", tid, exc)
                    break
            else:
                break  # 제거 가능한 종료 상태 task가 없음(전부 처리 중)

    def _run_worker(self, worker_index: int = 0):
        while True:
            task_id = self._queue.get()
            try:
                self._process(task_id, worker_index)
            except Exception as exc:
                log.error("작업 처리 중 예상치 못한 예외: %s (%s)", task_id, exc)
                with self._lock:
                    if task_id in self._tasks:
                        self._tasks[task_id]["status"] = "failed"
                        self._tasks[task_id]["error"] = str(exc)
                        self._safe_upsert(self._tasks[task_id])

    def _process(self, task_id: str, worker_index: int = 0):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task["status"] = "processing"
            user_input = task["input"]
            self._safe_upsert(task)

        try:
            context = memory.get_context()
            steps = plan(user_input, context=context)
        except Exception as exc:
            log.error("계획 생성 실패: %s (%s)", task_id, exc)
            with self._lock:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = str(exc)
                self._safe_upsert(self._tasks[task_id])
            return

        results = execute_steps(steps, profile_slot=worker_index)
        record = memory.save(user_input, results)
        with self._lock:
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["outcome"] = record["status"]
            self._tasks[task_id]["results"] = results
            self._safe_upsert(self._tasks[task_id])

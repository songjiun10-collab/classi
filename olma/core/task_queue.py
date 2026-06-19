"""HTTP로 들어오는 요청을 직렬로 처리하는 인프로세스 작업 큐.

Playwright는 영구 프로필(로그인 세션)을 쓰는 단일 브라우저 컨텍스트를 전제하므로,
여러 작업을 동시에 실행하면 같은 브라우저를 두 코드가 동시에 조작해 세션이 깨진다.
그래서 Redis/Celery 같은 분산 큐 대신, 워커 스레드 1개가 큐를 순서대로 비우는
가장 단순한 형태로 직렬성을 보장한다(main.py REPL의 plan→execute→save 흐름과 동일)."""
import queue
import threading
import uuid
from datetime import datetime, timezone

from config.config import TASK_QUEUE_MAX_TASKS
from core import memory
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
        self._worker = threading.Thread(target=self._run_worker, daemon=True)
        self._worker.start()

    def submit(self, user_input: str) -> str:
        task_id = uuid.uuid4().hex
        with self._lock:
            self._tasks[task_id] = {
                "task_id": task_id,
                "input": user_input,
                "status": "queued",
                "outcome": None,
                "results": None,
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._order.append(task_id)
            self._evict_old_tasks()
        self._queue.put(task_id)
        return task_id

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
                    break
            else:
                break  # 제거 가능한 종료 상태 task가 없음(전부 처리 중)

    def _run_worker(self):
        while True:
            task_id = self._queue.get()
            try:
                self._process(task_id)
            except Exception as exc:
                log.error("작업 처리 중 예상치 못한 예외: %s (%s)", task_id, exc)
                with self._lock:
                    if task_id in self._tasks:
                        self._tasks[task_id]["status"] = "failed"
                        self._tasks[task_id]["error"] = str(exc)

    def _process(self, task_id: str):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task["status"] = "processing"
            user_input = task["input"]

        try:
            context = memory.get_context()
            steps = plan(user_input, context=context)
        except Exception as exc:
            log.error("계획 생성 실패: %s (%s)", task_id, exc)
            with self._lock:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = str(exc)
            return

        results = execute_steps(steps)
        record = memory.save(user_input, results)
        with self._lock:
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["outcome"] = record["status"]
            self._tasks[task_id]["results"] = results

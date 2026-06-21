"""HTTP로 들어오는 요청을 처리하는 인프로세스 작업 큐.

Playwright는 영구 프로필(로그인 세션)을 쓰는 브라우저 컨텍스트를 전제하므로, 같은
프로필 디렉터리를 두 코드가 동시에 열면 충돌한다. 그래서 Redis/Celery 같은 분산 큐
대신, 기본은 워커 스레드 1개가 큐를 순서대로 비우는 가장 단순한 형태로 직렬성을
보장한다(main.py REPL의 plan→execute→save 흐름과 동일). `TASK_QUEUE_WORKERS`를
1보다 크게 설정하면 워커마다 독립된 브라우저 프로필(tools/browser.resolve_profile_dir)을
써서 충돌 없이 병렬로 처리한다."""
from __future__ import annotations

import queue
import threading
import uuid
from datetime import datetime, timezone

from config.config import (
    FAST_CHAT,
    TASK_AUTO_RECOVERY,
    TASK_QUEUE_MAX_TASKS,
    TASK_QUEUE_RESUME,
    TASK_QUEUE_WORKERS,
    TASK_RECOVERY_MAX,
    WORKFLOW_AUTO_REUSE,
)
from core import accounts, memory, profile, task_store, triage, workflow_planner, workflow_store
from core.logger import get_logger
from core.planner import plan
from executor.executor import execute_steps

log = get_logger("task_queue")

_TERMINAL_STATUSES = ("completed", "failed")


class TaskQueue:
    def __init__(self):
        self._tasks: dict = {}
        self._order: list = []
        # 우선순위 큐: (priority, seq, task_id). priority가 낮을수록 먼저 처리되고, 같은
        # priority면 seq(제출 순)로 FIFO 동작한다 — 즉 priority 미지정(0)이면 기존 FIFO와 동일.
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._seq = 0
        self._lock = threading.Lock()
        self._restore_from_store()
        self._workers = [
            threading.Thread(target=self._run_worker, args=(i,), daemon=True)
            for i in range(TASK_QUEUE_WORKERS)
        ]
        for worker in self._workers:
            worker.start()

    def _restore_from_store(self):
        """재시작 시 히스토리를 복원한다. 기본은 중단된 task를 failed로 정리하지만,
        TASK_QUEUE_RESUME=true면 중단된 task를 다시 큐에 넣어 처음부터 재실행한다(재시도형 재개)."""
        try:
            requeue_ids = task_store.requeue_interrupted() if TASK_QUEUE_RESUME else []
            if not TASK_QUEUE_RESUME:
                task_store.mark_interrupted_as_failed()
            for task in task_store.load_all():
                self._tasks[task["task_id"]] = task
                self._order.append(task["task_id"])
            for task_id in requeue_ids:
                self._seq += 1
                priority = self._tasks.get(task_id, {}).get("priority", 0)
                self._queue.put((priority, self._seq, task_id))
                log.info("중단된 작업 재개(재실행) 큐에 등록: %s", task_id)
        except Exception as exc:
            log.error("task store 복원 실패, 빈 상태로 시작: %s", exc)

    def submit(self, user_input: str, priority: int = 0) -> str:
        """작업을 큐에 넣는다. priority가 낮을수록 먼저 처리된다(기본 0; 급한 건 음수).
        같은 priority면 제출 순서(FIFO)를 따른다."""
        task_id = uuid.uuid4().hex
        with self._lock:
            self._seq += 1
            seq = self._seq
            task = {
                "task_id": task_id,
                "input": user_input,
                "status": "queued",
                "priority": priority,
                "outcome": None,
                "results": None,
                "error": None,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            self._tasks[task_id] = task
            self._order.append(task_id)
            self._evict_old_tasks()
        self._safe_upsert(task)
        self._queue.put((priority, seq, task_id))
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

    def _maybe_recover(self, task_id: str) -> bool:
        """실패한 작업을 자동 복구(재큐잉)한다. TASK_AUTO_RECOVERY가 켜져 있고 재시도 횟수가
        TASK_RECOVERY_MAX 미만일 때만, 같은 task를 한 단계 낮은 우선순위로 다시 큐에 넣는다.
        재큐잉했으면 True. 무한 루프 방지를 위해 recovery_count로 횟수를 센다."""
        if not TASK_AUTO_RECOVERY:
            return False
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return False
            count = task.get("recovery_count", 0)
            if count >= TASK_RECOVERY_MAX:
                return False
            task["recovery_count"] = count + 1
            task["status"] = "queued"
            self._seq += 1
            seq = self._seq
            priority = task.get("priority", 0) + 1   # 복구 재시도는 신규 작업보다 뒤로
            self._safe_upsert(task)
        self._queue.put((priority, seq, task_id))
        log.info("작업 자동 복구 재큐잉: %s (시도 %d/%d)", task_id, count + 1, TASK_RECOVERY_MAX)
        return True

    def _fail(self, task_id: str, error: str) -> None:
        """작업을 실패로 기록하되, 가능하면 먼저 자동 복구를 시도한다."""
        if self._maybe_recover(task_id):
            return
        with self._lock:
            if task_id in self._tasks:
                self._tasks[task_id]["status"] = "failed"
                self._tasks[task_id]["error"] = error
                self._safe_upsert(self._tasks[task_id])

    def _run_worker(self, worker_index: int = 0):
        while True:
            _priority, _seq, task_id = self._queue.get()
            try:
                self._process(task_id, worker_index)
            except Exception as exc:
                log.error("작업 처리 중 예상치 못한 예외: %s (%s)", task_id, exc)
                self._fail(task_id, str(exc))

    def _process(self, task_id: str, worker_index: int = 0):
        with self._lock:
            task = self._tasks.get(task_id)
            if task is None:
                return
            task["status"] = "processing"
            user_input = task["input"]
            self._safe_upsert(task)

        try:
            # WORKFLOW_AUTO_REUSE가 켜져 있고 요청과 충분히 맞는 '파라미터 없는' 저장
            # 템플릿이 있으면 다시 계획하지 않고 그 워크플로우를 재사용한다(없거나 꺼져
            # 있으면 평소대로 동적 생성). 실행/기억 경로는 아래에서 공통으로 처리한다.
            reuse = workflow_planner.select(user_input) if WORKFLOW_AUTO_REUSE else None
            if FAST_CHAT and not triage.needs_planning(user_input):
                # 간단한 질문은 계획을 건너뛰고 단일 LLM 답변으로 바로 처리한다.
                log.info("작업 %s: 간단한 질문 — 계획 생략", task_id)
                steps = triage.simple_steps(user_input)
            elif reuse and reuse["mode"] == "template":
                log.info("작업 %s: 저장 템플릿 재사용 '%s'", task_id, reuse["name"])
                steps = workflow_store.get(reuse["name"])["steps"]
            else:
                # Memory Profile: 계정의 누적 행동(최근 맥락+자주 쓰는 작업)과 선호/장기기억을
                # 한 번에 모아 플래너에 공급한다(파이프라인 account→profile→planner).
                context, facts = profile.planner_context(accounts.current_id(), user_input)
                examples = memory.successful_examples(user_input)
                steps = plan(user_input, context=context, examples=examples, facts=facts)
        except Exception as exc:
            log.error("계획 생성 실패: %s (%s)", task_id, exc)
            self._fail(task_id, str(exc))
            return

        results = execute_steps(steps, profile_slot=worker_index)
        record = memory.save(user_input, results)
        with self._lock:
            self._tasks[task_id]["status"] = "completed"
            self._tasks[task_id]["outcome"] = record["status"]
            self._tasks[task_id]["results"] = results
            self._safe_upsert(self._tasks[task_id])

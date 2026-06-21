"""Scheduler — 워크플로우를 주기적으로 실행한다(인프로세스).

Dynamic Workflow Engine 위에 "시간"을 얹는 레이어. due(만기) 스케줄을 찾아 runner로
실행하고 next_run을 interval만큼 미룬다. 핵심 로직(due/tick)은 시계(now_fn)와 실행기
(runner)를 주입받아 스레드·sleep 없이 결정론적으로 테스트된다 — 백그라운드 스레드는
그 위의 얇은 폴링 루프일 뿐이다.

기본 runner(default_runner)는 workflow 엔진을 직접 호출하되, API 작업 큐 워커(slot 0)와
다른 브라우저 프로필 슬롯(SCHEDULER_PROFILE_SLOT)을 써서 영구 프로필 충돌을 피한다."""
from __future__ import annotations

import threading
import time
import uuid

from config.config import SCHEDULER_POLL_SECONDS, SCHEDULER_PROFILE_SLOT
from core import scheduler_store, workflow
from core.logger import get_logger

log = get_logger("scheduler")


def default_runner(schedule: dict) -> str:
    """스케줄을 실제 워크플로우 실행으로 옮긴다. 반환: 실행 상태 문자열(done/partial/failed)."""
    kind = schedule["kind"]
    payload = schedule["payload"] or {}
    if kind == "dynamic":
        summary = workflow.run_dynamic(payload["request"], profile_slot=SCHEDULER_PROFILE_SLOT)
    elif kind == "template":
        summary = workflow.run_template(
            payload["name"], params=payload.get("params"), profile_slot=SCHEDULER_PROFILE_SLOT
        )
    else:
        raise ValueError(f"알 수 없는 스케줄 kind: {kind!r}")
    return summary["status"]


class Scheduler:
    def __init__(self, runner=default_runner, now_fn=time.time):
        self._runner = runner
        self._now = now_fn
        self._thread = None
        self._stop = threading.Event()

    def add(self, kind: str, payload: dict, interval_seconds: int,
            first_run_delay: float = 0.0, enabled: bool = True) -> str:
        """스케줄을 등록하고 id를 돌려준다. first_run_delay 뒤 첫 실행, 이후 interval마다 반복."""
        if kind not in ("dynamic", "template"):
            raise ValueError(f"지원하지 않는 kind: {kind!r} (dynamic|template)")
        if interval_seconds <= 0:
            raise ValueError("interval_seconds는 양수여야 합니다")
        sid = uuid.uuid4().hex
        scheduler_store.upsert({
            "id": sid, "kind": kind, "payload": payload,
            "interval_seconds": interval_seconds,
            "next_run": self._now() + first_run_delay,
            "enabled": enabled, "last_run": None, "last_status": None,
        })
        log.info("스케줄 등록: %s (%s, %ds마다)", sid, kind, interval_seconds)
        return sid

    def list(self) -> list:
        return scheduler_store.load_all()

    def remove(self, sid: str) -> bool:
        return scheduler_store.delete(sid)

    def set_enabled(self, sid: str, enabled: bool) -> bool:
        s = scheduler_store.get(sid)
        if s is None:
            return False
        s["enabled"] = enabled
        scheduler_store.upsert(s)
        return True

    def due(self, now: float | None = None) -> list:
        """지금(now) 기준 실행해야 할(활성 + next_run<=now) 스케줄들."""
        now = self._now() if now is None else now
        return [s for s in scheduler_store.load_all() if s["enabled"] and s["next_run"] <= now]

    def tick(self, now: float | None = None) -> list:
        """due 스케줄을 실행하고 next_run을 interval만큼 미룬다. 실행된 id 리스트 반환.

        runner가 예외를 던져도 그 스케줄만 last_status에 에러를 남기고 재예약한다 —
        한 스케줄의 실패가 루프 전체나 다른 스케줄을 막지 않는다."""
        now = self._now() if now is None else now
        ran = []
        for s in self.due(now):
            try:
                status = self._runner(s)
            except Exception as exc:
                log.error("스케줄 실행 실패: %s (%s)", s["id"], exc)
                status = f"error: {exc}"
            s["last_run"] = now
            s["last_status"] = status
            # 누적 드리프트를 막기 위해 now 기준으로 다음 실행을 잡는다(만기 시점이 아니라).
            s["next_run"] = now + s["interval_seconds"]
            scheduler_store.upsert(s)
            ran.append(s["id"])
        return ran

    def _loop(self, poll_seconds: float) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception as exc:
                log.error("스케줄러 tick 중 예외: %s", exc)
            self._stop.wait(poll_seconds)

    def start(self, poll_seconds: int = SCHEDULER_POLL_SECONDS) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(poll_seconds,), daemon=True)
        self._thread.start()
        log.info("스케줄러 시작(폴링 %ds)", poll_seconds)

    def stop(self) -> None:
        self._stop.set()

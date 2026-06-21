"""Event Trigger — "사건"이 일어나면 워크플로우를 실행한다(인프로세스).

Scheduler가 시간(주기)에 반응한다면 이쪽은 상태 변화에 반응한다: 파일이 생기거나
바뀌면 워크플로우를 깨운다. 핵심은 **엣지 트리거**다 — 조건이 "참인 동안" 계속 쏘는
게 아니라, 거짓→참(혹은 값의 변화)으로 넘어가는 *그 순간*에만 한 번 발화한다.
그래서 각 트리거는 마지막 관측 상태(state)를 저장하고, 폴링 때 새 상태와 비교한다.

감지기(checker)는 source 이름으로 CHECKERS에 등록한다. 시그니처:
    checker(config: dict, prev_state) -> (fired: bool, new_state)
사이트 URL·셀렉터처럼 임의로 추측할 수 없는 값은 코드에 박지 않고 config로 받는다
(웹 폴링 감지기는 그래서 기본 제공하지 않는다). 시계(now_fn)와 실행기(runner),
감지기 맵을 주입받아 스레드·sleep·실제 파일 없이 결정론적으로 테스트한다.
"""
from __future__ import annotations

import os
import threading
import time
import uuid

from config.config import EVENT_POLL_SECONDS
from core import event_store
from core.logger import get_logger
from core.scheduler import default_runner

log = get_logger("event_triggers")


def _check_file_exists(config: dict, prev_state):
    """파일이 '없다가 생기는' 순간에 발화. state=존재여부(bool)."""
    exists = os.path.exists(config["path"])
    fired = exists and prev_state is not True
    return fired, exists


def _check_file_changed(config: dict, prev_state):
    """파일 내용(수정 시각)이 바뀌는 순간에 발화. state=mtime(없으면 None).

    최초 관측은 기준선만 잡고 발화하지 않는다(등록 즉시 오발화 방지)."""
    path = config["path"]
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    fired = mtime is not None and prev_state is not None and mtime != prev_state
    return fired, mtime


# source 이름 → 감지기. 모호함 없는 로컬 파일 기반만 기본 제공한다.
CHECKERS = {
    "file_exists": _check_file_exists,
    "file_changed": _check_file_changed,
}


class EventEngine:
    def __init__(self, runner=default_runner, checkers=None, now_fn=time.time):
        self._runner = runner
        self._checkers = CHECKERS if checkers is None else checkers
        self._now = now_fn
        self._thread = None
        self._stop = threading.Event()

    def add(self, source: str, source_config: dict, kind: str, payload: dict,
            enabled: bool = True) -> str:
        """트리거를 등록하고 id를 반환. 등록 시점에 현재 상태를 기준선으로 잡아
        (발화 없이) 저장하므로, 이미 충족된 조건으로 즉시 쏘지 않는다."""
        if source not in self._checkers:
            raise ValueError(f"알 수 없는 이벤트 source: {source!r} ({list(self._checkers)})")
        if kind not in ("dynamic", "template"):
            raise ValueError(f"지원하지 않는 kind: {kind!r} (dynamic|template)")
        tid = uuid.uuid4().hex
        _, baseline = self._checkers[source](source_config, None)
        event_store.upsert({
            "id": tid, "source": source, "source_config": source_config,
            "kind": kind, "payload": payload, "enabled": enabled,
            "state": baseline, "last_fired": None, "last_status": None,
        })
        log.info("이벤트 트리거 등록: %s (%s)", tid, source)
        return tid

    def list(self) -> list:
        return event_store.load_all()

    def remove(self, tid: str) -> bool:
        return event_store.delete(tid)

    def set_enabled(self, tid: str, enabled: bool) -> bool:
        t = event_store.get(tid)
        if t is None:
            return False
        t["enabled"] = enabled
        event_store.upsert(t)
        return True

    def poll(self, now: float | None = None) -> list:
        """활성 트리거를 점검해 발화 조건이 충족된 것만 실행한다. 발화한 id 리스트 반환.

        감지기/러너가 예외를 던져도 그 트리거만 last_status에 남기고 넘어간다 —
        한 트리거의 실패가 다른 트리거나 폴링 루프를 막지 않는다. 상태는 발화 여부와
        무관하게 항상 갱신해 다음 비교 기준으로 삼는다."""
        now = self._now() if now is None else now
        fired_ids = []
        for t in event_store.load_all():
            if not t["enabled"]:
                continue
            checker = self._checkers.get(t["source"])
            if checker is None:
                log.error("등록된 감지기 없음: %s (트리거 %s)", t["source"], t["id"])
                continue
            try:
                fired, new_state = checker(t["source_config"], t["state"])
            except Exception as exc:
                log.error("이벤트 감지 실패: %s (%s)", t["id"], exc)
                t["state"] = None
                event_store.upsert(t)
                continue
            t["state"] = new_state
            if fired:
                try:
                    t["last_status"] = self._runner(t)
                except Exception as exc:
                    log.error("이벤트 워크플로우 실행 실패: %s (%s)", t["id"], exc)
                    t["last_status"] = f"error: {exc}"
                t["last_fired"] = now
                fired_ids.append(t["id"])
            event_store.upsert(t)
        return fired_ids

    def _loop(self, poll_seconds: float) -> None:
        while not self._stop.is_set():
            try:
                self.poll()
            except Exception as exc:
                log.error("이벤트 폴링 중 예외: %s", exc)
            self._stop.wait(poll_seconds)

    def start(self, poll_seconds: int = EVENT_POLL_SECONDS) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(poll_seconds,), daemon=True)
        self._thread.start()
        log.info("이벤트 엔진 시작(폴링 %ds)", poll_seconds)

    def stop(self) -> None:
        self._stop.set()

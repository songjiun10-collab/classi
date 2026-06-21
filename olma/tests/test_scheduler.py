"""Scheduler(core.scheduler) 테스트 — 주입한 시계(now_fn)와 모의 runner로 결정론적 검증.

스레드·sleep 없이 due/tick의 시간 로직만 본다: 등록, 만기 판정, 실행 후 재예약,
비활성 제외, runner 예외 격리.
"""
import importlib

import pytest


@pytest.fixture
def sched_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHEDULER_STORE_PATH", str(tmp_path / "schedules.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.scheduler_store as ss
    importlib.reload(ss)
    import core.scheduler as sched
    importlib.reload(sched)
    return sched


class _Clock:
    """주입용 가짜 시계 — t를 직접 굴려 시간 경과를 시뮬레이션한다."""
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t


def test_add_returns_id_and_persists(sched_env):
    clock = _Clock()
    s = sched_env.Scheduler(runner=lambda _: "done", now_fn=clock)
    sid = s.add("dynamic", {"request": "안녕"}, interval_seconds=60)
    assert isinstance(sid, str) and sid
    listed = s.list()
    assert len(listed) == 1
    assert listed[0]["id"] == sid
    assert listed[0]["kind"] == "dynamic"


def test_add_rejects_bad_kind_and_interval(sched_env):
    s = sched_env.Scheduler(runner=lambda _: "done", now_fn=_Clock())
    with pytest.raises(ValueError):
        s.add("weird", {}, interval_seconds=60)
    with pytest.raises(ValueError):
        s.add("dynamic", {"request": "x"}, interval_seconds=0)


def test_first_run_delay_defers_due(sched_env):
    clock = _Clock(1000.0)
    s = sched_env.Scheduler(runner=lambda _: "done", now_fn=clock)
    s.add("dynamic", {"request": "x"}, interval_seconds=60, first_run_delay=30)
    assert s.due() == []          # 아직 만기 아님(1030에 첫 실행)
    clock.t = 1030.0
    assert len(s.due()) == 1


def test_due_excludes_disabled(sched_env):
    clock = _Clock(1000.0)
    s = sched_env.Scheduler(runner=lambda _: "done", now_fn=clock)
    sid = s.add("dynamic", {"request": "x"}, interval_seconds=60, enabled=False)
    assert s.due() == []
    s.set_enabled(sid, True)
    assert len(s.due()) == 1


def test_tick_runs_due_and_records_status(sched_env):
    clock = _Clock(1000.0)
    calls = []
    s = sched_env.Scheduler(runner=lambda sc: calls.append(sc["id"]) or "done", now_fn=clock)
    sid = s.add("dynamic", {"request": "x"}, interval_seconds=60)
    ran = s.tick()
    assert ran == [sid]
    assert calls == [sid]
    got = s.list()[0]
    assert got["last_run"] == 1000.0
    assert got["last_status"] == "done"


def test_tick_reschedules_next_run_from_now(sched_env):
    clock = _Clock(1000.0)
    s = sched_env.Scheduler(runner=lambda _: "done", now_fn=clock)
    s.add("dynamic", {"request": "x"}, interval_seconds=60)
    s.tick()
    assert s.list()[0]["next_run"] == 1060.0   # now + interval
    # 다음 만기 전에는 다시 실행 안 됨
    clock.t = 1059.0
    assert s.tick() == []
    clock.t = 1060.0
    assert len(s.tick()) == 1


def test_tick_isolates_runner_exception(sched_env):
    clock = _Clock(1000.0)

    def boom(_):
        raise RuntimeError("터짐")

    s = sched_env.Scheduler(runner=boom, now_fn=clock)
    s.add("dynamic", {"request": "x"}, interval_seconds=60)
    ran = s.tick()                       # 예외가 tick을 막지 않는다
    assert len(ran) == 1
    got = s.list()[0]
    assert got["last_status"].startswith("error:")
    assert got["next_run"] == 1060.0     # 실패해도 재예약


def test_tick_one_failure_does_not_block_others(sched_env):
    clock = _Clock(1000.0)

    def runner(sc):
        if sc["payload"]["request"] == "bad":
            raise RuntimeError("nope")
        return "done"

    s = sched_env.Scheduler(runner=runner, now_fn=clock)
    s.add("dynamic", {"request": "bad"}, interval_seconds=60)
    good = s.add("dynamic", {"request": "good"}, interval_seconds=60)
    ran = s.tick()
    assert len(ran) == 2                  # 둘 다 시도됨
    statuses = {x["id"]: x["last_status"] for x in s.list()}
    assert statuses[good] == "done"


def test_remove(sched_env):
    s = sched_env.Scheduler(runner=lambda _: "done", now_fn=_Clock())
    sid = s.add("dynamic", {"request": "x"}, interval_seconds=60)
    assert s.remove(sid) is True
    assert s.list() == []
    assert s.remove(sid) is False


def test_set_enabled_missing_returns_false(sched_env):
    s = sched_env.Scheduler(runner=lambda _: "done", now_fn=_Clock())
    assert s.set_enabled("없음", True) is False


def test_default_runner_dispatches_by_kind(sched_env):
    from unittest.mock import patch
    with patch("core.scheduler.workflow.run_dynamic", return_value={"status": "done"}) as rd:
        status = sched_env.default_runner(
            {"kind": "dynamic", "payload": {"request": "안녕"}}
        )
    assert status == "done"
    rd.assert_called_once()

    with patch("core.scheduler.workflow.run_template", return_value={"status": "partial"}) as rt:
        status = sched_env.default_runner(
            {"kind": "template", "payload": {"name": "wf", "params": {"a": 1}}}
        )
    assert status == "partial"
    rt.assert_called_once()


def test_default_runner_unknown_kind_raises(sched_env):
    with pytest.raises(ValueError):
        sched_env.default_runner({"kind": "weird", "payload": {}})

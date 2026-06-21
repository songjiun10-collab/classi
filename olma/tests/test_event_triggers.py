"""Event Trigger 엔진(core.event_triggers) 테스트.

엣지 트리거(상태 변화 순간에만 발화)·기준선 캡처·예외 격리를, 주입한 가짜 감지기와
runner로 스레드 없이 검증한다. 실제 파일 감지기(file_exists/file_changed)는 tmp 파일로
별도 검증한다.
"""
import importlib

import pytest


@pytest.fixture
def ev_env(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_STORE_PATH", str(tmp_path / "events.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.event_store as es
    importlib.reload(es)
    import core.event_triggers as ev
    importlib.reload(ev)
    return ev


class _Flag:
    """주입용 가짜 감지기. value를 직접 바꿔 상태 변화를 흉내낸다.
    엣지 의미: prev_state != value 이면 발화, 새 상태는 value."""
    def __init__(self, value=0):
        self.value = value

    def __call__(self, config, prev_state):
        fired = prev_state is not None and prev_state != self.value
        return fired, self.value


def _engine(ev, checker, runner=None):
    runner = runner or (lambda t: "done")
    return ev.EventEngine(runner=runner, checkers={"flag": checker}, now_fn=lambda: 1000.0)


def test_add_captures_baseline_without_firing(ev_env):
    flag = _Flag(value=5)
    eng = _engine(ev_env, flag)
    eng.add("flag", {}, "dynamic", {"request": "x"})
    # 등록 시 기준선(5)만 잡고, 값이 그대로면 발화하지 않는다.
    assert eng.poll() == []
    assert eng.list()[0]["state"] == 5


def test_poll_fires_on_state_change(ev_env):
    flag = _Flag(value=5)
    calls = []
    eng = _engine(ev_env, flag, runner=lambda t: calls.append(t["id"]) or "done")
    tid = eng.add("flag", {}, "dynamic", {"request": "x"})
    flag.value = 6                       # 상태 변화 → 다음 poll에서 발화
    assert eng.poll() == [tid]
    assert calls == [tid]
    got = eng.list()[0]
    assert got["state"] == 6
    assert got["last_fired"] == 1000.0
    assert got["last_status"] == "done"


def test_poll_is_edge_triggered_not_level(ev_env):
    flag = _Flag(value=5)
    eng = _engine(ev_env, flag)
    eng.add("flag", {}, "dynamic", {"request": "x"})
    flag.value = 6
    assert len(eng.poll()) == 1          # 변화 순간 1회
    assert eng.poll() == []              # 같은 값 유지 → 다시 쏘지 않음
    flag.value = 7
    assert len(eng.poll()) == 1          # 또 바뀌면 다시 1회


def test_add_rejects_unknown_source_and_kind(ev_env):
    eng = _engine(ev_env, _Flag())
    with pytest.raises(ValueError):
        eng.add("없는소스", {}, "dynamic", {})
    with pytest.raises(ValueError):
        eng.add("flag", {}, "weird", {})


def test_disabled_trigger_not_polled(ev_env):
    flag = _Flag(value=5)
    eng = _engine(ev_env, flag)
    tid = eng.add("flag", {}, "dynamic", {"request": "x"}, enabled=False)
    flag.value = 6
    assert eng.poll() == []
    eng.set_enabled(tid, True)
    # 활성화 후에도 기준선은 비활성 동안 갱신되지 않았으므로(5), 6과 달라 발화한다.
    assert eng.poll() == [tid]


def test_poll_isolates_runner_exception(ev_env):
    flag = _Flag(value=5)

    def boom(t):
        raise RuntimeError("터짐")

    eng = _engine(ev_env, flag, runner=boom)
    tid = eng.add("flag", {}, "dynamic", {"request": "x"})
    flag.value = 6
    fired = eng.poll()                   # 예외가 poll을 막지 않는다
    assert fired == [tid]
    got = eng.list()[0]
    assert got["last_status"].startswith("error:")
    assert got["state"] == 6             # 상태는 갱신됨


def test_poll_isolates_checker_exception(ev_env):
    def bad_checker(config, prev_state):
        raise RuntimeError("감지 실패")

    eng = ev_env.EventEngine(
        runner=lambda t: "done", checkers={"bad": bad_checker}, now_fn=lambda: 1.0
    )
    # add는 기준선 캡처에서 예외가 나므로, 저장소에 직접 넣어 poll 격리만 본다.
    import core.event_store as es
    es.upsert({
        "id": "x", "source": "bad", "source_config": {}, "kind": "dynamic",
        "payload": {}, "enabled": True, "state": 1, "last_fired": None, "last_status": None,
    })
    assert eng.poll() == []              # 예외 삼키고 진행
    assert eng.list()[0]["state"] is None


def test_remove_and_set_enabled_missing(ev_env):
    eng = _engine(ev_env, _Flag())
    tid = eng.add("flag", {}, "dynamic", {"request": "x"})
    assert eng.remove(tid) is True
    assert eng.list() == []
    assert eng.remove(tid) is False
    assert eng.set_enabled("없음", True) is False


# --- 실제 파일 감지기 ---

def test_file_exists_checker_fires_on_creation(ev_env, tmp_path):
    target = tmp_path / "appears.txt"
    eng = ev_env.EventEngine(runner=lambda t: "done", now_fn=lambda: 1.0)
    tid = eng.add("file_exists", {"path": str(target)}, "dynamic", {"request": "x"})
    assert eng.poll() == []              # 아직 없음
    target.write_text("hi")
    assert eng.poll() == [tid]           # 생성 순간 발화
    assert eng.poll() == []              # 계속 존재해도 다시 안 쏨(엣지)


def test_file_changed_checker_fires_on_modification(ev_env, tmp_path):
    target = tmp_path / "watch.txt"
    target.write_text("v1")
    eng = ev_env.EventEngine(runner=lambda t: "done", now_fn=lambda: 1.0)
    tid = eng.add("file_changed", {"path": str(target)}, "dynamic", {"request": "x"})
    assert eng.poll() == []              # 기준선 잡힘, 변화 없음
    import os
    # mtime을 명시적으로 앞당겨 변경을 결정론적으로 만든다(같은 초 내 쓰기 방지).
    base = os.path.getmtime(target)
    target.write_text("v2")
    os.utime(target, (base + 10, base + 10))
    assert eng.poll() == [tid]

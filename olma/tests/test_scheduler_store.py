"""스케줄 저장소(core.scheduler_store) 테스트 — upsert/get/load_all/delete/영속성."""
import importlib


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("SCHEDULER_STORE_PATH", str(tmp_path / "schedules.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.scheduler_store as ss
    importlib.reload(ss)
    return ss


def _sched(sid="s1", next_run=10.0, enabled=True):
    return {
        "id": sid, "kind": "dynamic", "payload": {"request": "안녕"},
        "interval_seconds": 60, "next_run": next_run, "enabled": enabled,
        "last_run": None, "last_status": None,
    }


def test_upsert_and_get_roundtrip(tmp_path, monkeypatch):
    ss = _fresh(tmp_path, monkeypatch)
    ss.upsert(_sched())
    got = ss.get("s1")
    assert got["kind"] == "dynamic"
    assert got["payload"] == {"request": "안녕"}
    assert got["interval_seconds"] == 60
    assert got["next_run"] == 10.0
    assert got["enabled"] is True


def test_get_missing_returns_none(tmp_path, monkeypatch):
    ss = _fresh(tmp_path, monkeypatch)
    assert ss.get("없음") is None


def test_upsert_overwrites_same_id(tmp_path, monkeypatch):
    ss = _fresh(tmp_path, monkeypatch)
    ss.upsert(_sched())
    s = _sched()
    s["last_status"] = "done"
    s["next_run"] = 70.0
    ss.upsert(s)
    got = ss.get("s1")
    assert got["last_status"] == "done"
    assert got["next_run"] == 70.0
    assert len(ss.load_all()) == 1   # 덮어쓰기지 중복 아님


def test_load_all_sorted_by_next_run(tmp_path, monkeypatch):
    ss = _fresh(tmp_path, monkeypatch)
    ss.upsert(_sched("a", next_run=30.0))
    ss.upsert(_sched("b", next_run=10.0))
    ss.upsert(_sched("c", next_run=20.0))
    assert [s["id"] for s in ss.load_all()] == ["b", "c", "a"]


def test_delete(tmp_path, monkeypatch):
    ss = _fresh(tmp_path, monkeypatch)
    ss.upsert(_sched())
    assert ss.delete("s1") is True
    assert ss.get("s1") is None
    assert ss.delete("s1") is False   # 이미 없으면 False


def test_enabled_false_roundtrips(tmp_path, monkeypatch):
    ss = _fresh(tmp_path, monkeypatch)
    ss.upsert(_sched(enabled=False))
    assert ss.get("s1")["enabled"] is False


def test_persists_across_reload(tmp_path, monkeypatch):
    ss = _fresh(tmp_path, monkeypatch)
    ss.upsert(_sched())
    import core.scheduler_store as ss2
    importlib.reload(ss2)
    assert ss2.get("s1")["payload"] == {"request": "안녕"}

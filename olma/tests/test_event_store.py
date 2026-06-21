"""이벤트 트리거 저장소(core.event_store) 테스트 — upsert/get/load_all/delete/상태 영속성."""
import importlib


def _fresh(tmp_path, monkeypatch):
    monkeypatch.setenv("EVENT_STORE_PATH", str(tmp_path / "events.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.event_store as es
    importlib.reload(es)
    return es


def _trig(tid="t1", state=None):
    return {
        "id": tid, "source": "file_exists", "source_config": {"path": "/tmp/x"},
        "kind": "dynamic", "payload": {"request": "처리"}, "enabled": True,
        "state": state, "last_fired": None, "last_status": None,
    }


def test_upsert_and_get_roundtrip(tmp_path, monkeypatch):
    es = _fresh(tmp_path, monkeypatch)
    es.upsert(_trig())
    got = es.get("t1")
    assert got["source"] == "file_exists"
    assert got["source_config"] == {"path": "/tmp/x"}
    assert got["payload"] == {"request": "처리"}
    assert got["state"] is None


def test_state_none_and_value_roundtrip(tmp_path, monkeypatch):
    es = _fresh(tmp_path, monkeypatch)
    es.upsert(_trig(state=None))
    assert es.get("t1")["state"] is None
    t = _trig(state=1234.5)        # mtime 같은 값도 JSON으로 보존
    es.upsert(t)
    assert es.get("t1")["state"] == 1234.5
    t2 = _trig(state=True)         # bool 상태도 보존
    es.upsert(t2)
    assert es.get("t1")["state"] is True


def test_get_missing_returns_none(tmp_path, monkeypatch):
    es = _fresh(tmp_path, monkeypatch)
    assert es.get("없음") is None


def test_upsert_overwrites_same_id(tmp_path, monkeypatch):
    es = _fresh(tmp_path, monkeypatch)
    es.upsert(_trig())
    t = _trig(state=True)
    t["last_status"] = "done"
    es.upsert(t)
    got = es.get("t1")
    assert got["last_status"] == "done"
    assert got["state"] is True
    assert len(es.load_all()) == 1


def test_load_all_insertion_order(tmp_path, monkeypatch):
    es = _fresh(tmp_path, monkeypatch)
    es.upsert(_trig("a"))
    es.upsert(_trig("b"))
    es.upsert(_trig("c"))
    assert [t["id"] for t in es.load_all()] == ["a", "b", "c"]


def test_delete(tmp_path, monkeypatch):
    es = _fresh(tmp_path, monkeypatch)
    es.upsert(_trig())
    assert es.delete("t1") is True
    assert es.get("t1") is None
    assert es.delete("t1") is False


def test_persists_across_reload(tmp_path, monkeypatch):
    es = _fresh(tmp_path, monkeypatch)
    es.upsert(_trig(state=True))
    import core.event_store as es2
    importlib.reload(es2)
    assert es2.get("t1")["state"] is True

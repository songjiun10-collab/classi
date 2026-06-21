import importlib


def _fresh_store(tmp_path, monkeypatch):
    monkeypatch.setenv("TASK_STORE_PATH", str(tmp_path / "tasks.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.task_store as task_store
    importlib.reload(task_store)
    return task_store


def _sample_task(task_id="t1", status="completed", results=None):
    return {
        "task_id": task_id,
        "input": "작업 입력",
        "status": status,
        "outcome": "done",
        "results": results if results is not None else [{"action": "llm", "status": "ok", "result": "답변"}],
        "error": None,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


def test_upsert_then_load_all_round_trips(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    store.upsert(_sample_task())

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0]["task_id"] == "t1"
    assert loaded[0]["status"] == "completed"
    assert loaded[0]["results"] == [{"action": "llm", "status": "ok", "result": "답변"}]


def test_upsert_with_same_task_id_updates_in_place(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    store.upsert(_sample_task(status="queued", results=None))
    store.upsert(_sample_task(status="completed"))

    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0]["status"] == "completed"
    assert loaded[0]["results"] is not None


def test_delete_removes_task(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    store.upsert(_sample_task("a"))
    store.upsert(_sample_task("b"))
    store.delete("a")

    loaded = store.load_all()
    assert [t["task_id"] for t in loaded] == ["b"]


def test_mark_interrupted_as_failed_only_affects_unresolved_tasks(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    store.upsert(_sample_task("queued-task", status="queued", results=None))
    store.upsert(_sample_task("processing-task", status="processing", results=None))
    store.upsert(_sample_task("done-task", status="completed"))

    store.mark_interrupted_as_failed()

    by_id = {t["task_id"]: t for t in store.load_all()}
    assert by_id["queued-task"]["status"] == "failed"
    assert by_id["processing-task"]["status"] == "failed"
    assert by_id["done-task"]["status"] == "completed"
    assert by_id["queued-task"]["error"]


def test_load_all_orders_by_created_at_ascending(tmp_path, monkeypatch):
    store = _fresh_store(tmp_path, monkeypatch)
    task_a = _sample_task("a")
    task_a["created_at"] = "2026-01-01T00:00:00+00:00"
    task_b = _sample_task("b")
    task_b["created_at"] = "2026-01-02T00:00:00+00:00"
    store.upsert(task_b)
    store.upsert(task_a)

    loaded = store.load_all()
    assert [t["task_id"] for t in loaded] == ["a", "b"]

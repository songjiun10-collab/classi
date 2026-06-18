import importlib


def _fresh_memory(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.json"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.memory as memory
    importlib.reload(memory)
    return memory


def test_save_stores_task_state_with_steps(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)

    results = [
        {"action": "llm", "target": "ollama", "status": "ok", "attempts": 1, "duration": 0.1, "result": "답변"},
    ]
    record = memory.save("작업", results)

    assert record["status"] == "done"
    assert record["schema_version"] == 2
    assert len(record["steps"]) == 1
    assert record["steps"][0]["action"] == "llm"

    loaded = memory.load_all()
    assert loaded[0]["task"] == "작업"


def test_derive_status_partial_when_some_fail(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)

    results = [
        {"action": "browser_open", "status": "ok"},
        {"action": "browser_get_text", "status": "failed"},
    ]
    record = memory.save("부분 실패 작업", results)
    assert record["status"] == "partial"


def test_derive_status_failed_when_all_fail(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    record = memory.save("전부 실패", [{"action": "llm", "status": "failed"}])
    assert record["status"] == "failed"


def test_fallback_status_counts_as_success(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    record = memory.save("폴백 작업", [{"action": "browser_search", "status": "fallback"}])
    assert record["status"] == "done"


def test_get_context_summarizes_recent_tasks(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    memory.save("첫 작업", [{"action": "llm", "status": "ok"}])
    memory.save("둘째 작업", [{"action": "llm", "status": "failed"}])

    context = memory.get_context()
    assert "첫 작업" in context
    assert "둘째 작업" in context
    assert "[done]" in context
    assert "[failed]" in context


def test_find_matches_by_keyword(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    memory.save("날씨 알려줘", [{"action": "llm", "status": "ok"}])
    memory.save("뉴스 검색해줘", [{"action": "browser_search", "status": "ok"}])

    matched = memory.find("날씨")
    assert len(matched) == 1
    assert matched[0]["task"] == "날씨 알려줘"

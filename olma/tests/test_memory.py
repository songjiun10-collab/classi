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


def test_find_is_case_insensitive(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    memory.save("Python 공식 문서 열어줘", [{"action": "browser_open", "status": "ok"}])

    matched = memory.find("python")
    assert len(matched) == 1


def test_find_matches_step_action_and_result(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    memory.save("작업1", [{"action": "browser_search", "status": "ok", "result": "흥미로운 뉴스 결과"}])
    memory.save("작업2", [{"action": "llm", "status": "ok", "result": "다른 답변"}])

    by_action = memory.find("browser_search")
    assert [r["task"] for r in by_action] == ["작업1"]

    by_result = memory.find("뉴스")
    assert [r["task"] for r in by_result] == ["작업1"]


def test_find_returns_newest_match_first(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    memory.save("첫 매치 작업", [{"action": "llm", "status": "ok"}])
    memory.save("관련 없음", [{"action": "llm", "status": "ok"}])
    memory.save("둘째 매치 작업", [{"action": "llm", "status": "ok"}])

    matched = memory.find("매치")
    assert [r["task"] for r in matched] == ["둘째 매치 작업", "첫 매치 작업"]


def test_save_writes_atomically_leaving_no_tmp_file(tmp_path, monkeypatch):
    memory = _fresh_memory(tmp_path, monkeypatch)
    memory.save("작업", [{"action": "llm", "status": "ok"}])

    files = list(tmp_path.iterdir())
    assert [f.name for f in files] == ["memory.json"]


def test_corrupted_memory_file_self_heals_instead_of_raising(tmp_path, monkeypatch):
    memory_path = tmp_path / "memory.json"
    memory_path.write_text("{이건 깨진 JSON", encoding="utf-8")
    memory = _fresh_memory(tmp_path, monkeypatch)

    assert memory.load_all() == []
    assert memory.get_context() == ""
    # 손상된 원본은 지워지지 않고 타임스탬프가 붙어 백업된다.
    backups = list(tmp_path.glob("memory.json.corrupt-*"))
    assert len(backups) == 1


def test_save_recovers_after_corruption_instead_of_staying_broken(tmp_path, monkeypatch):
    memory_path = tmp_path / "memory.json"
    memory_path.write_text("not json at all", encoding="utf-8")
    memory = _fresh_memory(tmp_path, monkeypatch)

    # 손상 이후에도 save()는 죽지 않고 정상적으로 새 기록을 쌓는다(영구 장애 방지).
    record = memory.save("복구 후 작업", [{"action": "llm", "status": "ok"}])
    assert record["status"] == "done"
    assert [r["task"] for r in memory.load_all()] == ["복구 후 작업"]


def test_save_rotates_out_oldest_records_beyond_max(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_MAX_RECORDS", "3")
    memory = _fresh_memory(tmp_path, monkeypatch)

    for i in range(5):
        memory.save(f"작업{i}", [{"action": "llm", "status": "ok"}])

    loaded = memory.load_all()
    assert len(loaded) == 3
    assert [r["task"] for r in loaded] == ["작업2", "작업3", "작업4"]

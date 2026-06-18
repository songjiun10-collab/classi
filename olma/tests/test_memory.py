import importlib
import os


def test_save_includes_schema_version(tmp_path, monkeypatch):
    monkeypatch.setenv("MEMORY_PATH", str(tmp_path / "memory.json"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.memory as memory
    importlib.reload(memory)

    memory.save("작업", "결과")
    records = memory.load_all()

    assert len(records) == 1
    assert records[0]["task"] == "작업"
    assert records[0]["result"] == "결과"
    assert records[0]["schema_version"] == 1

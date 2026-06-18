"""작업 기록을 storage/memory.json 에 누적 저장한다."""
import json
import os
from datetime import datetime, timezone

from config.config import MEMORY_PATH
from core.schema import SCHEMA_VERSION


def _load_raw() -> list:
    if not os.path.exists(MEMORY_PATH):
        return []
    with open(MEMORY_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        return json.loads(content) if content else []


def save(task: str, result: str) -> None:
    records = _load_raw()
    records.append(
        {
            "task": task,
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
        }
    )
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def load_all() -> list:
    return _load_raw()


def recent(n: int = 5) -> list:
    return _load_raw()[-n:]

"""결정론적(temperature=0) Ollama 추론을 SQLite에 캐시한다 (성능/발열 절감).

temperature=0이면 같은 입력(모델+프롬프트+format+images)에 대해 출력이 결정론적이므로,
한 번 본 입력은 Ollama를 다시 호출하지 않고 저장된 응답을 그대로 돌려준다. 플래너 호출이
정확히 이 경우(temperature=0.0, 동일 시스템 프롬프트 골격)라 반복 작업에서 재추론 0회로
발열·지연을 크게 줄인다. temperature>0(예: 기본 0.7의 llm/summarize)은 비결정론적이라
절대 캐시하지 않는다 — 캐시 사용 여부는 호출자(ollama_client)가 temperature로 판단한다.

stdlib sqlite3만 쓴다(memory.py/task_store.py와 동일 패턴). 트랜잭션이 원자성을 보장하므로
별도 잠금/임시파일이 필요 없다."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager

from config.config import INFER_CACHE_MAX_RECORDS, INFER_CACHE_PATH
from core.logger import get_logger

log = get_logger("infer_cache")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS infer_cache (
    key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created_at INTEGER NOT NULL
)
"""

# 단조 증가하는 삽입 순서. created_at에 Date.now류를 쓰지 않는 이유는 결정론/테스트
# 안정성 — 회전(오래된 것부터 삭제)에는 삽입 순번만 있으면 충분하다.
_seq = 0


@contextmanager
def _connect():
    os.makedirs(os.path.dirname(INFER_CACHE_PATH), exist_ok=True)
    conn = sqlite3.connect(INFER_CACHE_PATH, timeout=5)
    try:
        conn.execute(_SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def make_key(
    model: str,
    prompt: str,
    format: dict | str | None,
    images: list | None,
) -> str:
    """캐시 키 = 출력에 영향을 주는 모든 입력의 안정적 해시.

    temperature=0이 전제이므로 temperature/seed는 키에 넣지 않는다(결정론적 동일 출력).
    images는 base64 문자열 리스트라 길 수 있어 해시에 그대로 포함한다."""
    payload = {
        "model": model,
        "prompt": prompt,
        "format": format,
        "images": images,
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def get(key: str) -> str | None:
    """캐시 히트면 저장된 응답, 미스면 None. 어떤 오류든 None으로 폴백(캐시는 보조수단)."""
    try:
        with _connect() as conn:
            row = conn.execute(
                "SELECT response FROM infer_cache WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None
    except Exception as exc:
        log.warning("추론 캐시 조회 실패(무시하고 직접 추론): %s", exc)
        return None


def put(key: str, response: str) -> None:
    """응답을 저장하고 초과분은 오래된(삽입 순번이 작은) 것부터 회전 삭제한다."""
    global _seq
    try:
        _seq += 1
        with _connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO infer_cache (key, response, created_at) VALUES (?, ?, ?)",
                (key, response, _seq),
            )
            conn.execute(
                "DELETE FROM infer_cache WHERE key NOT IN "
                "(SELECT key FROM infer_cache ORDER BY created_at DESC LIMIT ?)",
                (INFER_CACHE_MAX_RECORDS,),
            )
    except Exception as exc:
        log.warning("추론 캐시 저장 실패(무시): %s", exc)


def stats() -> dict:
    """캐시 항목 수(관측/디버깅용)."""
    try:
        with _connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM infer_cache").fetchone()[0]
        return {"entries": count}
    except Exception:
        return {"entries": 0}

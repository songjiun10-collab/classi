"""SQLite caches for classi (infer + extract). Pure storage + key derivation;
imports nothing from classifier_engine (config is passed in as args), so
classifier_engine can depend on this with zero cycle risk.

Dependency: classifier_engine → cache (one-directional only).
"""
import hashlib, json, os, pickle, sqlite3, threading, time
from pathlib import Path
from typing import List, Optional


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default


# ── infer cache ─────────────────────────────────────────────────────────────
_CACHE_ENABLED = os.environ.get("CLASSI_CACHE", "1") != "0"
_CACHE_DB = os.environ.get("CLASSI_CACHE_DB", str(Path.home() / ".classi" / "infer_cache.db"))
_cache_conn = None
# RLock 필수: get/put이 락을 쥔 채 _cache_connection()을 부르고, 연결 초기화도
# 같은 락으로 보호하므로(이중 초기화 방지) 재진입이 일어난다 — Lock이면 데드락.
_cache_lock = threading.RLock()


def cache_enabled() -> bool:
    """단일 진실원: 캐시 활성 여부. 테스트에서 set_cache_enabled()로 토글."""
    return _CACHE_ENABLED


def set_cache_enabled(v: bool) -> None:
    global _CACHE_ENABLED
    _CACHE_ENABLED = bool(v)


def _cache_connection():
    global _cache_conn
    with _cache_lock:
        if _cache_conn is None:
            Path(_CACHE_DB).parent.mkdir(parents=True, exist_ok=True)
            _cache_conn = sqlite3.connect(_CACHE_DB, check_same_thread=False)
            _cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS infer_cache "
                "(key TEXT PRIMARY KEY, value TEXT, ts REAL)")
            _cache_conn.commit()
    return _cache_conn


def infer_cache_key(model: str, image_bytes: bytes, prompt: str) -> str:
    """모델 추론을 결정하는 모든 입력(모델명·이미지·프롬프트)의 SHA-256 해시."""
    h = hashlib.sha256()
    h.update((model or "").encode("utf-8")); h.update(b"\x00")
    h.update(hashlib.sha256(image_bytes or b"").digest()); h.update(b"\x00")
    h.update((prompt or "").encode("utf-8"))
    return h.hexdigest()


def infer_cache_get(key: str) -> Optional[dict]:
    if not _CACHE_ENABLED:
        return None
    try:
        with _cache_lock:
            row = _cache_connection().execute(
                "SELECT value FROM infer_cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None
    except Exception:
        return None  # 캐시 장애가 분류를 막아선 안 된다


def infer_cache_put(key: str, value: dict) -> None:
    if not _CACHE_ENABLED:
        return
    try:
        with _cache_lock:
            conn = _cache_connection()
            conn.execute(
                "INSERT OR REPLACE INTO infer_cache (key, value, ts) VALUES (?,?,?)",
                (key, json.dumps(value, ensure_ascii=False), time.time()))
            conn.commit()
    except Exception:
        pass


# ── extract cache ────────────────────────────────────────────────────────────
_EXTRACT_CACHE_DB = os.environ.get(
    "CLASSI_EXTRACT_CACHE_DB", str(Path.home() / ".classi" / "extract_cache.db"))
_EXTRACT_CACHE_MAX_BYTES = _env_int("CLASSI_EXTRACT_CACHE_MAX_BYTES", 512 * 1024 * 1024)
_extract_cache_conn = None
_extract_cache_lock = threading.RLock()


def extract_cache_key(pdf_bytes: bytes, max_problems: int, *, version: str,
                      det_zoom, ocr_backend: str, ocr_zoom, scan_text: str) -> str:
    """추출 결과를 결정하는 모든 입력의 해시. config 값을 인자로 받아(엔진이 소유, 캐시가 해싱)
    key+storage를 같은 모듈에 두면서도 엔진을 import하지 않는다 — 순환 없음."""
    h = hashlib.sha256()
    for part in (version, str(det_zoom), ocr_backend, str(ocr_zoom), scan_text, str(max_problems)):
        h.update(part.encode("utf-8")); h.update(b"\x00")
    h.update(hashlib.sha256(pdf_bytes).digest())
    return h.hexdigest()


def _extract_cache_connection():
    global _extract_cache_conn
    with _extract_cache_lock:
        if _extract_cache_conn is None:
            Path(_EXTRACT_CACHE_DB).parent.mkdir(parents=True, exist_ok=True)
            _extract_cache_conn = sqlite3.connect(_EXTRACT_CACHE_DB, check_same_thread=False)
            _extract_cache_conn.execute(
                "CREATE TABLE IF NOT EXISTS extract_cache "
                "(key TEXT PRIMARY KEY, value BLOB, nbytes INTEGER, ts REAL)")
            _extract_cache_conn.commit()
    return _extract_cache_conn


def extract_problems_get(key: str) -> Optional[List[dict]]:
    """타입 있는 get: 캐시된 문항 목록(pickle 내부 처리)을 반환하거나 None."""
    if not _CACHE_ENABLED:
        return None
    try:
        with _extract_cache_lock:
            row = _extract_cache_connection().execute(
                "SELECT value FROM extract_cache WHERE key=?", (key,)).fetchone()
        return pickle.loads(row[0]) if row else None
    except Exception:
        return None


def extract_problems_put(key: str, problems: List[dict]) -> None:
    """타입 있는 put: pickle + INSERT + 바이트 상한 LRU 축출(기존 로직 그대로 이전)."""
    if not _CACHE_ENABLED:
        return
    try:
        blob = pickle.dumps(problems, protocol=4)
        with _extract_cache_lock:
            conn = _extract_cache_connection()
            conn.execute(
                "INSERT OR REPLACE INTO extract_cache (key, value, nbytes, ts) "
                "VALUES (?,?,?,?)", (key, blob, len(blob), time.time()))
            # 총 바이트 상한: 가장 오래된 항목부터 제거(방금 넣은 건 ts 최신이라 보존됨)
            while True:
                total = conn.execute(
                    "SELECT COALESCE(SUM(nbytes),0) FROM extract_cache").fetchone()[0]
                if total <= _EXTRACT_CACHE_MAX_BYTES:
                    break
                old = conn.execute(
                    "SELECT key FROM extract_cache ORDER BY ts ASC LIMIT 1").fetchone()
                if old is None or old[0] == key:
                    break  # 방금 항목 하나만으로 초과 — 그래도 이번 결과는 유지
                conn.execute("DELETE FROM extract_cache WHERE key=?", (old[0],))
            conn.commit()
    except Exception:
        pass

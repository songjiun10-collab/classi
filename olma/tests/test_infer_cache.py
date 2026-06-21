"""기능 A: 결정론적(temperature=0) 추론 캐시.

infer_cache 모듈의 get/put/회전과, ollama_client.generate가 temperature=0일 때만
캐시를 적용하는지(동일 입력 재호출 시 Ollama를 건너뛰는지)를 검증한다. 외부 의존 없이
임시 DB + 카운팅 HTTP 더미 서버로만 돈다.
"""
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


def _fresh_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("INFER_CACHE", "true")
    monkeypatch.setenv("INFER_CACHE_PATH", str(tmp_path / "infer_cache.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.infer_cache as ic
    importlib.reload(ic)
    return ic


def test_make_key_is_stable_and_input_sensitive(tmp_path, monkeypatch):
    ic = _fresh_cache(tmp_path, monkeypatch)
    k1 = ic.make_key("m", "prompt", None, None)
    k2 = ic.make_key("m", "prompt", None, None)
    k3 = ic.make_key("m", "다른 prompt", None, None)
    assert k1 == k2          # 동일 입력 → 동일 키
    assert k1 != k3          # 입력이 바뀌면 키가 바뀐다
    assert ic.make_key("m", "p", {"type": "object"}, None) != ic.make_key("m", "p", None, None)


def test_get_put_roundtrip_and_miss(tmp_path, monkeypatch):
    ic = _fresh_cache(tmp_path, monkeypatch)
    assert ic.get("없는키") is None
    ic.put("k", "저장된 응답")
    assert ic.get("k") == "저장된 응답"


def test_put_rotates_oldest_beyond_max(tmp_path, monkeypatch):
    monkeypatch.setenv("INFER_CACHE_MAX_RECORDS", "2")
    ic = _fresh_cache(tmp_path, monkeypatch)
    ic.put("a", "1")
    ic.put("b", "2")
    ic.put("c", "3")          # a가 회전돼 나가야 한다
    assert ic.get("a") is None
    assert ic.get("b") == "2"
    assert ic.get("c") == "3"
    assert ic.stats()["entries"] == 2


# --- ollama_client 통합: 호출 횟수를 세는 더미 Ollama 서버 ---

class _CountingHandler(BaseHTTPRequestHandler):
    calls = 0

    def do_POST(self):
        _CountingHandler.calls += 1
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        body = json.dumps({"response": "응답"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def counting_ollama(tmp_path, monkeypatch):
    _CountingHandler.calls = 0
    server = HTTPServer(("127.0.0.1", 0), _CountingHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()

    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("INFER_CACHE", "true")
    monkeypatch.setenv("INFER_CACHE_PATH", str(tmp_path / "infer_cache.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.infer_cache as ic
    importlib.reload(ic)
    import llm.ollama_client as oc
    importlib.reload(oc)
    yield oc
    server.shutdown()


def test_temp0_caches_second_call_skips_ollama(counting_ollama):
    oc = counting_ollama
    r1 = oc.generate("같은 질문", temperature=0.0)
    r2 = oc.generate("같은 질문", temperature=0.0)
    assert r1 == r2 == "응답"
    assert _CountingHandler.calls == 1     # 두 번째는 캐시 히트 → Ollama 1회만 호출


def test_nonzero_temp_is_not_cached(counting_ollama):
    oc = counting_ollama
    oc.generate("같은 질문", temperature=0.7)
    oc.generate("같은 질문", temperature=0.7)
    assert _CountingHandler.calls == 2     # 비결정론 → 매번 호출


def test_cache_disabled_flag_bypasses_cache(tmp_path, monkeypatch):
    _CountingHandler.calls = 0
    server = HTTPServer(("127.0.0.1", 0), _CountingHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
        monkeypatch.setenv("INFER_CACHE", "false")
        monkeypatch.setenv("INFER_CACHE_PATH", str(tmp_path / "infer_cache.db"))
        import config.config as cfg
        importlib.reload(cfg)
        import core.infer_cache as ic
        importlib.reload(ic)
        import llm.ollama_client as oc
        importlib.reload(oc)

        oc.generate("같은 질문", temperature=0.0)
        oc.generate("같은 질문", temperature=0.0)
        assert _CountingHandler.calls == 2   # 캐시 꺼짐 → 매번 호출
    finally:
        server.shutdown()

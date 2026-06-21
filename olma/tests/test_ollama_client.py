import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest


class _RecordingHandler(BaseHTTPRequestHandler):
    captured = {}

    def do_POST(self):
        length = int(self.headers["Content-Length"])
        _RecordingHandler.captured = json.loads(self.rfile.read(length))
        body = json.dumps({"response": "ok"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


@pytest.fixture
def dummy_ollama_server(monkeypatch, tmp_path):
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
    # 이 테스트들은 실제 POST 페이로드를 검증하므로 캐시를 꺼야 한다 — 캐시가 켜져 있으면
    # temperature=0 호출이 (이전 실행이 남긴) 캐시 히트로 HTTP를 건너뛰어 captured가 갱신되지 않는다.
    monkeypatch.setenv("INFER_CACHE", "false")
    monkeypatch.setenv("USAGE_STORE_PATH", str(tmp_path / "usage.db"))
    import config.config as cfg
    importlib.reload(cfg)
    import core.infer_cache as ic
    importlib.reload(ic)
    import core.usage_store as us
    importlib.reload(us)
    import core.usage as usage
    importlib.reload(usage)
    import llm.ollama_client as oc
    importlib.reload(oc)

    yield oc
    server.shutdown()


def test_generate_sends_format_and_options(dummy_ollama_server):
    oc = dummy_ollama_server
    result = oc.generate("hello", format={"type": "object"}, temperature=0.0, seed=42)

    assert result == "ok"
    payload = _RecordingHandler.captured
    assert payload["format"] == {"type": "object"}
    assert payload["options"] == {"temperature": 0.0, "seed": 42}


def test_generate_omits_format_and_options_when_not_given(dummy_ollama_server):
    oc = dummy_ollama_server
    oc.generate("hello")

    payload = _RecordingHandler.captured
    assert "format" not in payload
    assert "options" not in payload


def test_generate_sends_images_when_given(dummy_ollama_server):
    oc = dummy_ollama_server
    oc.generate("이 이미지를 설명해라", images=["YmFzZTY0aW1n"])

    payload = _RecordingHandler.captured
    assert payload["images"] == ["YmFzZTY0aW1n"]


def test_generate_omits_images_when_not_given(dummy_ollama_server):
    oc = dummy_ollama_server
    oc.generate("hello")

    payload = _RecordingHandler.captured
    assert "images" not in payload


def test_generate_fails_over_to_backup_model(dummy_ollama_server):
    """1순위 모델이 실패하면 fallback_models의 백업 모델로 전환해 성공한다."""
    oc = dummy_ollama_server
    calls = {"models": []}
    real_single = oc._generate_single

    def flaky_single(prompt, model, *args, **kwargs):
        calls["models"].append(model)
        if model == "primary":
            raise RuntimeError("primary 죽음")
        return real_single(prompt, model, *args, **kwargs)

    with patch.object(oc, "_generate_single", side_effect=flaky_single):
        result = oc.generate("hi", model="primary", fallback_models=["backup"])

    assert result == "ok"
    assert calls["models"] == ["primary", "backup"]   # 1순위 실패 → 백업 시도


def test_generate_raises_when_all_models_fail(dummy_ollama_server):
    oc = dummy_ollama_server

    def always_fail(prompt, model, *args, **kwargs):
        raise RuntimeError(f"{model} 죽음")

    with patch.object(oc, "_generate_single", side_effect=always_fail):
        with pytest.raises(RuntimeError):
            oc.generate("hi", model="primary", fallback_models=["backup"])


def test_generate_dedupes_and_skips_empty_fallbacks(dummy_ollama_server):
    oc = dummy_ollama_server
    seen = []

    def rec(prompt, model, *args, **kwargs):
        seen.append(model)
        raise RuntimeError("fail")

    with patch.object(oc, "_generate_single", side_effect=rec):
        with pytest.raises(RuntimeError):
            oc.generate("hi", model="m", fallback_models=["m", "", "m2"])

    assert seen == ["m", "m2"]   # 중복("m")과 빈 값은 건너뜀

import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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
def dummy_ollama_server(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _RecordingHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
    import config.config as cfg
    importlib.reload(cfg)
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

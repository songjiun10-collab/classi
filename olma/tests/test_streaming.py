"""기능 C: 스트리밍 생성. ollama_client.generate_stream이 NDJSON 청크를 순서대로
yield하는지, executor가 on_token으로 청크를 흘리며 최종 결과를 정확히 누적하는지 검증한다.
외부 의존 없이 NDJSON 스트림을 내려주는 더미 Ollama 서버로 돈다.
"""
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

import pytest

_CHUNKS = ["안녕", "하세", "요"]


class _StreamHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers["Content-Length"])
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for i, piece in enumerate(_CHUNKS):
            line = json.dumps({"response": piece, "done": i == len(_CHUNKS) - 1}) + "\n"
            self.wfile.write(line.encode())
            self.wfile.flush()

    def log_message(self, *args):
        pass


@pytest.fixture
def stream_ollama(monkeypatch):
    server = HTTPServer(("127.0.0.1", 0), _StreamHandler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("OLLAMA_HOST", f"http://127.0.0.1:{port}")
    import config.config as cfg
    importlib.reload(cfg)
    import llm.ollama_client as oc
    importlib.reload(oc)
    yield oc
    server.shutdown()


def test_generate_stream_yields_chunks_in_order(stream_ollama):
    oc = stream_ollama
    got = list(oc.generate_stream("인사해줘", temperature=0.7))
    assert got == _CHUNKS
    assert "".join(got) == "안녕하세요"


def test_executor_on_token_streams_and_accumulates():
    """executor가 on_token으로 청크를 흘리면서 최종 result로 합쳐 기록하는지."""
    from executor import executor

    received = []
    steps = [{"action": "llm", "input": "인사", "depends_on": None}]

    def fake_stream(prompt, **kwargs):
        yield from ["부분1", "부분2"]

    with patch("llm.ollama_client.generate_stream", side_effect=fake_stream):
        results = executor.execute_steps(steps, on_token=received.append)

    assert received == ["부분1", "부분2"]              # 청크가 콜백으로 흘러감
    assert results[0]["result"] == "부분1부분2"          # 최종 결과는 누적된 전체
    assert results[0]["status"] == "ok"


def test_executor_without_on_token_uses_nonstreaming():
    """on_token이 없으면 generate(비스트리밍)를 쓴다 — generate_stream을 부르지 않아야."""
    from executor import executor

    steps = [{"action": "llm", "input": "안녕", "depends_on": None}]
    with patch("llm.ollama_client.generate", return_value="비스트리밍 응답") as m_gen, \
         patch("llm.ollama_client.generate_stream") as m_stream:
        results = executor.execute_steps(steps)

    m_stream.assert_not_called()
    m_gen.assert_called_once()
    assert results[0]["status"] == "ok"
    assert results[0]["result"] == "비스트리밍 응답"

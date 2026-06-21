"""Context Compression(core.compression) 테스트 — 예산 압축의 결정론적 동작."""
from core import compression


def test_short_text_unchanged():
    assert compression.compress("짧은 텍스트", 100) == "짧은 텍스트"


def test_long_text_keeps_head_and_tail():
    text = "HEAD" + ("x" * 1000) + "TAIL"
    out = compression.compress(text, 100)
    assert len(out) <= 100 + 30          # 마커 여유
    assert out.startswith("HEAD")
    assert out.endswith("TAIL")
    assert "생략" in out


def test_zero_budget_returns_empty():
    assert compression.compress("뭐든", 0) == ""
    assert compression.compress("뭐든", -5) == ""


def test_none_text_safe():
    assert compression.compress(None, 100) == ""


def test_head_ratio_respected():
    text = "A" * 500 + "B" * 500
    out = compression.compress(text, 200, head_ratio=0.9)
    # 앞쪽(A)을 더 많이 보존한다.
    assert out.count("A") > out.count("B")


def test_compress_lines_drops_oldest():
    lines = [f"line{i}" for i in range(20)]
    out = compression.compress_lines(lines, 30)
    assert "생략" in out
    assert "line19" in out               # 최신 줄은 보존
    assert "line0" not in out            # 오래된 줄은 버려짐


def test_compress_lines_under_budget_keeps_all():
    lines = ["a", "b", "c"]
    out = compression.compress_lines(lines, 100)
    assert out == "a\nb\nc"
    assert "생략" not in out


def test_compress_lines_empty():
    assert compression.compress_lines([], 100) == ""
    assert compression.compress_lines(["a"], 0) == ""

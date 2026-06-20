import base64
import importlib
from unittest.mock import patch


def _fresh_ocr(monkeypatch, ocr_vlm_model=""):
    monkeypatch.setenv("OCR_VLM_MODEL", ocr_vlm_model)
    import config.config as cfg
    importlib.reload(cfg)
    import tools.ocr as ocr
    importlib.reload(ocr)
    return ocr


def test_extract_text_returns_tesseract_result_without_calling_vlm(monkeypatch, tmp_path):
    ocr = _fresh_ocr(monkeypatch, ocr_vlm_model="qwen2.5vl:7b")
    image_path = str(tmp_path / "x.png")

    with patch.object(ocr, "image_to_text", return_value="이미지 안의 텍스트") as tesseract, patch(
        "tools.ocr.ollama_client.generate"
    ) as vlm:
        result = ocr.extract_text(image_path)

    assert result == "이미지 안의 텍스트"
    tesseract.assert_called_once_with(image_path, ocr.TESSERACT_LANG)
    vlm.assert_not_called()


def test_extract_text_disabled_when_vlm_model_unset(monkeypatch, tmp_path):
    ocr = _fresh_ocr(monkeypatch, ocr_vlm_model="")
    image_path = str(tmp_path / "x.png")

    with patch.object(ocr, "image_to_text", return_value="") as tesseract, patch(
        "tools.ocr.ollama_client.generate"
    ) as vlm:
        result = ocr.extract_text(image_path)

    assert result == ""
    tesseract.assert_called_once()
    vlm.assert_not_called()


def test_extract_text_falls_back_to_vlm_when_tesseract_empty(monkeypatch, tmp_path):
    ocr = _fresh_ocr(monkeypatch, ocr_vlm_model="qwen2.5vl:7b")
    image_path = tmp_path / "x.png"
    image_path.write_bytes(b"fake-png-bytes")

    with patch.object(ocr, "image_to_text", return_value="   "), patch(
        "tools.ocr.ollama_client.generate", return_value="VLM이 읽은 텍스트"
    ) as vlm:
        result = ocr.extract_text(str(image_path))

    assert result == "VLM이 읽은 텍스트"
    vlm.assert_called_once()
    _, kwargs = vlm.call_args
    assert kwargs["model"] == "qwen2.5vl:7b"
    assert kwargs["images"] == [base64.b64encode(b"fake-png-bytes").decode("ascii")]


def test_extract_text_swallows_vlm_failure_and_returns_tesseract_result(monkeypatch, tmp_path):
    ocr = _fresh_ocr(monkeypatch, ocr_vlm_model="qwen2.5vl:7b")
    image_path = tmp_path / "x.png"
    image_path.write_bytes(b"fake-png-bytes")

    with patch.object(ocr, "image_to_text", return_value=""), patch(
        "tools.ocr.ollama_client.generate", side_effect=RuntimeError("연결 실패")
    ):
        result = ocr.extract_text(str(image_path))

    assert result == ""

"""스크린샷 이미지에서 텍스트를 추출한다 (pytesseract, 필요 시 로컬 VLM 폴백)."""
import base64

import pytesseract
from PIL import Image

from config.config import OCR_VLM_MODEL, TESSERACT_LANG
from core.logger import get_logger
from llm import ollama_client

log = get_logger("ocr")

_VLM_PROMPT = (
    "이 이미지에 보이는 모든 텍스트를 있는 그대로 정확히 전사(transcribe)해라. "
    "설명이나 해석을 덧붙이지 말고 텍스트만 출력해라."
)


def image_to_text(image_path: str, lang: str = TESSERACT_LANG) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang=lang)


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def extract_text(image_path: str, lang: str = TESSERACT_LANG) -> str:
    """tesseract로 먼저 시도하고, 결과가 비어 있으면(스캔 품질 문제 등) 설정된 경우에만
    로컬 Ollama 비전 모델로 한 번 더 시도한다. OCR_VLM_MODEL이 비어 있으면(기본값)
    VLM 호출 자체를 시도하지 않아 기존 동작과 완전히 동일하다."""
    text = image_to_text(image_path, lang)
    if text.strip() or not OCR_VLM_MODEL:
        return text

    log.info("tesseract 결과가 비어 있어 VLM OCR로 폴백: %s", OCR_VLM_MODEL)
    try:
        return ollama_client.generate(
            _VLM_PROMPT, model=OCR_VLM_MODEL, images=[_encode_image(image_path)]
        )
    except Exception as exc:
        log.warning("VLM OCR 폴백 실패, tesseract 결과를 그대로 반환: %s", exc)
        return text

"""스크린샷 이미지에서 텍스트를 추출한다 (pytesseract)."""
import pytesseract
from PIL import Image

from config.config import TESSERACT_LANG


def image_to_text(image_path: str, lang: str = TESSERACT_LANG) -> str:
    image = Image.open(image_path)
    return pytesseract.image_to_string(image, lang=lang)

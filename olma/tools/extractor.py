"""OCR로 추출한 원시 텍스트를 개별 메시지 단위로 분리한다."""


def extract_messages(raw_text: str) -> list:
    lines = [line.strip() for line in raw_text.splitlines()]
    return [line for line in lines if line]

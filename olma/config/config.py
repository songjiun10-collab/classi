"""Olma 전역 설정. 모든 값은 환경변수로 오버라이드 가능."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
OLLAMA_TEMPERATURE_DEFAULT = float(os.environ.get("OLLAMA_TEMPERATURE_DEFAULT", "0.7"))

# --- Memory ---
MEMORY_PATH = os.environ.get("MEMORY_PATH", str(BASE_DIR / "storage" / "memory.json"))

# --- Browser ---
BROWSER_HEADLESS = os.environ.get("BROWSER_HEADLESS", "false").lower() == "true"
BROWSER_USER_DATA_DIR = os.environ.get(
    "BROWSER_USER_DATA_DIR", str(BASE_DIR / "storage" / "browser_profile")
)
SCREENSHOT_DIR = os.environ.get("SCREENSHOT_DIR", str(BASE_DIR / "storage" / "screenshots"))

# --- Notification (카톡 등 읽기 전용) ---
# 정확한 카카오톡 웹 클라이언트 URL은 임의로 추측하지 않음.
# 사용자가 직접 사용할 페이지의 URL을 환경변수로 지정해야 함.
KAKAO_WEB_URL = os.environ.get("KAKAO_WEB_URL", "")

# --- Web AI (브라우저로 웹 AI 채팅을 열어 묻기. 공식 API 미사용, 카톡과 동일한 패턴) ---
# 어떤 웹 AI를 쓸지/입력창·전송·응답 영역 selector는 사이트마다 달라 임의로 추측하지 않음.
# 사용자가 직접 환경변수로 지정해야 동작한다(미설정 시 명확한 오류).
WEB_AI_URL = os.environ.get("WEB_AI_URL", "")
WEB_AI_INPUT_SELECTOR = os.environ.get("WEB_AI_INPUT_SELECTOR", "")  # 프롬프트 입력창 CSS 선택자
WEB_AI_SUBMIT_SELECTOR = os.environ.get("WEB_AI_SUBMIT_SELECTOR", "")  # 전송 버튼(비우면 Enter)
WEB_AI_RESPONSE_SELECTOR = os.environ.get("WEB_AI_RESPONSE_SELECTOR", "body")  # 응답 영역
WEB_AI_WAIT_MS = int(os.environ.get("WEB_AI_WAIT_MS", "8000"))  # 응답 생성 대기(ms)

# --- OCR ---
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "kor+eng")

# --- Executor ---
RETRY_COUNT = int(os.environ.get("RETRY_COUNT", "2"))  # step당 추가 재시도 횟수 (1~3 권장)
RETRY_BACKOFF = float(os.environ.get("RETRY_BACKOFF", "0.5"))  # 재시도 사이 대기(초), 시도마다 *2
STEP_TIMEOUT = int(os.environ.get("STEP_TIMEOUT", "60"))  # step 1회 실행 제한시간(초)
BROWSER_TIMEOUT = int(os.environ.get("BROWSER_TIMEOUT", "15000"))  # Playwright 동작 제한시간(ms)

# --- Logging ---
LOG_PATH = os.environ.get("LOG_PATH", str(BASE_DIR / "storage" / "olma.log"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

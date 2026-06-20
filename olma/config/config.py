"""Olma 전역 설정. 모든 값은 환경변수로 오버라이드 가능."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
OLLAMA_TEMPERATURE_DEFAULT = float(os.environ.get("OLLAMA_TEMPERATURE_DEFAULT", "0.7"))

# --- Memory (SQLite, storage/memory.db) ---
MEMORY_PATH = os.environ.get("MEMORY_PATH", str(BASE_DIR / "storage" / "memory.db"))
# 무한정 누적 방지: 초과분은 오래된 레코드부터 버린다(가장 단순한 회전 정책).
MEMORY_MAX_RECORDS = int(os.environ.get("MEMORY_MAX_RECORDS", "1000"))

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
# 응답 안정화 대기 상한(ms). 고정 대기가 아니라 "이 시간 안에 응답 텍스트 길이가
# 더 늘지 않으면 끝난 것으로 본다"의 상한값 — 빠른 모델은 더 일찍 반환되고,
# 느린 모델(예: 20초대 응답)은 이 상한까지 기다린다(tools/browser._wait_for_response_stable).
WEB_AI_WAIT_MS = int(os.environ.get("WEB_AI_WAIT_MS", "30000"))
# 응답 안정화 여부를 확인하는 폴링 간격(ms)과, 몇 번 연속 길이 변화가 없어야
# "스트리밍 종료"로 볼지(노이즈로 인한 1회성 미변화를 안정으로 오판하지 않기 위함).
WEB_AI_POLL_INTERVAL_MS = int(os.environ.get("WEB_AI_POLL_INTERVAL_MS", "1000"))
WEB_AI_STABLE_POLLS = int(os.environ.get("WEB_AI_STABLE_POLLS", "2"))
# 기본값(false)에서는 사용자가 명시적으로 요청했을 때만 web_ai_ask를 쓴다.
# true로 켜면, Planner가 로컬 LLM 능력을 넘는 고난도 작업이라고 판단할 때도
# (명시적 요청 없이) web_ai_ask를 선택할 수 있다. WEB_AI_URL 등이 미설정이면
# 어차피 실행 시 실패하고 라우터 폴백으로 로컬 Ollama가 받는다.
WEB_AI_AUTO_ESCALATE = os.environ.get("WEB_AI_AUTO_ESCALATE", "false").strip().lower() in (
    "1", "true", "yes",
)

# --- OCR ---
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "kor+eng")
# 비워두면(기본) VLM 폴백 비활성. tesseract가 빈 문자열을 반환할 때만(스캔 품질 문제 등)
# 이 Ollama 비전 모델로 한 번 더 시도한다(예: "qwen2.5vl:7b", "llava:13b").
# 로컬 Ollama 모델명이므로 외부 클라우드 호출을 추가하지 않는다.
OCR_VLM_MODEL = os.environ.get("OCR_VLM_MODEL", "")

# --- Executor ---
RETRY_COUNT = int(os.environ.get("RETRY_COUNT", "2"))  # step당 추가 재시도 횟수 (1~3 권장)
RETRY_BACKOFF = float(os.environ.get("RETRY_BACKOFF", "0.5"))  # 재시도 사이 대기(초), 시도마다 *2
STEP_TIMEOUT = int(os.environ.get("STEP_TIMEOUT", "60"))  # step 1회 실행 제한시간(초)
BROWSER_TIMEOUT = int(os.environ.get("BROWSER_TIMEOUT", "15000"))  # Playwright 동작 제한시간(ms)

# --- Logging ---
LOG_PATH = os.environ.get("LOG_PATH", str(BASE_DIR / "storage" / "olma.log"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")

# --- API ---
OLMA_API_HOST = os.environ.get("OLMA_API_HOST", "0.0.0.0")
OLMA_API_PORT = int(os.environ.get("OLMA_API_PORT", "8800"))
# 비워두면 인증 없음(로컬 단일 사용자 전제). 네트워크로 노출할 때는 반드시 설정할 것 —
# Olma는 읽기 전용 도구가 아니라 로그인된 브라우저 세션으로 클릭/입력까지 하는
# action-taking 에이전트라, 인증 없는 네트워크 노출은 단순 정보 유출보다 위험하다.
OLMA_API_KEY = os.environ.get("OLMA_API_KEY", "")

# --- Task Queue ---
TASK_QUEUE_MAX_TASKS = int(os.environ.get("TASK_QUEUE_MAX_TASKS", "200"))
# 기본 1 = 기존과 동일한 완전 직렬 처리. 1보다 크게 설정하면 워커마다 독립된 브라우저
# 프로필 디렉터리(첫 실행 시 기존 프로필을 복사해 로그인 세션을 물려받음)를 써서
# Playwright의 launch_persistent_context 프로필 잠금 충돌 없이 동시 실행한다.
TASK_QUEUE_WORKERS = int(os.environ.get("TASK_QUEUE_WORKERS", "1"))

# --- Task Store (SQLite, task 기록 영속화 — 재시작 시 히스토리 보존 목적.
# 처리 중이던 task는 안전하게 재개할 수 없으므로 재시작 시 failed로 전환된다) ---
TASK_STORE_PATH = os.environ.get("TASK_STORE_PATH", str(BASE_DIR / "storage" / "tasks.db"))

# --- Web AI Providers (다중 제공자 레지스트리. 비워두면 위 WEB_AI_* 단일 설정만
# "default" provider로 쓴다. 여러 웹 AI를 쓰려면 JSON 파일 경로를 지정한다 —
# 사이트별 URL/selector는 자주 바뀌고 임의로 추측할 수 없으므로 코드에 박지 않는다) ---
WEB_AI_PROVIDERS_PATH = os.environ.get("WEB_AI_PROVIDERS_PATH", "")

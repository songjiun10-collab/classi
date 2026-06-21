"""Olma 전역 설정. 모든 값은 환경변수로 오버라이드 가능."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Ollama ---
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:latest")
OLLAMA_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "60"))
OLLAMA_TEMPERATURE_DEFAULT = float(os.environ.get("OLLAMA_TEMPERATURE_DEFAULT", "0.7"))

# --- AI 역할별 모델 다중화 (core/ai_roles.py) ---
# 한 모델이 계획·대화·요약을 다 떠맡는 대신, 역할별로 다른 Ollama 모델을 쓸 수 있다.
# 비워두면 모두 OLLAMA_MODEL로 폴백하므로 기존 단일 모델 동작과 동일하다.
# - PLANNER: 계획 수립용(JSON 강제·결정론). 작고 빠른 모델로 충분한 경우가 많다.
# - REASONING: 외부 웹 AI가 받는 고난도 작업의 '로컬 폴백'용(웹 AI 실패 시). 가장 강한 로컬 모델 권장.
# - VLM: 이미지/화면 이해용 비전 모델(없으면 OCR_VLM_MODEL을 따라간다).
OLLAMA_PLANNER_MODEL = os.environ.get("OLLAMA_PLANNER_MODEL", "") or OLLAMA_MODEL
OLLAMA_REASONING_MODEL = os.environ.get("OLLAMA_REASONING_MODEL", "") or OLLAMA_MODEL
# 텍스트 역할(plan/chat/summarize/reason)의 백업 모델. 역할 모델이 실패하면(모델 없음·OOM·
# 오류) 이 모델로 페일오버한다. 비우면 백업 없음(단일 모델 동작 = 기존과 동일).
OLLAMA_BACKUP_MODEL = os.environ.get("OLLAMA_BACKUP_MODEL", "")

# --- Inference Cache (SQLite, storage/infer_cache.db) ---
# temperature=0(결정론적) 호출만 캐시한다 — 같은 입력은 Ollama 재호출 없이 저장 응답 반환.
# 플래너 호출(temperature=0.0)이 정확히 이 경우라 반복 작업의 재추론을 0회로 만든다.
# temperature>0(기본 llm/summarize)은 비결정론적이라 캐시하지 않는다. 끄려면 INFER_CACHE=0.
INFER_CACHE = os.environ.get("INFER_CACHE", "true").strip().lower() in ("1", "true", "yes")
INFER_CACHE_PATH = os.environ.get(
    "INFER_CACHE_PATH", str(BASE_DIR / "storage" / "infer_cache.db")
)
INFER_CACHE_MAX_RECORDS = int(os.environ.get("INFER_CACHE_MAX_RECORDS", "2000"))

# --- Memory (SQLite, storage/memory.db) ---
MEMORY_PATH = os.environ.get("MEMORY_PATH", str(BASE_DIR / "storage" / "memory.db"))
# 무한정 누적 방지: 초과분은 오래된 레코드부터 버린다(가장 단순한 회전 정책).
MEMORY_MAX_RECORDS = int(os.environ.get("MEMORY_MAX_RECORDS", "1000"))

# --- Long-term Memory (세션을 넘는 사실/선호 저장. 작업 이력(memory.py)과 별개) ---
LTM_PATH = os.environ.get("LTM_PATH", str(BASE_DIR / "storage" / "ltm.db"))
# 계획 시 플래너에 주입할 관련 사실 최대 개수(0이면 주입 안 함).
LTM_CONTEXT_LIMIT = int(os.environ.get("LTM_CONTEXT_LIMIT", "5"))

# --- Context Compression (프롬프트에 주입되는 맥락/사실 블록의 길이 예산) ---
# 플래너에 넣는 '최근 작업 맥락'과 '장기 기억' 블록 각각의 최대 글자 수. 초과하면
# 앞·뒤를 살리고 가운데를 생략 표시로 압축한다(프롬프트 비대화·토큰 낭비 방지).
CONTEXT_MAX_CHARS = int(os.environ.get("CONTEXT_MAX_CHARS", "1500"))

# --- Browser ---
BROWSER_HEADLESS = os.environ.get("BROWSER_HEADLESS", "false").lower() == "true"
BROWSER_USER_DATA_DIR = os.environ.get(
    "BROWSER_USER_DATA_DIR", str(BASE_DIR / "storage" / "browser_profile")
)
SCREENSHOT_DIR = os.environ.get("SCREENSHOT_DIR", str(BASE_DIR / "storage" / "screenshots"))

# 봇 탐지/캡차 완화(stealth). 자동화 브라우저는 navigator.webdriver=true·헤드리스·기본 UA로
# 쉽게 탐지돼 캡차가 걸린다. 아래로 흔한 탐지 신호를 줄인다(완전 회피는 불가능 — 가장 확실한
# 건 headed + 로그인된 영구 프로필 사용). BROWSER_STEALTH=false로 끌 수 있다.
BROWSER_STEALTH = os.environ.get("BROWSER_STEALTH", "true").strip().lower() not in ("0", "false", "no")
# 기본 UA를 실제 크롬처럼. 비우면 Playwright 기본 UA(HeadlessChrome 표식 포함 가능)를 쓴다.
BROWSER_USER_AGENT = os.environ.get(
    "BROWSER_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
)
BROWSER_LOCALE = os.environ.get("BROWSER_LOCALE", "ko-KR")
# 실제 설치된 Chrome으로 띄우면(Playwright 번들 Chromium 대신) Cloudflare의 패시브
# 핑거프린트(TLS/JA3·navigator·client-hints) 검사를 통과할 확률이 크게 오른다. 채널 사용 시
# UA를 강제하지 않는다(실제 Chrome UA와 일치해야 함). 빈 값이면 번들 Chromium으로 폴백.
BROWSER_CHANNEL = os.environ.get("BROWSER_CHANNEL", "chrome").strip()
# Cloudflare "Just a moment…" 같은 자동(관리형) 챌린지가 스스로 풀릴 때까지 기다리는 상한(ms).
# headed + 영구 프로필이면 보통 수 초 내 통과하고 cf_clearance 쿠키가 남아 다음부턴 건너뛴다.
CLOUDFLARE_WAIT_MS = int(os.environ.get("CLOUDFLARE_WAIT_MS", "25000"))

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

# 간단한 질문(작업성 신호가 없는 단순 질의·대화)은 planner LLM 호출을 건너뛰고 단일 로컬 LLM
# 답변으로 바로 처리한다(체감 응답성↑, 비용↓). 대화형 진입점(CLI·작업 큐)에만 적용되고,
# 예약/이벤트/레시피 워크플로는 의도된 작업이라 그대로 계획을 거친다. 끄면 모든 입력이 계획을
# 탄다. 판별은 core/triage.py(휴리스틱, LLM 없음)가 한다.
FAST_CHAT = os.environ.get("OLMA_FAST_CHAT", "true").strip().lower() not in ("0", "false", "no")

# --- Login (구글 세션을 타는 재로그인. 비번을 저장하지 않는다) ---
# 영구 프로필에 구글 로그인 세션이 살아 있으면, 대부분의 사이트는 "구글로 로그인" 버튼
# 한 번으로 재인증된다. 비번을 코드/로그/설정 어디에도 저장하지 않고 그 세션을 타는 방식.
# 사이트별 로그인 URL·확인 selector는 web_ai_providers와 동일하게 코드에 박지 않고
# 환경변수/JSON으로 등록한다(여러 사이트는 LOGIN_PROVIDERS_PATH JSON 사용).
LOGIN_PROVIDERS_PATH = os.environ.get("LOGIN_PROVIDERS_PATH", "")
# 단일 사이트용 레거시 기본 제공자(이름 "default").
LOGIN_URL = os.environ.get("LOGIN_URL", "")  # 로그인 페이지 URL
# 로그인됐을 때만 보이는 요소의 selector — 이미 로그인 상태인지/재로그인 성공인지 판별.
LOGIN_LOGGED_IN_SELECTOR = os.environ.get("LOGIN_LOGGED_IN_SELECTOR", "")
# "구글로 로그인" 버튼 selector(비우면 login_providers의 공통 기본 후보 사용).
LOGIN_GOOGLE_BUTTON_SELECTOR = os.environ.get("LOGIN_GOOGLE_BUTTON_SELECTOR", "")
# 로그인 벽(로그인 안 됐을 때만 뜨는 요소) selector — 자동 재로그인 트리거 감지에 쓴다.
# 비우면 해당 제공자는 자동 재로그인 대상에서 제외된다(오탐 방지).
LOGIN_WALL_SELECTOR = os.environ.get("LOGIN_WALL_SELECTOR", "")
LOGIN_WAIT_MS = int(os.environ.get("LOGIN_WAIT_MS", "30000"))  # 재로그인 후 확인 대기 상한(ms)
LOGIN_CHECK_MS = int(os.environ.get("LOGIN_CHECK_MS", "3000"))  # 로그인 여부 즉시 확인 대기(ms)
# AUTO_RELOGIN=true → 브라우저 step이 실패하고 현재 페이지에 로그인 벽이 감지되면
# default 제공자로 자동 재로그인 후 그 step을 1회 재시도한다. 기본 on이지만, default
# 제공자에 LOGIN_WALL_SELECTOR가 설정돼 있어야만 실제로 동작한다(미설정이면 no-op이라
# 켜둬도 안전). 끄려면 AUTO_RELOGIN=0.
AUTO_RELOGIN = os.environ.get("AUTO_RELOGIN", "true").strip().lower() in ("1", "true", "yes")

# --- OCR ---
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "kor+eng")
# 비워두면(기본) VLM 폴백 비활성. tesseract가 빈 문자열을 반환할 때만(스캔 품질 문제 등)
# 이 Ollama 비전 모델로 한 번 더 시도한다(예: "qwen2.5vl:7b", "llava:13b").
# 로컬 Ollama 모델명이므로 외부 클라우드 호출을 추가하지 않는다.
OCR_VLM_MODEL = os.environ.get("OCR_VLM_MODEL", "")
# 화면/이미지 '이해'(vision_describe 액션)용 비전 모델. 비우면 OCR_VLM_MODEL을 따른다 —
# OCR 전사와 화면 이해는 같은 VLM을 써도 무방하므로 한 모델만 받아도 둘 다 동작한다.
OLLAMA_VLM_MODEL = os.environ.get("OLLAMA_VLM_MODEL", "") or OCR_VLM_MODEL
# 비전 역할의 백업 비전 모델(VLM이 실패할 때). 텍스트 모델로는 폴백하지 않는다(이미지 이해 불가).
OLLAMA_VLM_BACKUP_MODEL = os.environ.get("OLLAMA_VLM_BACKUP_MODEL", "")

# --- Executor ---
RETRY_COUNT = int(os.environ.get("RETRY_COUNT", "2"))  # step당 추가 재시도 횟수 (1~3 권장)
RETRY_BACKOFF = float(os.environ.get("RETRY_BACKOFF", "0.5"))  # 재시도 사이 대기(초), 시도마다 *2
STEP_TIMEOUT = int(os.environ.get("STEP_TIMEOUT", "60"))  # step 1회 실행 제한시간(초)
BROWSER_TIMEOUT = int(os.environ.get("BROWSER_TIMEOUT", "15000"))  # Playwright 동작 제한시간(ms)
# 의존성 없는(depends_on=null) ollama step들을 동시에 실행해 멀티스텝 지연을 줄인다.
# 기본 on. 단일 GPU Ollama는 추론을 직렬화하는 경우가 많아 병렬 이득이 작거나 VRAM
# 경합/OOM을 부를 수 있으니, 그런 환경이면 OLLAMA_PARALLEL=0으로 끈다. 브라우저 step은
# stateful이라 항상 순차 실행(병렬 대상 아님)이며, 스트리밍 출력(on_token) 사용 시에도
# 출력 순서를 위해 병렬을 자동으로 끈다.
OLLAMA_PARALLEL = os.environ.get("OLLAMA_PARALLEL", "true").strip().lower() in (
    "1", "true", "yes",
)
OLLAMA_PARALLEL_MAX = int(os.environ.get("OLLAMA_PARALLEL_MAX", "4"))  # 동시 ollama step 상한

# --- Logging ---
LOG_PATH = os.environ.get("LOG_PATH", str(BASE_DIR / "storage" / "olma.log"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
# LOG_JSON=true → 로그를 한 줄 JSON으로 출력(로그 수집/관측 도구 연동용). 기본은 사람이 읽는 텍스트.
LOG_JSON = os.environ.get("LOG_JSON", "false").strip().lower() in ("1", "true", "yes")

# --- API ---
# 기본은 루프백(127.0.0.1)만 바인드한다 — Olma는 로그인 세션으로 클릭/입력까지 하는
# action-taking 에이전트라, 인증 기본값이 꺼져 있는 상태(OLMA_API_KEY="")에서 0.0.0.0으로
# 모든 인터페이스에 노출하면 같은 네트워크의 누구나 인증 없이 작업을 던질 수 있다.
# LAN/외부에 노출하려면 OLMA_API_HOST=0.0.0.0을 명시적으로 지정하고, 그때는 반드시
# OLMA_API_KEY도 함께 설정할 것.
OLMA_API_HOST = os.environ.get("OLMA_API_HOST", "127.0.0.1")
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
# TASK_QUEUE_RESUME=true → 재시작 시 중단된(queued/processing) task를 failed로 버리지 않고
# 다시 큐에 넣어 처음부터 재실행한다. mid-step 재개가 아니라 원래 입력으로 plan→execute를
# 다시 도는 "재시도형 재개"다(브라우저 세션은 새로 뜬다). 부작용이 있는 작업엔 주의.
# 기본(false)은 종전대로 중단 task를 failed로 정리한다.
TASK_QUEUE_RESUME = os.environ.get("TASK_QUEUE_RESUME", "false").strip().lower() in (
    "1", "true", "yes",
)

# --- Task Store (SQLite, task 기록 영속화 — 재시작 시 히스토리 보존 목적.
# 처리 중이던 task는 안전하게 재개할 수 없으므로 재시작 시 failed로 전환된다) ---
TASK_STORE_PATH = os.environ.get("TASK_STORE_PATH", str(BASE_DIR / "storage" / "tasks.db"))

# --- Workflow Store (SQLite, 재사용 가능한 워크플로우 템플릿 영속화) ---
WORKFLOW_STORE_PATH = os.environ.get(
    "WORKFLOW_STORE_PATH", str(BASE_DIR / "storage" / "workflows.db")
)

# --- Task Recovery (작업이 실패(예: 일시적 모델 다운)하면 자동 재큐잉) ---
# true면 실패한 작업을 최대 TASK_RECOVERY_MAX회까지 낮은 우선순위로 다시 큐에 넣어 재실행한다.
TASK_AUTO_RECOVERY = os.environ.get("TASK_AUTO_RECOVERY", "false").lower() == "true"
TASK_RECOVERY_MAX = int(os.environ.get("TASK_RECOVERY_MAX", "1"))

# --- Usage / Quota (모델별 호출·토큰 사용량 기록 + 일일 한도) ---
# 사용자 정의 스킬(자주 쓰는 작업의 이름표) 저장소.
SKILL_STORE_PATH = os.environ.get("SKILL_STORE_PATH", str(BASE_DIR / "storage" / "skills.db"))

USAGE_STORE_PATH = os.environ.get("USAGE_STORE_PATH", str(BASE_DIR / "storage" / "usage.db"))
# 모델 1개당 하루 호출 횟수 상한(0이면 무제한). 초과하면 그 모델 호출이 거부돼 백업 모델로 페일오버.
USAGE_DAILY_CALL_LIMIT = int(os.environ.get("USAGE_DAILY_CALL_LIMIT", "0"))
# 모델 1개당 하루 토큰(프롬프트+생성) 상한(0이면 무제한).
USAGE_DAILY_TOKEN_LIMIT = int(os.environ.get("USAGE_DAILY_TOKEN_LIMIT", "0"))

# --- Accounts (사용자 계정 = 파이프라인의 출발점. 단일 사용자 전제지만 다중 사용자 대비) ---
ACCOUNT_STORE_PATH = os.environ.get(
    "ACCOUNT_STORE_PATH", str(BASE_DIR / "storage" / "accounts.db")
)
# 계정을 지정하지 않은 요청이 귀속될 기본 계정 id(첫 사용 시 자동 생성).
DEFAULT_ACCOUNT = os.environ.get("DEFAULT_ACCOUNT", "local")

# --- Workflow Planner (요청을 보고 저장된 템플릿 재사용 vs 동적 생성 결정) ---
# true면 작업 처리 시 요청과 충분히 일치하는 '파라미터 없는' 저장 템플릿이 있으면 그것을
# 재사용하고, 없으면 평소대로 동적 생성한다. 기본 false(항상 동적 — 기존 동작 보존).
WORKFLOW_AUTO_REUSE = os.environ.get("WORKFLOW_AUTO_REUSE", "false").lower() == "true"
# 템플릿 자동 재사용 최소 일치 점수(0~1). 템플릿 이름/설명 토큰이 요청에 이 비율 이상 포함돼야.
WORKFLOW_REUSE_THRESHOLD = float(os.environ.get("WORKFLOW_REUSE_THRESHOLD", "0.6"))

# --- Scheduler (워크플로우 주기 실행) ---
SCHEDULER_STORE_PATH = os.environ.get(
    "SCHEDULER_STORE_PATH", str(BASE_DIR / "storage" / "schedules.db")
)
SCHEDULER_POLL_SECONDS = int(os.environ.get("SCHEDULER_POLL_SECONDS", "30"))  # due 점검 주기(초)
# 스케줄러가 워크플로우를 직접 실행할 때 쓸 브라우저 프로필 슬롯. API 큐 워커(0)와 다른
# 값으로 격리해 영구 프로필 잠금 충돌을 피한다(tools/browser.resolve_profile_dir).
SCHEDULER_PROFILE_SLOT = int(os.environ.get("SCHEDULER_PROFILE_SLOT", "100"))

# --- Event Triggers (조건이 충족되면 워크플로우 실행 — 시간이 아니라 "사건" 기반) ---
EVENT_STORE_PATH = os.environ.get(
    "EVENT_STORE_PATH", str(BASE_DIR / "storage" / "events.db")
)
EVENT_POLL_SECONDS = int(os.environ.get("EVENT_POLL_SECONDS", "15"))  # 이벤트 점검 주기(초)

# --- Human Approval Gate (위험·비가역 액션 실행 전 사람 승인 요구) ---
APPROVAL_STORE_PATH = os.environ.get(
    "APPROVAL_STORE_PATH", str(BASE_DIR / "storage" / "approvals.db")
)
# 마스터 스위치. 끄면(false) 어떤 액션도 승인 없이 바로 실행한다(기존 동작과 동일).
APPROVAL_GATE = os.environ.get("APPROVAL_GATE", "false").lower() == "true"
# 기본 위험도 외에 강제로 승인을 요구할 액션 목록(쉼표 구분). 예: "web_ai_ask,login".
APPROVAL_REQUIRED_ACTIONS = os.environ.get("APPROVAL_REQUIRED_ACTIONS", "")
# 승인 대기 step이 거절·만료 없이 기다리는 상한(초). 초과하면 거절로 처리해 큐를 막지 않는다.
APPROVAL_TIMEOUT_SECONDS = int(os.environ.get("APPROVAL_TIMEOUT_SECONDS", "3600"))

# --- Web AI Providers (다중 제공자 레지스트리. 비워두면 위 WEB_AI_* 단일 설정만
# "default" provider로 쓴다. 여러 웹 AI를 쓰려면 JSON 파일 경로를 지정한다 —
# 사이트별 URL/selector는 자주 바뀌고 임의로 추측할 수 없으므로 코드에 박지 않는다) ---
WEB_AI_PROVIDERS_PATH = os.environ.get("WEB_AI_PROVIDERS_PATH", "")
# default 제공자의 백업(페일오버) 제공자 이름. 예: 기본 웹 AI가 막히면 다른 웹 AI로 넘긴다.
# JSON 제공자는 각 항목의 "backup" 필드로 지정한다(예: claude → zai).
WEB_AI_BACKUP = os.environ.get("WEB_AI_BACKUP", "")
# 무료 웹 AI 프리셋(chatgpt/claude/gemini/perplexity)을 기본 등록할지. 진입 URL은 공개
# 정보지만 입력/응답 selector는 범용 후보라 사이트 변경 시 깨질 수 있다(정확한 값은
# WEB_AI_PROVIDERS_PATH JSON으로 덮어쓴다). specialties가 없어 명시적으로 고를 때만 쓰인다.
WEB_AI_PRESETS = os.environ.get("WEB_AI_PRESETS", "true").strip().lower() not in ("0", "false", "no")

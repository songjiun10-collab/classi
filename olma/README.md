# Olma V0.2

로컬 AI(Ollama) + 브라우저(Playwright) + OCR을 통합한 작업 실행 시스템.
사용자의 자연어 요청을 계획(Planner) → 실행 방식 결정(Router) → 로컬 LLM/브라우저 실행(Executor) → 상태 기억(Memory) 순서로 처리한다.

외부 클라우드 AI API는 사용하지 않으며, 모든 추론은 로컬 Ollama로 수행한다. 카톡 등 메신저는 읽기 전용으로만 다루며, 분류·채널 추천만 하고 자동 전송은 하지 않는다.

## V0.2 — "돌아가는 코드"에서 "안 죽는 시스템"으로

V0.2는 기능 추가가 아니라 **안정성/반복 성공률**을 위한 구조 강화다.

- **Planner = 제어 시스템**: 스키마 검증 + Ollama `format` 강제 + 1회 자기-교정 재시도 + 부분 step 복구. 그래도 실패하면 LLM을 전혀 쓰지 않는 **규칙 기반 폴백 플래너**(`core/fallback_planner.py`)가 키워드/URL로 의미 있는 step을 만든다(통째 폐기 안 함).
- **Executor = 내결함성 시스템**: step당 재시도(backoff) → 그래도 실패하면 **대체 타겟 폴백**(브라우저 실패를 LLM이 아는 선에서 받음) → 그래도 안 되면 실패로 기록하고 다음 step 계속. 모든 단계가 `storage/olma.log`에 구조적으로 기록된다.
- **Memory = 상태 시스템**: SQLite(`storage/memory.db`)에 `{task, status(done/failed/partial), steps[], timestamp}` 단위로 저장하고, `get_context()`로 최근 작업 맥락을 다음 계획에 다시 넣는다(context 재사용). 키워드 검색(`find()`)은 대소문자를 구분하지 않고 task 텍스트뿐 아니라 각 step의 action/result까지 검색한다.
- **Router = 정책 엔진**: 결정론적 action→타겟 매핑 위에 **대체 타겟(fallback chain)**과 입력 완결성 기반 **confidence**(휴리스틱)를 얹는다.
- **Browser = 안정화 레이어**: 명시적 wait 전략, **selector fallback**(후보 여러 개 순차 시도), DOM 텍스트가 비면 **OCR 폴백**, 실패 시 디버그 스크린샷. 배치 중간에 페이지/컨텍스트가 죽어도(탭이 닫히거나 크래시) `_ensure_page()`가 다음 step 실행 전에 자동 복구한다(page만 죽었으면 새 page만, context까지 죽었으면 전체 재시작).
- **Task Queue = 영속 큐**: 처리된 task는 SQLite(`storage/tasks.db`)에도 기록되어 재시작해도 히스토리가 남는다. 단, 재개(resume)는 아니다 — 재시작 시점에 `queued`/`processing`이던 task는 브라우저 세션·워커 스레드가 이미 사라져 안전하게 이어갈 수 없으므로 `failed`로 정리된다.

> timeout은 신호(signal)로 강제 종료하지 않고 I/O 계층(Ollama 요청 timeout, Playwright 동작 timeout)에서 적용한다 — sync Playwright를 강제 중단하면 브라우저 상태가 깨지기 때문.

## 사전 준비

1. **Ollama**: 로컬에 설치 후 모델 다운로드
   ```bash
   ollama pull qwen2.5:7b
   ollama serve   # 기본적으로 http://localhost:11434 에서 대기
   ```
2. **Python 의존성**
   ```bash
   cd olma
   pip install -r requirements.txt
   playwright install chromium
   ```
3. **OCR (Tesseract)**: 시스템 패키지로 설치 필요 (pip만으로는 부족)
   ```bash
   # Debian/Ubuntu 예시
   sudo apt-get install tesseract-ocr tesseract-ocr-kor
   ```

## 실행

```bash
cd olma
python main.py
```

`exit` 또는 `quit` 입력 시 종료된다.

## 환경변수 (config/config.py)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 사용할 모델 |
| `MEMORY_PATH` | `storage/memory.db` | task 기록(상태/step/맥락)을 저장할 SQLite 파일 경로 |
| `MEMORY_MAX_RECORDS` | `1000` | 보관할 최대 기록 수(초과분은 오래된 것부터 자동 삭제) |
| `BROWSER_HEADLESS` | `false` | 브라우저 헤드리스 여부 |
| `BROWSER_USER_DATA_DIR` | `storage/browser_profile` | 로그인 세션 유지를 위한 영구 프로필 경로 |
| `KAKAO_WEB_URL` | (없음) | 알림 분석 기능에서 열 메신저 웹 페이지 URL. 직접 지정 필요 |
| `WEB_AI_URL` | (없음) | `web_ai_ask`에서 열 웹 AI 채팅 페이지 URL. 직접 지정 필요 |
| `WEB_AI_INPUT_SELECTOR` | (없음) | 웹 AI 프롬프트 입력창 CSS 선택자 |
| `WEB_AI_SUBMIT_SELECTOR` | (없음) | 전송 버튼 선택자 (비우면 Enter 키 입력) |
| `WEB_AI_RESPONSE_SELECTOR` | `body` | 응답 텍스트를 읽을 영역 선택자 |
| `WEB_AI_WAIT_MS` | `30000` | 프롬프트 전송 후 응답 안정화 대기 상한(ms) |
| `WEB_AI_AUTO_ESCALATE` | `false` | `true`로 켜면 Planner가 사용자의 명시적 요청 없이도 "로컬 LLM 능력을 넘는 고난도 작업"이라고 판단할 때 `web_ai_ask`를 스스로 선택할 수 있게 한다 |
| `TESSERACT_LANG` | `kor+eng` | OCR 인식 언어 |
| `OCR_VLM_MODEL` | (없음) | tesseract 결과가 비었을 때(스캔 품질 문제 등) 한 번 더 시도할 로컬 Ollama 비전 모델명(예: `qwen2.5vl:7b`). 비워두면 VLM 폴백 비활성 |
| `RETRY_COUNT` | `2` | step 실패 시 추가 재시도 횟수 (1~3 권장) |
| `RETRY_BACKOFF` | `0.5` | 재시도 사이 대기(초), 시도마다 2배 증가 |
| `STEP_TIMEOUT` | `60` | step 1회 실행 제한시간(초, Ollama 요청에 적용) |
| `BROWSER_TIMEOUT` | `15000` | Playwright 동작 제한시간(ms) |
| `LOG_PATH` | `storage/olma.log` | 구조적 로그 파일 경로 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |
| `OLLAMA_TEMPERATURE_DEFAULT` | `0.7` | Planner/분류 외 일반 LLM 호출(`llm`/`summarize`)의 기본 temperature |
| `OLMA_API_HOST` | `0.0.0.0` | HTTP API 바인드 호스트 |
| `OLMA_API_PORT` | `8800` | HTTP API 바인드 포트 |
| `OLMA_API_KEY` | (없음) | 설정하면 모든 `/api/*` 요청에 `X-API-Key` 헤더 검증을 강제한다. 비워두면 인증 없음(로컬 단일 사용자 전제) — 네트워크로 노출할 때는 반드시 설정할 것 |
| `TASK_QUEUE_MAX_TASKS` | `200` | 작업 큐가 메모리에 보관하는 완료/실패 작업 기록 상한(초과분은 오래된 것부터 제거, 진행 중 작업은 보존) |
| `TASK_QUEUE_WORKERS` | `1` | 작업 큐 워커 스레드 수. 1보다 크게 설정하면 워커마다 독립된 브라우저 프로필(첫 실행 시 기존 프로필을 복사해 로그인 세션을 물려받음)을 써서 Playwright 프로필 잠금 충돌 없이 병렬 처리한다 |
| `TASK_STORE_PATH` | `storage/tasks.db` | task 기록을 영속화할 SQLite 파일 경로. 재시작 시 히스토리를 복원하지만, 그 시점에 `queued`/`processing`이던 task는 재개 불가로 판단해 `failed`로 정리한다 |
| `WEB_AI_PROVIDERS_PATH` | (없음) | 여러 웹 AI 제공자를 등록한 JSON 파일 경로. 비워두면 위 `WEB_AI_*` 단일 설정을 `"default"` 제공자 하나로만 사용한다(하위 호환) |

## HTTP API (선택)

REPL(`main.py`) 대신, Olma를 HTTP로 노출해 다른 기기/프로그램에서 작업을 제출할 수도 있다.

```bash
cd olma
python -m uvicorn api.server:app --host 0.0.0.0 --port 8800
```

브라우저로 `http://localhost:8800/` 을 열면 작업 제출/이력/메트릭을 보는 간단한 프런트엔드(`frontend/index.html`)가 뜬다.

엔드포인트:

| Method | Path | 설명 | 인증 |
|---|---|---|---|
| GET | `/health` | 헬스체크 | 불필요 |
| GET | `/` | 프런트엔드 | 불필요 |
| POST | `/api/task` | 새 작업 제출 (`{"input": "..."}`) → `{task_id, status}` | `OLMA_API_KEY` 설정 시 필요 |
| GET | `/api/task/{task_id}` | 작업 상태/결과 조회 | 〃 |
| GET | `/api/tasks?limit=20` | 최근 작업 목록 | 〃 |
| GET | `/api/memory/search?q=키워드&limit=10` | task 텍스트/step의 action·result에 키워드가 포함된 기록 검색(대소문자 무시, 최신순) | 〃 |
| GET | `/api/metrics` | 큐 깊이, 가동시간, 상태별 작업 수 | 〃 |

**왜 큐가 기본적으로 직렬(단일 워커)인가**: Playwright는 로그인 세션을 유지하는 영구 브라우저 프로필을 전제로 한다. 같은 프로필 디렉터리를 두 코드가 동시에 열면 충돌하므로, Redis/Celery 같은 분산 큐 대신 기본은 워커 스레드 1개가 큐를 순서대로 비우는 가장 단순한 구조(`core/task_queue.py`)를 쓴다 — 처리량보다 정확성이 우선이다. `TASK_QUEUE_WORKERS`를 1보다 크게 설정하면 워커마다 독립된 브라우저 프로필을 써서 충돌 없이 병렬로 처리할 수 있다.

**인증 설계**: Olma는 읽기 전용 도구가 아니라 로그인된 브라우저 세션으로 클릭/입력까지 하는 action-taking 에이전트다. 그래서 사용자 계정 시스템 대신 단일 공유 API 키(`OLMA_API_KEY`)로 충분하다고 보았다 — `/health`·`/`만 인증 없이 열려 있고 나머지 `/api/*`는 키가 설정된 순간부터 막힌다. 로컬에서만 쓸 거라면 비워두면 된다.

## Docker로 실행

Ollama까지 포함한 전체 스택을 컨테이너로 띄울 수 있다.

```bash
cd olma
OLMA_API_KEY=원하는키 docker compose up --build
```

`docker-compose.yml`은 `olma`(API 서버, 헤드리스 브라우저)와 `ollama` 두 서비스로 구성되며, `storage/`(메모리·브라우저 프로필·스크린샷)와 Ollama 모델 데이터를 각각 named volume으로 유지한다. 최초 실행 후 컨테이너 안에서 모델을 받아야 한다: `docker compose exec ollama ollama pull qwen2.5:7b`.

## Planner 출력 검증 & 폴백

Planner가 만드는 계획(Plan)은 `core/schema.py`의 Pydantic 스키마(`Plan`/`Step`, `schema_version` 포함)로 검증되며, Ollama 호출 시 이 스키마를 `format`으로 강제해 JSON 파싱 실패를 원천적으로 줄인다. 검증 실패 시 오류 내용을 포함해 1회 자기-교정 재시도 → 개별 step 부분 복구 → 그래도 복구할 step이 없으면 **규칙 기반 폴백 플래너**(`core/fallback_planner.py`)가 URL/키워드로 step을 구성한다(LLM을 쓰지 않으므로 Ollama가 죽어도 동작). step은 `depends_on`(이전 step의 인덱스)과 `{{result}}` 토큰으로 이전 결과를 참조할 수 있다. 외부에서 들어오는 텍스트(사용자 입력, 메시지, 이전 step 결과, 최근 작업 맥락)는 모두 구분자로 감싸 프롬프트 인젝션을 데이터로만 취급하도록 한다.

## 웹 AI 호출 (`web_ai_ask`, 브라우저 경유)

외부 클라우드 AI **API는 쓰지 않되**, 카톡과 동일한 방식으로 브라우저를 띄워 웹 AI 채팅(예: 사내 LLM 포털 등 사용자가 접근 권한을 가진 페이지)에 프롬프트를 입력하고 응답을 읽어 온다. "모든 추론은 로컬"이라는 기본 원칙과는 절충점이며, 명시적으로 켜야 동작한다.

- 어떤 웹 AI를 쓸지, 입력창/전송/응답 영역 selector는 사이트마다 다르므로 **코드에 박지 않고** 사용자가 직접 지정한다(미설정 시 명확한 오류). URL을 임의로 추측하지 않는다.
- 로그인 벽이 있으면 `BROWSER_HEADLESS=false`로 최초 1회 직접 로그인 → 영구 프로필에 세션 유지(카톡과 동일).
- 응답은 스트리밍이라 `wait_ms`만큼 기다린 뒤 텍스트를 읽고, DOM이 비면 OCR로 폴백한다. 웹 AI가 실패하면 라우터 폴백으로 로컬 Ollama가 "아는 선에서" 받는다.
- 대상 서비스의 이용약관에 자동화 제한이 있을 수 있으니 사용은 사용자 책임이다.

### 다중 제공자 등록 (`core/web_ai_providers.py`)

웹 AI를 하나만 쓴다면 위 `WEB_AI_*` 환경변수만 설정하면 된다(`"default"` 제공자로 동작). 여러 웹 AI를 등록해 작업별로 골라 쓰려면 `WEB_AI_PROVIDERS_PATH`로 JSON 파일을 가리킨다:

```json
{
  "internal_llm": {
    "url": "https://internal.example.com/chat",
    "input_selector": "#prompt-box",
    "submit_selector": "#send-btn",
    "response_selector": ".response",
    "wait_ms": 20000
  }
}
```

`url`/`input_selector`는 필수이고, `submit_selector`(없으면 Enter 키)/`response_selector`(기본 `body`)/`wait_ms`(기본 `WEB_AI_WAIT_MS`)는 선택이다. `web_ai_ask` step의 `input`은 기존 `browser_type`의 `"selector|||text"` 구분자 관례를 그대로 따라 `"provider_name|||prompt"` 형태로 쓴다 — `provider_name`이 등록된 이름이면 그 제공자를 쓰고, 아니면 전체 문자열을 `"default"` 제공자에 보낼 prompt로 취급한다.

### AI 역할 분담 (로컬 Ollama ↔ 웹 AI)

기본값(`WEB_AI_AUTO_ESCALATE=false`)에서는 Planner가 사용자가 "챗GPT/웹 AI한테 물어봐"처럼 **명시적으로 요청했을 때만** `web_ai_ask`를 선택하고, 그 외 모든 요청은 로컬 Ollama(`llm`)로 처리한다.

`WEB_AI_AUTO_ESCALATE=true`로 켜면 다음 **두 경우 중 하나**에 해당할 때 Planner가 스스로 `web_ai_ask`를 선택할 수 있다(하이브리드 정책):

1. 사용자가 외부 웹 AI 사용을 명시적으로 요청한 경우
2. 복잡한 코드 작성/디버깅, 최신 시사·실시간 정보, 여러 단계의 전문적 추론처럼 Planner가 보기에 **로컬 LLM 능력을 넘어선다고 판단**되는 경우

이 판단은 ML 기반 난이도 분류기가 아니라 Planner LLM 자신의 자기 평가이며, 어디까지나 "로컬 우선" 기본 원칙에 대한 절충점이므로 기본값은 꺼져 있다. `web_ai_ask`가 실행 시 실패하면(예: `WEB_AI_URL` 미설정) 기존 라우터 폴백에 따라 로컬 Ollama가 "아는 선에서" 대신 답한다.

## 테스트

```bash
cd olma
pip install -r requirements.txt pytest
pytest
```

Ollama 서버·Playwright 브라우저 없이도 동작하도록 전부 모킹 기반으로 작성되어 있다.

## 알림(Notification) 분석 기능 사용법

`notification_check` 액션은 Planner가 사용자의 요청에 메시지 확인 의도가 있을 때만 생성한다 (예: "카톡 메시지 확인해줘"). 동작 순서:

1. `KAKAO_WEB_URL`로 지정된 페이지를 브라우저로 열고 화면을 캡처
2. OCR로 텍스트 추출 후 메시지 단위로 분리
3. 각 메시지를 Ollama로 분류 (`school` / `personal` / `urgent` / `spam`, 우선순위)
4. 분류 결과에 따라 채널을 **추천만** 함 (자동 전송/이동 없음)

카톡 자동 로그인은 지원하지 않는다 — `BROWSER_HEADLESS=false` 상태로 최초 1회 실행해 직접 로그인하면, 영구 프로필에 세션이 저장되어 이후 자동으로 유지된다.

## 알려진 제약

- `browser_search`는 검색엔진 페이지를 직접 자동화하므로 안티봇 대응에 따라 실패할 수 있음.
- 외부 클라우드 AI API, 카카오 공식 API는 V0.1 범위에서 사용하지 않음 (Ollama + Playwright + OCR만 사용).

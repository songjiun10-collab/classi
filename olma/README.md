# Olma V0.2

로컬 AI(Ollama) + 브라우저(Playwright) + OCR을 통합한 작업 실행 시스템.
사용자의 자연어 요청을 계획(Planner) → 실행 방식 결정(Router) → 로컬 LLM/브라우저 실행(Executor) → 상태 기억(Memory) 순서로 처리한다.

외부 클라우드 AI API는 사용하지 않으며, 모든 추론은 로컬 Ollama로 수행한다. 카톡 등 메신저는 읽기 전용으로만 다루며, 분류·채널 추천만 하고 자동 전송은 하지 않는다.

## V0.2 — "돌아가는 코드"에서 "안 죽는 시스템"으로

V0.2는 기능 추가가 아니라 **안정성/반복 성공률**을 위한 구조 강화다.

- **Planner = 제어 시스템**: 스키마 검증 + Ollama `format` 강제 + 1회 자기-교정 재시도 + 부분 step 복구. 그래도 실패하면 LLM을 전혀 쓰지 않는 **규칙 기반 폴백 플래너**(`core/fallback_planner.py`)가 키워드/URL로 의미 있는 step을 만든다(통째 폐기 안 함).
- **Executor = 내결함성 시스템**: step당 재시도(backoff) → 그래도 실패하면 **대체 타겟 폴백**(브라우저 실패를 LLM이 아는 선에서 받음) → 그래도 안 되면 실패로 기록하고 다음 step 계속. 모든 단계가 `storage/olma.log`에 구조적으로 기록된다.
- **Memory = 상태 시스템**: `{task, status(done/failed/partial), steps[], timestamp}` 단위로 저장하고, `get_context()`로 최근 작업 맥락을 다음 계획에 다시 넣는다(context 재사용). 키워드 검색(`find()`)도 지원.
- **Router = 정책 엔진**: 결정론적 action→타겟 매핑 위에 **대체 타겟(fallback chain)**과 입력 완결성 기반 **confidence**(휴리스틱)를 얹는다.
- **Browser = 안정화 레이어**: 명시적 wait 전략, **selector fallback**(후보 여러 개 순차 시도), DOM 텍스트가 비면 **OCR 폴백**, 실패 시 디버그 스크린샷.

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
| `BROWSER_HEADLESS` | `false` | 브라우저 헤드리스 여부 |
| `BROWSER_USER_DATA_DIR` | `storage/browser_profile` | 로그인 세션 유지를 위한 영구 프로필 경로 |
| `KAKAO_WEB_URL` | (없음) | 알림 분석 기능에서 열 메신저 웹 페이지 URL. 직접 지정 필요 |
| `TESSERACT_LANG` | `kor+eng` | OCR 인식 언어 |
| `RETRY_COUNT` | `2` | step 실패 시 추가 재시도 횟수 (1~3 권장) |
| `RETRY_BACKOFF` | `0.5` | 재시도 사이 대기(초), 시도마다 2배 증가 |
| `STEP_TIMEOUT` | `60` | step 1회 실행 제한시간(초, Ollama 요청에 적용) |
| `BROWSER_TIMEOUT` | `15000` | Playwright 동작 제한시간(ms) |
| `LOG_PATH` | `storage/olma.log` | 구조적 로그 파일 경로 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |
| `OLLAMA_TEMPERATURE_DEFAULT` | `0.7` | Planner/분류 외 일반 LLM 호출(`llm`/`summarize`)의 기본 temperature |

## Planner 출력 검증 & 폴백

Planner가 만드는 계획(Plan)은 `core/schema.py`의 Pydantic 스키마(`Plan`/`Step`, `schema_version` 포함)로 검증되며, Ollama 호출 시 이 스키마를 `format`으로 강제해 JSON 파싱 실패를 원천적으로 줄인다. 검증 실패 시 오류 내용을 포함해 1회 자기-교정 재시도 → 개별 step 부분 복구 → 그래도 복구할 step이 없으면 **규칙 기반 폴백 플래너**(`core/fallback_planner.py`)가 URL/키워드로 step을 구성한다(LLM을 쓰지 않으므로 Ollama가 죽어도 동작). step은 `depends_on`(이전 step의 인덱스)과 `{{result}}` 토큰으로 이전 결과를 참조할 수 있다. 외부에서 들어오는 텍스트(사용자 입력, 메시지, 이전 step 결과, 최근 작업 맥락)는 모두 구분자로 감싸 프롬프트 인젝션을 데이터로만 취급하도록 한다.

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

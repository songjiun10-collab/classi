# Olma V0.1

로컬 AI(Ollama) + 브라우저(Playwright) + OCR을 통합한 작업 실행 시스템.
사용자의 자연어 요청을 계획(Planner) → 실행 방식 결정(Router) → 로컬 LLM/브라우저 실행(Executor) → 기록(Memory) 순서로 처리한다.

외부 클라우드 AI API는 사용하지 않으며, 모든 추론은 로컬 Ollama로 수행한다. 카톡 등 메신저는 읽기 전용으로만 다루며, 분류·채널 추천만 하고 자동 전송은 하지 않는다.

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
| `RETRY_COUNT` | `1` | step 실패 시 재시도 횟수 |
| `OLLAMA_TEMPERATURE_DEFAULT` | `0.7` | Planner/분류 외 일반 LLM 호출(`llm`/`summarize`)의 기본 temperature |

## Planner 출력 검증

Planner가 만드는 계획(Plan)은 `core/schema.py`의 Pydantic 스키마(`Plan`/`Step`, `schema_version` 포함)로 검증되며, Ollama 호출 시 이 스키마를 `format`으로 강제해 JSON 파싱 실패를 원천적으로 줄인다. 검증에 실패하면 오류 내용을 포함해 1회 자기-교정 재시도를 하고, 그래도 실패하면 개별 step만 부분 복구하며, 복구할 step이 전혀 없을 때만 단일 `llm` step으로 폴백한다. step은 `depends_on`(이전 step의 인덱스)과 `{{result}}` 토큰으로 이전 결과를 참조할 수 있다. 외부에서 들어오는 텍스트(사용자 입력, 메시지, 이전 step 결과)는 모두 구분자로 감싸 프롬프트 인젝션을 데이터로만 취급하도록 한다.

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

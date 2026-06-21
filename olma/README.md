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
- **Task Queue = 영속 큐**: 처리된 task는 SQLite(`storage/tasks.db`)에도 기록되어 재시작해도 히스토리가 남는다. 기본적으로 재시작 시점에 `queued`/`processing`이던 task는 브라우저 세션·워커 스레드가 이미 사라져 안전하게 이어갈 수 없으므로 `failed`로 정리된다. `TASK_QUEUE_RESUME=true`로 켜면 그런 task를 버리는 대신 다시 큐에 넣어 원래 입력으로 처음부터 재실행한다(mid-step 재개가 아니라 재시도형 재개 — 부작용 있는 작업엔 주의).

### 성능 (V0.2.1)

- **추론 캐시**: `temperature=0`(결정론적) Ollama 호출을 SQLite(`storage/infer_cache.db`)에 캐시한다. 같은 입력은 재추론 없이 저장 응답을 즉시 반환 — 플래너 호출이 정확히 이 경우라 **반복 작업의 재추론을 0회**로 만들어 발열·지연을 줄인다. `temperature>0`(비결정론, 기본 `llm`/`summarize`)은 캐시하지 않는다. `INFER_CACHE=0`으로 끔.
- **메모리 검색 = SQL**: `find()`/`successful_examples()`가 전체 레코드를 로드해 파이썬에서 거르던 풀스캔을 SQL `LIKE`로 내려 필요한 n건만 파싱한다(plan마다 호출되던 비용 제거).
- **독립 step 병렬**: 의존성 없는 ollama step들을 동시에 실행해 멀티스텝 지연을 줄인다(브라우저·의존성 체인은 순차 유지). 기본 on이며, 단일 GPU에서 VRAM 경합이 우려되면 `OLLAMA_PARALLEL=0`으로 끈다.
- **스트리밍 응답**: REPL(`main.py`)에서 ollama step의 생성 토큰을 즉시 화면에 흘려보낸다(체감 응답성↑). `generate_stream()`은 비결정론 사용처가 주 대상이라 캐시를 적용하지 않는다.

### 로그인 = 구글 세션 재활용 (V0.2.1)

비밀번호를 어디에도 저장하지 않는다. 영구 프로필에 **구글 로그인 세션이 살아 있으면**, 대부분의 사이트는 "구글로 로그인" 버튼 한 번으로 재인증된다 — 올마는 그 세션을 타는 방식만 쓴다.

- **명시적 `login` 액션**: `login` step(`input`은 등록된 제공자 이름, 비우면 `default`)을 실행하면, 이미 로그인돼 있으면 건너뛰고(`LOGIN_LOGGED_IN_SELECTOR`로 판별) 풀렸으면 "구글로 로그인" 버튼을 눌러 재인증한다(`browser.ensure_logged_in`). 2FA·기기 인증이 필요한 사이트는 자동화로 넘기지 않고 명확한 오류를 던진다(수동 로그인 필요).
- **자동 재로그인(`AUTO_RELOGIN=true`)**: 브라우저 step이 실패하고 현재 페이지에 로그인 벽(`LOGIN_WALL_SELECTOR`)이 감지되면, `default` 제공자로 자동 재로그인한 뒤 그 step을 1회 재시도한다. 로그인 벽이 아닌 다른 실패는 건드리지 않으며, `LOGIN_WALL_SELECTOR`가 설정돼야만 작동한다(오탐 방지).
- 사이트별 URL·selector는 `web_ai_providers`와 동일하게 코드에 박지 않고 `LOGIN_PROVIDERS_PATH` JSON 또는 `LOGIN_*` 환경변수로 등록한다. "구글로 로그인" 버튼만은 사이트 간 공통 패턴이 있어 기본 후보를 제공하되 덮어쓸 수 있다.

### AI 역할 분담 (V0.2.1)

한 모델이 전부를 떠맡는 대신, 각 AI가 잘하는 일에 배정한다. 역할 맵은 `core/ai_roles.py`가 단일 정의(SSOT)로 들고 있고, `/api/roles`로 확인할 수 있다.

| 역할 | 백엔드 | 모델(env) | 맡는 일 |
|------|--------|-----------|---------|
| `plan` | 로컬 | `OLLAMA_PLANNER_MODEL` | 계획 수립(JSON 강제·결정론·캐시) |
| `chat` | 로컬 | `OLLAMA_MODEL` | 일상 대화/간단 Q&A (`llm`) |
| `summarize` | 로컬 | `OLLAMA_MODEL` | 요약/정리 (`summarize`) |
| `reason` | **외부 웹 AI** | (폴백 시 `OLLAMA_REASONING_MODEL`) | 고난도 추론·코딩·최신정보·장문 (`web_ai_ask`) |
| `vision` | 로컬 VLM | `OLLAMA_VLM_MODEL`(없으면 `OCR_VLM_MODEL`) | 이미지/화면 이해 (`vision_describe`) |

- **로컬 모델 다중화**: 역할별로 다른 Ollama 모델을 지정할 수 있다(미지정 시 `OLLAMA_MODEL`로 폴백 → 기존 단일 모델 동작과 동일). 예: 계획은 작고 빠른 `qwen2.5:3b`, 로컬 폴백 추론은 강한 `qwen2.5:32b`.
- **외부 웹 AI 모델별 분할**: 여러 웹 AI를 등록하고 각자 특기(`specialties`)를 선언하면, `web_ai_ask`가 프롬프트를 분류해 특기 제공자로 자동 라우팅한다(예: 코딩→Claude, 최신검색→Gemini/Perplexity). `"이름|||프롬프트"`로 명시하면 그게 우선. 특기 제공자가 없으면 항상 `default`(하위 호환).
- **최신·실시간 정보는 반드시 외부 웹 AI**: 오늘/지금 기준 뉴스·시세·환율·날씨 등은 로컬 LLM의 지식 컷오프로 알 수 없으므로, Planner가 `WEB_AI_AUTO_ESCALATE` 설정과 무관하게 항상 `web_ai_ask`로 보낸다(로컬 `llm`으로 답하지 않음). 이 경우 웹 AI 체인이 모두 실패해도 **로컬로 폴백하지 않는다** — 엉뚱한 옛 정보를 주느니 명확히 실패하는 게 낫기 때문(코딩·추론 등 다른 web_ai_ask는 로컬 best-effort 폴백 유지).
- **페일오버 백업 체인**: 각 제공자가 `backup`(다른 제공자 이름)을 선언하면, 그 웹 AI가 실패할 때 백업으로 자동 전환한다(예: `claude` 실패 → `zai`). 체인은 순환·미등록 백업에서 안전하게 멈추고(최대 5단계), 끝까지 실패하면 로컬 `reason` 모델 폴백으로 내려간다. default 제공자의 백업은 `WEB_AI_BACKUP` 환경변수로도 지정 가능.
- **로컬 모델도 전부 백업**: 외부 웹 AI뿐 아니라 모든 로컬 역할(plan/chat/summarize/reason/vision)이 모델 페일오버 체인을 갖는다. 텍스트 역할은 `OLLAMA_BACKUP_MODEL`, 비전은 `OLLAMA_VLM_BACKUP_MODEL`로 1순위 모델 실패 시(모델 없음·OOM·오류) 자동 전환한다(`ai_roles.models_for`). 비전은 텍스트 모델로 내려가지 않는다(이미지 이해 불가). 백업 미설정이면 단일 모델 = 기존 동작.
- **VLM 역할 확장**: `vision_describe` 액션은 OCR(글자 전사)을 넘어 현재 화면을 VLM이 **이해/설명**한다(레이아웃·상태·의미). `browser_open`과 함께 쓰면 "페이지 열고 화면 설명"이 된다.
- **reason/vision은 텍스트 폴백 없음**: 로컬 텍스트 LLM은 로그인도 화면 이해도 못 하므로, `web_ai_ask`가 실패하면 `reason` 로컬 모델로만 받고, `vision_describe`는 거짓 성공을 만들지 않도록 폴백을 끈다.

웹 AI 특기 등록 예시(`WEB_AI_PROVIDERS_PATH` JSON):
```json
{
  "claude":     {"url": "...", "input_selector": "...", "response_selector": "...", "specialties": ["coding", "reasoning"], "backup": "zai"},
  "zai":        {"url": "https://z.ai", "input_selector": "...", "response_selector": "..."},
  "gemini":     {"url": "...", "input_selector": "...", "response_selector": "...", "specialties": ["search"], "backup": "perplexity"},
  "perplexity": {"url": "...", "input_selector": "...", "response_selector": "...", "specialties": ["search"]}
}
```
위 예시에서 코딩 작업은 `claude`(막히면 `zai`)로, **최신·실시간 정보는 `gemini`(막히면 `perplexity`)**로 라우팅된다. 최신정보는 로컬로 폴백하지 않으므로 이 둘이 모두 막히면 실패로 처리된다.

### Dynamic Workflow Engine (V0.3)

올마를 "도구 모음"에서 "실제로 일을 하는 시스템"으로 끌어올리는 중심축. **요청 → 동적 워크플로우 → 실행 → 기억** 파이프라인을 한 단위로 묶는다(`core/workflow.py`).

- **동적 실행** `run_dynamic(request)`: 자연어 요청을 planner로 step-DAG(워크플로우)로 만들어 executor(의존성 DAG + 병렬)로 돌리고 memory에 기록한다.
- **재사용 템플릿** `run_template(name, params)`: 저장된 고정 step-DAG에 파라미터를 채워 실행한다. 파라미터는 `{{param:KEY}}` 토큰으로 치환되며, executor의 실행 시점 토큰 `{{result}}`와 충돌하지 않는다(치환은 실행 '전' 단계).
- **템플릿 저장소** `core/workflow_store.py`(SQLite): 워크플로우 템플릿을 `{name, description, steps}`로 저장·재사용한다. 저장 시 `schema.Plan`으로 검증해 잘못된 워크플로우가 들어오는 것을 막는다. API: `GET/POST /api/workflows`, `DELETE /api/workflows/{name}`.

### Scheduler — 주기 실행 (V0.3)

워크플로우 위에 "시간"을 얹는다(`core/scheduler.py` + `core/scheduler_store.py`, SQLite). due(만기) 스케줄을 찾아 실행하고 `next_run`을 interval만큼 미룬다. 핵심 로직(`due`/`tick`)은 시계(`now_fn`)와 실행기(`runner`)를 주입받아 스레드·sleep 없이 결정론적으로 테스트된다 — 백그라운드 폴링 루프는 그 위의 얇은 층일 뿐. 한 스케줄의 실패는 `last_status`에 남기고 재예약하며 다른 스케줄을 막지 않는다. API 서버는 시작 시 폴링 루프를 띄운다(lifespan). API: `GET/POST /api/schedules`, `PATCH/DELETE /api/schedules/{id}`.

### Event Trigger — 사건 기반 실행 (V0.3)

Scheduler가 시간에 반응한다면 이쪽은 상태 변화에 반응한다(`core/event_triggers.py` + `core/event_store.py`). **엣지 트리거**: 조건이 "참인 동안" 계속 쏘지 않고 거짓→참(혹은 값 변화)으로 넘어가는 순간에만 한 번 발화한다(마지막 관측 상태를 저장해 비교). 기본 감지기는 모호함 없는 로컬 파일 기반(`file_exists`, `file_changed`)이며, 사이트 URL·셀렉터처럼 추측 불가한 값은 코드에 박지 않고 config로 받는다. 등록 시 현재 상태를 기준선으로 잡아 즉시 오발화하지 않는다. API: `GET/POST /api/events`, `PATCH/DELETE /api/events/{id}`.

### Parallel DAG Executor — 웨이브 병렬 (V0.3)

executor가 의존성 그래프를 깊이별 웨이브로 나눠, 각 웨이브의 독립 ollama step들을 동시에 선실행한다(`executor._precompute_waves`). depth 0(의존 없음)뿐 아니라 ollama→ollama로 이어지는 다단계 체인도 웨이브 단위로 병렬화된다(예: 두 입력을 각각 요약하는 두 summarize가 동시에 실행). 브라우저 step과 브라우저 결과에 의존하는 step은 상태·순서를 보존하기 위해 본 walk에서 순차 실행한다. 스트리밍(on_token) 시엔 출력 순서 보존을 위해 병렬을 끈다.

### Long-term Memory — 세션을 넘는 사실/선호 (V0.3)

작업 이력(`memory.py`, "무엇을 했나")과 별개로, 세션을 넘어 유지돼야 하는 사용자 사실·선호를 저장한다(`core/long_term_memory.py` + `core/ltm_store.py`, "무엇을 아는가"). `remember(content, kind, key)`로 기억하고(`key`를 주면 같은 키를 덮어써 변하는 단일 사실을 갱신, 없으면 누적), `recall(query)`로 떠올린다. 계획 시 요청 토큰으로 관련 사실을 검색해 플래너 프롬프트에 **참고 블록**으로 주입한다(지시가 아닌 참고; `EXTERNAL_DATA` 펜스로 감싼다). API: `GET/POST /api/facts`, `DELETE /api/facts/{id}`.

> 위 5개(Dynamic Workflow · Scheduler · Event Trigger · Parallel DAG · Long-term Memory)가 "요청 → 동적 Workflow 생성 → 병렬 실행 → 기억" 구조의 핵심 골격이다.

### 파이프라인 척추 (V0.4)

요청은 다음 한 줄 파이프라인을 따라 흐른다 — 각 레이어가 독립 모듈이고 `/api/*`로 조회된다:

```
User Account → Memory Profile → Planner → Capability Router → Workflow Builder → Agent Pool
```

- **User Account** (`core/accounts.py` + `account_store.py`): 파이프라인 출발점(identity). 단일 사용자 전제지만 다중 사용자(#88)를 대비해 계정을 1급 엔티티로 둔다. 기본 계정(`DEFAULT_ACCOUNT`, 기본 "local")은 시작 시 자동 생성되고, 자유 형식 `attributes`(선호)를 든다. API: `GET/POST /api/accounts`, `PATCH/DELETE /api/accounts/{id}`.
- **Memory Profile** (`core/profile.py`): 계정의 누적 행동을 한 프로파일로 합친다 — 작업 이력(통계·자주 쓰는 작업), 계정 선호, 관련 장기 기억. `planner_context(account, request)`가 플래너에 줄 (맥락, 사실) 쌍을 만든다. API: `GET /api/profile`.
- **Capability Router** (`core/router.py` + `core/capabilities.py`): 라우터가 step을 실행 target으로 보내고, Capability Layer가 각 액션의 위험도(safe/caution/dangerous)·되돌림가능·승인필요를 정의한다(SSOT). API: `GET /api/capabilities`.
- **Workflow Builder** (`core/workflow.py` + `core/workflow_planner.py`): `run_request(request)`가 먼저 Workflow Planner로 "저장된 템플릿 재사용 vs 동적 생성"을 고른 뒤 실행한다(파라미터 없는 템플릿만 자동 재사용; 결정론적 토큰 매칭). 작업 큐에서 `WORKFLOW_AUTO_REUSE=true`면 동일 로직으로 템플릿을 재사용한다(기본 off).
- **Agent Pool (dynamic)** (`core/agent_pool.py`): step을 실행하는 에이전트 풀. 내장 에이전트(local_llm/web_ai/vision/browser/notifier)는 capabilities에서 유도하고, `register()`로 전문 에이전트(#21~30)를 런타임에 끼울 수 있다. 각 step 결과에 담당 `agent`가 기록된다. API: `GET /api/agents`.

### 실사용성 · 에이전트 (V0.5)

- **Usage / Quota** (`core/usage.py` + `usage_store.py`): 모델별 호출·토큰 사용량을 일 단위로 기록하고(ollama 응답의 실토큰), `USAGE_DAILY_CALL_LIMIT`/`USAGE_DAILY_TOKEN_LIMIT` 초과 시 그 모델 호출을 막아 백업 모델로 페일오버한다. API: `GET /api/usage`(`?days=N`).
- **작업 우선순위 · 자동 복구** (`core/task_queue.py`): 큐가 우선순위 큐로 동작한다(`POST /api/task`의 `priority`, 낮을수록 먼저; 같으면 FIFO). `TASK_AUTO_RECOVERY=true`면 실패한 작업을 `TASK_RECOVERY_MAX`회까지 낮은 우선순위로 자동 재큐잉한다.
- **Recipes** (`core/recipes.py`): 자주 쓰는 작업을 파라미터화한 빌트인 워크플로우 템플릿(자동 리서치 `research` · 공지/페이지 요약 `webpage_summary` · GitHub PR 리뷰 `github_pr_review`). 사이트 특이값은 `{{param:KEY}}`로 받아 코드에 박지 않는다. `GET /api/recipes`, `POST /api/recipes/{name}`(설치=템플릿 저장).
- **Agent Catalog** (`core/agent_catalog.py`): 명명된 전문 에이전트 10종(Search/Research/Developer/Vision/School/Reviewer/Planner/Notification/Documentation/Browser)을 실제 구현에 매핑한다 — executor(Agent Pool 백엔드) 또는 recipe. `suggest(request)`로 요청에 맞는 에이전트를 추천한다. `GET /api/agent_catalog`(`?suggest_for=`).

### Provider Registry · Human Approval Gate · Context Compression (V0.4)

- **Provider Registry** (`core/providers.py`): 로컬 역할 모델 · 외부 웹 AI · 로그인 제공자를 한 카탈로그로 모으는 조회 레이어(SSOT 조회 표면). API: `GET /api/providers`.
- **Human Approval Gate** (`core/approval.py` + `approval_store.py`): `APPROVAL_GATE=true`면 위험·비가역 액션(기본 `browser_click`/`browser_type`)을 실행 전 사람 승인 앞에 세운다. 승인될 때까지 폴링 대기하고(거절/만료 시 그 step만 `rejected`), 만료로 큐가 영구히 막히지 않게 한다. API: `GET /api/approvals`, `POST /api/approvals/{id}`.
- **Context Compression** (`core/compression.py`): 플래너에 주입하는 맥락/사실 블록을 `CONTEXT_MAX_CHARS` 예산으로 압축한다(앞·뒤 보존, 가운데 생략) — 프롬프트 비대화·토큰 낭비 방지.

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

### 대화형 CLI (Claude Code 스타일)

`main.py`를 실행하면 Claude Code처럼 직관적인 대화형 REPL이 뜬다(`core/cli.py`, 렌더는 `core/ui.py`).

- **시작 배너**: 버전·모델·작업 위치와 "무엇부터 하면 되는지"를 한 박스에 보여준다.
- **슬래시 명령**: 기능을 외울 필요 없이 `/help` 한 번이면 전부 보인다(발견 가능성).
- **단계 표시**: 각 실행 단계를 도구 호출처럼 `●`(상태색) + 액션명 + 결과로 보여준다(성공 초록·실패 빨강·부분 노랑).
- **스트리밍 + 스피너**: 계획 수립 중에는 점멸 표시, 생성 토큰은 즉시 흘려보낸다. 색은 `NO_COLOR`/비-tty면 자동으로 끈다.

| 슬래시 명령 | 설명 |
|---|---|
| `/help` | 명령 목록 |
| `/status` | 누적 작업·성공률·상태별 통계 |
| `/history [n]` | 최근 작업 n건 |
| `/search <키워드>` | 작업 기록 검색 |
| `/model [번호\|이름\|default]` | 모델 선택/전환 (Claude Code의 `/model`) |
| `/skill [add\|rm\|<이름>] ...` | 스킬 보기/추가/삭제/실행 |
| `/memory` | 장기 기억(사실) 보기 |
| `/remember <내용>` | 사실을 장기 기억에 저장 |
| `/schedules` | 예약 작업 목록 |
| `/clear` | 화면 지우기 |
| `/exit`(=`/quit`) | 종료 |

#### 모델 선택 (`/model`)

Claude Code처럼 사용 모델을 런타임에 고를 수 있다. `/model`은 Ollama에 설치된 모델 목록(`GET /api/tags`)을 번호와 함께 보여주고 현재 모델을 `●`로 표시한다. `/model 2` 또는 `/model llama3:8b`로 전환, `/model default`로 config 기본값 복귀.

- 선택은 **텍스트 역할**(plan/chat/summarize, reason 로컬 폴백)에만 적용된다. 비전(VLM)은 이미지 이해 전용이라 그대로 유지한다.
- 단일 지점 `ai_roles.model_for()`에 런타임 오버라이드를 걸어 planner·executor 전체에 반영된다. 잘못된 모델명을 골라도 `models_for()` 페일오버 체인에 역할 본래 모델이 남아 안전하게 폴백한다.
- 같은 기능을 웹 UI 상단 **모델 셀렉터**와 API로도 제공한다: `GET /api/models`(installed/current/override/backend_mode/web_providers), `POST /api/model {model}`(빈 문자열이면 해제).

#### 로컬 ↔ 웹 AI 백엔드 토글 (coworker 참조, 무료만)

[accomplish-ai/coworker](https://github.com/accomplish-ai/coworker)처럼 "한 곳에서 모델 선택"을 무료 옵션으로만 통합한다. 웹 UI 상단 셀렉터 하나에서 **자동 / 로컬 Ollama 모델 / 웹 AI(브라우저)**를 고른다. 유료 API 키 클라우드는 쓰지 않는다.

- **자동**(기본): planner가 로컬/웹을 알아서 결정(최신·실시간 정보는 자동으로 웹 AI).
- **로컬**: 고른 Ollama 모델로 강제.
- **웹**: 일반 질문(`llm`)을 브라우저 웹 AI(`web_ai_ask`)로 보낸다. 반대로 로컬 모드는 `web_ai_ask`를 로컬로 내린다.
- 구현: `core/ai_roles.py`의 `backend_mode`/`effective_action`(action 재라우팅, executor 진입부에서 라우팅·병렬 판단 전에 적용). API: `POST /api/backend {mode}`. CLI: `/backend [auto|local|web [제공자]]`.

##### 웹 AI 제공자 선택 (about:blank 해결)

웹 모드에서 **어느 웹 AI**를 쓸지 고를 수 있다(웹 UI 셀렉터의 "웹 AI" 그룹 / CLI `/backend web <제공자>`). 무료 프리셋(chatgpt·claude·gemini·perplexity)의 **공개 진입 URL**을 기본 등록해, 고르면 실제 사이트가 열린다 — 예전엔 `WEB_AI_URL`이 비어 열 URL이 없어 **about:blank**만 떴다.

- 프리셋의 입력/응답 selector는 **범용 후보**라 사이트가 UI를 바꾸면 깨질 수 있다. 정확한 값은 `WEB_AI_PROVIDERS_PATH` JSON으로 덮어쓴다(코드에 사이트별 selector를 박지 않는 원칙 유지). `WEB_AI_PRESETS=false`로 프리셋을 끌 수 있다.
- 선택 상태는 `core/web_ai_providers.py`의 `set_active`/`active`가 들고, `web_ai_ask`가 제공자 미지정 시 active를 우선 쓴다. API: `GET /api/models`(web_providers·web_active), `POST /api/web_provider {name}`.

#### 캡차 완화 + 빈 응답(blank) 방지

웹 AI는 브라우저 자동화라 봇 탐지/캡차에 걸리고, 차단되면 응답이 비어 화면에 빈칸으로 떴다. 두 가지로 보완한다(`tools/browser.py`):

- **stealth**(`BROWSER_STEALTH=true` 기본): 자동화 플래그 제거(`--disable-blink-features=AutomationControlled`, `--enable-automation` 무시), 실제 크롬 같은 UA/로케일(`BROWSER_USER_AGENT`/`BROWSER_LOCALE`), `navigator.webdriver` 등 탐지 신호 위장(init script). 완전 회피는 불가능하므로 **headed(`BROWSER_HEADLESS=false`, 기본) + 로그인된 영구 프로필**이 가장 확실하다.
- **빈 응답/캡차 가드**: `ask_web_ai`가 응답이 비거나 캡차/로그인 신호가 보이면 빈칸을 돌려주지 않고 명확히 실패시킨다 → 라우터가 로컬로 폴백하거나 사용자에게 원인을 알린다(blank 방지).

#### 스킬 (`/skill`)

자주 쓰는 작업을 이름 붙여 저장해 두고 재실행하는 **사용자 정의 스킬**(Claude Code 스킬처럼). recipes(코드로 고정된 파라미터 워크플로)와 달리, 런타임에 자연어로 만들고 지우는 가벼운 단축이다.

- `/skill` 목록 · `/skill add <이름> <작업 내용>` 저장 · `/skill rm <이름>` 삭제 · `/skill <이름> [추가인자]` 실행.
- 실행하면 저장된 내용(필요 시 추가 인자를 덧붙여)을 그대로 작업 요청으로 돌린다(`core/skills.py`의 `resolve`). 저장은 SQLite(`storage/skills.db`).
- 웹 UI에서도 `/skill`을 지원하고, API: `GET /api/skills`, `POST /api/skills {name,body,description}`, `DELETE /api/skills/{name}`.

#### 간단한 질문은 계획 생략 (fast-chat)

작업성 신호(URL·액션 동사·멀티스텝 접속·최신정보·긴 문장)가 없는 단순 질의·대화는 planner(LLM 1회 추가 호출)를 건너뛰고 단일 로컬 LLM 답변으로 바로 처리한다 — 체감 응답성↑, 비용↓.

- 판별은 `core/triage.py`의 휴리스틱(LLM 없음, 보수적: 신호가 하나라도 있으면 계획을 태움). 최신·실시간 정보(오늘/지금/날씨/시세 등)는 로컬 LLM이 모르므로 계획을 거쳐 `web_ai_ask`로 간다.
- 대화형 진입점(CLI·작업 큐)에만 적용되고, 예약/이벤트/레시피 워크플로는 의도된 작업이라 그대로 계획한다. `OLMA_FAST_CHAT=false`로 끄면 모든 입력이 계획을 거친다(기본 on).

자연어(슬래시로 시작하지 않는 입력)는 그대로 작업 요청으로 계획→실행된다.

### 웹 UI

`frontend/index.html`은 폼 대시보드 대신 **대화형 화면**으로, CLI와 같은 시각 언어(`●` 글리프·상태색·슬래시 명령)를 쓴다. 하단 단일 입력(`Enter` 실행, `Shift+Enter` 줄바꿈), 작업은 사용자 말풍선 + 도구 호출식 단계로 흐름을 보여주며, 상단 모델 셀렉터로 모델을 바꿀 수 있다. 웹에서도 `/help`·`/status`·`/search`·`/model`·`/clear`를 지원한다. API 서버가 같은 출처로 서빙한다(`GET /`).

## 환경변수 (config/config.py)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 서버 주소 |
| `OLLAMA_MODEL` | `qwen2.5:7b` | 기본 모델(chat/summarize 역할 + 역할 미지정 폴백) |
| `OLLAMA_PLANNER_MODEL` | (= `OLLAMA_MODEL`) | `plan` 역할(계획 수립) 모델. 작고 빠른 모델로 분리 가능 |
| `OLLAMA_REASONING_MODEL` | (= `OLLAMA_MODEL`) | `reason` 역할의 **로컬 폴백** 모델(외부 웹 AI 실패 시). 가장 강한 로컬 모델 권장 |
| `OLLAMA_BACKUP_MODEL` | (없음) | 텍스트 역할(plan/chat/summarize/reason)의 **백업 모델**. 역할 모델이 실패하면(모델 없음·OOM·오류) 이 모델로 페일오버. 비우면 백업 없음 |
| `OLLAMA_VLM_MODEL` | (= `OCR_VLM_MODEL`) | `vision` 역할(`vision_describe`) 비전 모델. 비우면 `OCR_VLM_MODEL`을 따름 |
| `OLLAMA_VLM_BACKUP_MODEL` | (없음) | `vision` 역할의 **백업 비전 모델**(VLM 실패 시). 텍스트 모델로는 폴백하지 않음 |
| `INFER_CACHE` | `true` | `temperature=0`(결정론적) 추론을 SQLite에 캐시해 같은 입력은 Ollama 재호출 없이 즉시 반환한다(플래너 호출이 이에 해당 → 반복 작업 재추론 0회, 발열·지연↓). `temperature>0`(기본 `llm`/`summarize`)은 비결정론이라 캐시하지 않음. 끄려면 `0`/`false` |
| `INFER_CACHE_PATH` | `storage/infer_cache.db` | 추론 캐시 SQLite 파일 경로 |
| `INFER_CACHE_MAX_RECORDS` | `2000` | 캐시 보관 항목 상한(초과분은 오래된 것부터 자동 회전 삭제) |
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
| `LOGIN_PROVIDERS_PATH` | (없음) | 여러 사이트의 로그인 설정을 등록한 JSON 파일 경로(`web_ai_providers`와 동일 패턴). 비우면 아래 `LOGIN_*` 단일 설정을 `default` 제공자로 사용 |
| `LOGIN_URL` | (없음) | `login` 액션이 열 로그인 페이지 URL(default 제공자) |
| `LOGIN_LOGGED_IN_SELECTOR` | (없음) | 로그인됐을 때만 보이는 요소 selector — 이미 로그인/재로그인 성공 판별 |
| `LOGIN_GOOGLE_BUTTON_SELECTOR` | (없음) | "구글로 로그인" 버튼 selector. 비우면 공통 기본 후보(`button:has-text("Google")` 등)를 순서대로 시도 |
| `LOGIN_WALL_SELECTOR` | (없음) | 로그인 안 됐을 때만 뜨는 요소 selector. **자동 재로그인 트리거 감지에 사용** — 비우면 자동 재로그인이 동작하지 않는다(오탐 방지) |
| `LOGIN_WAIT_MS` | `30000` | 재로그인 후 로그인 확인 대기 상한(ms) |
| `LOGIN_CHECK_MS` | `3000` | 로그인 여부를 즉시 확인할 때의 대기(ms) |
| `AUTO_RELOGIN` | `true` | `true`로 켜면 브라우저 step이 실패하고 현재 페이지에 로그인 벽(`LOGIN_WALL_SELECTOR`)이 감지될 때 `default` 제공자로 자동 재로그인 후 그 step을 1회 재시도한다. default 제공자에 `LOGIN_WALL_SELECTOR`가 설정돼 있어야만 동작 |
| `TESSERACT_LANG` | `kor+eng` | OCR 인식 언어 |
| `OCR_VLM_MODEL` | (없음) | tesseract 결과가 비었을 때(스캔 품질 문제 등) 한 번 더 시도할 로컬 Ollama 비전 모델명(예: `qwen2.5vl:7b`). 비워두면 VLM 폴백 비활성 |
| `RETRY_COUNT` | `2` | step 실패 시 추가 재시도 횟수 (1~3 권장) |
| `RETRY_BACKOFF` | `0.5` | 재시도 사이 대기(초), 시도마다 2배 증가 |
| `STEP_TIMEOUT` | `60` | step 1회 실행 제한시간(초, Ollama 요청에 적용) |
| `BROWSER_TIMEOUT` | `15000` | Playwright 동작 제한시간(ms) |
| `OLLAMA_PARALLEL` | `true` | 의존성 없는(`depends_on=null`) `llm`/`summarize` step들을 동시에 실행해 멀티스텝 지연을 줄인다. 단일 GPU Ollama는 추론을 직렬화하는 경우가 많아 이득이 작거나 VRAM 경합/OOM 위험이 있으니 그런 환경이면 `0`으로 끈다. 캐시 히트가 섞인 배치나 다중 모델/CPU 환경에서 효과적. 브라우저 step과 의존성 체인은 항상 순차, REPL 스트리밍 사용 시엔 출력 순서를 위해 자동 비활성 |
| `OLLAMA_PARALLEL_MAX` | `4` | 동시에 실행할 ollama step 수 상한 |
| `LOG_PATH` | `storage/olma.log` | 구조적 로그 파일 경로 |
| `LOG_LEVEL` | `INFO` | 로그 레벨 |
| `LOG_JSON` | `false` | `true`로 켜면 로그를 한 줄 JSON으로 출력(로그 수집/관측 도구 연동용). 기본은 사람이 읽는 텍스트 |
| `OLLAMA_TEMPERATURE_DEFAULT` | `0.7` | Planner/분류 외 일반 LLM 호출(`llm`/`summarize`)의 기본 temperature |
| `OLMA_API_HOST` | `127.0.0.1` | HTTP API 바인드 호스트. 기본은 루프백만 — LAN/외부 노출이 필요하면 `0.0.0.0`으로 지정하고 반드시 `OLMA_API_KEY`도 함께 설정할 것 |
| `OLMA_API_PORT` | `8800` | HTTP API 바인드 포트 |
| `OLMA_API_KEY` | (없음) | 설정하면 모든 `/api/*` 요청에 `X-API-Key` 헤더 검증을 강제한다. 비워두면 인증 없음(로컬 단일 사용자 전제) — 네트워크로 노출할 때는 반드시 설정할 것 |
| `TASK_QUEUE_MAX_TASKS` | `200` | 작업 큐가 메모리에 보관하는 완료/실패 작업 기록 상한(초과분은 오래된 것부터 제거, 진행 중 작업은 보존) |
| `TASK_QUEUE_WORKERS` | `1` | 작업 큐 워커 스레드 수. 1보다 크게 설정하면 워커마다 독립된 브라우저 프로필(첫 실행 시 기존 프로필을 복사해 로그인 세션을 물려받음)을 써서 Playwright 프로필 잠금 충돌 없이 병렬 처리한다 |
| `TASK_QUEUE_RESUME` | `false` | `true`로 켜면 재시작 시 중단된(`queued`/`processing`) task를 `failed`로 버리지 않고 다시 큐에 넣어 원래 입력으로 **처음부터 재실행**한다(mid-step 재개가 아니라 재시도형 재개 — 브라우저 세션은 새로 뜸). 부작용이 있는 작업엔 주의 |
| `TASK_STORE_PATH` | `storage/tasks.db` | task 기록을 영속화할 SQLite 파일 경로. 재시작 시 히스토리를 복원하지만, 그 시점에 `queued`/`processing`이던 task는 재개 불가로 판단해 `failed`로 정리한다 |
| `WEB_AI_PROVIDERS_PATH` | (없음) | 여러 웹 AI 제공자를 등록한 JSON 파일 경로. 각 제공자는 `specialties`(예: `["coding"]`)로 특기별 자동 라우팅 대상이 되고, `backup`(다른 제공자 이름)으로 실패 시 페일오버 대상을 지정한다. 비워두면 위 `WEB_AI_*` 단일 설정을 `"default"` 제공자 하나로만 사용한다(하위 호환) |
| `WEB_AI_BACKUP` | (없음) | `default` 제공자가 실패할 때 넘어갈 백업 제공자 이름(페일오버). JSON 제공자는 각 항목의 `backup` 필드로 지정 |
| `WORKFLOW_STORE_PATH` | `storage/workflows.db` | 재사용 워크플로우 템플릿을 저장할 SQLite 파일 경로 |
| `SCHEDULER_STORE_PATH` | `storage/schedules.db` | 주기 실행 스케줄을 저장할 SQLite 파일 경로 |
| `SCHEDULER_POLL_SECONDS` | `30` | 스케줄러가 due(만기) 스케줄을 점검하는 주기(초) |
| `SCHEDULER_PROFILE_SLOT` | `100` | 스케줄러·이벤트 엔진이 워크플로우를 실행할 때 쓸 브라우저 프로필 슬롯. API 큐 워커(0)와 분리해 영구 프로필 잠금 충돌을 피한다 |
| `EVENT_STORE_PATH` | `storage/events.db` | 이벤트 트리거를 저장할 SQLite 파일 경로 |
| `EVENT_POLL_SECONDS` | `15` | 이벤트 엔진이 트리거 조건을 점검하는 주기(초) |
| `LTM_PATH` | `storage/ltm.db` | 장기 기억(사실/선호)을 저장할 SQLite 파일 경로 |
| `LTM_CONTEXT_LIMIT` | `5` | 계획 시 플래너에 주입할 관련 장기 기억 최대 개수(`0`이면 주입 안 함) |
| `CONTEXT_MAX_CHARS` | `1500` | 플래너에 주입하는 '최근 작업 맥락'·'장기 기억' 블록 각각의 최대 글자 수(초과 시 앞·뒤 보존, 가운데 생략) |
| `ACCOUNT_STORE_PATH` | `storage/accounts.db` | 계정을 저장할 SQLite 파일 경로 |
| `DEFAULT_ACCOUNT` | `local` | 계정 미지정 요청이 귀속될 기본 계정 id(시작 시 자동 생성) |
| `WORKFLOW_AUTO_REUSE` | `false` | `true`면 요청과 충분히 맞는 '파라미터 없는' 저장 템플릿이 있을 때 동적 생성 대신 재사용 |
| `WORKFLOW_REUSE_THRESHOLD` | `0.6` | 템플릿 자동 재사용 최소 일치 점수(0~1). 이름/설명 토큰이 요청에 이 비율 이상 포함돼야 |
| `APPROVAL_GATE` | `false` | `true`면 위험·비가역 액션을 실행 전 사람 승인 앞에 세운다(끄면 기존처럼 즉시 실행) |
| `APPROVAL_STORE_PATH` | `storage/approvals.db` | 승인 요청을 저장할 SQLite 파일 경로 |
| `APPROVAL_REQUIRED_ACTIONS` | (없음) | 기본 위험도 외에 강제로 승인을 요구할 액션 목록(쉼표 구분, 예: `web_ai_ask,login`) |
| `APPROVAL_TIMEOUT_SECONDS` | `3600` | 승인 대기 상한(초). 초과하면 거절로 처리해 큐가 막히지 않게 한다 |
| `USAGE_STORE_PATH` | `storage/usage.db` | 모델 사용량(호출·토큰) 저장 SQLite 경로 |
| `USAGE_DAILY_CALL_LIMIT` | `0` | 모델 1개당 하루 호출 상한(0=무제한). 초과 시 백업 모델로 페일오버 |
| `USAGE_DAILY_TOKEN_LIMIT` | `0` | 모델 1개당 하루 토큰(프롬프트+생성) 상한(0=무제한) |
| `TASK_AUTO_RECOVERY` | `false` | `true`면 실패한 작업을 자동 재큐잉(재실행)한다 |
| `TASK_RECOVERY_MAX` | `1` | 작업 자동 복구 최대 재시도 횟수 |

## HTTP API (선택)

REPL(`main.py`) 대신, Olma를 HTTP로 노출해 다른 기기/프로그램에서 작업을 제출할 수도 있다.

```bash
cd olma
# 기본은 루프백(127.0.0.1)만 — 같은 기기에서만 접속. 다른 기기에서 제출하려면
# --host 0.0.0.0으로 바꾸고 반드시 OLMA_API_KEY를 설정할 것.
python -m uvicorn api.server:app --host 127.0.0.1 --port 8800
```

브라우저로 `http://localhost:8800/` 을 열면 작업 제출/이력/메트릭을 보는 간단한 프런트엔드(`frontend/index.html`)가 뜬다.

엔드포인트:

| Method | Path | 설명 | 인증 |
|---|---|---|---|
| GET | `/health` | 헬스체크 | 불필요 |
| GET | `/` | 프런트엔드 | 불필요 |
| POST | `/api/task` | 새 작업 제출 (`{"input": "...", "priority"?}`, priority 낮을수록 먼저) → `{task_id, status}` | `OLMA_API_KEY` 설정 시 필요 |
| GET | `/api/task/{task_id}` | 작업 상태/결과 조회 | 〃 |
| GET | `/api/tasks?limit=20` | 최근 작업 목록 | 〃 |
| GET | `/api/memory/search?q=키워드&limit=10` | task 텍스트/step의 action·result에 키워드가 포함된 기록 검색(대소문자 무시, 최신순) | 〃 |
| GET | `/api/metrics` | 큐 깊이, 가동시간, 상태별 작업 수, 작업 성공률, action별 집계(횟수·성공률·평균 소요), 추론 캐시 항목 수(`infer_cache`) | 〃 |
| GET | `/api/roles` | AI 역할 분담 맵(로컬 역할별 백엔드·모델 + 외부 웹 AI 제공자별 특기) | 〃 |
| GET/POST | `/api/workflows` | 재사용 워크플로우 템플릿 목록 / 저장(`{name, description, steps}`, 저장 시 `schema.Plan` 검증) | 〃 |
| DELETE | `/api/workflows/{name}` | 워크플로우 템플릿 삭제 | 〃 |
| GET/POST | `/api/schedules` | 주기 실행 스케줄 목록 / 등록(`{kind, payload, interval_seconds, first_run_delay?, enabled?}`) | 〃 |
| PATCH/DELETE | `/api/schedules/{id}` | 스케줄 활성 토글(`{enabled}`) / 삭제 | 〃 |
| GET/POST | `/api/events` | 이벤트 트리거 목록(+사용 가능한 `sources`) / 등록(`{source, source_config, kind, payload, enabled?}`) | 〃 |
| PATCH/DELETE | `/api/events/{id}` | 트리거 활성 토글(`{enabled}`) / 삭제 | 〃 |
| GET/POST | `/api/facts` | 장기 기억 목록·검색(`?q=`) / 저장(`{content, kind?, key?, tags?}`, `key` 주면 덮어쓰기) | 〃 |
| DELETE | `/api/facts/{id}` | 장기 기억 삭제 | 〃 |
| GET/POST | `/api/accounts` | 계정 목록 / 생성(`{name, account_id?}`) | 〃 |
| PATCH/DELETE | `/api/accounts/{id}` | 계정 속성 설정(`{key, value}`) / 삭제(기본 계정 보호) | 〃 |
| GET | `/api/profile?account_id=` | 계정의 누적 행동 프로파일(통계+선호+장기기억) | 〃 |
| GET | `/api/providers` | 백엔드 제공자 카탈로그(로컬 모델+웹 AI+로그인)+요약 | 〃 |
| GET | `/api/capabilities` | 액션별 위험도·되돌림가능·승인필요(Capability Layer) | 〃 |
| GET | `/api/agents` | step 실행 에이전트 풀(내장+동적)과 각자 맡는 action·target | 〃 |
| GET/POST | `/api/approvals` | 승인 대기 목록(`?all=true`면 이력) / 결정(`POST /api/approvals/{id}` `{approved, reason?}`) | 〃 |
| GET | `/api/usage?days=N` | 모델별 사용량(호출·토큰)+한도. days=1이면 오늘 요약 | 〃 |
| GET/POST | `/api/recipes` | 빌트인 레시피 목록 / 설치(`POST /api/recipes/{name}`, 템플릿으로 저장) | 〃 |
| GET | `/api/agent_catalog` | 명명 에이전트 카탈로그(`?suggest_for=`면 요청에 맞는 에이전트 추천) | 〃 |

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

# CLAUDE.md

이 파일은 [Claude Code](https://claude.com/claude-code)가 이 저장소에서 작업할 때 참고하는 가이드입니다.

---

## 프로젝트 개요

**Classi**는 한국 수능·모의고사 PDF를 **과목 / 세부과목 / 단원**으로 자동 분류하는 증거 기반 문항 분류 엔진입니다.

흐름: `PDF → 문항 추출(PyMuPDF) → OCR(PaddleOCR/Tesseract) → 로컬 LLM(Ollama) 분류 → 신뢰도 보정 → 사람 검수 → 재학습`

자세한 사용법·구조는 [README.md](./README.md)를 참고하세요.

---

## 작업 원칙 (Karpathy Guidelines)

> Andrej Karpathy의 LLM 코딩 함정 관찰에서 도출된 4대 원칙. 이 저장소에서 코드를 쓰고·검토하고·리팩터링할 때 따릅니다.

### 1. 생각하고 코딩하기 (Think Before Coding)
- **가정을 명시**하고, 불확실하면 묻는다. 혼란을 숨기지 않는다.
- 한 가지로 단정하지 말고 **여러 해석을 제시**한다.
- 더 단순한 대안이 있으면 인정하고, 필요하면 **반대 의견을 낸다**.

### 2. 단순함 우선 (Simplicity First)
- **문제를 푸는 최소한의 코드**만 작성한다. 투기적(speculative) 코드는 금지.
- 요청하지 않은 기능, 한 번만 쓰는 추상화, 불필요한 유연성·설정, 일어날 수 없는 상황의 에러 처리를 만들지 않는다.
- 판단 기준: *"숙련된 엔지니어가 이걸 과하게 복잡하다고 볼까?"* → 그렇다면 단순화한다.

### 3. 외과적 변경 (Surgical Changes)
- **건드려야 할 것만 건드린다. 내가 만든 것만 치운다.**
- 무관한 코드·주석·포맷을 개선하지 않는다. 동작하는 코드를 리팩터링하지 않는다.
- **기존 스타일 관례를 따른다** (이 저장소는 한국어 주석·docstring을 적극 사용).
- 죽은 코드는 발견하면 **보고**하되, 요청 없이는 삭제하지 않는다.
- 내 변경이 고아로 만든 import/변수/함수만 제거한다.

### 4. 목표 주도 실행 (Goal-Driven Execution)
- **성공 기준을 정의하고, 검증될 때까지 반복**한다.
- 작업을 테스트 가능한 측정 목표로 바꾼다. 다단계 계획은 검증 체크포인트와 함께 제시한다.

### 5. 오류 기록 (Error Logging) — **필수**
- **코드를 작성·실행하다 오류가 발생하면, 무조건 아래 [오류 기록 로그](#오류-기록-로그)에 한 줄 추가한다.**
- 기록 항목: 날짜, 발생 위치(파일/명령), 증상, 원인, 해결 방법.
- **오류는 해결책(해결 방법)까지 반드시 채운다.** 해결 칸을 비워두지 말고, 어떻게 고쳤는지/우회했는지 남긴다.
- 같은 실수를 반복하지 않기 위한 장치이므로, 사소해 보여도 빠짐없이 남긴다.

### 6. LLM Wiki 기록 (변경 이력) — **필수**
- **파일을 수정하면, 그 수정 내용을 [`llm-wiki/`](./llm-wiki/)에 기록한다.** (Obsidian Vault로 열 수 있는 마크다운 노트 모음.)
- 기록 항목: 날짜, 수정한 파일, 변경 요약, 그리고 **전체 diff**.
- 원격 환경에서는 사용자 PC의 로컬 Obsidian Vault에 직접 쓸 수 없으므로, 저장소 안의 `llm-wiki/`에 남기고 Obsidian으로 열거나 동기화한다.

---

## 아키텍처 메모

> ⚠️ 현재 활성 백엔드 소스는 `backend/`가 아니라 타임스탬프 백업 디렉터리
> (`backend.bak.YYYYMMDD_HHMMSS/`)에 있습니다. 백엔드 작업 시 위치를 먼저 확인하세요.

- **`core/classifier_engine.py`** — 분류기 코어. PDF 추출, OCR 백엔드 선택, 문제 박스 검출, 프롬프트 생성, 추론 캐시.
- **`core/ontology.py`** — 교육과정 온톨로지. `SUBJECTS`, `CURRICULUM`, 키워드 세트, `SUBJECT_ALIASES`/`SUB_SUBJECT_ALIASES`(줄임말 정규화). **분류 체계를 바꾸면 여기가 단일 출처(source of truth)다.**
- **`core/confidence.py`** — pro/anti/경쟁 증거 기반 신뢰도 보정.
- **`core/retrieval.py`** — 학습 DB(SQLite) RAG 검색.
- **`api/server.py`** — FastAPI 서버. 분류는 백그라운드 태스크로 돌고, `TASKS`/`CAPTURES`는 `CLASSI_MAX_TASKS` 상한으로 evict된다.
- **`pipeline/auto_deeplearn.py`** — 현행 자동 딥러닝 오케스트레이션. (`gamma_consumer.py`는 DEPRECATED.)
- **`tools/english_analyzer.py`** — 독립 실행 영어 지문 분석 CLI.
- **`frontend/classi_index.html`** — 단일 페이지 웹 UI. `firebase-auth.js`로 소셜 로그인.

---

## 자주 쓰는 명령어

```bash
# 단위 테스트 (ollama·OCR 불필요, 결정적)
cd backend && python3 -m unittest tests.test_engine -v

# API 서버만 실행
cd backend/api && uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 전체 스택 (Ollama 먼저 실행되어 있어야 함)
cd scripts && ./run.sh
```

---

## 동작에 영향을 주는 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 서버 주소 |
| `CLASSI_OCR` | `paddle` | OCR 백엔드(`paddle`/`tesseract`) |
| `CLASSI_OCR_DET` | `mobile` | 검출기(`mobile` 빠름 / `server` 정확·느림 ~12배) |
| `CLASSI_FORMULA` | `0` | `1`이면 LaTeX 수식 인식 추가(느림·발열↑) |
| `CLASSI_MAX_TASKS` | `50` | 메모리 상한용 태스크 보관 개수 |

---

## 주의사항

- **크리덴셜을 커밋하지 마세요.** 텔레그램 다운로더는 `TELEGRAM_API_ID`/`TELEGRAM_API_HASH` 환경변수를 우선합니다. `firebase-auth.js`의 config는 플레이스홀더이며 실제 키로 채운 채 커밋하지 않습니다. `scripts/yubin_session.session`(텔레그램 세션)도 공유 금지.
- OCR/LLM 추론은 CPU·이벤트 루프 부하가 크므로, 서버에서는 `asyncio.to_thread`로 오프로드하는 기존 패턴을 유지하세요.
- 분류 결과 스키마는 `Classification`(pydantic) 모델을 따릅니다.

---

## 오류 기록 로그

> 작업 원칙 **5. 오류 기록**에 따라, 코딩 중 발생한 오류는 여기에 한 줄씩 누적합니다.
> (오래된 항목을 지우지 말고 계속 아래에 추가하세요.)

| 날짜 | 위치(파일/명령) | 증상 | 원인 | 해결 |
|------|------------------|------|------|------|
| _예시_ | `core/classifier_engine.py` | `ModuleNotFoundError: paddleocr` | OCR 의존성 미설치 | `pip install paddleocr` 또는 `CLASSI_OCR=tesseract`로 폴백 |

---

## LLM Wiki (변경 이력)

> 작업 원칙 **6. LLM Wiki 기록**에 따라, 파일을 수정할 때마다 변경 내용을 [`llm-wiki/`](./llm-wiki/)에 남깁니다.

- **위치**: 저장소 내 `llm-wiki/` 폴더(Obsidian Vault로 열거나 동기화).
- **형식**: 수정 1건당 노트 1개(또는 누적 로그)에 다음을 적는다 —
  - 날짜·시각, 수정한 파일 경로, 변경 요약, **전체 diff**(```diff 코드블록).
- **이유**: 원격(클라우드) 컨테이너에서는 로컬 Obsidian Vault에 직접 쓸 수 없어, 저장소 안에 두고 Obsidian으로 연다.

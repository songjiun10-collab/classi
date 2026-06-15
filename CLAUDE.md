# CLASSI

한국 수능/내신 PDF 문제를 비전 LLM으로 자동 분류하는 도구. PDF에서 문항을 추출해
과목·세부과목·단원·주제·난이도로 분류하고, 지식 그래프·검수 UI·Obsidian 연동을 제공한다.

## 아키텍처

```
PDF 업로드 → 문항 추출(PyMuPDF/OCR) → 비전 LLM 분류(Ollama) → 후처리(프라이어·신뢰도 보정) → 결과
```

- **frontend/classi_index.html** — 단일 HTML 파일 SPA(빌드 단계 없음). 모든 UI·로직·스타일이 한 파일에 인라인.
  - `frontend/firebase-auth.js` — 소셜 로그인 보조 스크립트
  - API 베이스: `file://`이면 `http://localhost:8000`, 아니면 `location.origin`
- **backend** — FastAPI 서버 + 비동기 분류 파이프라인
  - `api/server.py` — 엔드포인트: `POST /api/classify`(백그라운드 태스크), `GET /api/status/{id}`,
    `GET /api/captures/{id}`, `GET /api/models`, `POST /api/extract-problems`, `GET /health`
  - `core/classifier_engine.py` — 추출·프롬프트·프라이어·추론 캐시
  - `core/ontology.py` — 교육과정 온톨로지(`SUBJECTS`, `CURRICULUM`, 별칭)
  - `core/confidence.py` — `calibrate_confidence()` 신뢰도 보정
  - `pipeline/`, `downloaders/`, `review/` — 수집·소비·검수 도구
- **scripts/run.sh** — Ollama 확인 → 다운로더 → 컨슈머 → uvicorn 순으로 전체 기동

> ⚠️ **현재 백엔드 코드는 `backend.bak.20260601_174602/` 에 있고 `backend/`는 비어 있다.**
> `run.sh`는 `../backend/...` 경로를 참조하므로, 백엔드를 실제로 돌리려면 코드를 `backend/`로
> 복원하거나 경로를 맞춰야 한다.

## 실행

```bash
# 사전: Ollama 실행 + 비전 모델(gemma4 등) pull
ollama serve && ollama pull gemma4

# API 서버 (프런트엔드도 / 경로에서 함께 서빙됨)
cd backend/api && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload

# 또는 전체 파이프라인
bash scripts/run.sh
```

프런트엔드만 단독으로 열려면 `frontend/classi_index.html`을 브라우저로 직접 열면 된다
(이때 API는 `localhost:8000`을 가리킨다).

## Obsidian 연동

`docs/obsidian/README.md` 참고. 두 경로가 있다:

1. **기출 분류 결과 → 볼트** (앱): 설정 → 데이터 관리
   - "Obsidian 내보내기" — `.md` 노트 묶음 `.zip` 다운로드(모든 브라우저)
   - "볼트 연결" — 폴더 지정 후 분류 시마다 자동 저장(File System Access API, Chromium 전용)
   - 핵심 함수: `obsidianFiles()`(노트 생성), `buildZip()`(순수 JS ZIP), `syncObsidianVault()`
2. **Claude Code 개발 로그 → Obsidian** (훅): `.claude/settings.json`의 PostToolUse 훅이
   `scripts/hooks/obsidian_devlog.py`를 실행 → `docs/obsidian/dev-log/`에 변경 기록.
   `OBSIDIAN_VAULT` 환경변수 지정 시 로컬 볼트에도 미러.

## 규칙·관례

- **주석·UI 문자열은 한국어.** 기존 코드 톤(간결, 의도 설명 위주)을 따른다.
- 프런트엔드는 **단일 파일**이다. 외부 의존성/번들러를 추가하지 말고 인라인으로 유지한다.
- 비전 추론 실패는 문항 단위로 격리해 "미분류"로 폴백(전체 태스크를 실패시키지 않음).
- 결정론적 후처리(프라이어·신뢰도)는 캐시하지 않고 매번 재적용한다.
- 분류 레이블은 `core/ontology.py`의 `CURRICULUM`을 기준으로 한다.

## 작업 브랜치

기능 개발 브랜치: `claude/obsidian-notes-linking-3fyiod` (메인 PR: #3).

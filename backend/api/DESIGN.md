# Classi API 설계

`backend/api/server.py` (FastAPI) 기준. 코드가 단일 소스이고 이 문서는 지도다.

실행: `python3 -m uvicorn server:app --host 0.0.0.0 --port 8000` (backend/api/ 에서)

## 개요

PDF 시험지를 업로드하면 문항 단위로 잘라 OCR 텍스트와 캡처 이미지를 추출하고,
Ollama 비전 모델(`gemma4` 등)로 과목/세부과목/단원/난이도를 분류한다. 분류는
PDF당 즉시 `task_id`를 반환하고 백그라운드로 진행되며, 클라이언트는 `/api/status`로
폴링한다. 분류 결과는 신뢰도 보정(`core/confidence.py`)을 거치고, 저신뢰·격리 문항은
영속 리뷰 큐(`core/review_log.py`)에 쌓여 사람이 정답을 확정하면 보정 모델 재학습에
쓰인다(데이터 플라이휠, 자세한 수식은 `backend/docs/calibration_and_formula.md`).

## 보안 모델 (중요)

**인증/인가가 전혀 없다.** CORS는 `allow_origins=["*"]`로 전체 개방. 모든 엔드포인트가
요청자 구분 없이 동작하며 업로드/리뷰 데이터에 접근 제어가 없다. 로컬/사내망 단일 사용자
전제로 설계됨 — 공개 인터넷에 노출하려면 인증 미들웨어와 업로드 크기/레이트 제한을
먼저 추가해야 한다.

## 엔드포인트

### 분류
- `POST /api/classify` — `file`(PDF), `model`(기본 `gemma4`), `concurrency`(1~8, 기본 2),
  `force_classify`(해설서 오탐 시 일반 분류 강제). 즉시 `{task_id, status:"processing"}` 반환,
  실제 분류는 백그라운드(`_run_classify`)에서 진행.
- `GET /api/status/{task_id}` — `{status, progress, total, results, error?}`. `status`는
  `processing`→`completed`/`failed`. 진행 중 `results`는 경량화(text/evidence/rationale 제외),
  완료 시 전체 필드 포함.
- `GET /api/captures/{task_id}` — 문항별 캡처 PNG(base64) 목록. 태스크 없으면 404.
- `GET /api/needs-review/{task_id}?max_confidence=0.5` — 격리(`telemetry.needs_review`)·
  미분류·저신뢰 문항만 추려 반환(휴먼리뷰 우선순위용, 인메모리/세션 단위).

### 보조 추출
- `POST /api/extract-problems` — PDF만 받아 문항 박스 감지 후 페이지/문항별 캡처+텍스트
  반환(분류 없이 추출만, 디버깅·검수용).
- `POST /api/transcribe` — `file`(PDF), `zoom`(기본 3.0). 텍스트 레이어 있으면 무손실 추출,
  없으면 OCR. `.hwpx` 문서(base64) + 평문 텍스트 + 페이지별 문단 반환.

### 리뷰 로그 (영속, 데이터 플라이휠)
- `GET /api/review-log?limit=100&source_pdf=` — 누적 리뷰 대상(신뢰도 낮은 순), 세션 넘어 영속.
- `POST /api/review-log/resolve` — `source_pdf, page, problem_num, gold_subject?`. 사람이 정답
  확정 → 큐에서 제거, `gold_subject` 있으면 학습 타깃으로 기록.
- `GET /api/review-log/stats` — 전체/미해소/해소/사유별 집계.
- `POST /api/review-log/clear-resolved` — 해소 완료 항목 영구 삭제.

### 보정 학습
- `POST /api/calibration/train?min_samples=10` — review log의 gold 라벨로 신뢰도 보정
  로지스틱 가중치를 재학습(`source="gold"`)하고 즉시 메모리에 반영. 데이터 부족/단일
  클래스면 `{trained:false}`.

### 기타
- `GET /api/models` — Ollama에 설치된 비전 계열 모델 목록(화이트리스트 매칭, 없으면 설치된
  전체 반환; Ollama 불통 시 빈 목록).
- `GET /health` — `{status:"ok", ollama_host}`.
- `GET /` — `frontend/classi_index.html` 정적 서빙(별도 프런트 서버 불필요).

## 핵심 설계 노트

- **태스크 저장은 인메모리**(`TASKS`, `CAPTURES` dict, 재시작 시 소실). 개수 상한
  (`CLASSI_MAX_TASKS`, 기본 50)과 캡처 바이트 상한(`CLASSI_MAX_CAPTURE_BYTES`, 기본 256MB)을
  넘으면 가장 오래된 완료/실패 태스크부터 제거 — 장시간 구동 시 OOM 방지.
- **문항 단위 격리**: 한 문항의 모델/파싱 실패가 전체 PDF 분류를 실패시키지 않고
  `_fallback()`으로 "미분류"+`needs_review`로 표시해 리뷰 큐로 보냄.
- **스키마 강제**: Ollama 호출 시 `format=Classification.model_json_schema()`로 자유 텍스트
  출력을 차단해 파싱 실패로 인한 재추론(발열)을 줄임.
- **추론 캐시**: 동일 (모델, 이미지, 프롬프트) 조합은 sqlite 캐시에서 재사용. 캐시 IO는
  `asyncio.to_thread` + 전역 `Semaphore(1)`로 직렬화해 이벤트 루프를 막지 않으면서도
  동시 접근을 막음.
- **해설서(답지) 자동 스킵**: 표지/파일명으로 해설서로 판정되면 분류를 건너뛰고 캡처만
  보관 — 과목 분류가 고신뢰 오답으로 보정 데이터를 오염시키는 것을 방지.
- **업로드 경로는 항상 고정 파일명**(`upload.pdf`)으로 저장 — 원본 파일명을 그대로 쓰면
  path traversal에 노출되므로, 원본 파일명은 분류 프라이어용 메타데이터로만 별도 전달.

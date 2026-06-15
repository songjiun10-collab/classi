# classi

한국 수능·내신 학습자료(PDF·문제집)를 **과목 / 세부과목 단위로 자동 분류**하는 시스템.
스캔본·디지털 PDF에서 문항을 검출하고, 비전 모델과 교육과정 온톨로지 증거를 결합해
문항별로 `과목 → 세부과목`을 라벨링하고 신뢰도를 함께 산출한다.

## 구조

```
classi/
├── backend/        분류 엔진·온톨로지·API·파이프라인 (Python, 자체 git 저장소)
│   ├── core/       classifier_engine · ontology · confidence · retrieval · review_log
│   ├── api/        FastAPI 서버(server.py)
│   ├── downloaders/  소스 PDF 수집(kice · legendstudy · telegram)
│   ├── pipeline/   보정 학습(calibration_trainer) — 데이터 플라이휠
│   ├── tools/      PDF 추출·포맷 변환(docx/hwpx)·정확도 평가·OCR 벤치
│   ├── tests/      결정론적 단위 테스트(ollama·OCR 불필요)
│   └── docs/       설계·세션 로그
├── frontend/       단일 페이지 UI(classi_index.html) + Firebase 인증
├── scripts/        실행 스크립트(run.sh)
├── data/           raw / processed / exports
└── models/
```

## 동작 개요

1. **수집** — `downloaders/`가 출처별 PDF를 가져온다.
2. **문항 검출** — `core/classifier_engine`이 PyMuPDF·PaddleOCR로 텍스트/레이아웃을
   추출하고, 단(段) 인식·문제번호 LIS로 문항 박스를 잘라낸다(세트 문항·맨숫자 스타일 지원).
3. **사전 증거(pre_classify)** — `core/ontology`의 증거 테이블로 과목·세부과목 키워드,
   강사명·출판사 브랜드, 파일명·표지 신호를 스캔해 pro/anti 증거를 만든다.
4. **분류** — 비전 모델(ollama)이 문항 이미지를 보고 `과목/세부과목`을 JSON으로 반환,
   온톨로지 역방향 맵으로 과목/세부과목 혼동을 복구한다.
5. **신뢰도 보정** — `core/confidence`가 증거 카운트·경합·규칙충돌로 신뢰도를 보정한다.
   사람이 검수한 gold 라벨이 쌓이면 `pipeline/calibration_trainer`가 가중치를 학습한다.
6. **검수·학습 루프** — `core/review_log`가 예측·근거를 영속화하고, 검수 결과가 다시
   보정 학습으로 환류된다(데이터 플라이휠).

분류 체계와 도메인 지식(과목·세부과목·강사·브랜드·증거 키워드)은 모두
[`backend/core/ontology.py`](backend/core/ontology.py)에 단일 소스로 모여 있다.

## 실행

```bash
# 백엔드 테스트(결정론적 — 모델·OCR 없이 실행)
cd backend && python3 -m pytest tests/ -q

# API 서버
cd backend/api && python3 -m uvicorn server:app --host 0.0.0.0 --port 8000

# 전체 파이프라인(다운로더+소비자+서버)
scripts/run.sh   # ollama 실행 필요
```

> `backend/`는 자체 git 저장소다. 백엔드 변경은 `backend/` 안에서 커밋한다.

## 문서

- 설계·보정·수식 받아쓰기: [`backend/docs/`](backend/docs/)
- API 설계: [`backend/api/DESIGN.md`](backend/api/DESIGN.md)

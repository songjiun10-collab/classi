# Classi 📚

> 수능·모의고사 PDF를 **과목 / 세부과목 / 단원**으로 자동 분류하는 증거 기반(Evidence-based) 문항 분류 엔진

Classi는 PDF로 흩어져 있는 한국 수능(CSAT) 기출·모의고사 문제를 받아, OCR과 로컬 LLM(Ollama)으로 문항을 추출하고 교육과정 체계에 맞춰 자동 분류합니다. 분류 결과는 사람이 검수하고, 검수 데이터를 다시 학습에 반영하는 **자동 딥러닝 루프**로 정확도를 끌어올립니다.

---

## ✨ 주요 기능

- **PDF → 문항 추출**: PyMuPDF(fitz)로 페이지를 렌더링하고 문제 박스를 검출해 개별 문항으로 분리
- **OCR 다중 백엔드**: PaddleOCR(한국어·수식 강함, 기본) / Tesseract 폴백, 선택적 LaTeX 수식 인식
- **증거 기반 분류**: 교육과정 온톨로지(키워드·별칭) + 파일명/표지 prior + 로컬 LLM(Ollama) 비전 추론
- **신뢰도 보정(Confidence Calibration)**: pro/anti/경쟁 증거를 합산해 분류 신뢰도를 정밀 보정
- **RAG 검색**: 과거 학습 DB에서 관련 개념·오답 패턴·풀이 전략을 조회
- **자동 파이프라인**: 텔레그램 아카이브·평가원·전국연합 PDF 자동 수집 → 분류 → 검수 → 재학습
- **웹 UI + REST API**: FastAPI 서버와 단일 HTML 프런트엔드, Firebase 소셜 로그인
- **영어 지문 분석기**: 영어 문항을 구문 분석 + 한국어 번역 + 논리 흐름(Mermaid/SVG)으로 시각화

---

## 🗂️ 교육과정 분류 체계

| 과목 | 세부과목 |
|------|----------|
| 국어 | 독서, 문학, 화법과 작문, 언어와 매체 |
| 수학 | 수학Ⅰ, 수학Ⅱ, 미적분, 확률과 통계, 기하 |
| 영어 | 영어Ⅰ, 영어Ⅱ |
| 한국사 | 한국사 |
| 사회탐구 | 생활과 윤리, 윤리와 사상, 한국지리, 세계지리, 동아시아사, 세계사, 경제, 정치와 법, 사회·문화 |
| 과학탐구 | 물리학Ⅰ·Ⅱ, 화학Ⅰ·Ⅱ, 생명과학Ⅰ·Ⅱ, 지구과학Ⅰ·Ⅱ |
| 통합사회 | 인간·사회·환경, 자연환경과 인간, 생활공간과 사회, 인권과 정의, 시장경제, 세계화, 지속가능사회 |
| 통합과학 | 물질과 규칙성, 시스템과 상호작용, 변화와 다양성, 환경과 에너지, 과학과 미래사회 |
| 제2외국어 | 독일어Ⅰ, 프랑스어Ⅰ, 스페인어Ⅰ, 중국어Ⅰ, 일본어Ⅰ, 러시아어Ⅰ, 아랍어Ⅰ, 베트남어Ⅰ |
| 한문 | 한문Ⅰ |

`화작`, `언매`, `수1`, `확통`, `물리1` 같은 줄임말 별칭(alias)도 자동으로 정규화됩니다.

---

## 🏗️ 프로젝트 구조

```
classi/
├── frontend/
│   ├── classi_index.html      # 단일 페이지 웹 UI
│   └── firebase-auth.js       # Firebase 소셜 로그인(구글/애플/이메일)
├── backend/                   # 백엔드 코드(아래 모듈 구성)
│   ├── api/
│   │   └── server.py          # FastAPI 서버 (/api/classify 등)
│   ├── core/
│   │   ├── classifier_engine.py  # 분류기 코어(추출·OCR·프롬프트·캐시)
│   │   ├── ontology.py           # 교육과정 온톨로지(과목/단원/키워드)
│   │   ├── confidence.py         # 신뢰도 보정
│   │   └── retrieval.py          # 학습 DB RAG 검색
│   ├── downloaders/           # 텔레그램·평가원·레전드스터디 수집기
│   ├── pipeline/
│   │   ├── auto_deeplearn.py     # 자동 딥러닝 오케스트레이션
│   │   └── gamma_consumer.py     # (구버전) 감시·분류·검수 루프
│   ├── review/                # 검수 CLI
│   ├── tools/
│   │   └── english_analyzer.py   # 영어 지문 분석 CLI
│   └── tests/                 # 단위 테스트(ollama·OCR 불필요)
└── scripts/
    └── run.sh                 # 전체 스택 실행 스크립트
```

> ℹ️ 현재 활성 백엔드 소스는 타임스탬프가 붙은 백업 디렉터리
> (`backend.bak.YYYYMMDD_HHMMSS/`)에 보존되어 있습니다.

---

## 🚀 시작하기

### 사전 요구사항

- **Python 3.10+**
- **[Ollama](https://ollama.com)** — 로컬 비전 LLM 추론 (예: `ollama pull gemma3` 등 비전 모델)
- OCR 의존성: `paddleocr` 또는 `pytesseract`(+ Tesseract 바이너리)
- PDF 처리: `PyMuPDF`(fitz)

### 설치

```bash
git clone https://github.com/songjiun10-collab/classi.git
cd classi

# 가상환경 권장
python3 -m venv .venv && source .venv/bin/activate

# 핵심 의존성
pip install fastapi "uvicorn[standard]" pydantic ollama PyMuPDF \
            paddleocr pillow imagehash pytesseract \
            scikit-learn numpy aiohttp telethon
```

### 실행

**1) 빠르게 API 서버만 띄우기**

```bash
cd backend/api
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

브라우저에서 `http://localhost:8000` 접속 → 웹 UI에서 PDF 업로드 후 분류.

**2) 전체 스택 실행**

```bash
# Ollama가 먼저 실행되어 있어야 합니다
ollama serve &

cd scripts
./run.sh
```

`run.sh`는 ① Ollama 확인 → ② PDF 다운로더 → ③ 분류·검수·학습 소비자 → ④ API 서버를 순서대로 띄웁니다.

---

## 🔌 API 엔드포인트

| 메서드 | 경로 | 설명 |
|--------|------|------|
| `GET`  | `/` | 웹 UI(`classi_index.html`) 서빙 |
| `POST` | `/api/classify` | PDF 업로드 → 백그라운드 분류 태스크 시작 |
| `GET`  | `/api/status/{task_id}` | 분류 진행 상황·결과 조회 |
| `GET`  | `/api/captures/{task_id}` | 문항별 캡처 이미지(base64) 조회 |
| `POST` | `/api/extract-problems` | 문항 추출만 수행 |
| `GET`  | `/api/models` | 사용 가능한 Ollama 모델 목록 |
| `GET`  | `/health` | 헬스 체크 |

---

## ⚙️ 환경 변수

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama 서버 주소 |
| `CLASSI_OCR` | `paddle` | OCR 백엔드(`paddle` / `tesseract`) |
| `CLASSI_OCR_DET` | `mobile` | 검출기(`mobile` 빠름 / `server` 정확·느림) |
| `CLASSI_FORMULA` | `0` | `1`이면 영역별 LaTeX 수식 인식 추가 |
| `CLASSI_MAX_TASKS` | `50` | 메모리 상한을 위한 태스크 보관 개수 |
| `ENGLISH_ANALYZER_MODEL` | `gemma4` | 영어 분석기 LLM 모델 |
| `TELEGRAM_API_ID` / `TELEGRAM_API_HASH` | — | 텔레그램 다운로더 크리덴셜 |

---

## 🧪 테스트

ollama·OCR 없이 결정적으로 도는 단위 테스트가 포함되어 있습니다.

```bash
cd backend
python3 -m unittest tests.test_engine -v
```

---

## 📝 라이선스

별도 명시가 없는 한 본 저장소는 비공개/연구용입니다. 라이선스가 필요하면 추가해 주세요.

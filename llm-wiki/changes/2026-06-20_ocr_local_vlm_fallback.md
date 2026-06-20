# 2026-06-20 · OCR에 로컬 VLM(비전 모델) 3차 폴백 추가

## 배경
olma(ai-assistant 브랜치)가 도입한 "tesseract 결과가 비면 로컬 Ollama 비전 모델로 한 번 더
전사" 패턴을 classi OCR 계층에 이식했다. 스캔 품질 문제 등으로 Paddle·Tesseract가 모두 빈
문자열을 줄 때 마지막으로 한 번 더 건지기 위함. Karpathy 가이드라인(최소·외과적 변경,
게이트로 기본 동작 불변, 테스트 검증)에 맞춰 OCR 상류만 손대고 분류 로직은 건드리지 않았다.

## 원칙 준수
- **분류 로직 불변**: `core/ocr.py`(분류 상류, 단방향 의존)만 수정. classifier_engine·confidence·
  ontology 미변경.
- **"모든 추론은 로컬" 유지**: 외부 클라우드가 아니라 로컬 Ollama 비전 모델 호출(classi가 이미
  쓰는 `ollama` 패키지 재사용, `OLLAMA_HOST`).
- **기본 동작 바이트 불변**: `CLASSI_OCR_VLM` 미설정(기본)이면 폴백 자체가 비활성 — 기존
  Paddle→Tesseract 경로와 완전히 동일.

## 변경 파일
- `backend/core/ocr.py` — `CLASSI_OCR_VLM` 설정 + `_vlm_from_image()` 헬퍼 + `ocr_from_image()`에
  게이트된 3차 폴백 1블록.
- `backend/tests/test_ocr_vlm.py` — 신규, unittest 6건(외부 의존 없이 모킹).

## 변경 요약
1. `_OCR_VLM = os.environ.get("CLASSI_OCR_VLM", "").strip()` — 로컬 비전 모델명(예: `qwen2.5vl:7b`).
   비우면 폴백 off(기본).
2. `_vlm_from_image(img_bytes)` — `ollama.Client(host=OLLAMA_HOST).generate(model, prompt, images=[b64])`.
   `ollama`는 지연 import. 미설치/모델없음/타임아웃 등 어떤 예외도 `""`로 흡수해 OCR 경로를 막지
   않음.
3. `ocr_from_image()` — Paddle·Tesseract가 모두 빈 결과이고 `CLASSI_OCR_VLM`이 설정된 경우에만
   `_vlm_from_image`를 호출(`or txt`로 VLM도 실패하면 종전 빈 결과 유지).

## 검증
- `cd backend && python3 -m unittest tests.test_ocr_vlm -v` → 6 ok.
- 회귀 `python3 -m unittest tests.test_engine` → 179 ok.
- 테스트 커버: Paddle 히트시 VLM 미호출 / Tesseract 히트시 미호출 / 둘 다 비고 env 없으면 미호출 /
  둘 다 비고 env 있으면 호출·반환 / 전사 성공시 모델·이미지 전달 / ollama 실패시 "" 흡수.

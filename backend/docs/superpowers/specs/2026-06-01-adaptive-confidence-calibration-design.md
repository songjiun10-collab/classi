# 결정 기록 — 적응형 신뢰도 보정은 보류, 변환기 기능을 출하 (2026-06-01)

> 이 문서는 원래 "적응형 신뢰도 보정(학습 루프 폐쇄)" 설계서였으나, 구현 직전 코드 검증으로
> 전제가 뒤집혀 **보류 결정**으로 전환됨. 발견을 보존하기 위해 결정 기록으로 남긴다.

## 1. 무엇을 하려 했나
검수(gold)가 쌓일수록 신뢰도가 실제 정답률에 맞게 자동 보정되는 자기개선 루프를 닫으려 했다.
설계: `calibrate_confidence`가 `calibration_weights.json`을 게이팅·경계(±0.15)·폴백으로 소비,
트레이너는 `gold==pred`로 학습, `review_cli pending` 노출.

## 2. 왜 보류했나 (코드로 검증된 사실)
4관점 self-review + 직접 grep으로 확인:
- `server.py`는 문제 PNG를 디스크에 쓰지 않고 base64로 HTTP 반환만 한다(결과 dict에 **gold 없음**).
- `_telemetry.jsonl`을 쓰는 프로덕션 코드가 **없다**(테스트만, `tests/test_review_cli.py`).
- `review_cli`가 `ingest`로 읽는 **PNG 폴더 레이아웃 + telemetry 생산자**(과거 삭제된 `csat_v19_51.py`)가
  현 코드베이스에 부재.
- 그 결과 `gold==pred` 트레이너는 입력이 도달 불가, 소비자는 self-target 거부(S4)로 **영구 폴백**.
  유일한 gold 트레이너 `gamma_consumer`도 DEPRECATED + 누락 스크립트 가드로 도달 불가.

→ 즉 `retrieval.py`·`review_cli_v3`·`gold==pred 트레이너`·`calibration 소비자`는 **데이터 생산자가
사라진 잔존(vestigial) 인프라**다. 루프를 닫으려면 삭제된 배치 생산자를 **재건축**해야 한다.

## 3. CLAUDE.md 판정
이 기능의 최대 효과는 안전상 ±0.15 신뢰도 조정으로 제한된다. 그 payoff를 위해 삭제된 서브시스템을
재건하는 것은 CLAUDE.md **§2(Nothing speculative / minimum code) 위반**이고 **§1(더 단순한 길이 있으면
푸시백)** 대상이다. 사용자 제약("CLAUDE.md 위반 불가")에 따라 **보류**한다.
되살리려면 별도 마일스톤에서 "PDF→문제PNG폴더+telemetry(구조특징·raw_confidence 포함)" 배치 생산자를
먼저 복원해야 한다(그때 본 설계의 소비자/트레이너가 의미를 가진다).

## 4. 대신 출하한 것 — 변환기 기능 (살아있는 아키텍처)
`tools/pdf_to_hwpx.py` 변환기를 라이브 FastAPI에 연결하고 OCR 출력 품질을 개선.
- **`_clean_transcription`** (pdf_to_hwpx): U+FFFD(`�`)·PUA(엔진 `_PUA_RE` 재사용)·제로폭/제어문자 제거 +
  공백 정규화. 로마숫자 Ⅰ/Ⅱ 등 실제 글자는 보존(내용 보존 우선). `transcribe_pdf`에서 텍스트레이어·OCR
  두 경로 산출물에 일괄 적용하고 빈 문단은 버린다.
- **`POST /api/transcribe`** (server.py): PDF 업로드 → 평문 텍스트 + `.hwpx`(base64) 반환.
  `/api/extract-problems` 패턴 재사용(고정 파일명·청크 스트리밍·`asyncio.to_thread`·tmp 정리).
- 검증: `tests/test_hwpx.py`(+TestCleanTranscription 4, +TestTranscribePdfWiring 1),
  `tests/test_server.py`(+TestTranscribeEndpoint 1). 전체 **113개 그린**, ollama 무관·결정론적.

## 5. 사용
```
curl -F file=@입력.pdf -F zoom=2.0 http://localhost:8000/api/transcribe
# → {"pages":N,"paragraphs":M,"text":"...","hwpx_filename":"입력.hwpx","hwpx_base64":"..."}
```
CLI는 종전대로: `python3 tools/pdf_to_hwpx.py 입력.pdf [-o out.hwpx] [--zoom 2.0]`

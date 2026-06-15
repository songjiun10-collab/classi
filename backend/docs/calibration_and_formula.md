# 신뢰도 보정 플라이휠 & 수식 받아쓰기

이 세션(2026-06-02)에 완성/폐쇄한 두 서브시스템의 레퍼런스. 코드가 단일 소스이고, 이 문서는 지도다.

## 1. 신뢰도 보정 데이터 플라이휠 (폐쇄됨)

분류 신뢰도를 **사람 검수 gold**로 학습해 보정한다. 루프:

```
분류(server _run_classify)
  → calibrate_confidence: 증거로 신뢰도 보정 + telemetry(pro_match·anti_match·competing·
                          rule_conflict·raw_confidence) 산출            [core/confidence.py]
  → review_log.log_results: 저신뢰·미분류·격리 문항만 telemetry째 영속     [core/review_log.py]
  → 사람이 resolve(gold_subject) 로 정답 라벨링            [POST /api/review-log/resolve]
  → calibration_trainer: gold==예측을 타깃으로 로지스틱 학습 →
       source="gold" 가중치 저장                          [pipeline/calibration_trainer.py]
       (실행: python3 -m pipeline.calibration_trainer  또는  POST /api/calibration/train)
  → calibrate_confidence: gold 태그 가중치가 있으면 로지스틱으로 보정,
       없으면 하드코딩 폴백(무회귀)                         [_load_calib_weights / _learned_confidence]
```

### 핵심 설계
- **gold 태그 게이트(안전장치)**: 소비자는 `source=="gold"` 가중치만 신뢰한다. `auto_deeplearn`의
  자기예측(순환신호) 가중치는 `source="self_predicted"`라 **무시** → 순환신호가 추론을 오염시키지 않음.
- **no-op 폴백**: 가중치 파일 없음/형식오류/비-gold → 기존 하드코딩 보정과 바이트 동일. gold가 쌓이기
  전엔 휴면 상태로 안전.
- **특징 정합(중요)**: confidence 특징은 보정 *전* `raw_confidence`를 쓴다(telemetry에 적재). 소비자
  `_learned_confidence`도 raw_conf를 입력하므로 학습/추론 의미가 일치한다. (저장되는 res.confidence는
  보정 *후* 값이라 특징으로 부적합 — 트레이너는 raw_confidence 우선, 구행은 폴백.)
- **위치**: `~/.classi/calibration_weights.json`(또는 `$CLASSI_CALIB_WEIGHTS`). infer 캐시와 같은 `~/.classi/`.

### 한계 / 운용
- review_log는 **리뷰 대상(저신뢰·미분류·격리)만** 적재 → gold 학습셋이 그쪽으로 편향(보정이 가장 필요한
  영역이긴 함). 충분한 gold(기본 ≥10건, 두 클래스)가 쌓인 뒤 학습이 의미 있음.
- 죽은 `gamma_consumer.py`(없어진 스크립트 의존·고아 gold.jsonl)와 순환 `auto_deeplearn.train_calibration`을
  이 트레이너가 대체한다.

## 2. 스캔본 수식 → LaTeX 받아쓰기

`pdf_to_hwpx`(PDF→hwpx 받아쓰기)에서 스캔 페이지의 수식을 LaTeX로 복원해 인라인 병합한다.

```
스캔 페이지(텍스트레이어 없음) 렌더
  → LayoutDetection: 'formula' 영역 박스 검출            [core.classifier_engine._get_paddle_layout]
  → 각 영역 크롭 → FormulaRecognition → LaTeX            [core.classifier_engine.formula_regions]
  → _merge_formula_lines: 수식 bbox 안 텍스트 잔해 제거 + '$LaTeX$'를 좌표로 삽입  [tools/pdf_to_hwpx.py]
  → _assemble_reading_order가 인라인 위치에 자동 배치
```

- **옵트인**: `--formula` 플래그 또는 `CLASSI_FORMULA=1`(기본 off — 영역마다 인식이라 발열↑).
- **재사용**: 이미 있던 `FormulaRecognition`(분류 핫패스의 거친 전페이지 방식)을 받아쓰기에선 영역 단위로
  정밀 적용. paddleocr 내장(torch/pix2tex 불필요).
- born-digital PUA(□) 수식은 실코퍼스에 희박(검증)해 후순위. 텍스트레이어 PDF는 이 경로 미적용(무손실).
- 검증: 한완수 공통 p14에서 24식 인라인 복원(예: `$\frac{x^{2}-1}{x-1}=x+1$`).

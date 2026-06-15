# 2026-06-15 · autoresearch 학습용 복제 하니스 추가

## 배경
karpathy/autoresearch **이론**을 classi 코드베이스에서 학습용으로 복제(사용자 요청:
"학습용으로 복제해서 코드에 적용"). 범위는 **독립 학습용 하니스**로 확정 — 분류 본코드
(`core/`)를 일절 건드리지 않는 별도 모듈. (직전 영어 debate가 4/4로 '자동개선 플라이휠은
shelve가 옳다'고 결론냈으므로, 분류 보정 자동탐색이 아닌 격리된 교육용 복제로 한정.)

## autoresearch 이론 = 과학적 방법의 자동화
가설(코드 변경) → 통제된 실험(고정 예산) → 불변 측정(val_bpb) → 채택/기각 → 반복.
핵심 통제: 예산 고정(공정/compute-normalized 비교), per-byte 메트릭(게이밍 차단),
prepare(고정)/train(탐색) 분리.

## 추가 파일 (`backend/autoresearch/`, 신규)
- `__init__.py` — 패키지 선언.
- `prepare.py` — 불변부: 데이터·dataloader·불변 메트릭 `eval_bpb`(validation bits-per-byte).
- `train.py` — 탐색공간: `CONFIG`(context_len/hidden/lr/batch_size) + numpy char-MLP + `train()`(고정 예산 SGD).
- `loop.py` — 오케스트레이터: 고정예산 실험 반복 + 변이 언덕오르기 + 리더보드.
- `program.md` — 사람이 고치는 "연구 조직 코드"(목표·불변 규칙·탐색 백로그).
- `README.md` — 이론 ↔ 복제 매핑표 + 실행법 + classi 연결.
- `tests/test_autoresearch.py` — 스모크 3건(결정적·<1s).

## 이론 ↔ 복제 매핑
| 요소 | 원본 | 복제 |
|---|---|---|
| 고정 예산 | 5분/실험 | `DEFAULT_BUDGET_SEC`(기본 3초) |
| 불변 메트릭 | val_bpb | 동일(바이트 단위, vocab 비의존) |
| 고정/탐색 분리 | prepare.py/train.py | 동일 |
| 연구 조직 코드 | program.md | 동일 |
| 에이전트 | LLM이 train.py 수정 | loop.py 결정적 변이(LLM·ollama 불필요) |
| 백본 | 단일 GPU nanochat | numpy char-MLP(CPU) |

## 검증
- `python3 -m autoresearch.loop` → val_bpb 8.0(균등) → ~0.18로 하강, KEEP/drop 정상, 경고 0.
- `python3 -m unittest tests.test_autoresearch` → 3 OK(<1s).
- `python3 -m unittest tests.test_engine` → **OK(회귀 없음)** — 분류 본코드 무영향 확인.

## 설계 메모
- numpy 전용(torch 없음), CPU·결정적(seed 고정). ollama 미사용(장시간 추론 회피).
- 발산(높은 lr) 런은 errstate+nan_to_num+조기종료로 '유한하지만 나쁜 bpb'를 내고 루프가
  정상 기각 → 평가가 경고/예외로 깨지지 않음.

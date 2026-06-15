# autoresearch (학습용 복제)

[karpathy/autoresearch](https://github.com/karpathy/autoresearch)의 **이론**을 classi
코드베이스에서 CPU·numpy만으로 재현한 교육용 하니스. 분류 본코드(`core/`)와 완전히
분리된 독립 모듈이며, 어떤 분류 동작도 바꾸지 않는다.

## autoresearch 이론 = 과학적 방법의 자동화
가설(코드 변경) → 통제된 실험(고정 예산) → 불변 측정(val_bpb) → 채택/기각 → 반복.
통제 변인을 고정하고 단일 변인만 바꿔 측정한다.

| 이론 요소 | 원본 autoresearch | 이 복제 |
|---|---|---|
| 고정 시간 예산 | 5분/실험(wall-clock) | `prepare.DEFAULT_BUDGET_SEC`(기본 3초) |
| 불변 메트릭 | validation bits-per-byte | 동일 `val_bpb`(바이트 단위, vocab 비의존) |
| 고정부 vs 탐색부 | `prepare.py` / `train.py` | `prepare.py`(불변) / `train.py`(CONFIG 탐색) |
| 연구 조직 코드 | `program.md` | `program.md` |
| 에이전트 | LLM이 train.py 수정 | `loop.py`의 결정적 변이 탐색(LLM 불필요) |
| 백본 | 단일 GPU nanochat | numpy 1-은닉층 char-MLP(CPU) |

## 왜 이렇게 통제하나
- **예산 고정** → "더 큰 모델이 좋다"가 아니라 "**같은 컴퓨트에서** 뭐가 좋나"를 묻는
  공정(compute-normalized) 비교.
- **per-byte 메트릭** → vocab을 키워 점수를 올리는 게이밍 차단.
- **prepare 고정** → 평가 기준이 흔들리지 않아 변경 효과를 단독으로 측정.

## 실행
```bash
cd backend
python3 -m autoresearch.loop          # 기준 CONFIG → 변이 탐색 → val_bpb 리더보드
```
출력 예: 각 라운드의 `bpb`/`steps`와 KEEP/drop, 마지막에 best config·bpb(균등분포 8.0 대비).

## 파일
- `prepare.py` — 불변부: 데이터·dataloader·`eval_bpb`(불변 메트릭). **수정 금지.**
- `train.py` — 탐색공간: `CONFIG`(context_len/hidden/emb_dim/lr/batch_size/optimizer/
  momentum/warmup_frac) + 학습 임베딩 + Adam·모멘텀 옵티마이저 + LR 워밍업·감쇠 스케줄.
- `loop.py` — 오케스트레이터: 고정예산 실험 반복 + 언덕오르기.
- `program.md` — 사람이 고치는 연구 조직 코드(목표·규칙·탐색 백로그).

## classi와의 연결
classi의 `CLAUDE.md` 원칙 "4. 목표 주도 실행(성공 기준 정의 → 검증될 때까지 반복)"이
바로 이 루프의 축약판이다. 실제 분류 보정(`core/confidence.py`)도 같은 사고로 작업했다:
불변 메트릭 = `tests/test_engine.py`(결정적), 탐색 = 최소 변경, 채택 = 회귀 없는 개선.
"""

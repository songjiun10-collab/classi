# program.md — 연구 조직 코드 (사람이 고치는 지시문)

karpathy/autoresearch에서 사람은 Python을 직접 고치지 않고 이 마크다운을 반복 개선한다.
이 파일은 "에이전트가 어떻게 연구할지"를 자연어로 프로그래밍하는 메타 계층이다.
(학습용 복제에서는 LLM 대신 `loop.py`의 결정적 변이 탐색이 이 지시를 따른다.)

## 목표 (불변)
- **최소화 대상**: validation bits-per-byte(`val_bpb`). 낮을수록 좋다.
- **상한 기준선**: 256 균등분포 = 8.0 bits/byte. 학습은 이보다 낮춰야 한다.

## 규칙 (불변 — 어기면 비교가 무의미해진다)
1. `prepare.py`는 **절대 수정 금지**(데이터·dataloader·메트릭 = 통제 변인).
2. 모든 실험은 **동일한 고정 벽시계 예산**(`prepare.DEFAULT_BUDGET_SEC`)으로 돈다.
3. `val_bpb`로만 채점한다. vocab/메트릭을 바꿔 점수를 올리지 않는다(게이밍 금지).

## 탐색공간 (여기를 바꿔라 — `train.py`의 `CONFIG`)
- `context_len` 1..8 — 문맥 길이
- `hidden` 8..256 — 은닉 차원
- `lr` 0.01..4.0 — 학습률
- `batch_size` 8..256 — 미니배치

## 시도해볼 아이디어 (사람이 갱신하는 백로그)
- [x] 더 효율적인 옵티마이저(모멘텀/Adam)로 같은 예산에 더 많은 유효 스텝. → `optimizer` 노브.
- [x] 임베딩(one-hot → 학습 임베딩)으로 입력 차원 축소 → 스텝당 비용 절감. → `emb_dim` 노브.
- [x] learning-rate 스케줄(워밍업 후 감쇠). → `warmup_frac` + `_sched_lr`.
- [ ] `context_len`↑의 정확도 이득 vs 스텝당 비용 증가의 트레이드오프 탐색.
- [ ] 코퍼스 확장/실데이터, 2-은닉층, 가중치 감쇠(weight decay).

## 성공 기준 (goal-driven)
- 기준 CONFIG 대비 `val_bpb`가 유의미하게 낮아지면 채택(loop가 자동 KEEP).
- 같은 예산에서 더 낮은 bpb = 더 나은 "연구 결과".

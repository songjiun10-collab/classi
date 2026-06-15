"""autoresearch (학습용) — 탐색공간(SEARCH SPACE).

karpathy/autoresearch의 `train.py`에 대응한다. **에이전트/사람이 자유롭게 바꾸는 곳**:
모델 구조와 하이퍼파라미터(CONFIG). 규칙은 단 둘 —
  1) prepare.py(통제 변인)는 건드리지 않는다.
  2) 고정 예산(DEFAULT_BUDGET_SEC) 안에서 val_bpb를 최소화한다.

여기 모델은 바이트 문맥(context_len개)을 입력받아 다음 바이트(256분류)를 맞히는
1-은닉층 MLP다. 더 좋은 구조/학습법을 넣어 같은 예산에서 더 낮은 bpb를 내는 게 목표.
"""
import time
import numpy as np

from . import prepare

# ── 탐색 노브(에이전트가 바꾸는 대상) ──────────────────────────────────────────
CONFIG = {
    "context_len": 4,   # 직전 몇 바이트를 볼지
    "hidden": 64,       # 은닉 차원
    "lr": 0.5,          # 학습률
    "batch_size": 64,   # 미니배치 크기
    "momentum": 0.9,    # 모멘텀 계수(0=순수 SGD). program.md 백로그 적용분.
}


class CharMLP:
    """one-hot(context_len×256) → tanh(hidden) → softmax(256). numpy 전용."""

    def __init__(self, context_len, hidden, rng):
        self.C = context_len
        din = context_len * prepare.VOCAB
        # 표준편차를 fan-in에 맞춰 초기화(폭주 방지).
        self.W1 = rng.standard_normal((din, hidden)) * (1.0 / np.sqrt(din))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.standard_normal((hidden, prepare.VOCAB)) * (1.0 / np.sqrt(hidden))
        self.b2 = np.zeros(prepare.VOCAB)
        # 모멘텀 속도 버퍼(파라미터별). momentum=0이면 순수 SGD와 동일.
        self.vW1 = np.zeros_like(self.W1); self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2); self.vb2 = np.zeros_like(self.b2)

    def _onehot(self, X):
        B = len(X)
        oh = np.zeros((B, self.C * prepare.VOCAB))
        cols = X.astype(np.int64) + (np.arange(self.C) * prepare.VOCAB)[None, :]
        oh[np.arange(B)[:, None], cols] = 1.0
        return oh

    def _forward(self, X):
        oh = self._onehot(X)
        h = np.tanh(oh @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        return oh, h, logits

    def log_probs(self, X):
        """prepare.eval_bpb가 요구하는 인터페이스: 자연로그 분포 (B,256)."""
        # 발산(높은 lr)으로 가중치가 inf/nan이 된 모델도 '유한하지만 나쁜' bpb를 내게 한다
        # → 루프가 그 config를 정상적으로 기각(drop). 평가가 경고/예외로 깨지지 않는다.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _, _, logits = self._forward(X)
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            logits -= logits.max(axis=1, keepdims=True)
            z = np.exp(logits)
            return logits - np.log(z.sum(axis=1, keepdims=True))

    def step(self, X, Y, lr, momentum=0.0):
        """미니배치 1회 모멘텀 SGD 업데이트. 평균 NLL(nats) 반환.
        v ← momentum·v − lr·grad ; param ← param + v (momentum=0이면 순수 SGD)."""
        oh, h, logits = self._forward(X)
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        B = len(X)
        loss = float(-np.log(p[np.arange(B), Y] + 1e-12).mean())
        dlogits = p
        dlogits[np.arange(B), Y] -= 1.0
        dlogits /= B
        dW2 = h.T @ dlogits
        db2 = dlogits.sum(axis=0)
        dh = (dlogits @ self.W2.T) * (1.0 - h * h)
        dW1 = oh.T @ dh
        db1 = dh.sum(axis=0)
        self.vW2 = momentum * self.vW2 - lr * dW2; self.W2 += self.vW2
        self.vb2 = momentum * self.vb2 - lr * db2; self.b2 += self.vb2
        self.vW1 = momentum * self.vW1 - lr * dW1; self.W1 += self.vW1
        self.vb1 = momentum * self.vb1 - lr * db1; self.b1 += self.vb1
        return loss


def train(config, train_data, budget_sec, seed):
    """고정 벽시계 예산이 끝날 때까지 SGD를 돌리고 모델을 반환.

    더 빠른/효율적인 config는 같은 예산 안에서 더 많은 유효 업데이트를 해 bpb를 낮춘다.
    이것이 '예산을 고정하면 변경 간 비교가 공정하다'는 autoresearch 통제의 핵심이다.
    """
    rng = np.random.default_rng(seed)
    model = CharMLP(config["context_len"], config["hidden"], rng)
    steps = 0
    deadline = time.monotonic() + budget_sec
    # 발산은 예상된 결과(루프가 기각)다. overflow/invalid 경고를 소음으로 흘리지 않고
    # 무시하되, 손실이 비유한(inf/nan)이 되면 예산을 낭비하지 말고 조기 종료한다.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        while time.monotonic() < deadline:
            X, Y = prepare.get_batch(train_data, config["context_len"], config["batch_size"], rng)
            loss = model.step(X, Y, config["lr"], config.get("momentum", 0.0))
            steps += 1
            if not np.isfinite(loss):
                break
    return model, steps

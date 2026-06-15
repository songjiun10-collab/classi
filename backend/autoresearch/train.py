"""autoresearch (학습용) — 탐색공간(SEARCH SPACE).

karpathy/autoresearch의 `train.py`에 대응한다. **에이전트/사람이 자유롭게 바꾸는 곳**:
모델 구조와 하이퍼파라미터(CONFIG). 규칙은 단 둘 —
  1) prepare.py(통제 변인)는 건드리지 않는다.
  2) 고정 예산(DEFAULT_BUDGET_SEC) 안에서 val_bpb를 최소화한다.

program.md 백로그 반영분:
  - **학습 임베딩**: one-hot(256) → 학습 임베딩(emb_dim)으로 입력 차원 축소(스텝당 비용↓).
  - **옵티마이저**: Adam | 모멘텀 SGD 선택.
  - **LR 스케줄**: 예산 경과 비율 기준 워밍업(선형 상승) 후 선형 감쇠.

모델: 바이트 문맥(context_len개) → 임베딩 → tanh(hidden) → softmax(256).
"""
import time
import numpy as np

from . import prepare

# ── 탐색 노브(에이전트가 바꾸는 대상) ──────────────────────────────────────────
CONFIG = {
    "context_len": 4,     # 직전 몇 바이트를 볼지
    "hidden": 64,         # 은닉 차원
    "emb_dim": 16,        # 학습 임베딩 차원(one-hot 256 대비 입력 축소)
    "lr": 0.01,           # 기준 학습률(Adam 스케일)
    "batch_size": 64,     # 미니배치 크기
    "optimizer": "adam",  # "adam" | "sgd"
    "momentum": 0.9,      # sgd일 때 모멘텀 계수
    "warmup_frac": 0.1,   # 예산의 앞 비율만큼 워밍업
}


class CharLM:
    """임베딩 → tanh(hidden) → softmax(256). Adam/모멘텀 옵티마이저 내장. numpy 전용."""

    def __init__(self, config, rng):
        C = config["context_len"]; H = config["hidden"]
        D = int(config.get("emb_dim", 16)); V = prepare.VOCAB
        self.C, self.D = C, D
        din = C * D
        self.P = {
            "E":  rng.standard_normal((V, D)) * 0.1,
            "W1": rng.standard_normal((din, H)) * (1.0 / np.sqrt(din)),
            "b1": np.zeros(H),
            "W2": rng.standard_normal((H, V)) * (1.0 / np.sqrt(H)),
            "b2": np.zeros(V),
        }
        self.opt = config.get("optimizer", "sgd")
        self.momentum = float(config.get("momentum", 0.0))
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}  # adam 1차 / sgd 속도
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}  # adam 2차
        self.t = 0

    def _forward(self, X):
        Xi = X.astype(np.int64)
        flat = self.P["E"][Xi].reshape(len(Xi), -1)          # (B, C*D)
        h = np.tanh(flat @ self.P["W1"] + self.P["b1"])
        logits = h @ self.P["W2"] + self.P["b2"]
        return Xi, flat, h, logits

    def log_probs(self, X):
        """prepare.eval_bpb가 요구하는 인터페이스: 자연로그 분포 (B,256).
        발산 모델도 nan_to_num으로 '유한하지만 나쁜' bpb를 내 평가가 깨지지 않는다."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _, _, _, logits = self._forward(X)
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            logits -= logits.max(axis=1, keepdims=True)
            z = np.exp(logits)
            return logits - np.log(z.sum(axis=1, keepdims=True))

    def _grads(self, X, Y):
        Xi, flat, h, logits = self._forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        B = len(X)
        loss = float(-np.log(p[np.arange(B), Y] + 1e-12).mean())
        dlogits = p; dlogits[np.arange(B), Y] -= 1.0; dlogits /= B
        g = {"W2": h.T @ dlogits, "b2": dlogits.sum(axis=0)}
        dh = (dlogits @ self.P["W2"].T) * (1.0 - h * h)
        g["W1"] = flat.T @ dh
        g["b1"] = dh.sum(axis=0)
        demb = (dh @ self.P["W1"].T).reshape(B, self.C, self.D)
        gE = np.zeros_like(self.P["E"])
        np.add.at(gE, Xi, demb)                               # 임베딩 행으로 scatter-add
        g["E"] = gE
        return loss, g

    def step(self, X, Y, lr):
        """미니배치 1회 업데이트(Adam 또는 모멘텀 SGD). 평균 NLL(nats) 반환."""
        loss, g = self._grads(X, Y)
        self.t += 1
        if self.opt == "adam":
            for k in self.P:
                self.m[k] = self.beta1 * self.m[k] + (1 - self.beta1) * g[k]
                self.v[k] = self.beta2 * self.v[k] + (1 - self.beta2) * (g[k] * g[k])
                mhat = self.m[k] / (1 - self.beta1 ** self.t)
                vhat = self.v[k] / (1 - self.beta2 ** self.t)
                self.P[k] -= lr * mhat / (np.sqrt(vhat) + self.eps)
        else:  # 모멘텀 SGD: v ← momentum·v − lr·grad ; param ← param + v
            for k in self.P:
                self.m[k] = self.momentum * self.m[k] - lr * g[k]
                self.P[k] += self.m[k]
        return loss


def _sched_lr(base_lr, frac, warmup_frac):
    """LR 스케줄: frac∈[0,1]=예산 경과 비율. 워밍업(선형 상승) 후 선형 감쇠."""
    if warmup_frac > 0 and frac < warmup_frac:
        return base_lr * (frac / warmup_frac)
    return base_lr * max(0.0, (1.0 - frac) / max(1.0 - warmup_frac, 1e-9))


def train(config, train_data, budget_sec, seed):
    """고정 벽시계 예산이 끝날 때까지 학습하고 모델 반환.

    같은 예산 안에서 더 효율적인 config(임베딩으로 싼 스텝·맞는 옵티마이저·좋은 LR 스케줄)가
    더 많은 유효 업데이트를 해 bpb를 낮춘다 = autoresearch의 compute-normalized 비교.
    """
    rng = np.random.default_rng(seed)
    model = CharLM(config, rng)
    base_lr = config["lr"]; warmup = float(config.get("warmup_frac", 0.0))
    steps = 0
    start = time.monotonic(); deadline = start + budget_sec
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        while True:
            now = time.monotonic()
            if now >= deadline:
                break
            lr = _sched_lr(base_lr, (now - start) / budget_sec, warmup)
            X, Y = prepare.get_batch(train_data, config["context_len"], config["batch_size"], rng)
            loss = model.step(X, Y, lr)
            steps += 1
            if not np.isfinite(loss):
                break
    return model, steps

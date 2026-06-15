"""autoresearch (학습용) — 탐색공간(SEARCH SPACE).

karpathy/autoresearch의 `train.py`에 대응한다. **에이전트/사람이 자유롭게 바꾸는 곳**:
모델 구조와 하이퍼파라미터(CONFIG). 규칙은 단 둘 —
  1) prepare.py(통제 변인)는 한 번의 탐색 안에서 건드리지 않는다.
  2) 고정 예산(DEFAULT_BUDGET_SEC) 안에서 val_bpb를 최소화한다.

program.md 백로그 반영분:
  - **학습 임베딩**(emb_dim): one-hot(256) → 학습 임베딩으로 입력 차원 축소.
  - **다층 MLP**(n_layers): 1~N개 은닉층 일반화.
  - **옵티마이저**(optimizer): Adam | 모멘텀 SGD.
  - **LR 스케줄**(warmup_frac): 예산 경과 비율 기준 워밍업 후 선형 감쇠.
  - **weight decay**(weight_decay): 가중치 행렬에 L2 정규화(임베딩·bias 제외).

모델: 바이트 문맥(context_len개) → 임베딩 → [tanh(hidden)]×n_layers → softmax(256).
"""
import time
import numpy as np

from . import prepare

# ── 탐색 노브(에이전트가 바꾸는 대상) ──────────────────────────────────────────
CONFIG = {
    "context_len": 4,       # 직전 몇 바이트를 볼지
    "hidden": 64,           # 은닉 차원
    "n_layers": 1,          # 은닉층 개수
    "emb_dim": 16,          # 학습 임베딩 차원
    "lr": 0.01,             # 기준 학습률(Adam 스케일)
    "batch_size": 64,       # 미니배치 크기
    "optimizer": "adam",    # "adam" | "sgd"
    "momentum": 0.9,        # sgd일 때 모멘텀 계수
    "warmup_frac": 0.1,     # 예산의 앞 비율만큼 워밍업
    "weight_decay": 0.0,    # 가중치 L2 (0=없음)
}


class CharLM:
    """임베딩 → [tanh(hidden)]×n_layers → softmax(256). Adam/모멘텀 + weight decay. numpy 전용."""

    def __init__(self, config, rng):
        C = config["context_len"]; H = config["hidden"]
        D = int(config.get("emb_dim", 16)); V = prepare.VOCAB
        self.C, self.D = C, D
        self.n_layers = int(config.get("n_layers", 1))
        self.P = {"E": rng.standard_normal((V, D)) * 0.1}
        din = C * D
        for i in range(self.n_layers):       # 은닉층 i: din→H (이후엔 H→H)
            self.P[f"Wh{i}"] = rng.standard_normal((din, H)) * (1.0 / np.sqrt(din))
            self.P[f"bh{i}"] = np.zeros(H)
            din = H
        self.P["Wo"] = rng.standard_normal((H, V)) * (1.0 / np.sqrt(H))
        self.P["bo"] = np.zeros(V)

        self.opt = config.get("optimizer", "sgd")
        self.momentum = float(config.get("momentum", 0.0))
        self.wd = float(config.get("weight_decay", 0.0))
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8
        self.m = {k: np.zeros_like(v) for k, v in self.P.items()}  # adam 1차 / sgd 속도
        self.v = {k: np.zeros_like(v) for k, v in self.P.items()}  # adam 2차
        self.t = 0

    def _forward(self, X):
        Xi = X.astype(np.int64)
        a = self.P["E"][Xi].reshape(len(Xi), -1)         # (B, C*D)
        acts = [a]
        for i in range(self.n_layers):
            a = np.tanh(a @ self.P[f"Wh{i}"] + self.P[f"bh{i}"])
            acts.append(a)
        logits = a @ self.P["Wo"] + self.P["bo"]
        return Xi, acts, logits

    def log_probs(self, X):
        """prepare.eval_bpb 인터페이스: 자연로그 분포 (B,256). 발산 모델도 유한 bpb."""
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            _, _, logits = self._forward(X)
            logits = np.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
            logits -= logits.max(axis=1, keepdims=True)
            z = np.exp(logits)
            return logits - np.log(z.sum(axis=1, keepdims=True))

    def _grads(self, X, Y):
        Xi, acts, logits = self._forward(X)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits); p /= p.sum(axis=1, keepdims=True)
        B = len(X)
        loss = float(-np.log(p[np.arange(B), Y] + 1e-12).mean())
        dlogits = p; dlogits[np.arange(B), Y] -= 1.0; dlogits /= B
        g = {"Wo": acts[-1].T @ dlogits, "bo": dlogits.sum(axis=0)}
        da = dlogits @ self.P["Wo"].T
        for i in reversed(range(self.n_layers)):
            dz = da * (1.0 - acts[i + 1] ** 2)           # tanh'
            g[f"Wh{i}"] = acts[i].T @ dz
            g[f"bh{i}"] = dz.sum(axis=0)
            da = dz @ self.P[f"Wh{i}"].T
        demb = da.reshape(B, self.C, self.D)
        gE = np.zeros_like(self.P["E"]); np.add.at(gE, Xi, demb)
        g["E"] = gE
        if self.wd:                                      # weight decay: 가중치 행렬에만 L2
            for k in g:
                if k.startswith("W"):
                    g[k] = g[k] + self.wd * self.P[k]
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
    """고정 벽시계 예산이 끝날 때까지 학습하고 (model, steps) 반환."""
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

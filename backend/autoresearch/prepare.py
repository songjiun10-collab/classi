"""autoresearch (학습용) — 불변부(FIXED).

karpathy/autoresearch의 `prepare.py`에 대응한다. **탐색/에이전트가 절대 수정하지 않는
통제 변인**이다: 데이터 준비, dataloader, 그리고 불변 평가 메트릭(val_bpb).

왜 고정하나 — 이 부분이 실험마다 흔들리면 서로 다른 변경(train.py)의 결과를 공정하게
비교할 수 없다. autoresearch 이론의 핵심은 '단일 변인만 바꾸고 통제 변인은 고정'이다.
- 메트릭은 **validation bits-per-byte(val_bpb)**: 바이트 단위라 vocab/토크나이저에
  독립적 → 모델이 vocab을 키워 점수를 게이밍할 수 없다(불변 메트릭).
"""
import numpy as np

SEED = 1337
DEFAULT_BUDGET_SEC = 3.0  # 실험당 고정 벽시계 예산(통제). autoresearch의 '5분/실험'에 대응.
VOCAB = 256               # 바이트 단위 vocab(고정). val_bpb가 vocab 비의존인 이유.

# 학습 대상 코퍼스. 여러 주제 문단을 섞어 단순 타일링보다 풍부하게(난이도↑·현실성↑).
# 코퍼스를 바꾸면 val_bpb의 의미(=이 데이터에 대한 압축률)가 바뀌므로 과거 수치와 직접
# 비교 불가 — 벤치마크 재정의다. 한 번의 탐색(search) 안에서는 여전히 고정(통제 변인).
_PARAGRAPHS = [
    "autoresearch trains a tiny model under a fixed wall-clock budget, then measures "
    "validation bits-per-byte. the budget is fixed so that different code changes are "
    "compared at equal compute, not at equal effort.",
    "the metric is per-byte so that changing the vocabulary cannot game the score. "
    "prepare stays frozen; only the search space in train may change; the loop keeps "
    "whatever lowers val_bpb.",
    "classi classifies korean exam pdfs into subject and sub-subject at the level of "
    "individual problems, combining a vision model with ontology evidence and confidence "
    "calibration.",
    "the same scientific loop applies: fix the metric, fix the budget, iterate, keep the "
    "best. hill-climbing over hyperparameters finds coupled optima that a single change "
    "could not reveal on its own.",
]


def _corpus() -> bytes:
    """결정적 코퍼스(여러 문단을 결정적 순서로 반복해 학습 가능한 바이트 통계를 만든다)."""
    text = ""
    for r in range(6):
        for i, p in enumerate(_PARAGRAPHS):
            text += _PARAGRAPHS[(i + r) % len(_PARAGRAPHS)] + "\n"
    return text.encode("utf-8")


def load_data():
    """train/val 바이트 배열(np.uint8)을 90/10으로 분리해 반환. 분리 지점도 고정."""
    buf = np.frombuffer(_corpus(), dtype=np.uint8)
    cut = int(len(buf) * 0.9)
    return buf[:cut].copy(), buf[cut:].copy()


def _windows(data, context_len):
    """data에서 (X=직전 context_len 바이트, Y=다음 바이트) 슬라이딩 윈도우 전체를 만든다."""
    n = len(data) - context_len
    X = np.empty((n, context_len), dtype=np.uint8)
    for i in range(context_len):
        X[:, i] = data[i:i + n]
    Y = data[context_len:context_len + n].astype(np.int64)
    return X, Y.copy()


def get_batch(data, context_len, batch_size, rng):
    """train.py가 쓰는 미니배치 샘플러. 통제 변인이라 여기서 고정."""
    X, Y = _windows(data, context_len)
    idx = rng.integers(0, len(Y), size=batch_size)
    return X[idx], Y[idx]


def eval_bpb(model, val_data, context_len) -> float:
    """불변 메트릭: validation bits-per-byte = 평균(-log2 p(정답바이트)).

    model.log_probs(X)->(B,256) 자연로그 분포만 요구한다(모델 내부에 비의존).
    vocab(256) 고정이라 이 값은 토크나이저/모델 구조와 무관하게 공정 비교 가능.
    """
    X, Y = _windows(val_data, context_len)
    logp = model.log_probs(X)                       # (N, 256) 자연로그
    nll_nats = -logp[np.arange(len(Y)), Y]          # 정답 바이트의 음의 로그우도(nats)
    return float(nll_nats.mean() / np.log(2.0))     # nats → bits

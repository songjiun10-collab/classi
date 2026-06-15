"""autoresearch (학습용) — 오케스트레이터(LOOP).

karpathy/autoresearch의 야간 루프에 대응한다. 이론 4요소를 코드로 시연한다:
  1) 실험당 **고정 벽시계 예산** → 변경 간 공정(compute-normalized) 비교.
  2) **불변 메트릭 val_bpb** → vocab/구조로 점수 게이밍 불가.
  3) **prepare(고정)/train(탐색) 분리** → 통제 변인과 탐색 변인 격리.
  4) **언덕오르기**: 현재 최선 config를 변이(mutate)해 더 낮은 bpb를 찾으면 채택.

실제 autoresearch는 LLM이 train.py를 직접 고친다. 학습용 복제라 여기선 CONFIG 사전을
변이하는 결정적 탐색으로 대체했다(LLM·ollama 불필요, 재현 가능).

실행: `python3 -m autoresearch.loop`
"""
import copy
import time
import numpy as np

from . import prepare, train


def run_experiment(config, train_data, val_data, budget_sec, seed):
    """config 하나를 '고정 예산'으로 학습→평가. 결과 레코드 반환."""
    t0 = time.monotonic()
    model, steps = train.train(config, train_data, budget_sec, seed)
    bpb = prepare.eval_bpb(model, val_data, config["context_len"])
    return {"config": config, "bpb": bpb, "steps": steps,
            "wall": round(time.monotonic() - t0, 2)}


def _mutate(config, rng):
    """최선 config의 노브 하나를 작게 흔든다(이산 언덕오르기)."""
    c = copy.deepcopy(config)
    knob = rng.choice(["context_len", "hidden", "lr", "batch_size", "momentum"])
    if knob == "context_len":
        c["context_len"] = int(np.clip(c["context_len"] + rng.choice([-1, 1]), 1, 8))
    elif knob == "hidden":
        c["hidden"] = int(np.clip(c["hidden"] * rng.choice([0.5, 2.0]), 8, 256))
    elif knob == "lr":
        c["lr"] = float(np.clip(c["lr"] * rng.choice([0.5, 2.0]), 0.01, 4.0))
    elif knob == "batch_size":
        c["batch_size"] = int(np.clip(c["batch_size"] * rng.choice([0.5, 2.0]), 8, 256))
    else:
        c["momentum"] = float(np.clip(c.get("momentum", 0.9) + rng.choice([-0.2, 0.05]), 0.0, 0.98))
    return c


def search(rounds=6, budget_sec=None, seed=prepare.SEED, verbose=True):
    """기준 CONFIG에서 시작해 rounds회 변이 탐색. (best_config, best_bpb, history) 반환."""
    budget_sec = prepare.DEFAULT_BUDGET_SEC if budget_sec is None else budget_sec
    rng = np.random.default_rng(seed)
    train_data, val_data = prepare.load_data()

    base = dict(train.CONFIG)
    best = run_experiment(base, train_data, val_data, budget_sec, seed)
    history = [best]
    if verbose:
        print(f"[baseline] bpb={best['bpb']:.4f} steps={best['steps']} {best['config']}")

    for r in range(1, rounds + 1):
        cand_cfg = _mutate(best["config"], rng)
        cand = run_experiment(cand_cfg, train_data, val_data, budget_sec, seed)
        history.append(cand)
        kept = cand["bpb"] < best["bpb"]
        if kept:
            best = cand
        if verbose:
            mark = "KEEP" if kept else "drop"
            print(f"[round {r}] bpb={cand['bpb']:.4f} steps={cand['steps']} "
                  f"{mark}  {cand_cfg}")

    if verbose:
        uniform = 8.0  # 256 균등분포 = 8 bits/byte (학습 전 상한 기준선)
        print(f"\nbest val_bpb={best['bpb']:.4f}  (uniform baseline={uniform:.1f})  "
              f"config={best['config']}")
    return best["config"], best["bpb"], history


if __name__ == "__main__":
    search()

"""autoresearch 학습용 하니스 스모크 테스트(결정적·빠름).

분류 본코드와 무관한 독립 모듈 검증. 작은 예산으로 1초 내 끝난다.
goal-driven 검증: (1) 하니스가 끝까지 돌고, (2) 학습된 모델의 val_bpb가
균등분포 상한(8.0 bits/byte)보다 낮다(= 실제로 학습이 일어났다).
"""
import unittest

from autoresearch import prepare, train, loop


class TestAutoresearch(unittest.TestCase):
    def test_metric_is_finite_for_untrained_model(self):
        # 학습 전 무작위 모델도 평가가 깨지지 않고 유한·양수 bpb를 반환
        # (무작위 모델은 균등분포 8.0보다 나쁠 수 있으므로 상한은 단언하지 않는다)
        import math
        import numpy as np
        tr, val = prepare.load_data()
        m = train.CharLM({"context_len": 2, "hidden": 8, "emb_dim": 8}, np.random.default_rng(0))
        bpb = prepare.eval_bpb(m, val, 2)
        self.assertTrue(math.isfinite(bpb) and bpb > 0.0)

    def test_training_beats_uniform_baseline(self):
        tr, val = prepare.load_data()
        cfg = {"context_len": 3, "hidden": 16, "lr": 0.5, "batch_size": 32}
        model, steps = train.train(cfg, tr, budget_sec=0.3, seed=prepare.SEED)
        self.assertGreater(steps, 0)
        self.assertLess(prepare.eval_bpb(model, val, cfg["context_len"]), 8.0)

    def test_gradient_check_residual_layernorm(self):
        # 잔차+LayerNorm+2층 모델의 해석적 그래디언트를 유한차분과 대조해 backprop 정확성 보장.
        import numpy as np
        cfg = {"context_len": 2, "hidden": 5, "emb_dim": 3, "n_layers": 2,
               "residual": True, "layernorm": True, "weight_decay": 0.0}
        rng = np.random.default_rng(7)
        model = train.CharLM(cfg, rng)
        X = rng.integers(0, prepare.VOCAB, size=(6, cfg["context_len"])).astype(np.uint8)
        Y = rng.integers(0, prepare.VOCAB, size=6)

        def loss_of():
            lp = model.log_probs(X)
            return float(-lp[np.arange(len(Y)), Y].mean())

        _, g = model._grads(X, Y)
        eps = 1e-5
        for key in ("E", "Wh0", "Wh1", "Wo", "bo", "bh0"):
            P = model.P[key]
            flat = P.reshape(-1)
            for idx in rng.integers(0, flat.size, size=4):
                orig = flat[idx]
                flat[idx] = orig + eps; lp = loss_of()
                flat[idx] = orig - eps; lm = loss_of()
                flat[idx] = orig
                num = (lp - lm) / (2 * eps)
                ana = g[key].reshape(-1)[idx]
                self.assertAlmostEqual(num, ana, delta=1e-4,
                                       msg=f"{key}[{idx}] num={num} ana={ana}")

    def test_search_keeps_best(self):
        # 짧은 탐색이 끝까지 돌고, best가 history의 최소 bpb와 일치
        best_cfg, best_bpb, history = loop.search(
            rounds=2, budget_sec=0.2, seed=prepare.SEED, verbose=False)
        self.assertEqual(best_bpb, min(h["bpb"] for h in history))


if __name__ == "__main__":
    unittest.main()

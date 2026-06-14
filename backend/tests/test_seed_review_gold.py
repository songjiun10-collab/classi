#!/usr/bin/env python3
"""seed_review_gold 회귀 테스트 — 라벨 시험지 → review_log gold 시딩(주입식, 결정론적).

플라이휠 일괄 적재 경로를 고정한다: 분류·추출을 주입해 ollama/디스크 없이, 시딩된 행이
(1) gold로 resolve되고 (2) telemetry를 보존하며 (3) calibration_trainer가 실제로 소비해
가중치를 학습하는지(루프가 이어지는지)까지 확인한다.

Run:  cd backend && python3 -m unittest tests.test_seed_review_gold -v
"""
import os, sys, tempfile, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import review_log as RL
from pipeline import seed_review_gold as S
from pipeline import calibration_trainer as T


class _Cls:
    """Classification 더미(분류기 출력 대역)."""
    def __init__(self, subject, sub="기타", conf=0.9):
        self.subject, self.sub_subject, self.confidence = subject, sub, conf


def _fake_extract(n):
    return lambda path: [{"page": 1, "problem_num": str(i + 1),
                          "text": "", "cover_text": "", "image_bytes": b""} for i in range(n)]


class TestSeedReviewGold(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "seed.db")

    def test_seeds_gold_and_telemetry(self):
        entries = [{"path": "물리.pdf", "gold_subject": "과학탐구"}]
        # 분류기: 5문항 중 3개는 gold와 일치(과학탐구), 2개는 오분류(수학) — 두 클래스 확보
        preds = [_Cls("과학탐구", conf=0.95), _Cls("과학탐구", conf=0.9), _Cls("과학탐구", conf=0.8),
                 _Cls("수학", conf=0.7), _Cls("수학", conf=0.6)]
        seq = iter(preds)

        def classify_fn(prob, name):
            c = next(seq)
            tel = {"raw_confidence": c.confidence, "pro_match": 1, "anti_match": 0,
                   "competing": 0, "rule_conflict": False}
            return c, tel

        n = S.seed_from_manifest(entries, classify_fn, _fake_extract(5),
                                 db_path=self.db, max_per_file=8, log=lambda *a: None)
        self.assertEqual(n, 5)
        rows = RL.list_recent(limit=100, include_resolved=True, db_path=self.db)
        self.assertEqual(len(rows), 5)
        self.assertTrue(all(r["resolved"] == 1 for r in rows))                 # 전부 resolve
        self.assertTrue(all(r["gold_subject"] == "과학탐구" for r in rows))     # gold = 매니페스트값
        self.assertTrue(all((r["telemetry"] or "").strip() for r in rows))     # telemetry 보존

    def test_solution_book_skipped(self):
        entries = [{"path": "물리_해설지.pdf", "gold_subject": "과학탐구"}]
        called = {"n": 0}

        def classify_fn(prob, name):
            called["n"] += 1
            return _Cls("과학탐구"), {"raw_confidence": 0.9}

        n = S.seed_from_manifest(entries, classify_fn, _fake_extract(3),
                                 db_path=self.db, max_per_file=8, log=lambda *a: None)
        self.assertEqual(n, 0)            # 해설서는 시딩 제외
        self.assertEqual(called["n"], 0)  # 분류조차 호출 안 함

    def test_seeded_gold_is_trainable(self):
        """시드한 gold를 calibration_trainer가 실제로 소비해 가중치를 학습하는지(루프 폐쇄)."""
        entries = [{"path": "물리.pdf", "gold_subject": "과학탐구"}]
        # 두 클래스(정답 6·오답 6) + 분리 가능한 telemetry → 학습 성공 조건
        preds = ([(_Cls("과학탐구", conf=0.95), 2, 0)] * 6 +
                 [(_Cls("수학", conf=0.5), 0, 2)] * 6)
        seq = iter(preds)

        def classify_fn(prob, name):
            c, pro, anti = next(seq)
            return c, {"raw_confidence": c.confidence, "pro_match": pro, "anti_match": anti,
                       "competing": 0, "rule_conflict": False}

        S.seed_from_manifest(entries, classify_fn, _fake_extract(12),
                             db_path=self.db, max_per_file=20, log=lambda *a: None)
        out = os.path.join(os.path.dirname(self.db), "w.json")
        w = T.train_from_review_log(db_path=self.db, out_path=out, min_samples=10)
        self.assertIsNotNone(w)                      # 충분한 gold·두 클래스 → 학습 성공
        self.assertEqual(w["source"], "gold")
        self.assertEqual(w["n_samples"], 12)


if __name__ == "__main__":
    unittest.main()

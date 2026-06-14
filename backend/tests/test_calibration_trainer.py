#!/usr/bin/env python3
"""calibration_trainer 회귀 테스트 — review DB의 gold로 학습 → gold 태그 weights.

데이터 플라이휠 폐쇄 가드: 합성 gold(임시 DB)로 학습이 동작하고, 산출물을 소비자
(core.confidence.calibrate_confidence)가 실제로 받아들이는지(루프가 이어지는지)까지 확인한다.
sklearn 필요(설치돼 있음). ollama 무관·결정론적.

Run:  cd backend && python3 -m unittest tests.test_calibration_trainer -v
"""
import json, os, sys, tempfile, unittest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core import review_log as RL
from pipeline import calibration_trainer as T


def _log_resolve(db, num, subject, conf, tel, gold):
    RL.log_results("a.pdf", [{"page": int(num), "problem_num": str(num), "subject": subject,
                              "sub_subject": "기타", "confidence": conf, "telemetry": tel}], db_path=db)
    RL.resolve("a.pdf", int(num), str(num), gold_subject=gold, db_path=db)


class TestCalibrationTrainer(unittest.TestCase):
    def setUp(self):
        d = tempfile.mkdtemp()
        self.db = os.path.join(d, "rl.db")
        self.out = os.path.join(d, "w.json")

    def _seed(self, n_correct, n_wrong):
        i = 0
        for _ in range(n_correct):                                  # gold==pred → y=1
            i += 1
            _log_resolve(self.db, i, "수학", 0.3,
                         {"pro_match": 3, "anti_match": 0, "competing": 0, "rule_conflict": False,
                          "raw_confidence": 0.3}, "수학")
        for _ in range(n_wrong):                                    # gold!=pred → y=0
            i += 1
            _log_resolve(self.db, i, "과학탐구", 0.2,
                         {"pro_match": 0, "anti_match": 2, "competing": 1, "rule_conflict": True,
                          "raw_confidence": 0.2}, "수학")

    def test_trains_and_writes_gold_weights(self):
        self._seed(6, 6)
        w = T.train_from_review_log(db_path=self.db, out_path=self.out, min_samples=10)
        self.assertIsNotNone(w)
        self.assertEqual(w["source"], "gold")                       # 소비자가 신뢰하는 태그
        self.assertEqual(set(w["coefficients"]), set(T._FEATURES))  # 특징 키 = 소비자 기대치
        self.assertIn("intercept", w)
        self.assertEqual(w["n_samples"], 12)
        saved = json.loads(Path(self.out).read_text(encoding="utf-8"))
        self.assertEqual(saved["source"], "gold")

    def test_insufficient_data_returns_none(self):
        self._seed(2, 1)
        self.assertIsNone(T.train_from_review_log(db_path=self.db, out_path=self.out, min_samples=10))
        self.assertFalse(os.path.exists(self.out))                  # 부족하면 파일 안 씀

    def test_single_class_returns_none(self):
        self._seed(12, 0)                                           # 전부 정답 → 단일 클래스
        self.assertIsNone(T.train_from_review_log(db_path=self.db, out_path=self.out, min_samples=10))

    def test_unresolved_rows_excluded(self):
        self._seed(6, 6)                                            # 해소+gold 12건
        RL.log_results("a.pdf", [{"page": 99, "problem_num": "99", "subject": "수학",
                                  "sub_subject": "기타", "confidence": 0.1, "telemetry": {}}], db_path=self.db)
        rows = RL.list_recent(limit=1000, include_resolved=True, db_path=self.db)
        X, y = T._rows_to_xy(rows)
        self.assertEqual(len(X), 12)                                # 미해소(99번)는 제외

    def test_uses_raw_confidence_feature(self):
        # confidence 특징은 보정 전 raw_confidence를 써야(소비자와 의미 일치) — 저장 confidence가 아님.
        _log_resolve(self.db, 1, "수학", 0.1,                       # 저장 confidence=0.1(저신뢰→로깅됨)
                     {"pro_match": 1, "anti_match": 0, "competing": 0, "rule_conflict": False,
                      "raw_confidence": 0.4}, "수학")               # raw=0.4 (저장값과 다름)
        rows = RL.list_recent(limit=10, include_resolved=True, db_path=self.db)
        X, _ = T._rows_to_xy(rows)
        self.assertEqual(X[0][4], 0.4)                              # 저장 0.1이 아니라 raw 0.4

    def test_corrupt_telemetry_isolated(self):
        # 비-객체 JSON('null','[1,2,3]')·깨진 JSON은 .get 크래시로 전체 학습을 죽이면 안 된다.
        # 비-객체는 빈 telemetry로 흡수(행 유지), 깨진 JSON 행만 격리 스킵.
        rows = [
            {"resolved": 1, "gold_subject": "수학", "subject": "수학", "confidence": 0.3,
             "telemetry": '{"pro_match": 2, "raw_confidence": 0.3}'},   # 정상
            {"resolved": 1, "gold_subject": "수학", "subject": "수학", "confidence": 0.2,
             "telemetry": "null"},                                      # 비-객체 → tel={}
            {"resolved": 1, "gold_subject": "수학", "subject": "수학", "confidence": 0.1,
             "telemetry": "[1,2,3]"},                                   # 비-객체(list) → tel={}
            {"resolved": 1, "gold_subject": "수학", "subject": "수학", "confidence": 0.4,
             "telemetry": "{bad json"},                                 # 깨진 JSON → 행 스킵
        ]
        X, y = T._rows_to_xy(rows)
        self.assertEqual(len(X), 3)            # 정상 + 비-객체 2건 = 3, 깨진 JSON만 제외
        self.assertEqual(X[1][0], 0)           # 'null' 행은 pro_match 기본 0으로 흡수

    def test_unclassified_excluded_from_training(self):
        # 미분류는 소비자가 학습 경로 전에 단락 → 학습셋에서 빠져야 train/serve 정합이 맞다.
        rows = [
            {"resolved": 1, "gold_subject": "수학", "subject": "수학", "confidence": 0.3, "telemetry": "{}"},
            {"resolved": 1, "gold_subject": "수학", "subject": "미분류", "confidence": 0.0, "telemetry": "{}"},
        ]
        X, y = T._rows_to_xy(rows)
        self.assertEqual(len(X), 1)            # 미분류 행 제외

    def test_atomic_write_no_leftover_tmp(self):
        # 원자적 쓰기: 성공 후 임시 파일 잔존 없음 + 유효 JSON.
        self._seed(6, 6)
        T.train_from_review_log(db_path=self.db, out_path=self.out, min_samples=10)
        self.assertTrue(os.path.exists(self.out))
        self.assertFalse(os.path.exists(self.out + ".tmp"))
        json.loads(Path(self.out).read_text(encoding="utf-8"))

    def test_consumer_accepts_trained_weights(self):
        # 루프 폐쇄: 트레이너 산출물을 calibrate_confidence가 실제로 로드하는지.
        self._seed(6, 6)
        T.train_from_review_log(db_path=self.db, out_path=self.out, min_samples=10)
        import core.confidence as C
        os.environ["CLASSI_CALIB_WEIGHTS"] = self.out
        C._CALIB = "UNSET"
        try:
            w = C._load_calib_weights()
            self.assertIsNotNone(w)
            self.assertEqual(w["source"], "gold")
        finally:
            os.environ.pop("CLASSI_CALIB_WEIGHTS", None)
            C._CALIB = None


if __name__ == "__main__":
    unittest.main(verbosity=2)

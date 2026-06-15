#!/usr/bin/env python3
"""confidence 회귀 테스트 — calibrate_confidence가 경합 과목(alt_subjects)을 노출하는지.

server.py /api/classify 응답의 'alt_subjects'(왜 이 신뢰도인지 근거)의 데이터 소스를 가드한다.
ollama 무관·결정론적.

Run:  cd backend && python3 -m unittest tests.test_confidence -v
"""
import json, math, os, sys, tempfile, types, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core.confidence as C
from core.confidence import calibrate_confidence, rationale


class TestAltSubjects(unittest.TestCase):
    def test_competing_target_is_exposed(self):
        # 모델은 물리학Ⅰ이라 했으나 텍스트 증거(pro, conf≥0.7)가 화학Ⅰ을 가리킴 → 경합으로 노출
        res = types.SimpleNamespace(confidence=0.8, subject="과학탐구", sub_subject="물리학Ⅰ")
        pre = {"items": [{"polarity": "pro", "targets": ["화학Ⅰ"], "confidence": 0.9, "source": "text"}]}
        res2, telemetry, alt = calibrate_confidence(res, "", pre)
        self.assertIn("화학Ⅰ", alt)
        self.assertIsInstance(telemetry, dict)
        self.assertGreaterEqual(telemetry.get("competing", 0), 1)

    def test_no_competing_when_evidence_agrees(self):
        res = types.SimpleNamespace(confidence=0.8, subject="과학탐구", sub_subject="물리학Ⅰ")
        pre = {"items": [{"polarity": "pro", "targets": ["물리학Ⅰ"], "confidence": 0.9, "source": "text"}]}
        _, _, alt = calibrate_confidence(res, "", pre)
        self.assertEqual(alt, [])

    def test_parent_subject_target_is_support_not_competition(self):
        # 파일명 '물리'→과학탐구(대분류) 증거가, 과학탐구/물리학Ⅰ로 맞게 분류한 문항에서
        # 경합으로 집계돼 신뢰도를 깎던 버그(실측: 서바 물리 13문항 전부 '경합: 과학탐구') 가드.
        res = types.SimpleNamespace(confidence=0.8, subject="과학탐구", sub_subject="물리학Ⅰ")
        pre = {"items": [{"polarity": "pro", "targets": ["과학탐구"],
                          "confidence": 0.8, "source": "filename"}]}
        _, tel, alt = calibrate_confidence(res, "", pre)
        self.assertEqual(alt, [])
        self.assertEqual(tel["competing"], 0)
        self.assertEqual(tel["pro_match"], 1)


class TestGlobalAntiEvidence(unittest.TestCase):
    """브랜드/메타/비수능 anti 증거(targets=["미분류"])가 실제로 신뢰도에 작용하는지.

    소비자가 '전체' 타깃만 전역 신호로 처리해, emitter(pre_classify)가 내보내는
    '미분류' 타깃 anti는 anti_match가 영원히 0이던 죽은 신호 버그의 가드."""

    def setUp(self):
        self._saved = C._CALIB
        C._CALIB = None            # 하드코딩 폴백 경로로 고정(디스크 가중치 격리)

    def tearDown(self):
        C._CALIB = self._saved

    def _brand_anti(self):
        return {"items": [{"keyword": "메가스터디", "polarity": "anti",
                           "targets": ["미분류"], "confidence": 0.9, "source": "context"}]}

    def test_counts_anti_when_classified(self):
        # 과목으로 분류했는데 메타/브랜드 증거 존재 → 반박 1건, 신뢰도 -0.12
        res = types.SimpleNamespace(confidence=0.8, subject="과학탐구", sub_subject="물리학Ⅰ")
        res2, tel, _ = calibrate_confidence(res, "", self._brand_anti())
        self.assertEqual(tel["anti_match"], 1)
        self.assertAlmostEqual(res2.confidence, 0.68, places=6)

    def test_counts_pro_when_unclassified(self):
        # 미분류로 분류했고 메타/브랜드 증거 존재 → 미분류 판단을 지지하는 증거로 집계
        res = types.SimpleNamespace(confidence=0.2, subject="미분류", sub_subject="기타")
        _, tel, _ = calibrate_confidence(res, "", self._brand_anti())
        self.assertEqual(tel["pro_match"], 1)
        self.assertEqual(tel["anti_match"], 0)


class TestRationale(unittest.TestCase):
    """rationale(): 신뢰도 근거 한 줄 요약(근거 가시화)."""
    def test_competing_summary(self):
        s = rationale("과학탐구", 0.41,
                      {"pro_match": 1, "anti_match": 0, "competing": 1, "rule_conflict": False},
                      ["화학Ⅰ"])
        self.assertIn("신뢰도 0.41", s)
        self.assertIn("화학Ⅰ", s)

    def test_unclassified_no_evidence(self):
        s = rationale("미분류", 0.0, {}, [])
        self.assertIn("미분류", s)
        self.assertIn("증거 없음", s)


class TestLearnedCalibration(unittest.TestCase):
    """학습 가중치 소비자단(데이터 플라이휠): gold 태그만 소비, 없거나 비-gold면 하드코딩과 동일."""

    def setUp(self):
        self._saved = C._CALIB
        C._CALIB = None            # 디스크 격리 — 가중치 전무 상태로 고정

    def tearDown(self):
        C._CALIB = self._saved

    def _call(self):
        # 과학탐구/물리학Ⅰ, pro 증거 1건(물리학Ⅰ 일치, 가중1.0) → 하드코딩이면 +0.05.
        res = types.SimpleNamespace(confidence=0.8, subject="과학탐구", sub_subject="물리학Ⅰ")
        pre = {"items": [{"polarity": "pro", "targets": ["물리학Ⅰ"], "confidence": 0.9, "source": "text"}]}
        return calibrate_confidence(res, "", pre)

    def test_no_weights_identical_to_hardcoded(self):
        res, _, _ = self._call()
        self.assertAlmostEqual(res.confidence, 0.85, places=6)        # 0.8 + 0.05(pro 1건)

    def test_gold_weights_applied(self):
        C._CALIB = {"source": "gold", "intercept": -1.0,
                    "coefficients": {"pro_match": 1.0, "anti_match": -1.0,
                                     "competing": -0.5, "rule_conflict": -0.5, "confidence": 2.0}}
        res, _, _ = self._call()
        z = -1.0 + 1.0 * 1 + 2.0 * 0.8                                # pro_match=1, 나머지 0, raw_conf=0.8
        self.assertAlmostEqual(res.confidence, 1.0 / (1.0 + math.exp(-z)), places=6)
        self.assertNotAlmostEqual(res.confidence, 0.85, places=4)     # 하드코딩과 분명히 다름

    def test_unclassified_ignores_weights(self):
        # 미분류는 학습경로를 타지 않고 0.25 상한 유지(범주적 규칙 보존).
        C._CALIB = {"source": "gold", "intercept": 5.0, "coefficients": {}}
        res = types.SimpleNamespace(confidence=0.9, subject="미분류", sub_subject="기타")
        res2, _, _ = calibrate_confidence(res, "", {"items": []})
        self.assertLessEqual(res2.confidence, 0.25)

    def _write(self, payload):
        p = os.path.join(tempfile.mkdtemp(), "w.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f)
        return p

    def test_loader_rejects_non_gold(self):
        os.environ["CLASSI_CALIB_WEIGHTS"] = self._write(
            {"source": "self_predicted", "coefficients": {"pro_match": 5.0}, "intercept": 0.0})
        C._CALIB = "UNSET"
        try:
            self.assertIsNone(C._load_calib_weights())               # 순환신호 차단
        finally:
            os.environ.pop("CLASSI_CALIB_WEIGHTS", None)
            C._CALIB = None

    def test_reload_invalidates_cache(self):
        # /api/calibration/train이 새 가중치를 써도 1회 로드 캐시 때문에 재시작 전까지
        # 옛 값으로 분류하던 버그의 가드 — reload 후엔 디스크의 최신 가중치를 다시 읽어야 한다.
        os.environ["CLASSI_CALIB_WEIGHTS"] = self._write(
            {"source": "gold", "coefficients": {"pro_match": 1.0}, "intercept": 0.0})
        C._CALIB = None                                              # 학습 전: 가중치 없음으로 캐시됨
        try:
            self.assertIsNone(C._load_calib_weights())               # 캐시된 None 유지
            C.reload_calib_weights()                                 # 학습 성공 직후 호출되는 훅
            w = C._load_calib_weights()
            self.assertIsNotNone(w)                                  # 새 가중치가 즉시 보인다
            self.assertEqual(w["source"], "gold")
        finally:
            os.environ.pop("CLASSI_CALIB_WEIGHTS", None)
            C._CALIB = None

    def test_loader_accepts_gold(self):
        os.environ["CLASSI_CALIB_WEIGHTS"] = self._write(
            {"source": "gold", "coefficients": {"pro_match": 1.0}, "intercept": 0.0})
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

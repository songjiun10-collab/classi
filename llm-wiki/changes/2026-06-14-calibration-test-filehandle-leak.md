# 2026-06-14 · calibration 테스트 파일 핸들 누수 제거

> 백엔드 정확도 개선 야간 작업(브랜치 `improve/overnight-20260614`)의 변경 1건.

## 맥락
테스트가 json.load(open())로 파일을 안 닫아 ResourceWarning이 났다 — Path.read_text로 교체.

## 커밋
`bf26582` — test(calibration): 파일 핸들 누수 제거 — json.load(open()) → Path.read_text

## 변경된 파일
```
tests/test_calibration_trainer.py | 5 +++--
 1 file changed, 3 insertions(+), 2 deletions(-)
```

## 전체 diff

```diff
diff --git a/tests/test_calibration_trainer.py b/tests/test_calibration_trainer.py
index 997383e..3ede268 100644
--- a/tests/test_calibration_trainer.py
+++ b/tests/test_calibration_trainer.py
@@ -8,6 +8,7 @@ sklearn 필요(설치돼 있음). ollama 무관·결정론적.
 Run:  cd backend && python3 -m unittest tests.test_calibration_trainer -v
 """
 import json, os, sys, tempfile, unittest
+from pathlib import Path
 
 sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
 from core import review_log as RL
@@ -47,7 +48,7 @@ class TestCalibrationTrainer(unittest.TestCase):
         self.assertEqual(set(w["coefficients"]), set(T._FEATURES))  # 특징 키 = 소비자 기대치
         self.assertIn("intercept", w)
         self.assertEqual(w["n_samples"], 12)
-        saved = json.load(open(self.out, encoding="utf-8"))
+        saved = json.loads(Path(self.out).read_text(encoding="utf-8"))
         self.assertEqual(saved["source"], "gold")
 
     def test_insufficient_data_returns_none(self):
@@ -108,7 +109,7 @@ class TestCalibrationTrainer(unittest.TestCase):
         T.train_from_review_log(db_path=self.db, out_path=self.out, min_samples=10)
         self.assertTrue(os.path.exists(self.out))
         self.assertFalse(os.path.exists(self.out + ".tmp"))
-        json.load(open(self.out, encoding="utf-8"))
+        json.loads(Path(self.out).read_text(encoding="utf-8"))
 
     def test_consumer_accepts_trained_weights(self):
         # 루프 폐쇄: 트레이너 산출물을 calibrate_confidence가 실제로 로드하는지.
```

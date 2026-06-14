# 2026-06-14 · 영어 영역 미분류 복구 (eval 0%→교정)

> 백엔드 정확도 개선 야간 작업(브랜치 `improve/overnight-20260614`)의 변경 1건.

## 맥락
비전 모델이 영문 지문/한국어 듣기 발문을 '영어 영역'으로 못 잇고 미분류로 흘리던 실패. 듣기 안내문(영어 전용)·영어 기능어 신호로 미분류→영어 교정. 제2외국어 오탐도 동시 수정.

## 커밋
`c405e3a` — fix(classify): recover English exam from model 미분류 (was 0% in eval)

## 변경된 파일
```
api/server.py                |  6 +++--
 core/classifier_engine.py    | 54 +++++++++++++++++++++++++++++++++++++-------
 pipeline/seed_review_gold.py |  3 ++-
 tests/test_engine.py         | 50 ++++++++++++++++++++++++++++++++++++++++
 tools/eval_accuracy.py       |  7 +++---
 5 files changed, 106 insertions(+), 14 deletions(-)
```

## 전체 diff

```diff
diff --git a/api/server.py b/api/server.py
index 0002e11..9b73971 100644
--- a/api/server.py
+++ b/api/server.py
@@ -14,8 +14,8 @@ sys.path.append(str(Path(__file__).parent.parent))
 from core.classifier_engine import (
     extract_all_problems_cached, find_problem_boxes, pre_classify, make_compact_prompt,
     safe_json_parse, Classification, CURRICULUM, apply_filename_prior, apply_cover_prior,
-    downscale_for_model, infer_cache_key, infer_cache_get, infer_cache_put,
-    is_solution_book, save_solution_captures
+    apply_english_prior, downscale_for_model, infer_cache_key, infer_cache_get,
+    infer_cache_put, is_solution_book, save_solution_captures
 )
 from core.confidence import calibrate_confidence, rationale, reload_calib_weights
 from core import review_log
@@ -177,6 +177,8 @@ async def _run_classify(task_id: str, pdf_path: Path, filename: str, model: str,
                         # 단일 과목 시험은 표지가 사실상 정답 — 오염된 텍스트 레이어로
                         # 모델이 딴 과목을 골랐어도 표지 기준으로 교정한다.
                         cls = apply_cover_prior(cls, pre_evidence)
+                        # 영문 우세/듣기 안내 등 영어 적극 증거가 있으면 모델 미분류를 영어로 교정
+                        cls = apply_english_prior(cls, pre_evidence)
                         # alt_subjects(경합 과목)는 그동안 버려졌다 — 왜 이 신뢰도인지
                         # 사람이 알 수 있게 telemetry와 함께 응답에 노출한다(근거 가시화).
                         cls, telemetry, alt_subjects = calibrate_confidence(cls, ocr_text, pre_evidence)
diff --git a/core/classifier_engine.py b/core/classifier_engine.py
index 789a8ef..11b6133 100644
--- a/core/classifier_engine.py
+++ b/core/classifier_engine.py
@@ -488,25 +488,44 @@ def extract_cover_subject(cover_text: str) -> dict:
             items.append({"keyword": subj, "type": "cover", "polarity": "pro",
                           "targets": [subj], "hint": "표지에서 과목명 발견",
                           "confidence": 0.85, "source": "cover"})
+    # 4) 영어 듣기평가 안내문 — 수능에서 듣기가 있는 영역은 영어뿐이다(국어 등엔 없음).
+    #    cover_text는 한 파일의 모든 문항이 공유하므로, 듣기 문항(한국어 발문뿐이라
+    #    본문만으론 영어인지 알 수 없다)까지 문서 단위로 '영어'를 시사한다.
+    #    실측: '듣고 답하는 문제'는 영어 외 전 과목 표지에서 0건(국어의 '들려'는 지문어라 제외).
+    if "듣고답하는문제" in tn:
+        items.append({"keyword": "듣기평가 안내", "type": "cover", "polarity": "pro",
+                      "targets": ["영어"], "hint": "듣기평가 안내문 → 영어",
+                      "confidence": 0.92, "source": "cover"})
     return {"items": items}
 
 _HANGUL_RE = re.compile(r"[가-힣]")
 _LATIN_ALPHA_RE = re.compile(r"[A-Za-z]")
+# 영어 고유 기능어 — 독일어/프랑스어/스페인어엔 거의 없다(평가원 실측: 영어 본문 362회 vs
+# 독·프·스 각 0회). 라틴 글자 우세만으론 못 가르던 제2외국어를 분리하는 판별자.
+_EN_FUNCWORD_RE = re.compile(
+    r"\b(?:the|and|of|to|that|with|for|which|this|from|what|when|because|"
+    r"would|could|should|their|there|about|into|your|you|have|been|will)\b", re.IGNORECASE)
 
 
 def _english_evidence(text: str):
-    """영문이 한글을 압도하는 긴 텍스트 → '영어' 지문 시사(약한 pro). 보수적 임계로 오탐 방지:
-    영문 글자 ≥60 AND 영문이 한글의 5배 이상일 때만. 영어Ⅰ/Ⅱ는 어휘로 구분 불가라 대분류만 시사하며,
-    모델 프롬프트의 힌트로만 작용한다(하드 규칙 아님)."""
+    """영문이 한글을 압도하는 긴 텍스트 + 영어 기능어가 잦으면 '영어' 지문으로 본다(강한 pro).
+    임계: 영문 글자 ≥60 AND 영문이 한글의 5배 이상 AND 영어 기능어 ≥3회.
+    기능어 조건은 라틴 우세지만 영어가 아닌 제2외국어(독·프·스)를 영어로 오인하던 잠재 오탐을
+    막는다(종전엔 라틴 우세만 보아 독일어 지문도 '영어'로 시사). 영어Ⅰ/Ⅱ는 어휘로 구분
+    불가라 대분류만 시사한다. conf 0.9 — apply_english_prior가 모델 '미분류'를 영어로 교정하는
+    근거로도 쓰인다(영문 독해 문항을 모델이 한국 '영어 영역'으로 못 잇는 실패 복구)."""
     if not text:
         return []
     eng = len(_LATIN_ALPHA_RE.findall(text))
     han = len(_HANGUL_RE.findall(text))
-    if eng >= 60 and eng >= han * 5:
-        return [{"keyword": "영문 지문 우세", "type": "context", "polarity": "pro",
-                 "targets": ["영어"], "hint": f"en={eng}·ko={han} → 영어", "confidence": 0.6,
-                 "source": "context"}]
-    return []
+    if not (eng >= 60 and eng >= han * 5):
+        return []
+    func = len(_EN_FUNCWORD_RE.findall(text))
+    if func < 3:  # 라틴 우세지만 영어 기능어 부족 → 제2외국어 가능성 → 영어로 단정하지 않음
+        return []
+    return [{"keyword": "영문 지문 우세", "type": "context", "polarity": "pro",
+             "targets": ["영어"], "hint": f"en={eng}·ko={han}·func={func} → 영어",
+             "confidence": 0.9, "source": "context"}]
 
 
 @lru_cache(maxsize=16)
@@ -865,6 +884,25 @@ def apply_cover_prior(cls: "Classification", pre_evidence: dict) -> "Classificat
         cls.subject, cls.sub_subject = hit
     return cls
 
+
+def apply_english_prior(cls: "Classification", pre_evidence: dict) -> "Classification":
+    """비전 모델이 영어 영역을 '미분류'로 흘리는 실패를 교정한다.
+    영어는 '언어 우세'로 정의되는 유일 과목이라, 모델이 영문 독해 지문을 한국 교육과정의
+    '영어 영역'으로 잇지 못하거나(영문만 보임), 듣기 문항의 한국어 발문만으론 영역을 못
+    가리는 경우가 잦다(평가원 영어 5/5 미분류 관측). 영어를 적극 지목하는 고신뢰 증거
+    (영문 기능어 우세 본문 또는 듣기평가 안내문)가 있을 때만, 그리고 모델이 미분류일 때만
+    영어로 승격한다 — 다른 예측·다른 과목은 절대 건드리지 않는다(외과적).
+    영어Ⅰ/Ⅱ는 본문으로 구별 불가하므로 세부는 '기타'로 둔다."""
+    if cls.subject != "미분류":
+        return cls
+    for it in pre_evidence.get("items", []):
+        if (it.get("polarity") == "pro" and it.get("targets") == ["영어"]
+                and float(it.get("confidence", 0.0)) >= 0.9):
+            cls.subject, cls.sub_subject = "영어", "기타"
+            break
+    return cls
+
+
 def _env_int(name: str, default: int) -> int:
     """환경변수를 정수로 읽되 비정상 값이면 기본값으로 안전 폴백(import 시 예외 방지)."""
     try:
diff --git a/pipeline/seed_review_gold.py b/pipeline/seed_review_gold.py
index eef8fbb..d8b0579 100644
--- a/pipeline/seed_review_gold.py
+++ b/pipeline/seed_review_gold.py
@@ -24,7 +24,7 @@ from core import review_log
 from core.classifier_engine import (
     extract_all_problems_cached, pre_classify, make_compact_prompt, downscale_for_model,
     infer_cache_key, infer_cache_get, infer_cache_put, safe_json_parse, Classification,
-    apply_filename_prior, apply_cover_prior, is_solution_book)
+    apply_filename_prior, apply_cover_prior, apply_english_prior, is_solution_book)
 from core.confidence import calibrate_confidence
 
 DEFAULT_SEED_DB = Path.home() / ".classi" / "review_seed.db"
@@ -58,6 +58,7 @@ def classify_with_telemetry(prob, filename, model, use_filename=False):
     if use_filename:
         cls = apply_filename_prior(cls, pre)
     cls = apply_cover_prior(cls, pre)
+    cls = apply_english_prior(cls, pre)
     cls, telemetry, _alt = calibrate_confidence(cls, ocr_text, pre)
     return cls, telemetry
 
diff --git a/tests/test_engine.py b/tests/test_engine.py
index 3501968..c32d017 100644
--- a/tests/test_engine.py
+++ b/tests/test_engine.py
@@ -886,6 +886,56 @@ class TestEnglishEvidence(unittest.TestCase):
         text = "다음 글에서 화자의 정서로 가장 적절한 것을 고르시오. 시적 화자는 자연을 노래한다."
         self.assertEqual(E._english_evidence(text), [])
 
+    def test_latin_heavy_non_english_no_signal(self):
+        # 제2외국어(독일어 류): 라틴 글자는 우세하지만 영어 기능어가 없다 → 영어로 단정 안 함.
+        # 종전엔 라틴 우세만 보아 독일어 지문을 '영어'로 오시사하던 잠재 버그를 막는다.
+        text = ("Der Schueler liest ein Buch ueber die Geschichte der Stadt Berlin. " * 3)
+        # 'die/der' 등 독일어어휘는 영어 기능어 정규식에 없다(the/and/of…만 매칭)
+        self.assertEqual(E._english_evidence(text), [])
+
+    def test_listening_boilerplate_signals_english(self):
+        # 영어 듣기평가 안내문 — 수능에서 듣기 영역은 영어뿐. 표지(=문서 단위)에서 '영어' 시사.
+        cover = ("이 문제지에 관한 저작권은 한국교육과정평가원에 있습니다. "
+                 "1번부터 17번까지는 듣고 답하는 문제입니다. 한 번만 들려주고 방송을 잘 듣고 답하시오.")
+        items = E.extract_cover_subject(cover)["items"]
+        eng = [it for it in items if it["targets"] == ["영어"] and it["polarity"] == "pro"]
+        self.assertTrue(eng and eng[0]["confidence"] >= 0.9)
+
+    def test_listening_boilerplate_absent_in_korean(self):
+        # 국어 등 다른 과목 표지엔 듣기 안내문이 없다 → 영어 시사 없음(오탐 방지).
+        cover = "2026학년도 대학수학능력시험 문제지 국어 영역 다음 글을 읽고 물음에 답하시오."
+        items = E.extract_cover_subject(cover)["items"]
+        self.assertFalse([it for it in items if it.get("hint", "").startswith("듣기평가")])
+
+
+class TestApplyEnglishPrior(unittest.TestCase):
+    """모델 미분류를 영어 적극 증거가 있을 때만 영어로 교정(외과적)."""
+    def _pre(self, conf):
+        return {"items": [{"polarity": "pro", "targets": ["영어"], "confidence": conf,
+                           "source": "context"}]}
+
+    def test_promotes_unclassified_with_strong_evidence(self):
+        cls = E.Classification(subject="미분류", sub_subject="기타", confidence=0.0)
+        out = E.apply_english_prior(cls, self._pre(0.9))
+        self.assertEqual((out.subject, out.sub_subject), ("영어", "기타"))
+
+    def test_noop_without_english_evidence(self):
+        cls = E.Classification(subject="미분류", sub_subject="기타", confidence=0.0)
+        out = E.apply_english_prior(cls, {"items": []})
+        self.assertEqual(out.subject, "미분류")
+
+    def test_noop_when_evidence_weak(self):
+        # 0.9 미만 증거로는 승격하지 않는다(고신뢰 증거만 하드 교정).
+        cls = E.Classification(subject="미분류", sub_subject="기타", confidence=0.0)
+        out = E.apply_english_prior(cls, self._pre(0.6))
+        self.assertEqual(out.subject, "미분류")
+
+    def test_never_touches_confident_prediction(self):
+        # 모델이 이미 과목을 골랐으면(미분류 아님) 영어 증거가 있어도 건드리지 않는다.
+        cls = E.Classification(subject="국어", sub_subject="독서", confidence=0.8)
+        out = E.apply_english_prior(cls, self._pre(0.95))
+        self.assertEqual((out.subject, out.sub_subject), ("국어", "독서"))
+
 
 class TestFindProblemBoxes(unittest.TestCase):
     """find_problem_boxes 통합(텍스트레이어 경로, OCR 무관) — 핵심 오케스트레이터 회귀 가드.
diff --git a/tools/eval_accuracy.py b/tools/eval_accuracy.py
index a2445b3..6fee5b6 100644
--- a/tools/eval_accuracy.py
+++ b/tools/eval_accuracy.py
@@ -34,9 +34,9 @@ from pathlib import Path
 
 sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
 from core.classifier_engine import (  # noqa: E402
-    Classification, apply_cover_prior, apply_filename_prior, downscale_for_model,
-    extract_all_problems_cached, infer_cache_get, infer_cache_key, infer_cache_put,
-    is_solution_book, make_compact_prompt, pre_classify, safe_json_parse)
+    Classification, apply_cover_prior, apply_english_prior, apply_filename_prior,
+    downscale_for_model, extract_all_problems_cached, infer_cache_get, infer_cache_key,
+    infer_cache_put, is_solution_book, make_compact_prompt, pre_classify, safe_json_parse)
 from core.confidence import calibrate_confidence  # noqa: E402
 
 DEFAULT_MANIFEST = Path(__file__).parent / "eval_manifest.jsonl"
@@ -99,6 +99,7 @@ def classify_problem(prob: dict, filename: str, model: str, use_filename: bool):
     if use_filename:
         cls = apply_filename_prior(cls, pre)
     cls = apply_cover_prior(cls, pre)
+    cls = apply_english_prior(cls, pre)
     cls, _telemetry, _alt = calibrate_confidence(cls, ocr_text, pre)
     return cls
```

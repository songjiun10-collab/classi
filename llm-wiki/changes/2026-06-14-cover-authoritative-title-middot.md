# 2026-06-14 · 표지 권위 제목 + 가운뎃점 정규화

> 백엔드 정확도 개선 야간 작업(브랜치 `improve/overnight-20260614`)의 변경 1건.

## 맥락
표지에 본문이 섞여 한국지리의 '경제 수역'→경제, 생명과학의 '이화작용'→화작으로 오탐돼 단일과목 표지 인식이 모호해졌다. '영역(과목)' 권위 제목 우선 + 반각 가운뎃점(U+FF65) 정규화.

## 커밋
`43de1e7` — fix(cover): authoritative '영역(과목)' title + middot normalization

## 변경된 파일
```
core/classifier_engine.py | 30 ++++++++++++++++++++++++++++--
 tests/test_engine.py      | 20 ++++++++++++++++++++
 2 files changed, 48 insertions(+), 2 deletions(-)
```

## 전체 diff

```diff
diff --git a/core/classifier_engine.py b/core/classifier_engine.py
index 11b6133..4937cb3 100644
--- a/core/classifier_engine.py
+++ b/core/classifier_engine.py
@@ -191,6 +191,9 @@ _TERM_DENSITY_PATTERN = re.compile(r"[가-힣]{2,}")
 _ASCII_ROMAN_RE = re.compile(r"(?<=[가-힣])(iii|ii|i)(?![a-z])")
 _ASCII_ROMAN_MAP = {"i": "1", "ii": "2", "iii": "3"}
 
+_MIDDOT_TRANS = {ord(c): None for c in "·・･‧•"}  # 가운뎃점 변형 일괄 제거(translate용)
+
+
 def _norm_base(s: str) -> str:
     # 모든 과목명 정규화의 공통 토대. 유니코드 로마숫자(Ⅰ/Ⅱ/Ⅲ)는 .lower() 전에 치환한다
     # (str.lower()가 Ⅱ(U+2161)→ⅱ(U+2171)로 바꿔 이후 치환이 빗나가는 것을 방지).
@@ -200,7 +203,10 @@ def _norm_base(s: str) -> str:
     s = unicodedata.normalize("NFC", str(s or ""))
     s = s.replace("Ⅲ", "3").replace("Ⅱ", "2").replace("Ⅰ", "1")
     s = s.replace("ⅲ", "3").replace("ⅱ", "2").replace("ⅰ", "1")
-    return s.lower().replace(" ", "").replace("·", "")
+    # 가운뎃점 변형 모두 제거 — 캐논 '·'(U+00B7)뿐 아니라 표지 OCR/인코딩에서 들어오는
+    # 반각·전각 변형(･ U+FF65, ・ U+30FB, ‧ U+2027, • U+2022)도 통일해야 '사회·문화'가
+    # 매칭된다(실측: 평가원 사회·문화 표지는 U+FF65를 써 종전엔 과목명이 0건 매칭됐다).
+    return s.lower().replace(" ", "").translate(_MIDDOT_TRANS)
 
 def _norm(s: str) -> str:
     # 표시형 비교용 정규화: 공통 토대 + ASCII 로마숫자(I/II/III)를 '한글 접두 뒤'에서만 치환.
@@ -453,10 +459,24 @@ def extract_filename_meta(filename: str) -> dict:
             return {"items": items}
     return {"items": items}
 
+_COVER_TITLE_RE = re.compile(r"영역[(（]([^)）]+)[)）]")  # '사회탐구영역(한국지리)' 권위 제목
+
+
 def extract_cover_subject(cover_text: str) -> dict:
     items = []
     tn = _norm(cover_text)
     matched_norms: List[str] = []  # 이미 매칭된 더 구체적인 과목명의 정규형
+    # 0) 권위 제목 '{대분류}영역(세부과목)' — 평가원 탐구영역 표지의 확정 제목이다.
+    #    표지에 본문 텍스트가 섞여(예: 한국지리의 '배타적 경제 수역'→'경제', 생명과학의
+    #    '이화작용'→'화작') 다른 과목이 동시에 잡히는 오탐을 이 제목이 무력화한다(0.95·권위).
+    #    cover_single_subject가 권위 항목이 있으면 그것만 채택한다.
+    m = _COVER_TITLE_RE.search(tn)
+    if m:
+        hit = _SUB_CANON.get(_course_key(m.group(1)))
+        if hit:
+            items.append({"keyword": hit[1], "type": "cover", "polarity": "pro",
+                          "targets": [hit[1]], "hint": f"영역 괄호 제목 → {hit[1]}",
+                          "confidence": 0.95, "source": "cover", "authoritative": True})
     # 1) 정규 세부 과목명 직접 매칭 (예: "물리학Ⅱ", "확률과 통계") — 가장 구체적
     for _parent, subs in CURRICULUM.items():
         for ss in subs:
@@ -854,7 +874,9 @@ def _has_filename_prior(pre_evidence: dict) -> bool:
 def cover_single_subject(pre_evidence: dict) -> Optional[Tuple[str, str]]:
     """표지가 '하나의' 세부 과목만 명확히(conf≥0.9) 가리키면 (대분류, 세부과목)을 반환.
     표지가 여러 세부 과목을 담거나(예: 통합 문제집) 세부 과목을 못 잡으면 None.
-    단일 과목 시험의 표지는 사실상 정답이므로 reconciliation 프라이어로 쓴다."""
+    단일 과목 시험의 표지는 사실상 정답이므로 reconciliation 프라이어로 쓴다.
+    권위 제목('영역(세부과목)')이 있으면 본문 텍스트 오탐을 무시하고 그것만 채택한다."""
+    auth = set()
     pairs = set()
     for it in pre_evidence.get("items", []):
         if it.get("source") != "cover" or it.get("polarity") != "pro":
@@ -865,6 +887,10 @@ def cover_single_subject(pre_evidence: dict) -> Optional[Tuple[str, str]]:
             hit = _SUB_CANON.get(_course_key(t))
             if hit:
                 pairs.add(hit)
+                if it.get("authoritative"):
+                    auth.add(hit)
+    if auth:
+        return next(iter(auth)) if len(auth) == 1 else None
     return next(iter(pairs)) if len(pairs) == 1 else None
 
 
diff --git a/tests/test_engine.py b/tests/test_engine.py
index c32d017..019e9b8 100644
--- a/tests/test_engine.py
+++ b/tests/test_engine.py
@@ -172,6 +172,26 @@ class TestCoverSubject(unittest.TestCase):
         self.assertIn("물리학Ⅱ", t)
         self.assertNotIn("물리학Ⅰ", t)
 
+    def test_authoritative_title_overrides_body_collision(self):
+        # BUG: 표지에 본문이 섞여 한국지리 시험의 '배타적 경제 수역'이 사회탐구 '경제'로 잡혀
+        # cover_single_subject가 한국지리·경제 모호→None→교정 실패. 권위 제목이 이를 무력화.
+        cover = ("사회탐구영역(한국지리)\n제4 교시\n"
+                 "1. 다음 조건을 만족하는 곳은? <조건> 배타적 경제 수역에 위치")
+        pre = {"items": E.extract_cover_subject(cover)["items"]}
+        self.assertEqual(E.cover_single_subject(pre), ("사회탐구", "한국지리"))
+
+    def test_authoritative_title_overrides_alias_collision(self):
+        # 생명과학 본문 '이화작용'이 '화작'(화법과작문 별칭)으로 잡히던 충돌도 권위 제목이 이긴다.
+        cover = "과학탐구영역(생명과학Ⅰ)\n1. (가)에서 이화작용이 일어난다."
+        pre = {"items": E.extract_cover_subject(cover)["items"]}
+        self.assertEqual(E.cover_single_subject(pre), ("과학탐구", "생명과학Ⅰ"))
+
+    def test_halfwidth_middot_subject_name_matched(self):
+        # BUG: 평가원 사회·문화 표지는 반각 가운뎃점(U+FF65 '･')을 써 '사회문화'와 매칭 실패했다.
+        cover = "사회탐구영역(사회･문화)\n제4 교시"
+        self.assertEqual(E.cover_single_subject({"items": E.extract_cover_subject(cover)["items"]}),
+                         ("사회탐구", "사회·문화"))
+
     def test_csat_title_not_misread_as_math(self):
         # BUG: '대학수학능력시험' 속 '수학'이 모든 수능 PDF에서 가짜 수학 신호로 잡힘
         t = self._targets("2026학년도 대학수학능력시험 과학탐구영역 물리학II")
```

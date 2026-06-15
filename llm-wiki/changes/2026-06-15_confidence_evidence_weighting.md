# 2026-06-15 · 신뢰도 보정 fallback을 evidence-confidence 가중으로 교체

## 배경
`debate/*_en.md`(영어 멀티-AI 토론) 결론을 코드에 반영. autoresearch/Karpathy 가이드라인
(최소 코드·외과적 변경·측정 가능한 목표 + 테스트 검증)에 맞춰, 토론이 합의한 **최소·최우선
변경**만 적용했다.

## 영어 debate 결론 (한국어 본문 무시)
- **Q1 플라이휠**: 4/4 만장일치 "shelve 유지가 옳다"(±0.15 한정 효과·삭제된 producer·
  minimum-code·발열) → **코드 액션 없음**(producer 재건 금지).
- **Q2 핵심 결함 = D**: 하드코딩 fallback이 증거를 *가중 없는 정수 카운트*로 처리하고 각
  증거의 `conf`/출처 신뢰도를 무시(대표성 결함, 나머지는 증상).
- **perf debate**: 동일 결론 → **(c) evidence-confidence weighting 먼저 ship**,
  (d) 다문항 vstack 배칭은 이후 파일럿(발열/JSON 파싱 리스크)으로 미룸.
- 부수 결함 C: `기타` −0.12 패널티가 증거 무관 **무조건** 차감 → 희소 정답 recall 저하.

## 변경 파일
- `backend/core/confidence.py` — `calibrate_confidence` fallback 수정.
- `backend/tests/test_engine.py` — 결함 D/C 회귀 테스트 3건 추가.

## 변경 요약
1. **결함 D**: `pro_score/anti_score/competing_score` 누적을 `weight`(출처 가중) →
   `weight × conf`(출처 × 증거 신뢰도)로 변경. `conf`는 종전에 competing 게이트(>=0.7)에만
   쓰였음. **정수 카운트 `*_match`는 텔레메트리·gold 로지스틱 특징이라 그대로 int 유지**
   (test_match_counts_are_integers 보호). 계수(0.05/0.12/0.08)·`min(score,3)` 캡 불변
   → 총 이동폭 ~±0.15 유지(무회귀).
2. **결함 C**: `기타` 패널티를 무조건 → 조건부(`pro_score < 1.0`일 때만 −0.12). 강한
   파일명/표지 지지가 있는 정답 '기타'는 면제, 지지 없는 추정 '기타'는 종전대로 차감.
3. `_learned_confidence`(gold 경로)·텔레메트리 스키마·로지스틱 미변경 → gold 가중치 호환.

## 검증
`cd backend && python3 -m unittest tests.test_engine` → **179 OK** (기존 176 + 신규 3,
ollama·OCR 불필요·결정적). 신규: `test_strong_pro_beats_weak_pro`,
`test_other_penalty_waived_with_strong_pro`, `test_other_penalty_applies_without_support`.

## 적용하지 않은 것 (의도)
- 플라이휠 producer 재건(Q1 만장일치 shelve).
- (d) 다문항 vstack 배칭 / `core/cache.py` 분리 리팩터 — 토론이 "이후"로 미룬 더 큰 변경.

## diff
```diff
diff --git a/core/confidence.py b/core/confidence.py
@@ def calibrate_confidence(res, text: str, pre_evidence: dict) -> Tuple:
         hit = any(target_str == t or subject_str == t
                   or _norm(target_str) == _norm(t) or _norm(subject_str) == _norm(t)
                   for t in targets)
-        weight = 1.5 if source == "filename" else 1.0
-        if hit and pol == "pro": pro_match += 1; pro_score += weight; pro_targets_set.update(targets)
-        elif hit and pol == "anti": anti_match += 1; anti_score += weight; anti_targets_set.update(targets)
+        # 증거 강도 = 출처 가중(파일명 1.5×) × 그 증거 자신의 신뢰도(conf).
+        # 종전엔 conf를 competing 게이트(>=0.7)에만 쓰고 점수엔 무시 → 약한 OCR 증거 여러 개가
+        # 강한 증거 하나를 덮었다(영어 debate 결함 D: '가중 없는 정수 카운트'). conf로 가중해 교정.
+        # _match(정수 카운트)는 텔레메트리·gold 로지스틱 특징이므로 그대로 정수 유지.
+        strength = (1.5 if source == "filename" else 1.0) * conf
+        if hit and pol == "pro": pro_match += 1; pro_score += strength; pro_targets_set.update(targets)
+        elif hit and pol == "anti": anti_match += 1; anti_score += strength; anti_targets_set.update(targets)
         elif pol == "pro" and not hit:
-            if conf >= 0.7: competing += 1; competing_score += weight; competing_targets_set.update(targets)
+            if conf >= 0.7: competing += 1; competing_score += strength; competing_targets_set.update(targets)
         elif pol == "anti" and not hit: anti_targets_set.update(targets)
@@
         else:  # 하드코딩 폴백 = 현재 기본 동작(가중치 없음 → 무회귀)
             c += 0.05 * min(pro_score, 3); c -= 0.12 * min(anti_score, 3); c -= 0.08 * min(competing_score, 3)
             if rule_conflict: c -= 0.08
-            if res.sub_subject == "기타": c -= 0.12
+            # '기타' 패널티는 무조건이 아니라 '지지 증거가 약할 때'만(영어 debate 결함 C).
+            # 파일명/표지가 강하게 '기타'를 지지하는 정답까지 무조건 깎으면 희소 정답을
+            # 검수 임계 아래로 눌러 recall을 해친다 → pro_score가 한 건 분량 미만일 때만 차감.
+            if res.sub_subject == "기타" and pro_score < 1.0: c -= 0.12
```
(테스트 추가분 diff는 `git diff tests/test_engine.py` 참조.)

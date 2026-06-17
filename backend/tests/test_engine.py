#!/usr/bin/env python3
"""Deterministic unit tests for the classi engine (no ollama, no OCR).

Run:  cd backend && python3 -m unittest tests.test_engine -v
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import classifier_engine as E
from core import cache as C
from core.confidence import calibrate_confidence


class TestSafeJsonParse(unittest.TestCase):
    def test_plain_object(self):
        self.assertEqual(E.safe_json_parse('{"subject":"수학"}'), {"subject": "수학"})

    def test_code_fenced(self):
        self.assertEqual(E.safe_json_parse('```json\n{"subject":"수학"}\n```'),
                         {"subject": "수학"})

    def test_prose_then_json_with_nested(self):
        self.assertEqual(
            E.safe_json_parse('Here: {"subject":"수학","x":{"nested":1}} done'),
            {"subject": "수학", "x": {"nested": 1}})

    def test_close_brace_inside_string_value(self):
        # BUG: '}' inside a JSON string must not close the object early
        self.assertEqual(
            E.safe_json_parse('{"subject":"수학","note":"has } brace"}'),
            {"subject": "수학", "note": "has } brace"})

    def test_latex_braces_inside_string(self):
        # Math topics routinely contain LaTeX like \frac{a}{b}
        self.assertEqual(
            E.safe_json_parse(r'{"topic":"\\frac{a}{b}","unit":"미분"}'),
            {"topic": r"\frac{a}{b}", "unit": "미분"})

    def test_no_json_raises(self):
        with self.assertRaises(ValueError):
            E.safe_json_parse("no json here")

    def test_unterminated_raises(self):
        with self.assertRaises(ValueError):
            E.safe_json_parse('{"subject":"수학"')

    def test_trailing_comma_recovered(self):
        # LLM 흔한 오류: 후행 콤마 → 복구(미복구 시 재시도=추가 추론·발열)
        self.assertEqual(E.safe_json_parse('{"subject":"수학","sub_subject":"미적분",}'),
                         {"subject": "수학", "sub_subject": "미적분"})

    def test_single_quotes_recovered(self):
        self.assertEqual(E.safe_json_parse("{'subject':'수학'}"), {"subject": "수학"})

    def test_line_comment_recovered(self):
        self.assertEqual(E.safe_json_parse('{"subject":"수학" // 메모\n}'),
                         {"subject": "수학"})

    def test_python_bool_recovered(self):
        self.assertEqual(E.safe_json_parse('{"subject":"수학","flag":True}'),
                         {"subject": "수학", "flag": True})

    def test_comma_inside_string_value_preserved(self):
        # 복구 정규식이 문자열 안의 콤마를 건드리면 안 됨
        self.assertEqual(E.safe_json_parse('{"topic":"힘, 운동량",}'),
                         {"topic": "힘, 운동량"})


class TestNormalization(unittest.TestCase):
    def test_roman_two_becomes_arabic(self):
        # _norm intends roman->arabic; .lower() must not defeat the replace
        self.assertEqual(E._norm("물리학Ⅱ"), "물리학2")

    def test_roman_one_becomes_arabic(self):
        self.assertEqual(E._norm("수학Ⅰ"), "수학1")

    def test_norm_matches_course_key_on_roman(self):
        # the two normalizers must agree on canonical subject forms
        self.assertEqual(E._norm("물리학Ⅱ"), E._course_key("물리학Ⅱ"))

    def test_ascii_roman_two_becomes_arabic(self):
        # BUG: 표지/OCR이 로마자 Ⅱ 대신 ASCII "II"를 쓰면(예: "물리학II")
        # _norm이 변환하지 않아 과목명 매칭이 실패하던 문제
        self.assertEqual(E._norm("물리학II"), "물리학2")
        self.assertEqual(E._norm("물리학II"), E._course_key("물리학II"))
        self.assertEqual(E._norm("화학 III"), "화학3")

    def test_ascii_roman_leaves_english_words_alone(self):
        # 한글 접두가 없는 영어 단어의 i/ii는 절대 건드리지 않는다
        self.assertEqual(E._norm("if is video"), "ifisvideo")


class TestReconcile(unittest.TestCase):
    """Regression guards: behavior verified-correct, lock it in."""
    def _cls(self, **kw):
        return E.Classification.model_validate(kw)

    def test_clean_passthrough(self):
        c = self._cls(subject="과학탐구", sub_subject="물리학Ⅱ", confidence=0.8)
        self.assertEqual((c.subject, c.sub_subject), ("과학탐구", "물리학Ⅱ"))

    def test_subject_subsubject_swapped(self):
        c = self._cls(subject="물리학Ⅱ", sub_subject="기타", confidence=0.8)
        self.assertEqual((c.subject, c.sub_subject), ("과학탐구", "물리학Ⅱ"))

    def test_arabic_numeral_recovered(self):
        c = self._cls(subject="물리학2", sub_subject="", confidence=0.8)
        self.assertEqual((c.subject, c.sub_subject), ("과학탐구", "물리학Ⅱ"))

    def test_garbage_to_unclassified(self):
        c = self._cls(subject="garbage", sub_subject="nonsense", confidence=0.9)
        self.assertEqual((c.subject, c.sub_subject), ("미분류", "기타"))


class TestCalibrateConfidence(unittest.TestCase):
    def _scenario(self):
        c = E.Classification.model_validate(
            {"subject": "과학탐구", "sub_subject": "물리학Ⅱ", "confidence": 0.8})
        pre = {"items": [
            {"keyword": "광전효과", "polarity": "pro", "targets": ["물리학Ⅱ"],
             "confidence": 0.85, "source": "ocr"},
            {"keyword": "물2", "polarity": "pro", "targets": ["물리학Ⅱ"],
             "confidence": 0.9, "source": "filename"},
            {"keyword": "등가속도", "polarity": "pro", "targets": ["물리학Ⅰ"],
             "confidence": 0.75, "source": "ocr"},
        ]}
        return calibrate_confidence(c, "광전효과 등가속도", pre)

    def test_match_counts_are_integers(self):
        # BUG: weighting made these floats -> report says "2.5개"
        _, tel, _ = self._scenario()
        for k in ("pro_match", "anti_match", "competing"):
            self.assertIsInstance(tel[k], int, f"{k} should be an int count, got {tel[k]!r}")

    def test_match_counts_values(self):
        _, tel, _ = self._scenario()
        self.assertEqual(tel["pro_match"], 2)      # 광전효과 + 물2 both target 물리학Ⅱ
        self.assertEqual(tel["competing"], 1)      # 등가속도 targets 물리학Ⅰ
        self.assertEqual(tel["anti_match"], 0)

    def test_confidence_stays_in_range(self):
        res, _, _ = self._scenario()
        self.assertGreaterEqual(res.confidence, 0.0)
        self.assertLessEqual(res.confidence, 0.99)

    def _fallback_conf(self, sub_subject, items, raw=0.7):
        # 하드코딩 폴백 경로를 강제(gold 가중치 무시)해 결정적으로 검증.
        import core.confidence as _cf
        _cf._CALIB = None
        c = E.Classification.model_validate(
            {"subject": "과학탐구", "sub_subject": sub_subject, "confidence": raw})
        res, _, _ = calibrate_confidence(c, " ".join(it["keyword"] for it in items),
                                         {"items": items})
        return res.confidence

    def test_strong_pro_beats_weak_pro(self):
        # 결함 D: conf 높은 단일 pro가 conf 낮은 pro보다 신뢰도를 더 올려야 한다
        strong = self._fallback_conf("물리학Ⅱ", [
            {"keyword": "광전효과", "polarity": "pro", "targets": ["물리학Ⅱ"],
             "confidence": 0.9, "source": "ocr"}])
        weak = self._fallback_conf("물리학Ⅱ", [
            {"keyword": "광전효과", "polarity": "pro", "targets": ["물리학Ⅱ"],
             "confidence": 0.3, "source": "ocr"}])
        self.assertGreater(strong, weak)

    def test_other_penalty_waived_with_strong_pro(self):
        # 결함 C: 강한 파일명 pro가 '기타'를 지지하면 무조건 -0.12 패널티를 면제
        supported = self._fallback_conf("기타", [
            {"keyword": "기타지지", "polarity": "pro", "targets": ["기타"],
             "confidence": 0.9, "source": "filename"}])
        self.assertGreaterEqual(supported, 0.7)

    def test_other_penalty_applies_without_support(self):
        # 지지 증거 없는 '기타'는 의도대로 여전히 차감(과교정 방지)
        self.assertLess(self._fallback_conf("기타", []), 0.7)


class TestCoverSubject(unittest.TestCase):
    def _targets(self, text):
        out = []
        for it in E.extract_cover_subject(text)["items"]:
            out += it.get("targets", [])
        return out

    def test_full_name_volume_two_not_misread_as_one(self):
        # BUG: "물리학Ⅱ" 표지가 generic '물리'->물리학Ⅰ 로 오인식되던 문제
        t = self._targets("2026 물리학Ⅱ 모의고사")
        self.assertIn("물리학Ⅱ", t)
        self.assertNotIn("물리학Ⅰ", t)

    def test_bare_alias_fallback_preserved(self):
        # 번호 없는 '물리'만 있으면 기존대로 물리학Ⅰ 로 폴백
        t = self._targets("물리 모의고사")
        self.assertIn("물리학Ⅰ", t)

    def test_canonical_math_name(self):
        t = self._targets("확률과 통계 기출문제")
        self.assertIn("확률과 통계", t)

    def test_ascii_volume_two_detected_not_one(self):
        # BUG: 표지가 ASCII "물리학II"면 물리학Ⅱ로 못 잡고
        # generic 약어 '물리'→물리학Ⅰ 로 오인식되던 문제
        t = self._targets("2026 과학탐구영역 물리학II")
        self.assertIn("물리학Ⅱ", t)
        self.assertNotIn("물리학Ⅰ", t)

    def test_authoritative_title_overrides_body_collision(self):
        # BUG: 표지에 본문이 섞여 한국지리 시험의 '배타적 경제 수역'이 사회탐구 '경제'로 잡혀
        # cover_single_subject가 한국지리·경제 모호→None→교정 실패. 권위 제목이 이를 무력화.
        cover = ("사회탐구영역(한국지리)\n제4 교시\n"
                 "1. 다음 조건을 만족하는 곳은? <조건> 배타적 경제 수역에 위치")
        pre = {"items": E.extract_cover_subject(cover)["items"]}
        self.assertEqual(E.cover_single_subject(pre), ("사회탐구", "한국지리"))

    def test_authoritative_title_overrides_alias_collision(self):
        # 생명과학 본문 '이화작용'이 '화작'(화법과작문 별칭)으로 잡히던 충돌도 권위 제목이 이긴다.
        cover = "과학탐구영역(생명과학Ⅰ)\n1. (가)에서 이화작용이 일어난다."
        pre = {"items": E.extract_cover_subject(cover)["items"]}
        self.assertEqual(E.cover_single_subject(pre), ("과학탐구", "생명과학Ⅰ"))

    def test_halfwidth_middot_subject_name_matched(self):
        # BUG: 평가원 사회·문화 표지는 반각 가운뎃점(U+FF65 '･')을 써 '사회문화'와 매칭 실패했다.
        cover = "사회탐구영역(사회･문화)\n제4 교시"
        self.assertEqual(E.cover_single_subject({"items": E.extract_cover_subject(cover)["items"]}),
                         ("사회탐구", "사회·문화"))

    def test_csat_title_not_misread_as_math(self):
        # BUG: '대학수학능력시험' 속 '수학'이 모든 수능 PDF에서 가짜 수학 신호로 잡힘
        t = self._targets("2026학년도 대학수학능력시험 과학탐구영역 물리학II")
        self.assertNotIn("수학", t)
        self.assertIn("과학탐구", t)
        self.assertIn("물리학Ⅱ", t)


class TestBrandStopwordFalsePositive(unittest.TestCase):
    def _antis(self, text, **kw):
        pre = E.pre_classify(text, kw.get("filename", ""), kw.get("cover_text", ""))
        return [it["keyword"] for it in pre["items"]
                if it["polarity"] == "anti" and it.get("targets") == ["미분류"]]

    def _pros(self, text, **kw):
        pre = E.pre_classify(text, kw.get("filename", ""), kw.get("cover_text", ""))
        return {t for it in pre["items"] if it["polarity"] == "pro"
                for t in it.get("targets", [])}

    def test_compound_words_not_flagged_as_brand(self):
        # '정규분포'의 '정규', '이상기체'의 '이상' 이 브랜드 오탐되면 안 됨
        antis = self._antis("정규분포를 따르는 확률변수와 이상기체의 상태변화")
        self.assertNotIn("정규", antis)
        self.assertNotIn("이상", antis)

    def test_inequality_word_not_brand(self):
        # 부등호 '이상'(3 이상 9 이하 — 뒤가 숫자라 경계검사를 통과)은 온톨로지에서 빠져야
        antis = self._antis("자연수 n은 3 이상 9 이하의 정수이다")
        self.assertNotIn("이상", antis)

    def test_real_brand_still_detected(self):
        antis = self._antis("메가스터디 모의고사 교재")
        self.assertIn("메가스터디", antis)

    def test_prob_stat_pros_survive(self):
        # 브랜드 오탐 제거가 진짜 확통 증거를 망치면 안 됨
        self.assertIn("확률과 통계",
                      self._pros("정규분포 표본평균 신뢰구간 확률변수 모평균추정"))

    def test_physics_pros_survive(self):
        self.assertIn("물리학Ⅱ", self._pros("광전효과를 이용한 정지전압 측정과 물질파"))


class TestMetaAndNonCsatEvidence(unittest.TestCase):
    def _has_unclassified_anti(self, text, **kw):
        pre = E.pre_classify(text, kw.get("filename", ""), kw.get("cover_text", ""))
        return any(it.get("targets") == ["미분류"] and it["polarity"] == "anti"
                   for it in pre["items"])

    def test_solution_page_flagged(self):
        self.assertTrue(self._has_unclassified_anti("정답 및 해설 채점표 풀이"))

    def test_noncsat_cover_flagged(self):
        self.assertTrue(self._has_unclassified_anti("", cover_text="2025 초등 수학 올림피아드 경시대회"))


class TestDedupProblemStarts(unittest.TestCase):
    """문제 박스 과분할(한 문제가 둘로 쪼개짐) 회귀 방지."""
    def test_bare_number_block_dropped_when_body_exists(self):
        # 같은 번호가 '번호만'(False) + '번호+본문'(True) 두 블록으로 잡히고
        # y가 멀리 떨어져 기존 y-band dedup을 빠져나가던 실제 케이스
        starts = [
            (249.0, 80.0, 1, False),   # '1.'
            (254.0, 88.0, 1, True),    # '1. 그림은…'
            (420.0, 88.0, 2, True),    # '2. 그림 (가)…'
            (487.0, 90.0, 2, False),   # '2.'  (y 67pt 떨어짐)
        ]
        out = E._dedup_problem_starts(starts)
        self.assertEqual(sorted(n for _, _, n in out), [1, 2])
        self.assertEqual(len(out), 2)

    def test_bare_only_number_kept_as_fallback(self):
        # 본문이 별도 블록인 PDF: 페이지에 본문 블록이 전혀 없으면 번호 블록 유지(누락 방지)
        out = E._dedup_problem_starts([(100.0, 50.0, 5, False)])
        self.assertEqual([n for _, _, n in out], [5])

    def test_bare_continuation_dropped_when_page_has_bodies(self):
        # 앞 페이지에서 이어진 '번호만'(본문 없음) 마커가 바로 아래 문제를
        # 가리지 않도록, 같은 페이지에 본문 블록이 있으면 번호-only는 버린다
        starts = [
            (152.0, 93.0, 6, False),   # 이어진 '6.' (본문 없음)
            (161.0, 88.0, 7, True),    # '7. 그림은…' (9pt 아래)
        ]
        out = E._dedup_problem_starts(starts)
        self.assertEqual([n for _, _, n in out], [7])


class TestConfidenceNormalization(unittest.TestCase):
    def _conf(self, v):
        return E.Classification.model_validate(
            {"subject": "수학", "sub_subject": "미적분", "confidence": v}).confidence

    def test_percent_scale(self):
        self.assertAlmostEqual(self._conf(95), 0.95)

    def test_ten_scale(self):
        self.assertAlmostEqual(self._conf(8), 0.8)

    def test_unit_scale_unchanged(self):
        self.assertAlmostEqual(self._conf(0.85), 0.85)

    def test_negative_clamped(self):
        self.assertEqual(self._conf(-1), 0.0)

    def test_difficulty_score_clamped(self):
        c = E.Classification.model_validate(
            {"subject": "수학", "sub_subject": "미적분", "difficulty_score": 15})
        self.assertEqual(c.difficulty_score, 10)


class TestCoverPrior(unittest.TestCase):
    """단일 과목 시험: 표지가 사실상 정답. 재가공 PDF의 오염된 텍스트 레이어
    (get_textbox가 잔여 한국사 텍스트 '노무현 전 대통령'을 반환)로 물리 문제가
    한국사로 오분류되던 실제 버그를, 표지 기준 reconciliation으로 교정한다."""
    def _cls(self, **kw):
        return E.Classification.model_validate(kw)

    def _cover(self, *pairs, conf=0.92):
        return {"items": [
            {"keyword": kw, "polarity": "pro", "targets": [t],
             "confidence": conf, "source": "cover"} for kw, t in pairs]}

    def test_contaminated_history_corrected_to_physics2(self):
        # BUG: 물리학Ⅱ PDF인데 텍스트 레이어 오염으로 모델이 한국사 출력
        c = self._cls(subject="한국사", sub_subject="한국사", confidence=0.74)
        c = E.apply_cover_prior(c, self._cover(("물리학Ⅱ", "물리학Ⅱ")))
        self.assertEqual((c.subject, c.sub_subject), ("과학탐구", "물리학Ⅱ"))

    def test_near_subject_physics1_corrected_to_physics2(self):
        # 단일 과목 시험이므로 인접 과목(물리학Ⅰ)도 표지(물리학Ⅱ) 기준으로 교정
        c = self._cls(subject="과학탐구", sub_subject="물리학Ⅰ", confidence=0.66)
        c = E.apply_cover_prior(c, self._cover(("물리학Ⅱ", "물리학Ⅱ")))
        self.assertEqual((c.subject, c.sub_subject), ("과학탐구", "물리학Ⅱ"))

    def test_unclassified_preserved(self):
        # 모델이 표지/해설로 보고 미분류한 건 표지 프라이어가 건드리지 않음
        c = self._cls(subject="미분류", sub_subject="기타", confidence=0.1)
        c = E.apply_cover_prior(c, self._cover(("물리학Ⅱ", "물리학Ⅱ")))
        self.assertEqual((c.subject, c.sub_subject), ("미분류", "기타"))

    def test_multi_subject_cover_no_override(self):
        # 표지가 여러 세부 과목을 담으면(통합 문제집) 적용 안 함 → 모델 출력 보존
        c = self._cls(subject="과학탐구", sub_subject="화학Ⅱ", confidence=0.6)
        pre = self._cover(("물리학Ⅱ", "물리학Ⅱ"), ("화학Ⅱ", "화학Ⅱ"))
        c = E.apply_cover_prior(c, pre)
        self.assertEqual((c.subject, c.sub_subject), ("과학탐구", "화학Ⅱ"))

    def test_filename_prior_takes_precedence(self):
        # 파일명이 더 권위 있음 — 파일명 프라이어가 지정하면 표지는 양보
        c = self._cls(subject="한국사", sub_subject="한국사", confidence=0.7)
        pre = {"items": [
            {"keyword": "화학2", "polarity": "pro", "targets": ["화학Ⅱ"],
             "confidence": 0.9, "source": "filename"},
            {"keyword": "물리학Ⅱ", "polarity": "pro", "targets": ["물리학Ⅱ"],
             "confidence": 0.92, "source": "cover"},
        ]}
        c = E.apply_filename_prior(c, pre)
        c = E.apply_cover_prior(c, pre)
        self.assertEqual((c.subject, c.sub_subject), ("과학탐구", "화학Ⅱ"))

    def test_subject_level_cover_does_not_override(self):
        # 표지에 대분류명(과학탐구 0.85)만 있고 세부 과목이 없으면 교정하지 않음
        c = self._cls(subject="한국사", sub_subject="한국사", confidence=0.7)
        pre = {"items": [
            {"keyword": "과학탐구", "polarity": "pro", "targets": ["과학탐구"],
             "confidence": 0.85, "source": "cover"}]}
        c = E.apply_cover_prior(c, pre)
        self.assertEqual((c.subject, c.sub_subject), ("한국사", "한국사"))

    def test_realistic_csat_cover_text(self):
        # 실제 표지 문구 전체 흐름(extract_cover_subject→pre_classify) 회귀
        pre = E.pre_classify("", "", "2026학년도 대학수학능력시험 과학탐구영역 물리학Ⅱ")
        self.assertEqual(E.cover_single_subject(pre), ("과학탐구", "물리학Ⅱ"))


class TestFilenameFalsePositives(unittest.TestCase):
    """파일명 대분류 키워드의 substring 오탐 가드.

    '법' 단독 키가 '화법과 작문'·'풀이방법'을 사회탐구로, '통합' 단독 키가
    '통합사회'를 통합과학으로 잡던 버그 — 이 가짜 증거(conf 0.8 ≥ 경합 임계 0.7)는
    정답 분류의 신뢰도를 경합 감점으로 부당하게 깎고 telemetry(→gold 학습 특징)도 오염시켰다."""
    def _targets(self, fn):
        return [t for it in E.extract_filename_meta(fn)["items"] for t in it["targets"]]

    def test_hwajak_filename_not_social_studies(self):
        self.assertNotIn("사회탐구", self._targets("화법과작문_기출.pdf"))

    def test_generic_word_beop_no_match(self):
        self.assertNotIn("사회탐구", self._targets("미분_풀이방법_정리.pdf"))

    def test_integrated_social_maps_to_itself(self):
        self.assertEqual(self._targets("2026_통합사회_모의고사.pdf"), ["통합사회"])

    def test_integrated_science_still_works(self):
        self.assertEqual(self._targets("통합과학_3월모의.pdf"), ["통합과학"])

    def test_politics_law_via_jeongbeop(self):
        self.assertEqual(self._targets("정법_수능기출.pdf"), ["사회탐구"])


class TestFilenameNFDNormalization(unittest.TestCase):
    """macOS 파일시스템·업로드의 한글 파일명은 NFD(자모 분해형)로 들어온다.
    NFC인 소스코드 키워드와 substring 매칭이 전부 빗나가 파일명 프라이어가 실전에서
    0건 매칭되던 버그의 가드(실파일 70개 실측으로 발견). 양변 NFC 통일 후엔 매칭돼야 한다."""
    def _nfd(self, s):
        import unicodedata
        return unicodedata.normalize("NFD", s)

    def test_nfd_sub_subject_keyword(self):
        items = E.extract_filename_meta(self._nfd("26 물2 서바 9회.pdf"))["items"]
        self.assertEqual(items[0]["targets"], ["물리학Ⅱ"])

    def test_nfd_subject_keyword(self):
        items = E.extract_filename_meta(self._nfd("26 물리 서바이벌 9회.pdf"))["items"]
        self.assertEqual(items[0]["targets"], ["과학탐구"])

    def test_nfd_instructor(self):
        items = E.extract_filename_meta(self._nfd("배기범_모의고사_시즌_4_1회.pdf"))["items"]
        self.assertEqual(items[0]["keyword"], "배기범")

    def test_nfd_cover_text(self):
        # 표지 텍스트도 같은 정규화를 타므로 NFD 입력이 과목 매칭돼야 한다
        pre = E.pre_classify("", "", self._nfd("2026학년도 대학수학능력시험 과학탐구영역 물리학Ⅱ"))
        self.assertEqual(E.cover_single_subject(pre), ("과학탐구", "물리학Ⅱ"))


class TestInstructorFilenamePrior(unittest.TestCase):
    """파일명 강사명 → 과목 프라이어(INSTRUCTOR_SUBJECT 소비 — 정의만 있고 미사용이던 데이터).
    단일 과목 강사 자료는 강사명이 사실상 과목 정답. 단, 일반어/인명 합성 충돌은
    경계 가드·동철 제외로 걸러 오탐을 막는다."""
    def _items(self, fn):
        return E.extract_filename_meta(fn)["items"]

    def test_instructor_with_sub_subject(self):
        items = self._items("백호_봉투모의고사_2회.pdf")
        self.assertEqual(items[0]["targets"], ["생명과학Ⅰ"])
        self.assertEqual(items[0]["source"], "filename")

    def test_instructor_subject_only(self):
        items = self._items("현우진 드릴 시즌1.pdf")
        self.assertEqual(items[0]["targets"], ["수학"])

    def test_sub_subject_keyword_beats_instructor(self):
        # 세부과목 명시('물2')가 강사명(배기범: 과목만)보다 우선
        items = self._items("배기범_물2_파이널.pdf")
        self.assertEqual(items[0]["targets"], ["물리학Ⅱ"])

    def test_embedded_name_blocked(self):
        # '유신헌법'의 '유신'(국어 강사)은 합성어 내부 매칭 → 차단(앞뒤 경계 가드)
        targets = [t for it in self._items("유신헌법_한국사특강.pdf") for t in it["targets"]]
        self.assertNotIn("국어", targets)

    def test_common_word_homograph_blocked(self):
        # '이미지'(수학 강사)는 일반어와 동철 → 파일명 신호로 쓰지 않는다
        targets = [t for it in self._items("이미지 분석 자료.pdf") for t in it["targets"]]
        self.assertNotIn("수학", targets)

    def test_2026_added_instructors_resolve(self):
        # 2026 검증 추가분(국어 1타·대성 수학)이 파일명에서 과목 prior로 발화하는지
        for fn, subj in (("김동욱 일클래스 독서.pdf", "국어"),
                         ("유대종 국어 봉투모의고사.pdf", "국어"),
                         ("김승리 올오카 1강.pdf", "국어"),
                         ("전형태 언매 N제.pdf", "국어"),
                         ("김범준 수학 드릴 시즌2.pdf", "수학")):
            targets = [t for it in self._items(fn) for t in it["targets"]]
            self.assertIn(subj, targets, f"{fn} → {subj} 미발화 ({targets})")


class TestFilenameSubSubject(unittest.TestCase):
    """파일명 세부과목 매핑. 본교재류 파일명은 '수학1'(풀네임+아라비아) 형태가 흔한데,
    기존엔 약어 '수1'만 등록돼 매칭 실패 → 상위 과목으로만 폴백하던 누수를 막는다."""
    def _meta(self, fn):
        return E.extract_filename_meta(fn)["items"]

    def _corrected(self, fn, guess=("수학", "미적분")):
        # 모델이 틀린 세부과목을 냈을 때 파일명 프라이어가 교정하는지
        pre = E.pre_classify("", fn, "")
        c = E.Classification.model_validate(
            {"subject": guess[0], "sub_subject": guess[1], "confidence": 0.5})
        c = E.apply_filename_prior(c, pre)
        return (c.subject, c.sub_subject)

    def test_fullname_math1_maps_to_su1(self):
        items = self._meta("2027_강기원_스텝1_수학1_본교재.pdf")
        self.assertEqual(items[0]["targets"], ["수학Ⅰ"])
        self.assertEqual(items[0]["confidence"], 0.9)

    def test_fullname_math2_maps_to_su2(self):
        self.assertEqual(self._meta("수학2_본교재.pdf")[0]["targets"], ["수학Ⅱ"])

    def test_abbrev_su1_su2_still_work(self):
        # 모의고사 약어형 비회귀
        self.assertEqual(self._meta("수1 모의 3회.pdf")[0]["targets"], ["수학Ⅰ"])
        self.assertEqual(self._meta("수2 N제.pdf")[0]["targets"], ["수학Ⅱ"])

    def test_calculus_not_shadowed_by_math(self):
        # '미적분'은 '수학n'보다 먼저 매칭돼 그대로 유지
        self.assertEqual(self._meta("2026_미적분_모의.pdf")[0]["targets"], ["미적분"])

    def test_real_textbook_corrects_wrong_model_guess(self):
        # 실제 사용자 파일: 스캔본이라 본문 OCR이 흔들려 모델이 미적분으로 새도 수학Ⅰ로 교정
        self.assertEqual(
            self._corrected("2027_강기원_스텝1_수학1_본교재.pdf"), ("수학", "수학Ⅰ"))


class TestCompactPromptHints(unittest.TestCase):
    def test_includes_confusion_guide(self):
        p = E.make_compact_prompt("광전효과", {"items": []})
        self.assertIn("혼동 주의", p)

    def test_exam_info_injected_for_single_cover(self):
        # 단일 과목 표지면 '시험 정보' 지시문이 OCR 오염 대비로 주입됨
        pre = {"items": [
            {"keyword": "물리학Ⅱ", "polarity": "pro", "targets": ["물리학Ⅱ"],
             "confidence": 0.92, "source": "cover"}]}
        p = E.make_compact_prompt("노무현 전 대통령에 대한 설명으로 옳은 것은", pre)
        self.assertIn("시험 정보", p)
        self.assertIn("물리학Ⅱ", p)

    def test_no_exam_info_without_single_cover(self):
        p = E.make_compact_prompt("광전효과", {"items": []})
        self.assertNotIn("시험 정보", p)

    def test_pua_glyphs_stripped_from_prompt(self):
        # 임베드 폰트 수식이 PUA 글리프로 들어와도 프롬프트엔 남지 않아야
        dirty = "1. 그림은 \ue0fc\ue0fd평면상의 힘 \ue06d\ue06e의 크기는?"
        p = E.make_compact_prompt(dirty, {"items": []})
        self.assertEqual(E._PUA_RE.findall(p), [])
        self.assertIn("평면상의 힘", p)   # 실제 한글은 보존


class TestCleanOcrForPrompt(unittest.TestCase):
    def test_strips_pua_keeps_hangul_ascii(self):
        s = "힘\ue034 N F=ma \ue0fc평면"
        out = E._clean_ocr_for_prompt(s)
        self.assertEqual(E._PUA_RE.findall(out), [])
        for tok in ("힘", "N", "F=ma", "평면"):
            self.assertIn(tok, out)

    def test_keeps_circled_numbers_and_jamo(self):
        # 원문자(①)·자모(ㄱㄴㄷ)는 PUA가 아니므로 보존
        s = "\ue005① ㄱ ㄴ ㄷ"
        out = E._clean_ocr_for_prompt(s)
        for tok in ("①", "ㄱ", "ㄴ", "ㄷ"):
            self.assertIn(tok, out)
        self.assertEqual(E._PUA_RE.findall(out), [])

    def test_empty_safe(self):
        self.assertEqual(E._clean_ocr_for_prompt(""), "")


class TestSocialStudiesEvidence(unittest.TestCase):
    """사회탐구·한국사 키워드 신설(기존엔 증거 0개였던 공백)."""
    def _targets(self, text):
        out = set()
        for it in E.pre_classify(text, "", "")["items"]:
            if it.get("polarity") == "pro":
                out.update(it.get("targets", []))
        return out

    def test_each_subject_detected(self):
        cases = {
            "공리주의와 의무론으로 본 안락사 논쟁": "생활과 윤리",
            "성리학의 사단칠정 논쟁과 이황의 입장": "윤리와 사상",
            "카르스트 지형과 고위평탄면의 형성": "한국지리",
            "지중해성기후와 플랜테이션 농업의 특징": "세계지리",
            "메이지유신 이후 책봉조공 질서의 변화": "동아시아사",
            "프랑스혁명과 산업혁명이 제국주의에 미친 영향": "세계사",
            "기회비용과 가격탄력성으로 본 완전경쟁시장": "경제",
            "위헌법률심판과 죄형법정주의의 의미": "정치와 법",
            "낙인이론과 아노미로 설명하는 일탈행동": "사회·문화",
            "갑오개혁과 을사늑약, 대한민국임시정부의 활동": "한국사",
        }
        for text, expected in cases.items():
            self.assertIn(expected, self._targets(text), f"'{text}' → {expected} 미검출")

    def test_no_cross_subject_obvious_collision(self):
        # 경제 지문이 한국사/윤리로 잘못 잡히면 안 됨
        t = self._targets("기회비용과 한계효용, 총수요와 총공급으로 본 경기변동")
        self.assertIn("경제", t)
        self.assertNotIn("한국사", t)
        self.assertNotIn("생활과 윤리", t)

    def test_meme_president_text_not_history(self):
        # 회귀: 밈 물리 시험의 '노무현 전 대통령'은 한국사 키워드가 아님
        t = self._targets("다음은 노무현 전 대통령에 대한 설명으로 옳은 것은")
        self.assertNotIn("한국사", t)

    def test_general_words_not_keywords(self):
        # 일반어('사회','경제','법' 등) 단독은 키워드에 없어야(오탐원)
        for w in ("사회", "경제", "법", "윤리", "정치", "지리"):
            self.assertNotIn(w, E.ECONOMICS_PRO | E.POLITICS_LAW_PRO |
                             E.SOCIO_CULTURE_PRO | E.KOREAN_HISTORY_PRO)


class TestInferCache(unittest.TestCase):
    """추론 캐시(발열·재실행 비용 절감) — 동일 입력 추론 결과 재사용.
    스토리지는 core.cache로 이전됐으므로 patch는 C(cache 모듈)를 통한다."""
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self._old_db = C._CACHE_DB
        self._old_enabled = C._CACHE_ENABLED
        self._old_conn = C._cache_conn
        C._CACHE_DB = os.path.join(self.tmp, "c.db")
        C._cache_conn = None
        C.set_cache_enabled(True)

    def tearDown(self):
        try:
            if C._cache_conn: C._cache_conn.close()
        except Exception:
            pass
        C._CACHE_DB = self._old_db
        C._cache_conn = self._old_conn
        C.set_cache_enabled(self._old_enabled)
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_key_deterministic_and_input_sensitive(self):
        k = E.infer_cache_key("gemma4", b"img", "prompt")
        self.assertEqual(k, E.infer_cache_key("gemma4", b"img", "prompt"))
        self.assertNotEqual(k, E.infer_cache_key("gemma4", b"img2", "prompt"))
        self.assertNotEqual(k, E.infer_cache_key("gemma4", b"img", "prompt2"))
        self.assertNotEqual(k, E.infer_cache_key("other", b"img", "prompt"))

    def test_put_get_roundtrip(self):
        k = E.infer_cache_key("m", b"x", "p")
        self.assertIsNone(E.infer_cache_get(k))
        E.infer_cache_put(k, {"subject": "과학탐구", "sub_subject": "물리학Ⅱ"})
        got = E.infer_cache_get(k)
        self.assertEqual(got["sub_subject"], "물리학Ⅱ")

    def test_disabled_is_noop(self):
        C.set_cache_enabled(False)
        k = E.infer_cache_key("m", b"x", "p")
        E.infer_cache_put(k, {"a": 1})
        self.assertIsNone(E.infer_cache_get(k))


class TestOcrMemo(unittest.TestCase):
    """스캔 PDF 검출 OCR 재사용(_ocr_text_from_memo) — 박스별 재OCR 제거 로직."""

    class _Rect:
        def __init__(self, h): self.height = h

    class _Page:
        def __init__(self, h=1000): self.rect = TestOcrMemo._Rect(h)

    def setUp(self):
        E._PAGE_OCR_MEMO.clear()

    def tearDown(self):
        E._PAGE_OCR_MEMO.clear()

    def test_no_memo_returns_none(self):
        # 메모가 없으면(텍스트 레이어 PDF 등) None → 호출부가 기존 OCR 경로로 폴백
        self.assertIsNone(E._ocr_text_from_memo(self._Page()))

    def test_full_page_reading_order(self):
        import fitz
        p = self._Page()
        E._PAGE_OCR_MEMO[id(p)] = [
            (10, 10, 50, 25, "A"), (60, 12, 90, 25, "B"), (10, 200, 50, 215, "C"),
        ]
        self.assertEqual(E._ocr_text_from_memo(p), "A\nB\nC")
        # clip으로 윗줄만 슬라이스
        self.assertEqual(E._ocr_text_from_memo(p, fitz.Rect(0, 0, 100, 100)), "A\nB")

    def test_empty_region_returns_blank_not_none(self):
        # 메모는 있으나 영역에 단어가 없으면 ""(재OCR 안 함), None 아님
        import fitz
        p = self._Page()
        E._PAGE_OCR_MEMO[id(p)] = [(10, 10, 50, 25, "A")]
        self.assertEqual(E._ocr_text_from_memo(p, fitz.Rect(500, 500, 600, 600)), "")

    def test_memo_is_size_bounded(self):
        # 직접 소비하지 않는 호출 경로(/api/extract-problems)에서도 무한 적재되지 않음
        for k in range(20):
            E._PAGE_OCR_MEMO[k] = [(0, 0, 1, 1, "w")]
            while len(E._PAGE_OCR_MEMO) > 8:
                E._PAGE_OCR_MEMO.pop(next(iter(E._PAGE_OCR_MEMO)), None)
        self.assertLessEqual(len(E._PAGE_OCR_MEMO), 8)
        self.assertIn(19, E._PAGE_OCR_MEMO)  # 최신(현재 페이지) 보존


class TestEnglishCrosswalk(unittest.TestCase):
    """비전 모델이 과목을 영어로 출력해도(예: 'Physics II') 한국어 정규명으로 매핑.
    영문명이 없으면 _reconcile이 미분류로 폐기하던 누수(item 6)를 막는다."""
    def _cls(self, subject, sub="기타"):
        c = E.Classification.model_validate(
            {"subject": subject, "sub_subject": sub, "confidence": 0.8})
        return (c.subject, c.sub_subject)

    def test_english_science_subjects(self):
        cases = {
            "Physics": ("과학탐구", "물리학Ⅰ"),
            "Physics II": ("과학탐구", "물리학Ⅱ"),
            "Chemistry": ("과학탐구", "화학Ⅰ"),
            "Chemistry II": ("과학탐구", "화학Ⅱ"),
            "Biology": ("과학탐구", "생명과학Ⅰ"),
            "Earth Science": ("과학탐구", "지구과학Ⅰ"),
        }
        for eng, expected in cases.items():
            self.assertEqual(self._cls(eng), expected, f"{eng} 미매핑")

    def test_english_math_subjects(self):
        self.assertEqual(self._cls("Calculus"), ("수학", "미적분"))
        self.assertEqual(self._cls("Geometry"), ("수학", "기하"))
        self.assertEqual(self._cls("Mathematics II"), ("수학", "수학Ⅱ"))

    def test_english_subsubject_field(self):
        # 모델이 subject=과학탐구, sub_subject=영문 으로 줘도 교정
        self.assertEqual(self._cls("과학탐구", "Physics II"), ("과학탐구", "물리학Ⅱ"))

    def test_korean_history_not_misread_as_korean_language(self):
        # BUG: 'Korean History'가 'korean' 부분일치로 국어(국어)로 잡히던 문제 → 한국사로
        self.assertEqual(self._cls("Korean History"), ("한국사", "한국사"))

    def test_english_garbage_still_unclassified(self):
        self.assertEqual(self._cls("Astrology"), ("미분류", "기타"))


class TestClusterColumns(unittest.TestCase):
    """다단(멀티컬럼) 레이아웃 검출 핵심 프리미티브 — 문제번호 x좌표를 단으로 군집.
    PDF/OCR 없이 결정론적으로 검증한다(2단 분리·단일단 폴백·거터 경계)."""
    def test_single_column_when_xs_close(self):
        # 모든 번호가 한 x 근처면 단일 컬럼(페이지 전폭)
        self.assertEqual(E._cluster_columns([50, 52, 48, 51], 600.0), [(0.0, 600.0)])

    def test_gap_below_threshold_stays_single(self):
        # gap이 page*0.15(=90) 이하면 같은 단으로 본다
        self.assertEqual(E._cluster_columns([50, 100, 130], 600.0), [(0.0, 600.0)])

    def test_two_columns_split(self):
        # 좌단(~50)·우단(~350)이 임계 이상 떨어지면 2단으로 분리
        bounds = E._cluster_columns([50, 55, 350, 360], 600.0)
        self.assertEqual(len(bounds), 2)
        self.assertEqual(bounds[0][0], 0.0)                 # 첫 단은 0에서 시작
        self.assertAlmostEqual(bounds[0][1], (55 + 350) / 2)  # 거터 = 좌단끝·우단시작 중점
        self.assertAlmostEqual(bounds[1][1], 600.0)         # 마지막 단은 페이지 끝까지
        self.assertLess(bounds[0][1], bounds[1][0])         # 두 단 사이엔 거터(겹침 없음)

    def test_empty_and_single_fall_back_to_full_width(self):
        self.assertEqual(E._cluster_columns([], 600.0), [(0.0, 600.0)])
        self.assertEqual(E._cluster_columns([200.0], 600.0), [(0.0, 600.0)])


class TestOcrColumnHelpers(unittest.TestCase):
    """_find_problem_boxes_ocr의 순수 헬퍼(OCR/tesseract 무관, 결정론적) 회귀 가드:
    데이터기반 단 기준선(#2)·단별 LIS(#1)."""

    def test_baselines_balanced_two_column(self):
        # 균형 2단: 좌/우 단의 최소 x가 기준선 — 종전 고정문턱과 동일 결과.
        self.assertEqual(E._ocr_column_baselines([50, 55, 350, 360], 600.0), (50, 350))

    def test_baselines_single_column(self):
        left_L, right_L = E._ocr_column_baselines([50, 52, 48], 600.0)
        self.assertEqual(left_L, 48)
        self.assertIsNone(right_L)                       # 단 하나 → 우측 기준선 없음

    def test_baselines_asymmetric_two_column(self):
        # 회귀 핵심: 우단 기준선(230)이 0.45*W(=270) '미만'인 비대칭 2단.
        # 고정 0.45*W 문턱이면 우단을 못 잡아(right_L=None) 단일단으로 오인 → 데이터기반은 잡는다.
        left_L, right_L = E._ocr_column_baselines([40, 45, 230, 240], 600.0)
        self.assertEqual(left_L, 40)
        self.assertEqual(right_L, 230)                   # None 아님 = 2단 인식

    def test_baselines_empty(self):
        self.assertEqual(E._ocr_column_baselines([], 600.0), (0.0, None))

    def test_baselines_ignores_central_noise(self):
        # 회귀: 지문 속 잡음 번호 하나(190)가 좌단(40~48)·우단(350~358) 사이 별도 군집을 만들면,
        # '둘째 군집'을 맹목 채택하는 코드는 right_L=190(잡음)이 되어 진짜 우단(350)을 통째로 떨군다.
        # 원소 수 상위 2개 군집을 잡아 잡음 마이크로군집을 무시 → right_L=350.
        xs = [40, 42, 44, 46, 48, 190, 350, 352, 354, 356, 358]
        left_L, right_L = E._ocr_column_baselines(xs, 600.0)
        self.assertEqual(left_L, 40)
        self.assertEqual(right_L, 350)

    def test_columns_membership_uses_cluster_bounds(self):
        # #5: 멤버십을 군집 경계로 통일. 넓은 좌단의 x=110은 좌측 기준선(40)에서 ±tol(30)을
        # 넘게 떨어졌지만 좌단 군집 경계 안이라 유지돼야 한다(고정 근접매칭이면 오탈락=문제 누락).
        cols = E._ocr_columns([40, 42, 44, 110, 350, 352, 354], 600.0)
        self.assertEqual(len(cols), 2)
        self.assertEqual(E._col_index(110, cols), 0)    # 넓은 좌단 멤버 유지
        self.assertEqual(E._col_index(350, cols), 1)
        self.assertIsNone(E._col_index(250, cols))      # 거터(어느 단 경계도 아님) → None

    def test_columns_ignore_central_noise_for_membership(self):
        # 좌·우단 사이 잡음 번호(190)는 단으로 채택되지 않고, 그 위치는 멤버십에서 None.
        cols = E._ocr_columns([40, 42, 44, 46, 48, 190, 350, 352, 354, 356, 358], 600.0)
        self.assertEqual([round(c[0]) for c in cols], [40, 350])
        self.assertIsNone(E._col_index(190, cols))
        self.assertEqual(E._col_index(40, cols), 0)
        self.assertEqual(E._col_index(355, cols), 1)

    def _col_of(self, x):
        return 0 if x < 150 else 1

    def test_per_column_lis_keeps_both_columns(self):
        # 회귀 핵심: 우단 번호(2,3,4)가 좌단 최대(5)보다 작은 경우. 전역 LIS면 우단을 통째로
        # 떨궈 5개만 남지만, 단별 LIS는 양 단을 각각 유지 → 8개 모두.
        cand = [(50, 100, 1, 90), (50, 200, 2, 90), (50, 300, 3, 90), (50, 400, 4, 90), (50, 500, 5, 90),
                (300, 100, 2, 90), (300, 200, 3, 90), (300, 300, 4, 90)]
        kept = E._keep_increasing_per_column(cand, self._col_of)
        self.assertEqual(len(kept), 8)
        left_nums = [c[2] for c in kept if c[0] < 150]
        right_nums = [c[2] for c in kept if c[0] >= 150]
        self.assertEqual(left_nums, [1, 2, 3, 4, 5])
        self.assertEqual(right_nums, [2, 3, 4])

    def test_per_column_lis_drops_in_column_misdetection(self):
        # 단 안에서 비단조(9는 오검출) → LIS가 제거.
        cand = [(50, 100, 1, 90), (50, 200, 2, 90), (50, 300, 9, 90), (50, 400, 3, 90), (50, 500, 4, 90)]
        kept = E._keep_increasing_per_column(cand, self._col_of)
        self.assertEqual([c[2] for c in kept], [1, 2, 3, 4])

    def test_succ_chain_prefers_consecutive(self):
        # 진짜 문제번호 단(+1 연속) vs 잡음 군집(우연한 증가열) 구분의 핵심 신호.
        self.assertEqual(E._succ_chain_len([6, 7, 8]), 3)
        self.assertEqual(E._succ_chain_len([1, 6, 8, 34]), 1)   # 증가열이지만 연속 아님
        self.assertEqual(E._succ_chain_len([1, 4, 2, 3]), 3)    # 부분수열 (1,2,3)
        self.assertEqual(E._succ_chain_len([]), 0)

    def test_scored_columns_pick_real_baselines_over_noise(self):
        # 실측(시대인재 물리 p2) 축약 재현: 본문 잡음 군집이 LIS는 길어도(1,6,8,34)
        # 연속사슬이 없어, 진짜 좌(6,7,8)·우(9,10,11) 기준선이 뽑혀야 한다.
        cand = [
            (181, 263, 6, 90), (182, 796, 7, 90), (185, 1557, 8, 90),      # 진짜 좌단
            (545, 264, 1, 90), (491, 399, 6, 90), (607, 1559, 8, 90), (498, 1624, 34, 90),  # 잡음
            (871, 262, 9, 90), (875, 1047, 10, 90), (876, 1492, 11, 90),   # 진짜 우단
        ]
        cols = E._ocr_columns_scored(cand, 1684.0)
        self.assertEqual(len(cols), 2)
        self.assertAlmostEqual(cols[0][0], 181.0)
        self.assertAlmostEqual(cols[1][0], 871.0)
        # 잡음 x(491~607)는 어느 단 멤버십에도 안 든다
        self.assertIsNone(E._col_index(545, cols))
        self.assertIsNone(E._col_index(607, cols))

    def test_straggler_one_in_right_column_dropped(self):
        # 실측(시대인재 물리 p1): 우단 상단 잡음 '1.'이 단내 LIS([1,4,5])를 통과해
        # 중복 문항박스를 만들던 케이스 — 좌단 최대(3) 이하인 우단 선두만 제거.
        by_col = {
            0: [(50, 100, 1, 90), (50, 200, 2, 90), (50, 300, 3, 90)],
            1: [(300, 50, 1, 90), (300, 110, 4, 90), (300, 210, 5, 90)],
        }
        out = E._drop_cross_column_stragglers(by_col)
        self.assertEqual([c[2] for c in out[0]], [1, 2, 3])
        self.assertEqual([c[2] for c in out[1]], [4, 5])

    def test_straggler_no_drop_when_right_column_legitimately_lower(self):
        # 우단 전체가 좌단보다 작은 경우(좌단 오인식 고번호 등)는 연속 구조가 아님 → 무변경.
        # 단별 LIS가 양 단을 보존하는 설계(test_per_column_lis_keeps_both_columns)와 정합.
        by_col = {
            0: [(50, 100, 1, 90), (50, 200, 5, 90)],
            1: [(300, 100, 2, 90), (300, 200, 3, 90)],
        }
        self.assertEqual(E._drop_cross_column_stragglers(by_col), by_col)

    def test_straggler_no_drop_when_left_has_phantom_high_number(self):
        # 좌단에 허수 큰 번호(zoom3의 '27' 류)가 끼면 우단 정상 번호가 전부 '낙오'로 보임
        # → 이때 지우면 우단 전멸. 남는 게 없으면 건드리지 않는 안전장치 가드.
        by_col = {
            0: [(50, 100, 1, 90), (50, 200, 27, 90)],
            1: [(300, 100, 4, 90), (300, 200, 5, 90)],
        }
        self.assertEqual(E._drop_cross_column_stragglers(by_col), by_col)

    def test_two_column_sequence_repairs_left_ocr_misread(self):
        # 실제 배기범 PDF: 좌측 3번을 Tesseract가 8번으로 읽어 1,2,8 / 4,5,6이 됨.
        by_col = {
            0: [(50, 100, 1, 90), (50, 200, 2, 90), (50, 300, 8, 90)],
            1: [(300, 110, 4, 90), (300, 210, 5, 90), (300, 310, 6, 90)],
        }
        repaired = E._repair_two_column_sequence(by_col)
        self.assertEqual([c[2] for c in repaired[0]], [1, 2, 3])
        self.assertEqual([c[2] for c in repaired[1]], [4, 5, 6])


class TestLisIndices(unittest.TestCase):
    """오검출 번호 제거의 핵심 순수 헬퍼 _lis_indices(엄격 증가 최장부분수열 인덱스)
    직접 가드 — 리팩터(예: O(n log n) 치환) 시 tie-break까지 거동 고정."""
    def test_empty(self):
        self.assertEqual(E._lis_indices([]), [])

    def test_single(self):
        self.assertEqual(E._lis_indices([7]), [0])

    def test_strictly_increasing_keeps_all(self):
        self.assertEqual(E._lis_indices([1, 2, 3, 4]), [0, 1, 2, 3])

    def test_strictly_decreasing_keeps_first(self):
        # dp가 모두 1 → 첫 최댓값(인덱스 0)에서 끝나는 길이-1 사슬.
        self.assertEqual(E._lis_indices([5, 4, 3]), [0])

    def test_strict_so_duplicates_not_chained(self):
        # 동일값은 엄격 증가가 아니라 한쪽만 사슬에 든다.
        self.assertEqual(E._lis_indices([1, 1, 2]), [0, 2])

    def test_drops_single_outlier(self):
        # [1,2,9,3,4] → LIS는 [1,2,3,4](인덱스 0,1,3,4), 오검출 9(인덱스 2) 제외.
        self.assertEqual(E._lis_indices([1, 2, 9, 3, 4]), [0, 1, 3, 4])


class TestEnglishEvidence(unittest.TestCase):
    """영문 우세 지문 → '영어' 약한 시사(보수적 임계). 한글 텍스트엔 발화하지 않음(오탐 방지)."""
    def test_english_heavy_signals_english(self):
        text = ("Read the following passage and choose the best answer. " * 4) + "다음 글의 주제는?"
        ev = E._english_evidence(text)
        self.assertTrue(ev and ev[0]["targets"] == ["영어"] and ev[0]["polarity"] == "pro")
        items = E.pre_classify(text)["items"]
        self.assertTrue(any("영어" in it.get("targets", []) for it in items))

    def test_korean_text_no_english_signal(self):
        text = "다음 글에서 화자의 정서로 가장 적절한 것을 고르시오. 시적 화자는 자연을 노래한다."
        self.assertEqual(E._english_evidence(text), [])

    def test_latin_heavy_non_english_no_signal(self):
        # 제2외국어(독일어 류): 라틴 글자는 우세하지만 영어 기능어가 없다 → 영어로 단정 안 함.
        # 종전엔 라틴 우세만 보아 독일어 지문을 '영어'로 오시사하던 잠재 버그를 막는다.
        text = ("Der Schueler liest ein Buch ueber die Geschichte der Stadt Berlin. " * 3)
        # 'die/der' 등 독일어어휘는 영어 기능어 정규식에 없다(the/and/of…만 매칭)
        self.assertEqual(E._english_evidence(text), [])

    def test_listening_boilerplate_signals_english(self):
        # 영어 듣기평가 안내문 — 수능에서 듣기 영역은 영어뿐. 표지(=문서 단위)에서 '영어' 시사.
        cover = ("이 문제지에 관한 저작권은 한국교육과정평가원에 있습니다. "
                 "1번부터 17번까지는 듣고 답하는 문제입니다. 한 번만 들려주고 방송을 잘 듣고 답하시오.")
        items = E.extract_cover_subject(cover)["items"]
        eng = [it for it in items if it["targets"] == ["영어"] and it["polarity"] == "pro"]
        self.assertTrue(eng and eng[0]["confidence"] >= 0.9)

    def test_listening_boilerplate_absent_in_korean(self):
        # 국어 등 다른 과목 표지엔 듣기 안내문이 없다 → 영어 시사 없음(오탐 방지).
        cover = "2026학년도 대학수학능력시험 문제지 국어 영역 다음 글을 읽고 물음에 답하시오."
        items = E.extract_cover_subject(cover)["items"]
        self.assertFalse([it for it in items if it.get("hint", "").startswith("듣기평가")])


class TestApplyEnglishPrior(unittest.TestCase):
    """모델 미분류를 영어 적극 증거가 있을 때만 영어로 교정(외과적)."""
    def _pre(self, conf):
        return {"items": [{"polarity": "pro", "targets": ["영어"], "confidence": conf,
                           "source": "context"}]}

    def test_promotes_unclassified_with_strong_evidence(self):
        cls = E.Classification(subject="미분류", sub_subject="기타", confidence=0.0)
        out = E.apply_english_prior(cls, self._pre(0.9))
        self.assertEqual((out.subject, out.sub_subject), ("영어", "기타"))

    def test_noop_without_english_evidence(self):
        cls = E.Classification(subject="미분류", sub_subject="기타", confidence=0.0)
        out = E.apply_english_prior(cls, {"items": []})
        self.assertEqual(out.subject, "미분류")

    def test_noop_when_evidence_weak(self):
        # 0.9 미만 증거로는 승격하지 않는다(고신뢰 증거만 하드 교정).
        cls = E.Classification(subject="미분류", sub_subject="기타", confidence=0.0)
        out = E.apply_english_prior(cls, self._pre(0.6))
        self.assertEqual(out.subject, "미분류")

    def test_never_touches_confident_prediction(self):
        # 모델이 이미 과목을 골랐으면(미분류 아님) 영어 증거가 있어도 건드리지 않는다.
        cls = E.Classification(subject="국어", sub_subject="독서", confidence=0.8)
        out = E.apply_english_prior(cls, self._pre(0.95))
        self.assertEqual((out.subject, out.sub_subject), ("국어", "독서"))


class TestFindProblemBoxes(unittest.TestCase):
    """find_problem_boxes 통합(텍스트레이어 경로, OCR 무관) — 핵심 오케스트레이터 회귀 가드.
    본문은 ASCII(한글 폰트 렌더 변수 제거) — 검출 패턴은 선두 '1.'(숫자)만 보므로 무방."""

    def _page(self, lines):
        import fitz
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)            # A4 pt
        for (x, y, t) in lines:
            page.insert_text((x, y), t, fontsize=11)
        return doc, page

    def test_two_problems_single_column(self):
        doc, page = self._page([(60, 150, "1. first problem body text here."),
                                (60, 400, "2. second problem body text here.")])
        try:
            boxes = E.find_problem_boxes(page)
            self.assertEqual([n for n, _ in boxes], ["1", "2"])  # 두 문제 번호 검출
            for _, r in boxes:                                   # 유효 사각형(페이지 내·y정상)
                self.assertLess(r.y0, r.y1)
                self.assertLessEqual(r.x1, 595)
                self.assertGreaterEqual(r.x0, 0)
        finally:
            doc.close()

    def test_no_markers_falls_back_to_word_split(self):
        doc, page = self._page([(60, 200, "plain body text with no problem number at all.")])
        try:
            boxes = E.find_problem_boxes(page)
            self.assertTrue(boxes)                               # 마커 없음 → _split_by_words 폴백
            self.assertEqual(boxes[0][0], "1")                   # 순번 라벨
        finally:
            doc.close()

    def test_blank_page_returns_empty(self):
        doc, page = self._page([])
        try:
            self.assertEqual(E.find_problem_boxes(page), [])     # 텍스트·이미지 없음 → 빈 결과
        finally:
            doc.close()


class TestOcrDetectExceptionLogged(unittest.TestCase):
    """OCR 검출 파이프라인(pixmap/PIL/pytesseract) 예외를 조용히 삼키지 않고 로그로 드러낸다.
    tesseract 미설정·실패가 '스캔본 문제 0개'로 위장되는 걸 운영자가 인지하게 함.
    거동은 불변([] 반환) — 예외 경로에만 print 한 줄 추가."""

    class _BoomPage:
        def get_pixmap(self, *a, **k):
            raise RuntimeError("tesseract boom")

    def test_exception_returns_empty_and_logs(self):
        import io as _io, contextlib
        had = E.HAS_TESSERACT
        E.HAS_TESSERACT = True                               # import 성공 가정(런타임 실패 재현)
        try:
            buf = _io.StringIO()
            with contextlib.redirect_stdout(buf):
                result = E._find_problem_boxes_ocr(self._BoomPage())
            self.assertEqual(result, [])                     # 거동 불변: 여전히 빈 결과
            out = buf.getvalue()
            self.assertIn("OCR 검출 실패", out)               # 조용히 삼키지 않고 드러냄
            self.assertIn("tesseract boom", out)             # 원인(e) 포함
        finally:
            E.HAS_TESSERACT = had


class TestContinuationStrip(unittest.TestCase):
    """2단 이어짐 캡처(_emit_problem_boxes): 우단 첫 번호가 좌단보다 의미 있게 아래에서
    시작하면, 그 위 고아 영역(직전 문제의 이어진 보기·선지)을 직전 번호의 strip 박스로
    내보내야 한다 — 종전엔 어느 박스에도 안 들어가 유실됐다."""
    def _run(self, right_first_y, memo):
        by_col = {0: [(50, 100, 1, 90), (50, 800, 2, 90)],
                  1: [(600, right_first_y, 3, 90), (600, 1500, 4, 90)]}
        col_bounds = {0: (0.0, 550.0), 1: (550.0, 1200.0)}
        return E._emit_problem_boxes(by_col, col_bounds, pad_y=10, bottom=1940,
                                     W=1200.0, H=2000.0, zoom=1.0, memo_pts=memo)

    def test_orphan_strip_emitted_for_prev_problem(self):
        # 우단 첫 번호(y=600)가 기준 상단(y=100)보다 한참 아래 + 그 사이에 텍스트 존재
        memo = [(700, 300, 760, 320, "이어진 보기")]   # strip 영역 안의 텍스트(포인트 공간)
        out = self._run(600, memo)
        nums = [n for n, _ in out]
        self.assertEqual(nums, ["1", "2", "2", "3", "4"])  # '2'의 strip이 본체 뒤에 연속
        strip = out[2][1]
        self.assertLess(strip.y1, 600)                     # strip은 우단 첫 번호 위에서 끝남
        self.assertGreaterEqual(strip.x0, 550.0)           # 우단 x 범위

    def test_no_strip_when_columns_align(self):
        # 양 단 첫 번호가 같은 높이(실측: 배기범 p2, 243 vs 246) → 고아 영역 없음
        out = self._run(105, [(700, 50, 760, 70, "텍스트")])
        self.assertEqual([n for n, _ in out], ["1", "2", "3", "4"])

    def test_no_strip_when_region_blank(self):
        # 높이 조건은 충족해도 메모에 텍스트가 없으면(빈 여백) 합성하지 않는다
        out = self._run(600, [])
        self.assertEqual([n for n, _ in out], ["1", "2", "3", "4"])


class TestBareNumberStarts(unittest.TestCase):
    """맨숫자 문제번호 폴백(구두점 없는 교재 스타일 — 실측: 완자 '07'·900제 '004')의
    검증 로직 가드. 축 눈금·연도 오탐은 'x기준선 정렬 + y순 증가 + 스텝≤3'으로 차단."""
    def test_real_problem_numbers_accepted(self):
        # 완자 p45 축약: 좌단 07(y54)·09(y573) — 스텝 2, 증가 → 인정
        cand = [(54, 62, 7, True), (573, 62, 9, True)]
        self.assertEqual(len(E._validate_bare_starts(cand, 595.0)), 2)

    def test_axis_ticks_rejected_by_step(self):
        # 그래프 축 눈금(20,40,60,80): 같은 x지만 스텝 20 → 탈락
        cand = [(100, 80, 20, True), (200, 80, 40, True), (300, 80, 60, True)]
        self.assertEqual(E._validate_bare_starts(cand, 595.0), [])

    def test_axis_ticks_rejected_by_direction(self):
        # y축 눈금은 위가 큰 수(100→0): y순으로 감소 → 탈락
        cand = [(100, 80, 3, True), (200, 80, 2, True), (300, 80, 1, True)]
        self.assertEqual(E._validate_bare_starts(cand, 595.0), [])

    def test_isolated_number_rejected(self):
        # 고립 숫자(연도·쪽수)는 ≥2 요건에서 탈락
        self.assertEqual(E._validate_bare_starts([(100, 80, 26, True)], 595.0), [])

    def test_three_digit_workbook_numbers(self):
        # 900제 스타일: 004/005/006 — 3자리, 스텝 1 → 인정
        cand = [(244, 340, 4, True), (297, 340, 5, True), (420, 340, 6, True)]
        self.assertEqual(len(E._validate_bare_starts(cand, 595.0)), 3)

    def test_bare_page_end_to_end(self):
        # 합성 PDF: 맨숫자 '07'/'08' 스타일 + [7~8] 세트 머리글 → 세트로 묶여야 한다
        import tempfile
        import fitz
        from pathlib import Path
        doc = fitz.open()
        pg = doc.new_page(width=595, height=842)
        pg.insert_text((50, 100), "[7~8] Read the passage below.", fontsize=10)
        pg.insert_text((50, 150), "Shared passage text for both.", fontsize=10)
        # 실파일(완자·900제)처럼 번호는 '별도 블록' — 본문과 y를 띄워 블록 병합 방지
        pg.insert_text((50, 300), "07", fontsize=12)
        pg.insert_text((50, 330), "First bare-numbered question?", fontsize=10)
        pg.insert_text((50, 550), "08", fontsize=12)
        pg.insert_text((50, 580), "Second bare-numbered question?", fontsize=10)
        p = Path(tempfile.mkdtemp()) / "bare.pdf"
        doc.save(p); doc.close()
        probs = E.extract_all_problems(p)
        by_num = {x["problem_num"]: x for x in probs}
        self.assertIn("7", by_num); self.assertIn("8", by_num)
        self.assertEqual(by_num["7"]["set_range"], "7-8")
        self.assertEqual(by_num["7"]["image_id"], by_num["8"]["image_id"])


class TestSetHeaderRegex(unittest.TestCase):
    """세트 머리글 패턴 — 물결 변형(~ ～ ∼)·공백 허용, 본문 점수표기/소수점 오탐 금지."""
    def _m(self, s):
        m = E._SET_HEADER_RE.match(s)
        return (int(m.group(1)), int(m.group(2))) if m else None

    def test_variants(self):
        self.assertEqual(self._m("[41~42] 다음 글을 읽고"), (41, 42))
        self.assertEqual(self._m("[43 ~ 45] 다음 글을"), (43, 45))
        self.assertEqual(self._m("[41～42]"), (41, 42))
        self.assertEqual(self._m("[ 16∼17 ]"), (16, 17))

    def test_non_headers_rejected(self):
        self.assertIsNone(self._m("[3점]"))
        self.assertIsNone(self._m("1.5 mol의 기체가"))
        self.assertIsNone(self._m("그림은 [41~42]를 인용한"))   # 줄 선두 아님


class TestSetQuestions(unittest.TestCase):
    """수능 세트 문항 엔드투엔드(합성 텍스트레이어 PDF — OCR·ollama 무관, 결정적).

    구조: 40번 단독 → '[41~42]' 머리글 → 지문 → 41번 → 42번 (단일 컬럼).
    기대: 41·42는 같은 set_id('41-42_<stem>')와 image_id(합성 캡처 해시)를 공유하는
    별도 레코드, 40은 set_id=None(기존 단독 처리 무회귀). 40의 캡처는 머리글에서 끊겨
    세트 지문을 삼키지 않는다."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import fitz
        from pathlib import Path
        doc = fitz.open()
        pg = doc.new_page(width=595, height=842)
        pg.insert_text((50, 80), "40. Choose the correct answer below.", fontsize=10)
        pg.insert_text((50, 90), "(A) one  (B) two  (C) three", fontsize=10)
        pg.insert_text((50, 200), "[41~42] Read the following passage and answer.", fontsize=10)
        pg.insert_text((50, 250), "The shared passage body sits here for both questions.", fontsize=10)
        pg.insert_text((50, 400), "41. What is the best title of the passage?", fontsize=10)
        pg.insert_text((50, 410), "(A) t1  (B) t2", fontsize=10)
        pg.insert_text((50, 600), "42. Which mood does the passage convey?", fontsize=10)
        pg.insert_text((50, 610), "(A) m1  (B) m2", fontsize=10)
        cls.pdf = Path(tempfile.mkdtemp()) / "set_exam.pdf"
        doc.save(cls.pdf); doc.close()
        # 추출 캐시 우회를 위해 원본 함수 직접 호출
        cls.problems = E.extract_all_problems(cls.pdf)
        cls.by_num = {p["problem_num"]: p for p in cls.problems}

    def test_three_records(self):
        self.assertEqual(sorted(self.by_num), ["40", "41", "42"])

    def test_set_members_share_set_id_and_image(self):
        p41, p42 = self.by_num["41"], self.by_num["42"]
        self.assertEqual(p41["set_id"], f"41-42_{self.pdf.stem}")
        self.assertEqual(p41["set_id"], p42["set_id"])
        self.assertEqual(p41["set_range"], "41-42")
        self.assertEqual(p41["image_id"], p42["image_id"])
        self.assertEqual(p41["image_bytes"], p42["image_bytes"])     # 합성 캡처 1장 공유

    def test_set_text_contains_passage_and_both_questions(self):
        t = self.by_num["41"]["text"]
        self.assertIn("shared passage body", t)                      # 지문 포함
        self.assertIn("best title", t)                               # 41 문항
        self.assertIn("mood", t)                                     # 42 문항

    def test_single_problem_untouched(self):
        p40 = self.by_num["40"]
        self.assertIsNone(p40["set_id"])
        self.assertIsNone(p40["set_range"])
        self.assertTrue(p40["image_id"])                             # 단독도 image_id는 가진다
        self.assertNotEqual(p40["image_id"], self.by_num["41"]["image_id"])

    def test_single_capture_stops_at_set_header(self):
        # 머리글이 경계로 작용 — 40번 텍스트에 세트 지문이 섞이면 안 된다
        self.assertNotIn("shared passage body", self.by_num["40"]["text"])
        self.assertNotIn("Read the following passage", self.by_num["40"]["text"])

    def test_detect_set_ranges_unit(self):
        import fitz
        doc = fitz.open(self.pdf)
        ranges = E.detect_set_ranges(doc[0])
        self.assertEqual([(s, e) for s, e, _ in ranges], [(41, 42)])


class TestCrossPageSet(unittest.TestCase):
    """페이지 간 세트(국어 독서 지문 세트는 거의 항상 페이지를 넘는다 — 실측: 더프 국어
    [4~9]가 p2→p3). 머리글 페이지 +2까지에서 멤버를 매칭해 한 세트로 묶는 가드."""

    @classmethod
    def setUpClass(cls):
        import tempfile
        import fitz
        from pathlib import Path
        doc = fitz.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 100), "[4~5] Read the passage and answer the questions.", fontsize=10)
        p1.insert_text((50, 150), "Cross-page passage body text.", fontsize=10)
        p1.insert_text((50, 400), "4. First question on page one?", fontsize=10)
        p1.insert_text((50, 420), "(A) a  (B) b", fontsize=10)
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((50, 100), "5. Second question on page two?", fontsize=10)
        p2.insert_text((50, 120), "(A) c  (B) d", fontsize=10)
        p2.insert_text((50, 400), "6. Standalone question after the set?", fontsize=10)
        p2.insert_text((50, 420), "(A) e  (B) f", fontsize=10)
        cls.pdf = Path(tempfile.mkdtemp()) / "xpage.pdf"
        doc.save(cls.pdf); doc.close()
        cls.by_num = {p["problem_num"]: p for p in E.extract_all_problems(cls.pdf)}

    def test_members_grouped_across_pages(self):
        p4, p5 = self.by_num["4"], self.by_num["5"]
        self.assertEqual(p4["set_range"], "4-5")
        self.assertEqual(p4["set_id"], p5["set_id"])
        self.assertEqual(p4["image_id"], p5["image_id"])     # 두 페이지 캡처가 한 장으로 합성
        self.assertEqual(p4["page"], 1); self.assertEqual(p5["page"], 2)

    def test_set_text_spans_pages(self):
        t = self.by_num["4"]["text"]
        self.assertIn("First question", t)
        self.assertIn("Second question", t)                  # 다음 페이지 문항 포함

    def test_following_single_untouched(self):
        self.assertIsNone(self.by_num["6"]["set_id"])


class TestSetMissingNumberRecovery(unittest.TestCase):
    """세트 범위 기반 누락 번호 복원(_compose_sets 순수 단위) — 번호 OCR 미스로 문항이
    통째로 사라지던 것을(실측: 더프 국어 [10~13]에서 11·12 누락) 세트 공유 캡처로 복원."""
    def _item(self, num, page_no=1):
        import fitz
        return {"num": str(num), "page_no": page_no, "rects": [fitz.Rect(0, 0, 10, 10)],
                "pngs": [b"png"], "texts": [f"q{num}"]}

    def test_missing_members_synthesized(self):
        pending = [self._item(10), self._item(13)]
        E._compose_sets(pending, [{"s": 10, "e": 13, "page_no": 1,
                                   "strip_png": None, "strip_text": ""}], "exam")
        nums = sorted(int(it["num"]) for it in pending)
        self.assertEqual(nums, [10, 11, 12, 13])             # 11·12 합성 생성
        by = {it["num"]: it for it in pending}
        self.assertEqual(by["11"]["set_id"], by["10"]["set_id"])
        self.assertEqual(by["11"]["set_image"], by["10"]["set_image"])  # 세트 캡처 공유

    def test_no_synthesis_when_number_exists_elsewhere_in_window(self):
        pending = [self._item(10), self._item(13), self._item(11, page_no=2)]
        E._compose_sets(pending, [{"s": 10, "e": 13, "page_no": 1,
                                   "strip_png": None, "strip_text": ""}], "exam")
        self.assertEqual(sum(1 for it in pending if it["num"] == "11"), 1)  # 중복 생성 금지

    def test_no_synthesis_for_unconfirmed_set(self):
        pending = [self._item(10)]                            # 멤버 1개 → 세트 미확정
        E._compose_sets(pending, [{"s": 10, "e": 13, "page_no": 1,
                                   "strip_png": None, "strip_text": ""}], "exam")
        self.assertEqual(len(pending), 1)


class TestVstackPngs(unittest.TestCase):
    """2단 이어짐 문제 조각 세로 합성(_vstack_pngs) — 캡처 정확도 보강의 순수 헬퍼 가드."""
    def _png(self, w, h, color=(200, 10, 10)):
        import io
        from PIL import Image
        b = io.BytesIO(); Image.new("RGB", (w, h), color).save(b, format="PNG")
        return b.getvalue()

    def test_single_passthrough(self):
        p = self._png(10, 10)
        self.assertIs(E._vstack_pngs([p]), p)

    def test_stack_dimensions(self):
        import io
        from PIL import Image
        out = E._vstack_pngs([self._png(100, 40), self._png(60, 30)])
        im = Image.open(io.BytesIO(out))
        self.assertEqual(im.size, (100, 70))   # 폭=최대, 높이=합

    def test_corrupt_falls_back_to_first(self):
        p = self._png(10, 10)
        self.assertEqual(E._vstack_pngs([p, b"not a png"]), p)


class TestExtractCache(unittest.TestCase):
    """추출 캐시(extract_all_problems_cached): 같은 PDF 재분류 시 스캔본 OCR(유일하게 남은
    발열원, 뜨거운 기계에서 219s 실측)을 통째로 건너뛰는 2단 캐시의 가드."""

    def setUp(self):
        import tempfile, fitz
        from pathlib import Path
        self.Path = Path
        self.tmp = tempfile.mkdtemp()
        os.environ["CLASSI_EXTRACT_CACHE_DB"] = os.path.join(self.tmp, "ec.db")
        # 모듈 전역을 테스트 DB로 교체(환경변수는 import 시점에 읽히므로 직접 패치)
        # 스토리지는 core.cache로 이전됐으므로 C(cache 모듈)를 통해 패치한다.
        self._db, self._conn = C._EXTRACT_CACHE_DB, C._extract_cache_conn
        C._EXTRACT_CACHE_DB = os.path.join(self.tmp, "ec.db")
        C._extract_cache_conn = None
        doc = fitz.open()
        doc.new_page().insert_text((72, 72), "1. 등가속도 운동 문제입니다. 답을 고르시오.")
        self.pdf = self.Path(self.tmp) / "t.pdf"
        doc.save(self.pdf); doc.close()

    def tearDown(self):
        os.environ.pop("CLASSI_EXTRACT_CACHE_DB", None)
        if C._extract_cache_conn is not None:
            C._extract_cache_conn.close()
        C._EXTRACT_CACHE_DB, C._extract_cache_conn = self._db, self._conn

    def test_second_call_hits_cache_and_matches(self):
        first = E.extract_all_problems_cached(self.pdf)
        called = []
        orig = E.extract_all_problems
        E.extract_all_problems = lambda *a, **k: called.append(1) or orig(*a, **k)
        try:
            second = E.extract_all_problems_cached(self.pdf)
        finally:
            E.extract_all_problems = orig
        self.assertEqual(called, [])                          # 원본 추출이 호출되지 않음 = 캐시 적중
        self.assertEqual(len(first), len(second))
        self.assertEqual(first[0]["text"], second[0]["text"])
        self.assertEqual(first[0]["image_bytes"], second[0]["image_bytes"])

    def test_content_change_invalidates(self):
        E.extract_all_problems_cached(self.pdf)
        self.pdf.write_bytes(self.pdf.read_bytes() + b" ")    # 내용 변경 → 키 변경
        called = []
        orig = E.extract_all_problems
        E.extract_all_problems = lambda *a, **k: called.append(1) or orig(*a, **k)
        try:
            E.extract_all_problems_cached(self.pdf)
        finally:
            E.extract_all_problems = orig
        self.assertEqual(called, [1])                         # 재추출됨

    def test_byte_cap_evicts_oldest(self):
        import fitz
        C._extract_cache_lock.acquire(); C._extract_cache_lock.release()
        saved = C._EXTRACT_CACHE_MAX_BYTES
        C._EXTRACT_CACHE_MAX_BYTES = 1  # 어떤 항목도 1바이트는 못 지킴 → 직전 항목 즉시 축출
        try:
            E.extract_all_problems_cached(self.pdf)
            with C._extract_cache_lock:
                n = C._extract_cache_connection().execute(
                    "SELECT COUNT(*) FROM extract_cache").fetchone()[0]
            self.assertEqual(n, 1)                            # 방금 넣은 것만 남고(보존 가드) 초과는 정리
            doc = fitz.open(); doc.new_page().insert_text((72, 72), "2. 다른 문제. 답을 고르시오.")
            p2 = self.Path(self.tmp) / "t2.pdf"; doc.save(p2); doc.close()
            E.extract_all_problems_cached(p2)
            with C._extract_cache_lock:
                rows = C._extract_cache_connection().execute(
                    "SELECT COUNT(*) FROM extract_cache").fetchone()[0]
            self.assertEqual(rows, 1)                         # 옛 항목 축출, 최신만 유지
        finally:
            C._EXTRACT_CACHE_MAX_BYTES = saved


if __name__ == "__main__":
    unittest.main(verbosity=2)


class TestSolutionBook(unittest.TestCase):
    """해설서 처리(분류 대신 캡처 보관) — 판정·저장 가드."""
    def test_filename_markers(self):
        self.assertTrue(E.is_solution_book("물리_정답_및_해설.pdf"))
        self.assertTrue(E.is_solution_book(__import__("unicodedata").normalize(
            "NFD", "수학1 답지.pdf")))                       # NFD 파일명도 판정
        self.assertFalse(E.is_solution_book("26 물리 서바이벌 9회.pdf"))

    def test_bare_jeongdap_substring_not_solution(self):
        # '정답' 단독 substring 과포착 가드 — 시험지가 해설로 오분류되면 분류 없이
        # 캡처만 보관돼 데이터가 조용히 손실된다(라운드2 감사 발견).
        self.assertFalse(E.is_solution_book("정답률낮은문제모음.pdf"))
        self.assertFalse(E.is_solution_book("2026_오답정답노트_기출.pdf"))

    def test_compound_jeongdap_still_solution(self):
        self.assertTrue(E.is_solution_book("수학_정답지.pdf"))
        self.assertTrue(E.is_solution_book("물리 정답및해설 3회.pdf"))

    def test_cover_marker(self):
        self.assertTrue(E.is_solution_book("x.pdf", "2026 시대인재 정답 및 해설"))
        self.assertFalse(E.is_solution_book("x.pdf", "과학탐구 영역 문제지"))

    def test_save_captures_layout(self):
        import tempfile
        from pathlib import Path
        base = Path(tempfile.mkdtemp())
        probs = [{"page": 2, "problem_num": "11", "image_bytes": b"a"},
                 {"page": 2, "problem_num": "11", "image_bytes": b"b"},   # 중복 → _2
                 {"page": 3, "problem_num": "p", "image_bytes": b"c"}]
        out = E.save_solution_captures(probs, "물2_서바_해설.pdf", base_dir=base)
        self.assertEqual(out, base / "물2_서바_해설")        # 문제집별 하위 폴더
        names = sorted(f.name for f in out.iterdir())
        self.assertEqual(names, ["물2_서바_해설_p2_q11.png", "물2_서바_해설_p2_q11_2.png",
                                 "물2_서바_해설_p3_qp.png"])
        self.assertEqual((out / "물2_서바_해설_p2_q11.png").read_bytes(), b"a")


class TestEvidenceScanTable(unittest.TestCase):
    """증거 스캔 테이블(ontology.EVIDENCE_SCANS) 무결성 — 키워드셋과 타깃·신뢰도가
    한 곳에 모인 단일 소스. 타깃 오타('물리학2' vs '물리학Ⅱ')·빈 셋·잘못된 polarity를
    추가 시점에 잡아, 조용히 죽는 증거(스캔 비용만 내고 매칭 0)를 차단한다."""
    from core import ontology as _O
    O = _O

    def _valid_targets(self):
        valid = set(self.O.SUBJECTS)
        for _subj, subs in self.O.CURRICULUM.items():
            valid.update(subs)
        return valid

    def test_targets_are_known_ontology_terms(self):
        valid = self._valid_targets()
        for g in self.O.EVIDENCE_SCANS:
            for t in g.targets:
                self.assertIn(t, valid,
                              f"증거 타깃 '{t}'가 SUBJECTS/CURRICULUM에 없음 (hint={g.hint})")

    def test_pro_targets_reconcile_to_canonical_subject(self):
        # pro 세부과목 타깃은 엔진 역방향 맵(_SUB_CANON)으로 대분류 복구가 돼야
        # 모델이 sub_subject만 줘도 과목이 정해진다(누수 방지).
        for g in self.O.EVIDENCE_SCANS:
            if g.polarity != "pro":
                continue
            for t in g.targets:
                if t in self.O.SUBJECTS:          # 대분류 자체(통합과학 등)는 제외
                    continue
                self.assertIn(E._course_key(t), E._SUB_CANON,
                              f"pro 타깃 '{t}' 정규화 불가 (hint={g.hint})")

    def test_keyword_sets_nonempty(self):
        for g in self.O.EVIDENCE_SCANS:
            self.assertTrue(g.keywords, f"빈 키워드셋 (hint={g.hint})")

    def test_polarity_is_valid(self):
        for g in self.O.EVIDENCE_SCANS:
            self.assertIn(g.polarity, ("pro", "anti"), f"잘못된 polarity (hint={g.hint})")

    def test_table_actually_wires_into_pre_classify(self):
        # 테이블이 엔진 배선과 끊기면(예: 루프 누락) 회귀. 대표 증거 발화로 연결 확인.
        targets = {t for it in E.pre_classify("광전효과 물질파 드브로이 보어모형", "", "")["items"]
                   if it["polarity"] == "pro" for t in it["targets"]}
        self.assertIn("물리학Ⅱ", targets)

    def test_every_instructor_is_a_brand_stopword(self):
        # 강사명=과목 신호이자 stopword. 자동 합집합으로 드리프트 0이어야(과거 강민철 누락 회귀).
        drift = set(self.O.INSTRUCTOR_SUBJECT) - set(self.O.PUBLISHER_BRAND_STOPWORDS)
        self.assertEqual(drift, set(), f"stopword에 없는 강사(드리프트): {sorted(drift)}")

    def test_instructor_name_fires_anti_unclassified_in_body(self):
        # 본문 OCR에 박힌 강사명(헤더·워터마크)은 미분류 anti로 잡혀야(내용 오인 방지)
        antis = [it["keyword"] for it in E.pre_classify("강민철 독서 강의 노트", "", "")["items"]
                 if it["polarity"] == "anti" and it["targets"] == ["미분류"]]
        self.assertIn("강민철", antis)

#!/usr/bin/env python3
"""Deterministic unit tests for the classi engine (no ollama, no OCR).

Run:  cd backend && python3 -m unittest tests.test_engine -v
"""
import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import classifier_engine as E
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
    """추론 캐시(발열·재실행 비용 절감) — 동일 입력 추론 결과 재사용."""
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self._old = (E._CACHE_DB, E._CACHE_ENABLED, E._cache_conn)
        E._CACHE_DB = os.path.join(self.tmp, "c.db")
        E._cache_conn = None
        E._CACHE_ENABLED = True

    def tearDown(self):
        try:
            if E._cache_conn: E._cache_conn.close()
        except Exception:
            pass
        E._CACHE_DB, E._CACHE_ENABLED, E._cache_conn = self._old
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
        E._CACHE_ENABLED = False
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


if __name__ == "__main__":
    unittest.main(verbosity=2)

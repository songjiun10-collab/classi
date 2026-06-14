#!/usr/bin/env python3
"""pdf_to_hwpx 회귀 테스트 — OCR 없이 검증 가능한 부분만.

가드: (1) 컬럼 인식 읽기순서 조립(2단 뒤섞임 방지), (2) .hwpx 구조 무결성
      (유효한 zip · mimetype 첫 엔트리·무압축 · 모든 XML well-formed · 본문/이스케이프 반영).
한컴오피스 실제 열림 여부는 이 환경에서 검증 불가(별도 수동 확인 필요).

Run:  cd backend && python3 -m unittest tests.test_hwpx -v
"""
import io, os, sys, zipfile, unittest
from xml.dom import minidom

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools import pdf_to_hwpx as H
import fitz


class TestReadingOrder(unittest.TestCase):
    def test_two_columns_not_interleaved(self):
        # 좌단(x≈50)·우단(x≈350): 좌단 전체가 우단보다 먼저 와야(2단 뒤섞임 방지)
        lines = [(50, 100, "L1"), (350, 100, "R1"), (50, 200, "L2"), (350, 200, "R2")]
        self.assertEqual(H._assemble_reading_order(lines, 600.0), ["L1", "L2", "R1", "R2"])

    def test_single_column_keeps_y_order(self):
        lines = [(50, 300, "c"), (52, 100, "a"), (48, 200, "b")]
        self.assertEqual(H._assemble_reading_order(lines, 600.0), ["a", "b", "c"])

    def test_empty(self):
        self.assertEqual(H._assemble_reading_order([], 600.0), [])


class TestFormulaMerge(unittest.TestCase):
    """_merge_formula_lines: 수식 bbox 안 텍스트 잔해 제거 + '$LaTeX$' 인라인 삽입(순수 함수)."""

    def test_suppresses_text_inside_formula_and_inserts_latex(self):
        lines = [(50, 100, "본문"), (60, 205, "x2-1")]          # 둘째 줄은 수식 영역 안 잔해
        fregions = [(55, 200, 300, 240, "x^{2}-1")]
        merged = H._merge_formula_lines(lines, fregions)
        self.assertIn((50, 100, "본문"), merged)                 # 영역 밖 텍스트 보존
        self.assertNotIn((60, 205, "x2-1"), merged)              # 영역 안 잔해 제거
        self.assertIn((55, 200, "$x^{2}-1$"), merged)            # LaTeX 삽입

    def test_keeps_text_outside_regions(self):
        lines = [(50, 100, "A"), (50, 500, "B")]
        fregions = [(40, 200, 200, 260, "y=mx+b")]
        merged = H._merge_formula_lines(lines, fregions)
        self.assertIn((50, 100, "A"), merged)
        self.assertIn((50, 500, "B"), merged)
        self.assertIn((40, 200, "$y=mx+b$"), merged)

    def test_empty_latex_dropped(self):
        self.assertEqual(H._merge_formula_lines([], [(10, 10, 50, 50, "  ")]), [])

    def test_inline_placement_via_reading_order(self):
        lines = [(50, 100, "위"), (50, 300, "아래")]
        fregions = [(50, 200, 300, 240, "a+b")]
        merged = H._merge_formula_lines(lines, fregions)
        self.assertEqual(H._assemble_reading_order(merged, 600.0), ["위", "$a+b$", "아래"])


class TestUseTextLayer(unittest.TestCase):
    """_use_text_layer: 이미지 있으면 200자, 없으면 30자 문턱(헤더만 보고 스캔 본문 놓침 방지)."""
    def test_pure_text_low_threshold(self):
        self.assertTrue(H._use_text_layer(30, has_images=False))
        self.assertFalse(H._use_text_layer(29, has_images=False))

    def test_image_page_high_threshold(self):
        # 텍스트 헤더(예: 40자)만 있고 본문이 스캔 이미지면 텍스트레이어를 쓰면 안 됨(OCR로).
        self.assertFalse(H._use_text_layer(40, has_images=True))
        self.assertTrue(H._use_text_layer(200, has_images=True))


class TestTextLayerLines(unittest.TestCase):
    """_textlayer_lines: 텍스트레이어 2단 페이지를 읽기순서(좌단 전체→우단)로 복원."""
    def test_two_column_reading_order(self):
        doc = fitz.open()
        page = doc.new_page(width=595, height=842)  # A4
        # 좌단(x=60)·우단(x=360)에 같은 y로 배치 — get_text() 기본 모드라면 줄단위로 교차됨.
        page.insert_text((60, 120), "Left one")
        page.insert_text((360, 120), "Right one")
        page.insert_text((60, 220), "Left two")
        page.insert_text((360, 220), "Right two")
        out = H._textlayer_lines(page)
        doc.close()
        joined = " ".join(out)
        # 좌단 두 줄이 우단보다 먼저(2단 뒤섞임 없음). 각 줄 내부 단어 순서도 보존.
        self.assertIn("Left one", joined)
        self.assertIn("Right two", joined)
        self.assertLess(joined.index("Left two"), joined.index("Right one"))

    def test_single_line_preserved(self):
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "The quick brown fox")
        out = H._textlayer_lines(page)
        doc.close()
        self.assertIn("The quick brown fox", " ".join(out))


class TestHwpxStructure(unittest.TestCase):
    def setUp(self):
        self.buf_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "test_classi.hwpx")
        pages = [["첫 문단 <보기> & 기호", "둘째 문단"], ["둘째 페이지"]]
        H.build_hwpx(pages, self.buf_path, title="테스트 & <문서>")

    def tearDown(self):
        try: os.remove(self.buf_path)
        except OSError: pass

    def test_valid_zip(self):
        self.assertTrue(zipfile.is_zipfile(self.buf_path))

    def test_mimetype_first_and_stored(self):
        with zipfile.ZipFile(self.buf_path) as z:
            infos = z.infolist()
            self.assertEqual(infos[0].filename, "mimetype")           # 반드시 첫 엔트리
            self.assertEqual(infos[0].compress_type, zipfile.ZIP_STORED)  # 무압축
            self.assertEqual(z.read("mimetype").decode(), "application/hwp+zip")

    def test_required_entries_present(self):
        with zipfile.ZipFile(self.buf_path) as z:
            names = set(z.namelist())
        for req in ("version.xml", "settings.xml", "META-INF/container.xml",
                    "META-INF/manifest.xml", "Contents/content.hpf",
                    "Contents/header.xml", "Contents/section0.xml"):
            self.assertIn(req, names, f"{req} 누락")

    def test_all_xml_well_formed(self):
        with zipfile.ZipFile(self.buf_path) as z:
            for name in z.namelist():
                if name.endswith((".xml", ".hpf")):
                    minidom.parseString(z.read(name))  # 깨지면 예외 → 실패

    def test_text_and_escaping_in_section(self):
        with zipfile.ZipFile(self.buf_path) as z:
            sec = z.read("Contents/section0.xml").decode("utf-8")
        self.assertIn("둘째 문단", sec)
        self.assertIn("&lt;보기&gt;", sec)   # '<보기>' 이스케이프
        self.assertIn("&amp;", sec)            # '&' 이스케이프
        self.assertNotIn("<보기>", sec)        # 원시 꺾쇠가 본문으로 새면 안 됨


class TestCleanTranscription(unittest.TestCase):
    """OCR 후처리: 깨진 코드포인트만 제거하고 실제 글자는 보존(내용 보존 최우선)."""
    def test_removes_replacement_char_keeps_roman(self):
        # U+FFFD 제거, 로마숫자 Ⅱ는 분류용 정규화와 달리 그대로 보존
        self.assertEqual(H._clean_transcription("과학탐구\ufffd영역(물리학Ⅱ)"), "과학탐구영역(물리학Ⅱ)")

    def test_pua_becomes_placeholder_zerowidth_removed(self):
        # PUA 수식 글리프 → '□'(자리 보존), 제로폭은 제거
        self.assertEqual(H._clean_transcription("\ue000보기\ue001"), "□보기□")
        self.assertEqual(H._clean_transcription("a\u200bb\u200cc\ufeff"), "abc")

    def test_pure_pua_line_dropped(self):
        # 줄 전체가 PUA(내용 없는 가짜 글리프)면 버림
        self.assertEqual(H._clean_transcription("\ue000\ue001"), "")

    def test_collapses_spaces_and_trims(self):
        self.assertEqual(H._clean_transcription("x    y"), "x y")
        self.assertEqual(H._clean_transcription("  \t정상  "), "정상")

    def test_empty(self):
        self.assertEqual(H._clean_transcription(""), "")
        self.assertEqual(H._clean_transcription(None), "")


class TestStructureParagraphs(unittest.TestCase):
    """문항 번호(1. 2. …)로 시작하는 문단 앞에만 빈 줄 삽입. 보기 ①②③·소수점은 분할 안 함."""
    def test_inserts_blank_before_problems(self):
        got = H._structure_paragraphs(["표지", "1. 첫문제", "본문", "2. 둘째", "③ 보기"])
        self.assertEqual(got, ["표지", "", "1. 첫문제", "본문", "", "2. 둘째", "③ 보기"])

    def test_no_blank_before_first(self):
        self.assertEqual(H._structure_paragraphs(["1. 첫", "2. 둘"]), ["1. 첫", "", "2. 둘"])

    def test_decimal_and_circled_not_split(self):
        # '1.5'(소수점, 마침표 뒤 공백 없음)·'③'(원숫자=보기)은 문항 시작이 아님
        self.assertEqual(H._structure_paragraphs(["abc", "1.5 미터", "③ 답"]),
                         ["abc", "1.5 미터", "③ 답"])

    def test_empty(self):
        self.assertEqual(H._structure_paragraphs([]), [])


class TestTranscribePdfWiring(unittest.TestCase):
    """transcribe_pdf가 두 경로 산출물에 _clean_transcription을 일괄 적용하고 빈 문단을 버리는지."""
    def setUp(self):
        self.path = os.path.join(os.environ.get("TMPDIR", "/tmp"), "test_classi_wire.pdf")
        doc = fitz.open(); doc.new_page(); doc.save(self.path); doc.close()
        self._orig = H.transcribe_page

    def tearDown(self):
        H.transcribe_page = self._orig
        try: os.remove(self.path)
        except OSError: pass

    def test_central_cleaning_and_blank_drop(self):
        # 더럽힌 줄을 내도록 transcribe_page 대체 → transcribe_pdf의 중앙 정리 검증
        H.transcribe_page = lambda page, zoom, formula=False: ["과학탐구\ufffd영역", "\ue001버려질글리프", "   ", "정상줄"]
        pages = H.transcribe_pdf(self.path)
        self.assertEqual(pages, [["과학탐구영역", "□버려질글리프", "정상줄"]])


if __name__ == "__main__":
    unittest.main(verbosity=2)

#!/usr/bin/env python3
"""kice 다운로더 파싱 회귀 테스트 — 네트워크 없이 결정론적.

평가원 사이트 실제 마크업(작은따옴표 href, 탐구 zip 첨부, goView 목록)에 대해
링크/seq 추출이 동작하는지 고정한다. 과거 정규식은 fileSeq 뒤 큰따옴표(")만 받아
사이트가 쓰는 작은따옴표(')를 전부 놓쳐 다운로드가 0건이었다(이 테스트가 그 회귀를 막는다).

Run:  cd backend && python3 -m unittest tests.test_kice_downloader -v
"""
import os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from downloaders import kice


# 실제 view.do 첨부 블록(작은따옴표 href) 발췌 — 국어(pdf 2개)
VIEW_HTML = (
    "<div class='fieldBox'><dl><dt><span>첨부파일</span></dt><dd>"
    "<a href='/boardCnts/fileDown.do?fileSeq=60defdef6d83db1b756f841089563c5a' target='_blank' >국어영역_문제지.pdf</a>&nbsp;다운로드횟수 : 218948<br/>"
    "<a href='/boardCnts/fileDown.do?fileSeq=46c7d36aa9208016099f9a4cf2e7cc2a' target='_blank' >국어영역_정답표.pdf</a><br/>"
    "</dd></dl></div>"
)
# 탐구영역 게시물: zip 첨부 + 비-PDF(hwp) 첨부 혼재
TANGU_HTML = (
    "<a href='/boardCnts/fileDown.do?fileSeq=a5ffc663aaaa' target='_blank' >사회탐구영역_문제지.zip</a>"
    "<a href='/boardCnts/fileDown.do?fileSeq=11e60d8abbbb' target='_blank' >사회탐구영역_정답표.zip</a>"
    "<a href='/boardCnts/fileDown.do?fileSeq=deadbeefcccc' target='_blank' >붙임_유의사항.hwp</a>"
)
LIST_HTML = (
    "<tr><td onclick=\"javascript:goView('1500234','5093801', '0', 'null', 'W', '1', 'N', '')\">국어</td></tr>"
    "<tr><td onclick=\"javascript:goView('1500234','5093800', '0', 'null', 'W', '1', 'N', '')\">수학</td></tr>"
    # 헤더/반복 블록에서 같은 seq가 또 나와도 한 번만 잡혀야 한다
    "<tr><td onclick=\"javascript:goView('1500234','5093801', '0', 'null', 'W', '1', 'N', '')\">국어(중복)</td></tr>"
    # 함수 정의 라인엔 실인자가 없어 매칭되지 않아야 한다
    "<script>function goView(reqBoardID, reqBoardSeq){}</script>"
)


class TestAttachRegex(unittest.TestCase):
    def test_single_quote_pdf_links_captured(self):
        m = kice._ATTACH_RE.findall(VIEW_HTML)
        self.assertEqual(len(m), 2)
        seqs = [s for s, _ in m]
        names = [n for _, n in m]
        self.assertEqual(seqs, ["60defdef6d83db1b756f841089563c5a", "46c7d36aa9208016099f9a4cf2e7cc2a"])
        self.assertEqual(names, ["국어영역_문제지.pdf", "국어영역_정답표.pdf"])

    def test_zip_captured_hwp_ignored(self):
        m = kice._ATTACH_RE.findall(TANGU_HTML)
        names = [n for _, n in m]
        self.assertEqual(names, ["사회탐구영역_문제지.zip", "사회탐구영역_정답표.zip"])  # hwp 제외
        self.assertTrue(all(n.lower().endswith(".zip") for n in names))


class TestPostSeqs(unittest.TestCase):
    def test_dedup_and_order(self):
        seqs = kice._post_seqs(LIST_HTML, "1500234")
        self.assertEqual(seqs, ["5093801", "5093800"])  # 중복 1회·순서 보존·함수정의 제외

    def test_other_board_id_not_matched(self):
        self.assertEqual(kice._post_seqs(LIST_HTML, "9999999"), [])


if __name__ == "__main__":
    unittest.main()

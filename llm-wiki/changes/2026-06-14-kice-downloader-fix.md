# 2026-06-14 · 평가원(KICE) 다운로더 0건 버그 수정 + 탐구 zip 해제

> 백엔드 정확도 개선 야간 작업(브랜치 `improve/overnight-20260614`)의 변경 1건.

## 맥락
사이트가 href에 작은따옴표를 써 정규식(큰따옴표 전제)이 0건 매칭 → 다운로드 불가였다. 양따옴표 허용 + 페이지 크롤 + 탐구 zip의 pdf 멤버 해제.

## 커밋
`b2933c0` — fix(downloaders/kice): 사이트 변경 대응 — 다운로드 0건 버그 수정 + 탐구 zip 해제

## 변경된 파일
```
downloaders/kice.py           | 169 ++++++++++++++++++++++++++++++------------
 tests/test_kice_downloader.py |  65 ++++++++++++++++
 2 files changed, 187 insertions(+), 47 deletions(-)
```

## 전체 diff

```diff
diff --git a/downloaders/kice.py b/downloaders/kice.py
index e1bb33c..06ad9e2 100644
--- a/downloaders/kice.py
+++ b/downloaders/kice.py
@@ -1,73 +1,148 @@
 #!/usr/bin/env python3
-"""download_kice.py — 평가원 수능/모의평가 기출 PDF 다운로더"""
-import argparse, asyncio, os, re
-from datetime import datetime
+"""download_kice.py — 평가원 수능/모의평가 기출 PDF 다운로더
+
+게시판(list.do)의 게시물 제목이 전부 '문제 및 정답'이라, 과거 방식인 제목 연도검색
+(searchStr=연도)은 항상 0건이 된다 → 제목 연도필터 대신 페이지(page=N)를 순회해 최신
+게시물부터 받는다. 게시물 하나는 한 과목 시험이며(국어영역_문제지.pdf 등), 탐구영역은
+세부과목 PDF들이 zip 한 개로 묶여 배포되므로 zip을 받아 내부 PDF를 풀어 저장한다
+(첨부 파일명이 곧 과목 라벨 — eval·보정 gold로 바로 쓸 수 있다).
+"""
+import argparse, asyncio, io, os, re, zipfile
 from pathlib import Path
 import aiohttp
 
 BASE_URL = "https://www.suneung.re.kr"
-OUTPUT_DIR = Path("pdfs/평가원")
+OUTPUT_DIR = Path("data/raw/평가원")  # backend/.gitignore가 data/ 를 제외 → 산출물이 git에 안 섞인다
 BOARDS = {"1500234": "수능_기출", "1500236": "수능_모의평가", "1500237": "수능_예시문항"}
 
-async def download_file(session, url, filepath):
-    if filepath.exists() and filepath.stat().st_size > 0:
-        return True  # 0바이트(이전 실패 잔재)면 다시 받는다
+# 첨부 앵커: fileSeq(16진) 뒤 따옴표는 사이트가 작은따옴표를 쓴다(과거 정규식은 큰따옴표만
+# 받아 모든 링크를 놓쳤다 → 다운로드 0건이던 핵심 버그). pdf와 zip(탐구 묶음) 모두 캡처한다.
+_ATTACH_RE = re.compile(
+    r"fileDown\.do\?fileSeq=([a-f0-9]+)['\"][^>]*>([^<]+\.(?:pdf|zip))", re.IGNORECASE)
+
+
+def _post_seqs(html: str, board_id: str):
+    """목록 HTML의 goView('boardID','seq',...) 호출에서 게시물 seq를 추출(헤더 중복 제거)."""
+    seqs = re.findall(r"goView\('" + re.escape(board_id) + r"','(\d+)'", html)
+    return list(dict.fromkeys(seqs))  # 순서 보존하며 중복 제거
+
+
+def _safe_write(body: bytes, filepath: Path):
+    """잘린 다운로드가 다음 실행에서 '완료'로 오인되지 않게 .part에 먼저 쓰고 성공 시 원자적 교체."""
     tmp = filepath.with_suffix(filepath.suffix + ".part")
+    tmp.write_bytes(body)
+    os.replace(tmp, filepath)
+
+
+async def _get_bytes(session, url):
     try:
         async with session.get(url) as resp:
             if resp.status != 200:
-                return False
-            body = await resp.read()
-        # 잘린 다운로드가 최종 경로에 남아 다음 실행에서 '완료'로 오인되는 것을 막기 위해
-        # .part에 먼저 쓰고 성공 시에만 원자적으로 교체한다.
-        tmp.write_bytes(body)
-        os.replace(tmp, filepath)
-        return True
+                return None
+            return await resp.read()
     except Exception:
-        tmp.unlink(missing_ok=True)  # 부분 파일 제거
-        return False
+        return None
 
-async def get_pdf_links(session, board_id, board_seq):
-    url = f"{BASE_URL}/boardCnts/view.do?boardID={board_id}&boardSeq={board_seq}&lev=0&m=0403"
+
+async def _get_text(session, url):
     async with session.get(url) as resp:
-        text = await resp.text()
-    # seq와 파일명을 같은 앵커에서 함께 캡처한다. 따로 findall 하면 비-PDF 첨부(HWP 등)가
-    # 섞일 때 두 리스트의 길이/순서가 어긋나 zip이 엉뚱한 seq를 엉뚱한 이름으로 묶거나 누락시킨다.
-    return re.findall(r'fileDown\.do\?fileSeq=([a-f0-9]+)"[^>]*>([^<]+\.pdf)', text, re.IGNORECASE)
+        return await resp.text()
 
-async def download_board(session, board_id, board_name, years):
+
+async def download_pdf(session, file_seq, filename, base_dir):
+    """단일 PDF 첨부를 받는다. 이미 있으면(>0바이트) 건너뛴다. 반환: 새로 받은 개수(0/1)."""
+    filepath = base_dir / filename
+    if filepath.exists() and filepath.stat().st_size > 0:
+        return 0
+    body = await _get_bytes(session, f"{BASE_URL}/boardCnts/fileDown.do?fileSeq={file_seq}")
+    if not body:
+        return 0
+    _safe_write(body, filepath)
+    return 1
+
+
+async def download_zip_pdfs(session, file_seq, zip_name, base_dir):
+    """탐구 zip을 받아 내부 .pdf 멤버만 푼다. 멤버명(예: '01 생활과 윤리_문제.pdf')이
+    과목 라벨이므로 그대로 보존한다. zip 손상/비-zip이면 건너뛴다(형제 작업 보호). 반환: 푼 PDF 수."""
+    body = await _get_bytes(session, f"{BASE_URL}/boardCnts/fileDown.do?fileSeq={file_seq}")
+    if not body or body[:2] != b"PK":
+        return 0
+    sub = base_dir / Path(zip_name).stem
+    sub.mkdir(parents=True, exist_ok=True)
+    try:
+        zf = zipfile.ZipFile(io.BytesIO(body))
+    except zipfile.BadZipFile:
+        return 0
+    n = 0
+    for info in zf.infolist():
+        name = os.path.basename(info.filename)
+        if not name.lower().endswith(".pdf"):
+            continue
+        # zip이 UTF-8 플래그 없이 저장됐으면 cp437→euc-kr로 한글 복원(평가원 zip은 UTF-8이나 방어적으로).
+        if not (info.flag_bits & 0x800):
+            try:
+                name = name.encode("cp437").decode("euc-kr")
+            except Exception:
+                pass
+        out = sub / name
+        if out.exists() and out.stat().st_size > 0:
+            continue
+        try:
+            _safe_write(zf.read(info), out)
+            n += 1
+        except Exception:
+            continue  # 멤버 하나 실패가 나머지 해제를 막지 않게
+    return n
+
+
+async def get_attachments(session, board_id, board_seq):
+    """게시물 view 페이지에서 (fileSeq, 파일명) 첨부 목록(pdf·zip)을 추출."""
+    url = f"{BASE_URL}/boardCnts/view.do?boardID={board_id}&boardSeq={board_seq}&lev=0&m=0403"
+    return _ATTACH_RE.findall(await _get_text(session, url))
+
+
+async def get_post_seqs(session, board_id, page):
+    url = f"{BASE_URL}/boardCnts/list.do?boardID={board_id}&m=0403&s=suneung&page={page}"
+    return _post_seqs(await _get_text(session, url), board_id)
+
+
+async def download_board(session, board_id, board_name, pages, limit):
     base_dir = OUTPUT_DIR / board_name
     base_dir.mkdir(parents=True, exist_ok=True)
-    total = 0
-    for year in years:
-        try:
-            list_url = f"{BASE_URL}/boardCnts/list.do?boardID={board_id}&m=0403&s=suneung&searchStr={year}"
-            async with session.get(list_url) as resp:
-                seqs = re.findall(r"goView\('" + board_id + r"','(\d+)'", await resp.text())
+    saved = posts = 0
+    for page in range(1, pages + 1):
+        for seq in await get_post_seqs(session, board_id, page):
+            if limit and posts >= limit:
+                return saved
+            posts += 1
             tasks = []
-            for seq in seqs:
-                for file_seq, filename in await get_pdf_links(session, board_id, seq):
-                    tasks.append(download_file(session, f"{BASE_URL}/boardCnts/fileDown.do?fileSeq={file_seq}", base_dir / filename))
+            for file_seq, name in await get_attachments(session, board_id, seq):
+                name = name.strip()
+                if name.lower().endswith(".zip"):
+                    tasks.append(download_zip_pdfs(session, file_seq, name, base_dir))
+                else:
+                    tasks.append(download_pdf(session, file_seq, name, base_dir))
             if tasks:
-                # 한 다운로드 실패가 형제 작업을 취소시키지 않도록 격리하고 성공 건수만 집계
-                done = await asyncio.gather(*tasks, return_exceptions=True)
-                total += sum(1 for r in done if r is True)
-        except Exception as e:
-            print(f"⚠ {board_name} {year} 실패: {e}")  # 한 해 실패가 전체 크롤을 끊지 않음
-            continue
-    return total
+                # 한 첨부 실패가 형제를 취소하지 않도록 격리하고 성공 건수만 합산
+                for r in await asyncio.gather(*tasks, return_exceptions=True):
+                    if isinstance(r, int):
+                        saved += r
+    return saved
+
 
 async def main(args):
     OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
-    target_boards = {args.board: BOARDS[args.board]} if args.board else BOARDS
-    years = [str(y) for y in range(int(args.year.split("-")[0]), int(args.year.split("-")[1])+1)] if args.year and "-" in args.year else [args.year] if args.year else [str(y) for y in range(2005, datetime.now().year+1)]
+    targets = {args.board: BOARDS[args.board]} if args.board else BOARDS
     async with aiohttp.ClientSession() as session:
-        for bid, bname in target_boards.items():
-            await download_board(session, bid, bname, years)
+        for bid, bname in targets.items():
+            n = await download_board(session, bid, bname, args.pages, args.limit)
+            print(f"{bname}: PDF {n}개 저장 → {OUTPUT_DIR / bname}")
     print("완료")
 
+
 if __name__ == "__main__":
-    parser = argparse.ArgumentParser()
-    parser.add_argument("--board", choices=BOARDS.keys())
-    parser.add_argument("--year")
-    asyncio.run(main(parser.parse_args()))
+    ap = argparse.ArgumentParser(description="평가원 기출/모평 PDF 다운로더(탐구 zip 자동 해제)")
+    ap.add_argument("--board", choices=BOARDS.keys(), help="기본: 전체 게시판")
+    ap.add_argument("--pages", type=int, default=1, help="목록 페이지 수(최신부터, 기본 1)")
+    ap.add_argument("--limit", type=int, default=0, help="게시판당 최대 게시물 수(0=무제한)")
+    asyncio.run(main(ap.parse_args()))
diff --git a/tests/test_kice_downloader.py b/tests/test_kice_downloader.py
new file mode 100644
index 0000000..d358c20
--- /dev/null
+++ b/tests/test_kice_downloader.py
@@ -0,0 +1,65 @@
+#!/usr/bin/env python3
+"""kice 다운로더 파싱 회귀 테스트 — 네트워크 없이 결정론적.
+
+평가원 사이트 실제 마크업(작은따옴표 href, 탐구 zip 첨부, goView 목록)에 대해
+링크/seq 추출이 동작하는지 고정한다. 과거 정규식은 fileSeq 뒤 큰따옴표(")만 받아
+사이트가 쓰는 작은따옴표(')를 전부 놓쳐 다운로드가 0건이었다(이 테스트가 그 회귀를 막는다).
+
+Run:  cd backend && python3 -m unittest tests.test_kice_downloader -v
+"""
+import os, sys, unittest
+
+sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
+from downloaders import kice
+
+
+# 실제 view.do 첨부 블록(작은따옴표 href) 발췌 — 국어(pdf 2개)
+VIEW_HTML = (
+    "<div class='fieldBox'><dl><dt><span>첨부파일</span></dt><dd>"
+    "<a href='/boardCnts/fileDown.do?fileSeq=60defdef6d83db1b756f841089563c5a' target='_blank' >국어영역_문제지.pdf</a>&nbsp;다운로드횟수 : 218948<br/>"
+    "<a href='/boardCnts/fileDown.do?fileSeq=46c7d36aa9208016099f9a4cf2e7cc2a' target='_blank' >국어영역_정답표.pdf</a><br/>"
+    "</dd></dl></div>"
+)
+# 탐구영역 게시물: zip 첨부 + 비-PDF(hwp) 첨부 혼재
+TANGU_HTML = (
+    "<a href='/boardCnts/fileDown.do?fileSeq=a5ffc663aaaa' target='_blank' >사회탐구영역_문제지.zip</a>"
+    "<a href='/boardCnts/fileDown.do?fileSeq=11e60d8abbbb' target='_blank' >사회탐구영역_정답표.zip</a>"
+    "<a href='/boardCnts/fileDown.do?fileSeq=deadbeefcccc' target='_blank' >붙임_유의사항.hwp</a>"
+)
+LIST_HTML = (
+    "<tr><td onclick=\"javascript:goView('1500234','5093801', '0', 'null', 'W', '1', 'N', '')\">국어</td></tr>"
+    "<tr><td onclick=\"javascript:goView('1500234','5093800', '0', 'null', 'W', '1', 'N', '')\">수학</td></tr>"
+    # 헤더/반복 블록에서 같은 seq가 또 나와도 한 번만 잡혀야 한다
+    "<tr><td onclick=\"javascript:goView('1500234','5093801', '0', 'null', 'W', '1', 'N', '')\">국어(중복)</td></tr>"
+    # 함수 정의 라인엔 실인자가 없어 매칭되지 않아야 한다
+    "<script>function goView(reqBoardID, reqBoardSeq){}</script>"
+)
+
+
+class TestAttachRegex(unittest.TestCase):
+    def test_single_quote_pdf_links_captured(self):
+        m = kice._ATTACH_RE.findall(VIEW_HTML)
+        self.assertEqual(len(m), 2)
+        seqs = [s for s, _ in m]
+        names = [n for _, n in m]
+        self.assertEqual(seqs, ["60defdef6d83db1b756f841089563c5a", "46c7d36aa9208016099f9a4cf2e7cc2a"])
+        self.assertEqual(names, ["국어영역_문제지.pdf", "국어영역_정답표.pdf"])
+
+    def test_zip_captured_hwp_ignored(self):
+        m = kice._ATTACH_RE.findall(TANGU_HTML)
+        names = [n for _, n in m]
+        self.assertEqual(names, ["사회탐구영역_문제지.zip", "사회탐구영역_정답표.zip"])  # hwp 제외
+        self.assertTrue(all(n.lower().endswith(".zip") for n in names))
+
+
+class TestPostSeqs(unittest.TestCase):
+    def test_dedup_and_order(self):
+        seqs = kice._post_seqs(LIST_HTML, "1500234")
+        self.assertEqual(seqs, ["5093801", "5093800"])  # 중복 1회·순서 보존·함수정의 제외
+
+    def test_other_board_id_not_matched(self):
+        self.assertEqual(kice._post_seqs(LIST_HTML, "9999999"), [])
+
+
+if __name__ == "__main__":
+    unittest.main()
```

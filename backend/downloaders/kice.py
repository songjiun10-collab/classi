#!/usr/bin/env python3
"""download_kice.py — 평가원 수능/모의평가 기출 PDF 다운로더

게시판(list.do)의 게시물 제목이 전부 '문제 및 정답'이라, 과거 방식인 제목 연도검색
(searchStr=연도)은 항상 0건이 된다 → 제목 연도필터 대신 페이지(page=N)를 순회해 최신
게시물부터 받는다. 게시물 하나는 한 과목 시험이며(국어영역_문제지.pdf 등), 탐구영역은
세부과목 PDF들이 zip 한 개로 묶여 배포되므로 zip을 받아 내부 PDF를 풀어 저장한다
(첨부 파일명이 곧 과목 라벨 — eval·보정 gold로 바로 쓸 수 있다).
"""
import argparse, asyncio, io, os, re, zipfile
from pathlib import Path
import aiohttp

BASE_URL = "https://www.suneung.re.kr"
OUTPUT_DIR = Path("data/raw/평가원")  # backend/.gitignore가 data/ 를 제외 → 산출물이 git에 안 섞인다
BOARDS = {"1500234": "수능_기출", "1500236": "수능_모의평가", "1500237": "수능_예시문항"}

# 첨부 앵커: fileSeq(16진) 뒤 따옴표는 사이트가 작은따옴표를 쓴다(과거 정규식은 큰따옴표만
# 받아 모든 링크를 놓쳤다 → 다운로드 0건이던 핵심 버그). pdf와 zip(탐구 묶음) 모두 캡처한다.
_ATTACH_RE = re.compile(
    r"fileDown\.do\?fileSeq=([a-f0-9]+)['\"][^>]*>([^<]+\.(?:pdf|zip))", re.IGNORECASE)


def _post_seqs(html: str, board_id: str):
    """목록 HTML의 goView('boardID','seq',...) 호출에서 게시물 seq를 추출(헤더 중복 제거)."""
    seqs = re.findall(r"goView\('" + re.escape(board_id) + r"','(\d+)'", html)
    return list(dict.fromkeys(seqs))  # 순서 보존하며 중복 제거


def _safe_write(body: bytes, filepath: Path):
    """잘린 다운로드가 다음 실행에서 '완료'로 오인되지 않게 .part에 먼저 쓰고 성공 시 원자적 교체."""
    tmp = filepath.with_suffix(filepath.suffix + ".part")
    tmp.write_bytes(body)
    os.replace(tmp, filepath)


async def _get_bytes(session, url):
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return None
            return await resp.read()
    except Exception:
        return None


async def _get_text(session, url):
    async with session.get(url) as resp:
        return await resp.text()


async def download_pdf(session, file_seq, filename, base_dir):
    """단일 PDF 첨부를 받는다. 이미 있으면(>0바이트) 건너뛴다. 반환: 새로 받은 개수(0/1)."""
    filepath = base_dir / filename
    if filepath.exists() and filepath.stat().st_size > 0:
        return 0
    body = await _get_bytes(session, f"{BASE_URL}/boardCnts/fileDown.do?fileSeq={file_seq}")
    if not body:
        return 0
    _safe_write(body, filepath)
    return 1


async def download_zip_pdfs(session, file_seq, zip_name, base_dir):
    """탐구 zip을 받아 내부 .pdf 멤버만 푼다. 멤버명(예: '01 생활과 윤리_문제.pdf')이
    과목 라벨이므로 그대로 보존한다. zip 손상/비-zip이면 건너뛴다(형제 작업 보호). 반환: 푼 PDF 수."""
    body = await _get_bytes(session, f"{BASE_URL}/boardCnts/fileDown.do?fileSeq={file_seq}")
    if not body or body[:2] != b"PK":
        return 0
    sub = base_dir / Path(zip_name).stem
    sub.mkdir(parents=True, exist_ok=True)
    try:
        zf = zipfile.ZipFile(io.BytesIO(body))
    except zipfile.BadZipFile:
        return 0
    n = 0
    for info in zf.infolist():
        name = os.path.basename(info.filename)
        if not name.lower().endswith(".pdf"):
            continue
        # zip이 UTF-8 플래그 없이 저장됐으면 cp437→euc-kr로 한글 복원(평가원 zip은 UTF-8이나 방어적으로).
        if not (info.flag_bits & 0x800):
            try:
                name = name.encode("cp437").decode("euc-kr")
            except Exception:
                pass
        out = sub / name
        if out.exists() and out.stat().st_size > 0:
            continue
        try:
            _safe_write(zf.read(info), out)
            n += 1
        except Exception:
            continue  # 멤버 하나 실패가 나머지 해제를 막지 않게
    return n


async def get_attachments(session, board_id, board_seq):
    """게시물 view 페이지에서 (fileSeq, 파일명) 첨부 목록(pdf·zip)을 추출."""
    url = f"{BASE_URL}/boardCnts/view.do?boardID={board_id}&boardSeq={board_seq}&lev=0&m=0403"
    return _ATTACH_RE.findall(await _get_text(session, url))


async def get_post_seqs(session, board_id, page):
    url = f"{BASE_URL}/boardCnts/list.do?boardID={board_id}&m=0403&s=suneung&page={page}"
    return _post_seqs(await _get_text(session, url), board_id)


async def download_board(session, board_id, board_name, pages, limit):
    base_dir = OUTPUT_DIR / board_name
    base_dir.mkdir(parents=True, exist_ok=True)
    saved = posts = 0
    for page in range(1, pages + 1):
        for seq in await get_post_seqs(session, board_id, page):
            if limit and posts >= limit:
                return saved
            posts += 1
            tasks = []
            for file_seq, name in await get_attachments(session, board_id, seq):
                name = name.strip()
                if name.lower().endswith(".zip"):
                    tasks.append(download_zip_pdfs(session, file_seq, name, base_dir))
                else:
                    tasks.append(download_pdf(session, file_seq, name, base_dir))
            if tasks:
                # 한 첨부 실패가 형제를 취소하지 않도록 격리하고 성공 건수만 합산
                for r in await asyncio.gather(*tasks, return_exceptions=True):
                    if isinstance(r, int):
                        saved += r
    return saved


async def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = {args.board: BOARDS[args.board]} if args.board else BOARDS
    async with aiohttp.ClientSession() as session:
        for bid, bname in targets.items():
            n = await download_board(session, bid, bname, args.pages, args.limit)
            print(f"{bname}: PDF {n}개 저장 → {OUTPUT_DIR / bname}")
    print("완료")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="평가원 기출/모평 PDF 다운로더(탐구 zip 자동 해제)")
    ap.add_argument("--board", choices=BOARDS.keys(), help="기본: 전체 게시판")
    ap.add_argument("--pages", type=int, default=1, help="목록 페이지 수(최신부터, 기본 1)")
    ap.add_argument("--limit", type=int, default=0, help="게시판당 최대 게시물 수(0=무제한)")
    asyncio.run(main(ap.parse_args()))

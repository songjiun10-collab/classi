#!/usr/bin/env python3
"""download_kice.py — 평가원 수능/모의평가 기출 PDF 다운로더"""
import argparse, asyncio, re
from datetime import datetime
from pathlib import Path
import aiohttp

BASE_URL = "https://www.suneung.re.kr"
OUTPUT_DIR = Path("pdfs/평가원")
BOARDS = {"1500234": "수능_기출", "1500236": "수능_모의평가", "1500237": "수능_예시문항"}

async def download_file(session, url, filepath):
    if filepath.exists(): return True
    async with session.get(url) as resp:
        if resp.status == 200: filepath.write_bytes(await resp.read()); return True
    return False

async def get_pdf_links(session, board_id, board_seq):
    url = f"{BASE_URL}/boardCnts/view.do?boardID={board_id}&boardSeq={board_seq}&lev=0&m=0403"
    async with session.get(url) as resp:
        text = await resp.text()
    file_seqs = re.findall(r"fileDown\.do\?fileSeq=([a-f0-9]+)", text)
    names = re.findall(r'fileDown\.do\?fileSeq=[a-f0-9]+"[^>]*>([^<]+\.pdf)</a>', text, re.IGNORECASE)
    return list(zip(file_seqs, names))

async def download_board(session, board_id, board_name, years):
    base_dir = OUTPUT_DIR / board_name
    base_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for year in years:
        list_url = f"{BASE_URL}/boardCnts/list.do?boardID={board_id}&m=0403&s=suneung&searchStr={year}"
        async with session.get(list_url) as resp:
            seqs = re.findall(r"goView\('" + board_id + r"','(\d+)'", await resp.text())
        tasks = []
        for seq in seqs:
            for file_seq, filename in await get_pdf_links(session, board_id, seq):
                tasks.append(download_file(session, f"{BASE_URL}/boardCnts/fileDown.do?fileSeq={file_seq}", base_dir / filename))
        if tasks: total += sum(await asyncio.gather(*tasks))
    return total

async def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target_boards = {args.board: BOARDS[args.board]} if args.board else BOARDS
    years = [str(y) for y in range(int(args.year.split("-")[0]), int(args.year.split("-")[1])+1)] if args.year and "-" in args.year else [args.year] if args.year else [str(y) for y in range(2005, datetime.now().year+1)]
    async with aiohttp.ClientSession() as session:
        for bid, bname in target_boards.items():
            await download_board(session, bid, bname, years)
    print("완료")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--board", choices=BOARDS.keys())
    parser.add_argument("--year")
    asyncio.run(main(parser.parse_args()))

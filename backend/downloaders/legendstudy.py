#!/usr/bin/env python3
"""download_legendstudy.py — legendstudy.com 전국연합학력평가 PDF 다운로더"""
import argparse, asyncio, os, re
from pathlib import Path
import aiohttp

BASE_URL = "https://legendstudy.com"
OUTPUT_DIR = Path("pdfs/전국연합")

async def download_file(session, url, filepath):
    if filepath.exists() and filepath.stat().st_size > 0:
        return True  # 0바이트(이전 실패 잔재)면 다시 받는다
    tmp = filepath.with_suffix(filepath.suffix + ".part")
    try:
        async with session.get(url) as resp:
            if resp.status != 200:
                return False
            body = await resp.read()
        # .part에 먼저 쓰고 성공 시에만 원자적으로 교체 → 잘린 파일이 최종 경로에 남지 않음
        tmp.write_bytes(body)
        os.replace(tmp, filepath)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False

async def get_post_pdfs(session, post_id):
    async with session.get(f"{BASE_URL}/{post_id}") as resp:
        if resp.status != 200: return []
        text = await resp.text()
    return [(link, link.split('/')[-1]) for link in re.findall(r'href="(https?://[^"]+\.pdf)"', text)]

async def main(args):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    async with aiohttp.ClientSession() as session:
        for post_id in range(args.start, args.end + 1):
            try:
                pdfs = await get_post_pdfs(session, post_id)
                if pdfs:
                    tasks = [download_file(session, url, OUTPUT_DIR / name) for url, name in pdfs]
                    # 단일 전송 실패가 형제 다운로드를 취소하거나 전체 크롤을 끊지 않도록 격리
                    done = await asyncio.gather(*tasks, return_exceptions=True)
                    total += sum(1 for r in done if r is True)
            except Exception as e:
                print(f"⚠ post {post_id} 실패: {e}")  # 한 게시물 실패는 로그 후 계속
            await asyncio.sleep(0.3)
    print(f"완료: {total}개")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--year")
    asyncio.run(main(parser.parse_args()))

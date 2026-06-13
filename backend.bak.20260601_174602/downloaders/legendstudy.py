#!/usr/bin/env python3
"""download_legendstudy.py — legendstudy.com 전국연합학력평가 PDF 다운로더"""
import argparse, asyncio, re
from pathlib import Path
import aiohttp

BASE_URL = "https://legendstudy.com"
OUTPUT_DIR = Path("pdfs/전국연합")

async def download_file(session, url, filepath):
    if filepath.exists(): return True
    async with session.get(url) as resp:
        if resp.status == 200: filepath.write_bytes(await resp.read()); return True
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
            pdfs = await get_post_pdfs(session, post_id)
            if pdfs:
                tasks = [download_file(session, url, OUTPUT_DIR / name) for url, name in pdfs]
                total += sum(await asyncio.gather(*tasks))
            await asyncio.sleep(0.3)
    print(f"완료: {total}개")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=2000)
    parser.add_argument("--year")
    asyncio.run(main(parser.parse_args()))

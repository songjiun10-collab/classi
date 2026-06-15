#!/usr/bin/env python3
"""
유빈 아카이브 PDF 병렬 다운로더 (최대 8배속)
실행: python3 telegram_yubin_parallel.py
"""

import asyncio, os
from pathlib import Path
from telethon import TelegramClient

# 크리덴셜은 환경변수 우선(없으면 기존 값으로 폴백 → 실행 방식이 깨지지 않음).
# 완전 외부화하려면 TELEGRAM_API_ID / TELEGRAM_API_HASH 를 export 하고
# 아래 폴백 리터럴을 지운다(평문 노출 제거).
API_ID = int(os.environ.get("TELEGRAM_API_ID", "32139356"))
API_HASH = os.environ.get("TELEGRAM_API_HASH", "801a7c2bee2f4543cf71159e2aeaf530")
CHANNEL = os.environ.get("TELEGRAM_CHANNEL", "@yubin_MPGA")
DOWNLOAD_DIR = Path("pdfs/yubin/downloaded")
DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_CONCURRENT = 8

async def main():
    client = TelegramClient('yubin_session', API_ID, API_HASH)
    await client.start()

    print(f"📥 '{CHANNEL}' 채널에서 PDF 병렬 다운로드 시작 (최대 {MAX_CONCURRENT}개 동시)...\n")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    count = 0
    total = 0

    async def download_one(message):
        nonlocal count
        async with semaphore:
            # attributes[0]이 항상 DocumentAttributeFilename은 아니다(이미지 크기 등일 수 있음).
            # 그 경우 AttributeError가 gather(return_exceptions=True)에 조용히 삼켜져
            # 해당 파일이 무음 스킵되던 버그 → 속성 전체에서 file_name을 찾는다.
            filename = next((a.file_name for a in message.document.attributes
                             if getattr(a, "file_name", None)), None) or f"{message.document.id}.pdf"
            filepath = DOWNLOAD_DIR / filename

            if filepath.exists() and filepath.stat().st_size > 0:
                print(f"⏭️ {filename} (이미 있음)")
                return

            tmp = filepath.with_suffix(filepath.suffix + ".part")
            try:
                # 부분 다운로드가 최종 경로에 남아 다음 실행에서 영구 스킵되는 것을 막기 위해
                # .part에 받고 성공 시에만 교체한다.
                await client.download_media(message.document, str(tmp))
                os.replace(tmp, filepath)
            except Exception as e:
                try: tmp.unlink()
                except OSError: pass
                print(f"⚠ {filename} 실패: {e}")  # 한 건 실패가 배치 전체를 취소하지 않도록 격리
                return
            count += 1
            print(f"✅ [{count}] {filename}")

    messages = []
    async for message in client.iter_messages(CHANNEL):
        if message.document and 'pdf' in (message.document.mime_type or ''):
            messages.append(message)
            total += 1

    print(f"📊 총 {total}개 PDF 발견, 병렬 다운로드 시작...\n")

    tasks = [asyncio.create_task(download_one(msg)) for msg in messages]
    await asyncio.gather(*tasks, return_exceptions=True)  # 한 건 실패가 배치 전체를 취소하지 않게

    print(f"\n🎉 완료! 총 {count}개 새 PDF 다운로드됨")
    print(f"📁 저장 위치: {DOWNLOAD_DIR.absolute()}")

asyncio.run(main())

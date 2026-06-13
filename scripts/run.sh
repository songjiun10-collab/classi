#!/bin/bash
echo "🚀 Classi 전체 실행"
echo "1. Ollama 확인 중..."
pgrep ollama || (echo "Ollama 실행 필요" && exit 1)
echo "2. 유빈 다운로더 시작..."
nohup python3 ../backend/downloaders/telegram_yubin_parallel.py > /tmp/download.log 2>&1 &
echo "3. 소비자(분류+검수+학습) 시작..."
nohup python3 ../backend/pipeline/gamma_consumer.py > /tmp/consumer.log 2>&1 &
echo "4. API 서버 시작..."
cd ../backend/api && uvicorn server:app --host 0.0.0.0 --port 8000 --reload

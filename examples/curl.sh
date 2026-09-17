#!/usr/bin/env bash
# Foydalanish: API_URL=https://xxxx.ngrok-free.app ./examples/curl.sh audio.wav
API_URL="${API_URL:-http://127.0.0.1:8000}"
FILE="${1:?audio fayl yo'lini bering}"

curl -X POST "$API_URL/transcribe" \
  -H "ngrok-skip-browser-warning: true" \
  -F "file=@$FILE"

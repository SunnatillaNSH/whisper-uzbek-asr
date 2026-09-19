#!/usr/bin/env python3
"""RunPod endpoint'ini real audio bilan sinash.

Foydalanish:
    export RUNPOD_API_KEY="rpa_..."          # RunPod → Settings → API Keys
    python3 test_runpod.py /yo'l/audio.aac

Faqat standart kutubxonalardan foydalanadi (pip install kerak emas).
"""

import base64
import json
import os
import sys
import time
import urllib.request

ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID", "vwobkifazoyyh1")
API_KEY = os.environ.get("RUNPOD_API_KEY")
BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"


def post(path, payload):
    req = urllib.request.Request(
        f"{BASE}/{path}",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def get(path):
    req = urllib.request.Request(
        f"{BASE}/{path}", headers={"Authorization": f"Bearer {API_KEY}"}
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def main():
    if not API_KEY:
        sys.exit("RUNPOD_API_KEY o'rnatilmagan. Avval: export RUNPOD_API_KEY=\"rpa_...\"")
    if len(sys.argv) < 2:
        sys.exit("Foydalanish: python3 test_runpod.py /yo'l/audio.aac")

    path = sys.argv[1]
    raw = open(path, "rb").read()
    print(f"Fayl: {path}  ({len(raw)/1024:.0f} KB)")

    print("Yuborilmoqda...")
    job = post("run", {"input": {"audio_base64": base64.b64encode(raw).decode()}})
    job_id = job["id"]
    print(f"Job ID: {job_id}")

    started = time.time()
    while True:
        res = get(f"status/{job_id}")
        status = res.get("status")
        print(f"  [{time.time()-started:5.0f}s] {status}")
        if status in ("COMPLETED", "FAILED", "CANCELLED", "TIMED_OUT"):
            break
        time.sleep(5)

    print()
    out = res.get("output") or {}
    if out.get("status") == "success":
        print(f"Audio uzunligi   : {out.get('duration_sec')} sek")
        print(f"Qayta ishlash    : {out.get('processing_time_sec')} sek")
        if out.get("realtime_factor"):
            print(f"Tezlik           : {out['realtime_factor']}x realtime")
        print(f"Navbatda kutish  : {res.get('delayTime', 0)/1000:.1f} sek")
        print()
        print("─" * 70)
        print(out.get("text", ""))
        print("─" * 70)
    else:
        print("Xato yoki kutilmagan javob:")
        print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

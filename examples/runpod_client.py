"""RunPod Serverless endpoint'iga so'rov yuborish misoli.

Foydalanish:
    export RUNPOD_API_KEY=...        # RunPod → Settings → API Keys
    export RUNPOD_ENDPOINT_ID=...    # endpoint sahifasida ko'rinadi
    python examples/runpod_client.py audio.wav

DIQQAT: RunPod Serverless API'si bizning FastAPI'dan farq qiladi —
u multipart fayl emas, JSON qabul qiladi va audio base64 ko'rinishida
yuboriladi.
"""

import base64
import os
import sys

import requests

API_KEY = os.environ.get("RUNPOD_API_KEY")
ENDPOINT_ID = os.environ.get("RUNPOD_ENDPOINT_ID")


def transcribe(audio_path: str) -> str:
    if not API_KEY or not ENDPOINT_ID:
        raise RuntimeError("RUNPOD_API_KEY va RUNPOD_ENDPOINT_ID o'rnatilishi shart")

    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    # /runsync — javobni kutib turadi (qisqa audiolar uchun qulay).
    # Uzun audiolar uchun /run ishlatib, keyin /status/{id} orqali
    # natijani so'rab olish tavsiya etiladi.
    resp = requests.post(
        f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={"input": {"audio_base64": audio_b64}},
        timeout=600,
    )
    resp.raise_for_status()
    data = resp.json()

    # RunPod javobni {"status": "COMPLETED", "output": {...}} ko'rinishida qaytaradi
    output = data.get("output") or {}
    if output.get("status") != "success":
        raise RuntimeError(f"Xato: {output.get('message') or data}")
    return output["text"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Foydalanish: python examples/runpod_client.py audio.wav")
    print(transcribe(sys.argv[1]))

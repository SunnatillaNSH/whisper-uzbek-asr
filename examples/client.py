"""Python mijoz: python examples/client.py audio.wav
API_URL muhit o'zgaruvchisi orqali ngrok manzilini bering."""
import os
import sys
import requests

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")


def transcribe(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        resp = requests.post(
            f"{API_URL}/transcribe",
            files={"file": (os.path.basename(audio_path), f)},
            headers={"ngrok-skip-browser-warning": "true"},
            timeout=300,
        )
    data = resp.json()
    if data.get("status") != "success":
        raise RuntimeError(f"API xatosi ({resp.status_code}): {data.get('message')}")
    return data["text"]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("Foydalanish: python examples/client.py audio.wav")
    print(transcribe(sys.argv[1]))

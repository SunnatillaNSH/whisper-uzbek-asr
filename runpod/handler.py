# ==============================================================
# RunPod Serverless handler — fine-tuned Whisper modeli uchun
# ==============================================================
# DIQQAT: bu fayl app.py (FastAPI) dan FARQ QILADI. RunPod Serverless
# HTTP server emas, balki `handler(job)` funksiyasini kutadi: platforma
# so'rovni qabul qilib, uni shu funksiyaga uzatadi va javobni qaytaradi.
#
# Kirish (job["input"]) uchun uchta variant qo'llab-quvvatlanadi:
#   {"audio_base64": "<base64>"}        — audio fayl base64 ko'rinishida
#   {"audio_url": "https://..."}        — audio faylga havola
#   {"audio_base64": "...", "language": "uzbek"}  — tilni majburiy belgilash
#
# Javob:
#   {"status": "success", "text": "...", "duration_sec": 5.2,
#    "processing_time_sec": 0.8}
#   yoki {"status": "error", "message": "..."}

import base64
import io
import os
import subprocess
import tempfile
import time

import librosa
import numpy as np
import requests
import soundfile as sf
import torch
from transformers import pipeline

import runpod

# ---------------- Sozlamalar ----------------
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
LANGUAGE = os.environ.get("ASR_LANGUAGE", "uzbek")
SAMPLING_RATE = 16000
MAX_AUDIO_MB = int(os.environ.get("MAX_AUDIO_MB", "50"))

# ---------------- Modelni bir marta yuklash ----------------
# Bu kod konteyner ishga tushganda (cold start) BIR MARTA bajariladi.
# Keyingi barcha so'rovlar allaqachon yuklangan modeldan foydalanadi.
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[handler] Model yuklanmoqda: {MODEL_DIR} (device={DEVICE}, dtype={DTYPE})")
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    device=DEVICE,
    torch_dtype=DTYPE,
    chunk_length_s=30,   # 30 soniyadan uzun audio avtomatik bo'laklanadi
)
print("[handler] Model tayyor")


def _decode_audio(raw: bytes) -> np.ndarray:
    """Audio baytlarini 16 kHz mono float32 massivga o'giradi.

    Avval soundfile bilan urinadi (wav, flac, ogg, ko'pincha mp3 ham).
    Agar u o'qiy olmasa (m4a, aac, webm kabi formatlar), ffmpeg orqali
    wav'ga o'giriladi."""
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, "input")
            dst = os.path.join(tmp, "out.wav")
            with open(src, "wb") as f:
                f.write(raw)
            subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                 "-ac", "1", "-ar", str(SAMPLING_RATE), dst],
                check=True,
            )
            audio, sr = sf.read(dst, dtype="float32")

    if audio.ndim > 1:            # stereo → mono
        audio = audio.mean(axis=1)
    if sr != SAMPLING_RATE:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
    return audio


def _get_audio_bytes(inp: dict) -> bytes:
    """job["input"] ichidan audio baytlarini oladi."""
    if inp.get("audio_base64"):
        return base64.b64decode(inp["audio_base64"])

    if inp.get("audio_url"):
        resp = requests.get(inp["audio_url"], timeout=120)
        resp.raise_for_status()
        return resp.content

    raise ValueError("'audio_base64' yoki 'audio_url' berilishi shart")


def handler(job):
    started = time.time()
    try:
        inp = job.get("input") or {}
        raw = _get_audio_bytes(inp)

        size_mb = len(raw) / (1024 * 1024)
        if size_mb > MAX_AUDIO_MB:
            return {
                "status": "error",
                "message": f"Fayl juda katta ({size_mb:.1f} MB). Limit: {MAX_AUDIO_MB} MB",
            }

        audio = _decode_audio(raw)
        if audio is None or len(audio) == 0:
            return {"status": "error", "message": "Audio bo'sh yoki o'qib bo'lmadi"}

        result = asr(
            {"raw": audio, "sampling_rate": SAMPLING_RATE},
            generate_kwargs={
                "language": inp.get("language", LANGUAGE),
                "task": "transcribe",
            },
        )

        return {
            "status": "success",
            "text": result["text"].strip(),
            "duration_sec": round(len(audio) / SAMPLING_RATE, 2),
            "processing_time_sec": round(time.time() - started, 2),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


runpod.serverless.start({"handler": handler})

# ==============================================================
# 3.1 FASTAPI ILOVASI — fine-tuned Whisper modelini API sifatida xizmat qilish
# ==============================================================
import os
import time
import shutil
import tempfile
import threading
import subprocess

import torch
import librosa
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import pipeline

# ---------------- Sozlamalar ----------------
MODEL_DIR     = os.environ.get("MODEL_DIR", "/content/whisper-small-uz")
LANGUAGE      = os.environ.get("ASR_LANGUAGE", "uzbek")
SAMPLING_RATE = 16000
MAX_FILE_MB   = 50
ALLOWED_EXT   = {".wav", ".mp3", ".m4a", ".ogg", ".oga", ".flac", ".webm", ".opus", ".aac"}

# ---------------- Modelni bir marta yuklash ----------------
DEVICE = 0 if torch.cuda.is_available() else -1
DTYPE  = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"[app] Model yuklanmoqda: {MODEL_DIR} (device={DEVICE}, dtype={DTYPE})")
asr = pipeline(
    task="automatic-speech-recognition",
    model=MODEL_DIR,
    device=DEVICE,
    torch_dtype=DTYPE,
    chunk_length_s=30,      # 30 soniyadan uzun audiolarni bo'laklab transkripsiya qiladi
)
print("[app] Model tayyor")

# GPU'da bir vaqtda faqat bitta inference bo'lishi uchun qulf
_infer_lock = threading.Lock()

# ---------------- FastAPI ----------------
app = FastAPI(
    title="Uzbek Whisper ASR API",
    description="Fine-tuned Whisper modeli orqali o'zbek tilidagi audioni matnga o'girish",
    version="1.0.0",
)

# Boshqa domenlardan (frontend, mobil ilova) so'rov yuborishga ruxsat
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Barcha xatolarni bir xil JSON formatda qaytaramiz: {"status": "error", "message": "..."}
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content={"status": "error", "message": exc.detail})

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"status": "error", "message": str(exc)})


def load_audio(path: str):
    """Audio faylni 16 kHz mono numpy massiviga o'qiydi.
    librosa o'qiy olmasa (masalan webm/m4a), ffmpeg orqali wav'ga o'giradi."""
    try:
        audio, _ = librosa.load(path, sr=SAMPLING_RATE, mono=True)
        return audio
    except Exception:
        wav_path = path + ".16k.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", path,
             "-ac", "1", "-ar", str(SAMPLING_RATE), wav_path],
            check=True,
        )
        try:
            audio, _ = librosa.load(wav_path, sr=SAMPLING_RATE, mono=True)
            return audio
        finally:
            if os.path.exists(wav_path):
                os.remove(wav_path)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": MODEL_DIR,
        "device": "cuda" if DEVICE == 0 else "cpu",
        "language": LANGUAGE,
    }


# `async def` emas, `def` — FastAPI uni threadpool'da ishga tushiradi va
# og'ir GPU inference event loop'ni bloklamaydi.
@app.post("/transcribe")
def transcribe(file: UploadFile = File(...)):
    filename = file.filename or "audio"
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXT:
        raise HTTPException(
            status_code=400,
            detail=f"Qo'llab-quvvatlanmaydigan format '{ext}'. Ruxsat etilgan: {sorted(ALLOWED_EXT)}",
        )

    tmp_path = None
    try:
        # 1) Faylni vaqtinchalik saqlaymiz
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        if size_mb > MAX_FILE_MB:
            raise HTTPException(status_code=413, detail=f"Fayl juda katta ({size_mb:.1f} MB). Limit: {MAX_FILE_MB} MB")

        # 2) Audio → 16 kHz massiv
        audio = load_audio(tmp_path)
        if audio is None or len(audio) == 0:
            raise HTTPException(status_code=400, detail="Audio bo'sh yoki o'qib bo'lmadi")

        # 3) Transkripsiya (til va vazifa majburiy belgilanadi)
        t0 = time.time()
        with _infer_lock:
            result = asr(
                {"raw": audio, "sampling_rate": SAMPLING_RATE},
                generate_kwargs={"language": LANGUAGE, "task": "transcribe"},
            )
        elapsed = time.time() - t0

        # 4) JSON javob
        return {
            "status": "success",
            "text": result["text"].strip(),
            "filename": filename,
            "duration_sec": round(len(audio) / SAMPLING_RATE, 2),
            "processing_time_sec": round(elapsed, 2),
        }
    finally:
        # 5) Vaqtinchalik faylni tozalaymiz
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

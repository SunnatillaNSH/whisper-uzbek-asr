# ==============================================================
# RunPod Serverless handler — faster-whisper (CTranslate2)
# ==============================================================
# `handler.py` (transformers pipeline) o'rniga. Farqi:
#   * CTranslate2 dvigateli — bir xil model, bir necha barobar tez.
#     Serverless soniyabay to'lanadi, ya'ni tezlik to'g'ridan-to'g'ri pul.
#   * torch kerak emas — obraz ~8 GB dan ~2 GB ga tushadi, sovuq start qisqaradi.
#   * VAD, takrorlash himoyasi va gallyutsinatsiya darvozalari mavjud.
#
# Kirish (job["input"]):
#   {"audio_base64": "<base64>"}  yoki  {"audio_url": "https://..."}
#   ixtiyoriy: {"language": "uz", "beam_size": 5, "vad": true, "segments": true}
#
# Javob:
#   {"status":"success","text":"...","duration_sec":..,"processing_time_sec":..,
#    "realtime_factor":..}

import base64
import io
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import requests
import soundfile as sf
from faster_whisper import WhisperModel

import runpod

# ---------------- Sozlamalar ----------------
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model")
LANGUAGE = os.environ.get("ASR_LANGUAGE", "uz")          # CT2 ISO kodini kutadi
COMPUTE_TYPE = os.environ.get("ASR_COMPUTE_TYPE", "float16")
BEAM_SIZE = int(os.environ.get("ASR_BEAM_SIZE", "5"))

# Domen lug'ati — STANDART HOLDA O'CHIQ.
#
# Dastlab u standart qilib qo'yilgandi: mantiq shundaki, "Face ID" kabi
# atamalar umumiy nutqda kam uchraydi va model ularni almashtirib yuboradi.
# Mantiq to'g'ri edi, natija esa teskari chiqdi.
#
# Bir xil 60 namunada o'lchandi:
#     hotwords yoqiq    WER 48.27%   so'z qamrovi 88.2%
#     hotwords o'chiq   WER 44.38%   so'z qamrovi 95.1%
#
# Sababi: faster-whisper hotwords'ni prompt sifatida beradi, ya'ni dekoderni
# ro'yxatdagi so'zlar tomon og'diradi. Ro'yxat qo'ng'iroq mazmuniga mos
# kelmasa — va 30 soniyalik bo'lakda ko'pincha mos kelmaydi — model eshitgan
# so'zini tashlab, kutilayotgan so'zga tortiladi yoki umuman chiqarmaydi.
#
# Foydasi ham bor: gallyutsinatsiya kamayadi (insertion 195 → 116). Lekin
# evaziga haqiqiy so'zlar yo'qoladi, sof hisobda zarari kattaroq.
#
# Kerak bo'lganda so'rovda berilsin: {"hotwords": "Face ID, davomat dasturi"}.
# Butun trafik uchun majburiy standart sifatida emas.
SUGGESTED_HOTWORDS = (
    "Face ID, davomat dasturi, xodim, xodimlar, apparat, kontrol, "
    "oylik to'lov, shartnoma, buxgalteriya, o'rnatish, ro'yxatdan o'tkazish, "
    "million so'm, dollar, plyus"
)
HOTWORDS = os.environ.get("ASR_HOTWORDS", "")
INITIAL_PROMPT = os.environ.get("ASR_INITIAL_PROMPT", "")
SAMPLING_RATE = 16000
MAX_AUDIO_MB = int(os.environ.get("MAX_AUDIO_MB", "50"))

# ---------------- Modelni bir marta yuklash ----------------
# Model MAXFIY HF repo'sida turadi, RunPod esa образ qurayotganda token
# bera olmaydi (deploy interfeysida build-argument maydoni yo'q). Shuning
# uchun model образga "pishirilmaydi", balki konteyner ishga tushganda
# yuklab olinadi. Token RunPod endpoint sozlamalaridagi HF_TOKEN dan olinadi.
#
# Bu faqat SOVUQ START da sodir bo'ladi (~30 s, RunPod tarmog'i tez).
# Keyingi barcha so'rovlar allaqachon yuklangan modeldan foydalanadi.
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "")
if HF_MODEL_ID and not os.path.isdir(MODEL_DIR):
    from huggingface_hub import snapshot_download
    print(f"[handler] Model yuklab olinmoqda: {HF_MODEL_ID}", flush=True)
    t0 = time.time()
    snapshot_download(HF_MODEL_ID, local_dir=MODEL_DIR,
                      token=os.environ.get("HF_TOKEN") or None)
    print(f"[handler] Yuklab olindi ({time.time() - t0:.0f} s)", flush=True)

print(f"[handler] Model yuklanmoqda: {MODEL_DIR} ({COMPUTE_TYPE})", flush=True)
model = WhisperModel(MODEL_DIR, device="cuda", compute_type=COMPUTE_TYPE)
print("[handler] Model tayyor", flush=True)

# Whisper jim yoki shovqinli qismlarda o'zidan matn "eshitadi". VAD buni
# kamaytiradi, lekin butunlay yo'qotmaydi — quyidagi iboralar trening
# ma'lumotidan kelib chiqqan qoldiqlar bo'lib, qo'ng'iroq matnida uchramaydi.
HALLUCINATIONS = [
    re.compile(r"^\W*(obuna bo['‘’]?ling[^.!?]*)[.!?]?\W*$", re.I),
    re.compile(r"^\W*(subscribe|thanks? for watching)[^.!?]*[.!?]?\W*$", re.I),
    re.compile(r"^\W*(продолжение следует)[^.!?]*[.!?]?\W*$", re.I),
]


def _clean(text: str) -> str:
    """Takroriy va gallyutsinatsion segmentlarni chiqaradi."""
    parts, out, prev = re.split(r"(?<=[.!?])\s+", text), [], None
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if any(rx.match(p) for rx in HALLUCINATIONS):
            continue
        # Bir xil jumla ketma-ket takrorlansa (Whisper halqasi) — bittasini qoldiramiz
        if prev is not None and p.lower() == prev.lower():
            continue
        out.append(p)
        prev = p
    return " ".join(out).strip()


def _decode_audio(raw: bytes) -> np.ndarray:
    """Audio baytlarini 16 kHz mono float32 massivga o'giradi.

    soundfile wav/flac/ogg va ko'pincha mp3 ni o'qiydi; m4a/aac/webm uchun
    ffmpeg zaxira yo'l sifatida ishlatiladi.
    """
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst = os.path.join(tmp, "in"), os.path.join(tmp, "out.wav")
            with open(src, "wb") as f:
                f.write(raw)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                            "-ac", "1", "-ar", str(SAMPLING_RATE), dst], check=True)
            audio, sr = sf.read(dst, dtype="float32")

    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SAMPLING_RATE:
        # Faqat resampling uchun librosa/torch olib kelmaymiz — oddiy chiziqli
        # interpolyatsiya yetarli, chunki manba deyarli har doim 16 kHz.
        n = int(round(len(audio) * SAMPLING_RATE / sr))
        audio = np.interp(np.linspace(0, len(audio) - 1, n),
                          np.arange(len(audio)), audio).astype(np.float32)
    return np.ascontiguousarray(audio, dtype=np.float32)


def _get_audio_bytes(inp: dict) -> bytes:
    if inp.get("audio_base64"):
        return base64.b64decode(inp["audio_base64"])
    if inp.get("audio_url"):
        # User-Agent majburiy: ba'zi saytlar kutubxonaning standart
        # "python-requests/..." UA'sini bloklab 403 qaytaradi.
        resp = requests.get(
            inp["audio_url"], timeout=300, allow_redirects=True,
            headers={"User-Agent": "whisper-uzbek-asr/2.0 "
                                   "(+https://github.com/SunnatillaNSH/whisper-uzbek-asr)"},
        )
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
            return {"status": "error",
                    "message": f"Fayl juda katta ({size_mb:.1f} MB). Limit: {MAX_AUDIO_MB} MB"}

        audio = _decode_audio(raw)
        if audio is None or len(audio) == 0:
            return {"status": "error", "message": "Audio bo'sh yoki o'qib bo'lmadi"}

        # So'rov darajasida bekor qilish mumkin; bo'sh satr = o'chirish
        hot = inp.get("hotwords", HOTWORDS) or None
        prompt = inp.get("initial_prompt", INITIAL_PROMPT) or None
        if prompt:
            hot = None          # faster-whisper ikkalasini birga qabul qilmaydi

        segments, info = model.transcribe(
            audio,
            language=inp.get("language", LANGUAGE),
            task="transcribe",
            beam_size=int(inp.get("beam_size", BEAM_SIZE)),
            hotwords=hot,
            initial_prompt=prompt,
            vad_filter=bool(inp.get("vad", True)),
            vad_parameters=dict(
                # Standart qiymatlar (2000 / 400). Avval bu yerda 500 / 200
                # turgandi — ya'ni VAD standartdan to'rt barobar agressiv edi
                # va pauzalar orasidagi qisqa tasdiq so'zlarini nutq emas deb
                # kesib tashlayotgandi.
                #
                # 254 qo'ng'iroqda o'lchandi: xatolarning 44.3% i tushib
                # qolish (odatdagi ASR'da 15-20%), model Muxlisa'dan 19% kam
                # so'z chiqarardi. VAD butunlay o'chirilganda so'z qamrovi
                # 80.7% dan 87.8% ga ko'tarildi, lekin jimlikdagi
                # gallyutsinatsiya 1.6 barobar oshdi — shuning uchun
                # o'chirish emas, standart ostonaga qaytarish.
                min_silence_duration_ms=int(os.environ.get("VAD_MIN_SILENCE_MS", "2000")),
                speech_pad_ms=int(os.environ.get("VAD_SPEECH_PAD_MS", "400")),
            ),
            # Standart qiymati True va Whisper'ning eng mashhur nuqsonini
            # keltirib chiqaradi: shovqinli joydan keyin model o'z matnini
            # qayta-qayta takrorlash halqasiga tushadi.
            condition_on_previous_text=False,
            # Dekodlash chalkashsa (siqilish nisbati yoki log-ehtimollik
            # chegaradan chiqsa), yuqoriroq temperatura bilan qayta uriniladi.
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
                 "text": s.text.strip()} for s in segments]
        text = _clean(" ".join(s["text"] for s in segs))

        dur = len(audio) / SAMPLING_RATE
        proc = time.time() - started
        out = {
            "status": "success",
            "text": text,
            "duration_sec": round(dur, 2),
            "processing_time_sec": round(proc, 2),
            "realtime_factor": round(dur / proc, 1) if proc > 0 else None,
        }
        if inp.get("segments"):
            out["segments"] = segs
        return out

    except Exception as e:
        return {"status": "error", "message": f"{type(e).__name__}: {e}"}


runpod.serverless.start({"handler": handler})

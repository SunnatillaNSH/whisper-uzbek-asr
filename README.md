# Whisper — O'zbek tili uchun Fine-Tuning + FastAPI + ngrok

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SunnatillaNSH/whisper-uzbek-asr/blob/main/notebooks/whisper_uzbek_finetune_colab.ipynb)

`openai/whisper-large-v3` (eng kuchli Whisper modeli) ni o'zbek tili uchun Google Colab (T4 GPU) da **LoRA (PEFT)** yordamida fine-tune qilish, uni FastAPI orqali `POST /transcribe` API sifatida ishga tushirish va ngrok orqali tashqi loyihalarga ulash uchun to'liq pipeline.

## Tezkor boshlash

1. Yuqoridagi **Open in Colab** tugmasini bosing.
2. `Runtime → Change runtime type` — **A100 GPU** tanlang (T4 ham ishlaydi, lekin sekinroq; pastdagi sozlamalarni moslashtiring).
3. Colab **Secrets** (🔑) ga `NGROK_AUTH_TOKEN` qo'shing.
4. Katakchalarni yuqoridan pastga ishga tushiring — barcha datasetlar avtomatik yuklanadi (pastga qarang), token/login shart emas.

## Dataset — 7 ta ochiq o'zbekcha manba birlashtirilgan

Notebook 1.1-bo'limda quyidagi manbalarni **avtomatik** yuklab, birlashtiradi (barchasi Hugging Face'da ochiq, login/token shart emas):

| Manba | Cheklov |
|---|---:|
| [`yakhyo/mozilla-common-voice-uzbek`](https://huggingface.co/datasets/yakhyo/mozilla-common-voice-uzbek) | 8 000 |
| [`DavronSherbaev/uzbekvoice-filtered`](https://huggingface.co/datasets/DavronSherbaev/uzbekvoice-filtered) | 20 000 |
| [`mrmuminov/uzbek_voice`](https://huggingface.co/datasets/mrmuminov/uzbek_voice) | 30 000 |
| [`shunyalabs/uzbek-speech-dataset`](https://huggingface.co/datasets/shunyalabs/uzbek-speech-dataset) | hammasi (~2 943) |
| [`islomov/news_youtube_uzbek_speech_dataset`](https://huggingface.co/datasets/islomov/news_youtube_uzbek_speech_dataset) | 15 000 |
| [`islomov/it_youtube_uzbek_speech_dataset`](https://huggingface.co/datasets/islomov/it_youtube_uzbek_speech_dataset) | 10 000 |
| [`BoburAmirov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset`](https://huggingface.co/datasets/BoburAmirov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset) | hammasi (~14 547) |

Jami cheklov bo'yicha ~100 000 namuna. Har biri avtomatik yuklanadi, sifat filtri (up/down vote, mavjud bo'lsa) va bo'sh matnli qatorlarni chiqarib tashlash qo'llaniladi, audio `.wav` (16 kHz) sifatida `/content/data/audio/` ga, transkriptlar `/content/data/train.csv` ga yoziladi. `SOURCES` ro'yxatidagi har bir manbaning cheklovini o'zgartirish mumkin.

**O'z datasetingiz bilan ishlashni istasangiz** (masalan real qo'ng'iroq yozuvlari), 1.1-katakchani o'tkazib yuboring va `train.csv` + `audio/` ni qo'lda tayyorlang — format pastda.

## Repo tarkibi

| Fayl | Vazifasi |
|---|---|
| `notebooks/whisper_uzbek_finetune_colab.ipynb` | Asosiy notebook: setup → fine-tuning → API → ngrok |
| `app.py` | FastAPI ilovasi (Colab'dan tashqarida ham ishlaydi: `uvicorn app:app --port 8000`) |
| `requirements.txt` | Kutubxonalar |
| `data/train.csv.example` | Dataset formati namunasi |
| `examples/curl.sh` | cURL mijoz |
| `examples/client.py` | Python `requests` mijoz |
| `examples/client.js` | Node.js / Cloudflare Workers `fetch` mijoz |
| `examples/browser.html` | Brauzer `fetch` + fayl tanlash misoli |

## Dataset formati

```
data/
├── train.csv        # ustunlar: path,sentence
└── audio/
    ├── 0001.wav
    └── 0002.mp3
```

```csv
path,sentence
0001.wav,Assalomu alaykum, qanday yordam bera olaman?
0002.mp3,Buyurtmangiz ertaga yetkazib beriladi.
```

`path` — `audio/` ga nisbatan yoki absolyut yo'l. Har bir audio 30 soniyadan qisqa bo'lishi kerak.

## API

| Method | Path | Body | Javob |
|---|---|---|---|
| `GET` | `/health` | — | `{"status":"ok","model":"...","device":"cuda"}` |
| `POST` | `/transcribe` | `multipart/form-data`, field **`file`** | `{"status":"success","text":"...","duration_sec":5.2,"processing_time_sec":0.8}` |

Xato: `{"status":"error","message":"..."}` (HTTP 400 / 413 / 500).
Formatlar: wav, mp3, m4a, ogg, flac, webm, opus, aac. Limit: 50 MB.

```bash
API_URL=https://xxxx.ngrok-free.app ./examples/curl.sh audio.wav
API_URL=https://xxxx.ngrok-free.app python examples/client.py audio.wav
API_URL=https://xxxx.ngrok-free.app node examples/client.js audio.wav
```

## Serverni Colab'siz ishga tushirish

```bash
pip install -r requirements.txt
MODEL_DIR=/path/to/whisper-large-v3-uz uvicorn app:app --host 0.0.0.0 --port 8000
```

## Trening sozlamalari (LoRA)

| Parametr | Qiymat |
|---|---|
| Model | `whisper-large-v3` (1.5 mlrd parametr) |
| Usul | LoRA (PEFT) — asosiy model muzlatilgan, faqat `q_proj`/`v_proj` adapterlari (~0.3% parametr) o'qitiladi |
| LoRA r / alpha / dropout | 32 / 64 / 0.05 |
| Batch × accum (A100) | 8 × 2 = 16 — T4'da 2 × 8 = 16 ga qaytaring |
| fp16 (autocast) + gradient checkpointing | yoqilgan |
| Learning rate / warmup | 1e-4 / 500 |
| max_steps / eval_steps | 10000 / 500 |
| Eval baholash | faqat **loss** orqali (predict_with_generate ishlatilmaydi — PEFT bilan mos kelmaydigan ma'lum xato beradi). Yakuniy WER trening tugagach qo'lda hisoblanadi |

To'liq fine-tuning large-v3 uchun T4'ga sig'maydi (optimizator holati ~25 GB talab qiladi) — shuning uchun LoRA ishlatiladi. Trening tugagach adapterlar asosiy modelga birlashtiriladi (`merge_and_unload`), fp16'ga o'tkaziladi va oddiy Whisper modeli sifatida saqlanadi — `app.py` hech qanday qo'shimcha o'zgarishsiz ishlayveradi.

**Vaqt**: A100'da 10000 qadam taxminan 5-6 soat davom etadi (T4'da bir necha baravar sekinroq). Colab Pro sessiyasi ~24 soatgacha, bepul versiya ~12 soatdan keyin uziladi. Uzilsa: `trainer.train(resume_from_checkpoint=True)`. Modelni Drive'ga nusxalash katakchasi notebook ichida bor — uzoq trening uchun buni albatta oching.

**Natija (birinchi urinish, ~23 ming namuna bilan):** WER 34.01%. Aniq/formal nutqda sifat yaxshi, real qo'ng'iroq audiosida (shovqin, tabiiy nutq) sezilarli xatolar bor edi (masalan ism nomuvofiqligi). Sabab topildi va tuzatildi: dataset yig'ish kodida ba'zi manbalarning ko'p qatori matn topilmagani sabab jim tashlab yuborilar edi — endi bu tuzatilgan va diagnostika qo'shilgan (`Yozildi: X | O'tkazib yuborildi: Y`).

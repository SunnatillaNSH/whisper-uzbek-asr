# Whisper — O'zbek tili uchun Fine-Tuning + FastAPI + ngrok

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SunnatillaNSH/whisper-uzbek-asr/blob/main/notebooks/whisper_uzbek_finetune_colab.ipynb)

`openai/whisper-small` modelini o'zbek tili uchun Google Colab (T4 GPU) da fine-tune qilish, uni FastAPI orqali `POST /transcribe` API sifatida ishga tushirish va ngrok orqali tashqi loyihalarga ulash uchun to'liq pipeline.

## Tezkor boshlash

1. Yuqoridagi **Open in Colab** tugmasini bosing.
2. `Runtime → Change runtime type → T4 GPU` tanlang.
3. Colab **Secrets** (🔑) ga `NGROK_AUTH_TOKEN` qo'shing.
4. Katakchalarni yuqoridan pastga ishga tushiring — dataset avtomatik yuklanadi (pastga qarang), token/login shart emas.

## Dataset — Mozilla Common Voice (o'zbek)

Notebook 1.1-bo'limda [`yakhyo/mozilla-common-voice-uzbek`](https://huggingface.co/datasets/yakhyo/mozilla-common-voice-uzbek) datasetini **avtomatik** yuklaydi — bu Mozilla Common Voice loyihasining o'zbekcha qismi, Hugging Face'da ochiq (login/token shart emas):

- `validated` split — odamlar tomonidan tasdiqlangan yozuvlar (~86 ming qator)
- Sifat filtri (`up_votes >= 1`, `down_votes == 0`) va `MAX_SAMPLES` (standart: 8000) orqali hajm cheklanadi
- Audio `.wav` (16 kHz) sifatida `/content/data/audio/` ga, transkriptlar `/content/data/train.csv` ga yoziladi

`MAX_SAMPLES` ni oshirsangiz sifat yaxshilanadi, trening vaqti uzayadi. Litsenziya: Common Voice yozuvlari CC0 (public domain) ostida tarqatiladi.

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
MODEL_DIR=/path/to/whisper-small-uz uvicorn app:app --host 0.0.0.0 --port 8000
```

## T4 uchun trening sozlamalari

| Parametr | Qiymat |
|---|---|
| Model | `whisper-small` (`medium` uchun batch 4, accum 4) |
| Batch × accum | 8 × 2 = 16 |
| fp16 + gradient checkpointing | yoqilgan |
| Learning rate / warmup | 1e-5 / 100 |
| max_steps / eval_steps | 2000 / 250 |

Colab uzilsa: `trainer.train(resume_from_checkpoint=True)`. Modelni Drive'ga nusxalash katakchasi notebook ichida bor.

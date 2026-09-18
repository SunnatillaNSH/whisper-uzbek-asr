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

## RunPod Pod'da trening (Colab o'rniga)

Colab'da sessiya uzilishi, compute unit tugashi va 235 GB disk cheklovi bor. RunPod Pod'da bularning hech biri yo'q — sessiya cheksiz, diskni o'zingiz tanlaysiz.

| Fayl | Vazifasi |
|---|---|
| `train.py` | Mustaqil trening skripti (Colab'ga bog'liq emas) |
| `requirements-train.txt` | Trening kutubxonalari |
| `eval_wer.py` | Tayyor modelning WER'ini hisoblash |
| `upload_to_hf.py` | Modelni Hugging Face Hub'ga yuklash |

### Qadamlar

1. **Pod yarating**: RunPod → Pods → Deploy. Tavsiya: **L40S (48 GB, ~$1.09/soat)** yoki **A40 (48 GB, ~$0.49/soat, sekinroq)**. Template: PyTorch. Volume: **100 GB** (`/workspace`).
2. **Ulaning** (web terminal yoki SSH) va tayyorlang:
   ```bash
   git clone https://github.com/SunnatillaNSH/whisper-uzbek-asr.git
   cd whisper-uzbek-asr
   pip install -r requirements-train.txt
   ```
3. **Ishga tushiring** (uzilishdan himoyalanish uchun `nohup` bilan):
   ```bash
   nohup python train.py > /workspace/train.log 2>&1 &
   tail -f /workspace/train.log
   ```
4. **Uzilib qolsa** — checkpoint'lar `/workspace` da qoladi:
   ```bash
   python train.py --resume
   ```
5. **Baholash va yuklash**:
   ```bash
   MODEL_DIR=/workspace/whisper-large-v3-uz python eval_wer.py
   HF_TOKEN=hf_... HF_REPO=<foydalanuvchi>/whisper-large-v3-uz-v2 python upload_to_hf.py
   ```

### Sozlamalar (muhit o'zgaruvchilari)

```bash
MAX_STEPS=10000 BATCH_SIZE=16 GRAD_ACCUM=1 LR=1e-4 python train.py
```

Standart manbalar — **tabiiy suhbat** nutqi (Toshkent podkastlari + YouTube yangiliklari, ~35 000 namuna). O'qib yozdirilgan datasetlar (Common Voice, UzbekVoice) `train.py` ichidagi `SOURCES` ro'yxatida izohlangan holda turadi.

**Nima uchun aynan shunday:** qo'ng'iroq tahlili uchun tabiiy, erkin suhbat nutqi o'qib yozdirilgan toza nutqdan muhimroq. UzbekVoice va Common Voice — odamlar jumlalarni mikrofonga o'qib bergan yozuvlar; telefon qo'ng'irog'i esa shovqinli, siqilgan va tez. Datasetni kattalashtirish o'zi sifatni oshirmaydi — **mos turdagi** ma'lumot kerak.

## RunPod Serverless'ga joylashtirish

Colab + ngrok — sinov uchun. Doimiy ishlashi uchun model RunPod Serverless'ga joylashtiriladi: so'rov kelganda konteyner uyg'onadi, bo'sh turganda to'lov yo'q.

| Fayl | Vazifasi |
|---|---|
| `handler.py` | RunPod Serverless handler (FastAPI emas — platforma `handler(job)` funksiyasini chaqiradi) |
| `Dockerfile` | Konteyner образi; `HF_MODEL_ID` build-arg berilsa, modelni образ ichiga "pishiradi" |
| `requirements-runpod.txt` | Konteyner kutubxonalari |
| `examples/runpod_client.py` | Endpoint'ga so'rov yuborish misoli (Python) |
| `examples/runpod_worker.js` | Cloudflare Worker'dan chaqirish misoli (R2 va to'g'ridan-to'g'ri havola) |

### Qadamlar

1. **Modelni Hugging Face Hub'ga yuklang** (Colab'da, trening tugagach):
   ```python
   from huggingface_hub import HfApi
   HfApi().create_repo("<foydalanuvchi>/whisper-large-v3-uz", private=True, exist_ok=True)
   HfApi().upload_folder(folder_path=OUTPUT_DIR, repo_id="<foydalanuvchi>/whisper-large-v3-uz")
   ```
2. RunPod → **Serverless → Deploy from a GitHub repository** → shu repo'ni tanlang.
3. Dockerfile yo'li: `/Dockerfile` (repo ildizida), build arg: `HF_MODEL_ID=<foydalanuvchi>/whisper-large-v3-uz`.
   Model private bo'lsa, RunPod Secrets'ga `HF_TOKEN` qo'shing.
4. GPU sifatida **T4** tanlang — model fp16'da ~3 GB VRAM oladi, T4 (16 GB) ortig'i bilan yetadi.

### So'rov yuborish

```bash
export RUNPOD_API_KEY=...
export RUNPOD_ENDPOINT_ID=...
python examples/runpod_client.py audio.wav
```

```bash
curl -X POST "https://api.runpod.ai/v2/$RUNPOD_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"input\": {\"audio_base64\": \"$(base64 -i audio.wav)\"}}"
```

Kirish sifatida `audio_base64` yoki `audio_url` berish mumkin. Javob: `{"status": "success", "text": "...", "duration_sec": ..., "processing_time_sec": ...}`.

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
| max_steps / eval_steps | 6000 / 500 |
| Eval baholash | faqat **loss** orqali (predict_with_generate ishlatilmaydi — PEFT bilan mos kelmaydigan ma'lum xato beradi). Yakuniy WER trening tugagach qo'lda hisoblanadi |

To'liq fine-tuning large-v3 uchun T4'ga sig'maydi (optimizator holati ~25 GB talab qiladi) — shuning uchun LoRA ishlatiladi. Trening tugagach adapterlar asosiy modelga birlashtiriladi (`merge_and_unload`), fp16'ga o'tkaziladi va oddiy Whisper modeli sifatida saqlanadi — `app.py` hech qanday qo'shimcha o'zgarishsiz ishlayveradi.

**Vaqt**: A100'da 6000 qadam taxminan 3-4 soat davom etadi (T4'da bir necha baravar sekinroq). Colab Pro sessiyasi ~24 soatgacha, bepul versiya ~12 soatdan keyin uziladi. Uzilsa: `trainer.train(resume_from_checkpoint=True)`. Modelni Drive'ga nusxalash katakchasi notebook ichida bor — uzoq trening uchun buni albatta oching.

**Tozalash**: notebook boshida "0. Tozalash" bo'limi bor — u oldingi ishga tushirishdan qolgan audio fayllar, `train.csv` va eski modelni o'chiradi, lekin **yuklab olingan datasetlar keshini saqlab qoladi** (qayta yuklamaslik uchun). Hamma narsani, jumladan keshni ham o'chirish kerak bo'lsa, katakcha ichidagi `WIPE_HF_CACHE` ni `True` qiling. Google Drive'dagi checkpoint'larga hech qachon tegilmaydi.

**Xotira va disk**: mel-spektrogrammalar oldindan hisoblab saqlanmaydi — ular trening paytida, har bir batch uchun data collator ichida joyida hisoblanadi. `whisper-large-v3` uchun bitta namunaning spektrogrammasi ~1.5 MB bo'lgani sababli, ~98 000 namunani oldindan hisoblash ~143 GB disk va juda katta RAM talab qilardi (va "session crashed after using all available RAM" xatosiga olib kelardi). Joyida hisoblash bu muammoni butunlay yo'q qiladi va tezlikka ta'sir qilmaydi.

**Disk xavfsizligi**: dataset yig'ish paytida bo'sh joy `MIN_FREE_DISK_GB` (standart 20 GB) dan pastga tushsa, qolgan manbalar o'tkazib yuboriladi va mavjud namunalar bilan trening davom etadi — "No space left on device" bilan yiqilib qolish o'rniga.

**Natija (birinchi urinish, ~23 ming namuna bilan):** WER 34.01%. Aniq/formal nutqda sifat yaxshi, real qo'ng'iroq audiosida (shovqin, tabiiy nutq) sezilarli xatolar bor edi (masalan ism nomuvofiqligi). Sabab topildi va tuzatildi: dataset yig'ish kodida ba'zi manbalarning ko'p qatori matn topilmagani sabab jim tashlab yuborilar edi — endi bu tuzatilgan va diagnostika qo'shilgan (`Yozildi: X | O'tkazib yuborildi: Y`).

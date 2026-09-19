# Whisper O'zbek ASR — API hujjati

RunPod Serverless'da joylashtirilgan o'zbek nutqni matnga o'girish xizmati.
Model real telefon qo'ng'iroqlarida fine-tune qilingan.

| | |
|---|---|
| Endpoint ID | `vwobkifazoyyh1` |
| Bazaviy manzil | `https://api.runpod.ai/v2/vwobkifazoyyh1` |
| Model | `Sunnat0091/whisper-large-v3-uz-calls-ct2` (maxfiy) |
| Dvigatel | faster-whisper 1.2.1 (CTranslate2), GPU, float16 |
| Tezlik | ~19x realtime (60 s audio → ~3 s) |
| Sovuq start | ~25 s (birinchi so'rov) |

---

## 1. Autentifikatsiya

Har bir so'rovda RunPod API kaliti kerak:

```
Authorization: Bearer rpa_XXXXXXXXXXXXXXXX
```

Kalitni RunPod → **Settings → API Keys** dan olasiz. Kalitni mijoz tomonida
(brauzer, mobil ilova) saqlamang — faqat serveringizdan chaqiring.

---

## 2. Ikki usul: `/runsync` va `/run`

| | `/runsync` | `/run` |
|---|---|---|
| Ishlashi | javobni kutib turadi | job ID qaytaradi, natijani keyin olasiz |
| Qachon | audio < 2 daqiqa | uzun audio yoki sovuq start ehtimoli bor |
| Chegara | ~90 s dan uzun kutmaydi | cheklov yo'q |

**Tavsiya:** qo'ng'iroq tahlili uchun `/run` ishlating — sovuq start 25 soniya
olishi mumkin va `/runsync` uzilib qolishi mumkin.

---

## 3. So'rov formati

Hamma narsa `input` obyekti ichida beriladi.

### 3.1. Audio berishning ikki yo'li

**a) Havola orqali (TAVSIYA ETILADI)**

Audio allaqachon internetda bo'lsa (R2, S3, MoyZvonki havolasi) — shu yo'l
eng tejamkor, chunki fayl sizning serveringiz orqali o'tmaydi:

```json
{
  "input": {
    "audio_url": "https://example.com/call-12345.mp3"
  }
}
```

**b) Base64 orqali**

Fayl sizning qo'lingizda bo'lsa:

```json
{
  "input": {
    "audio_base64": "UklGRiQAAABXQVZFZm10..."
  }
}
```

Chegara: 50 MB (`MAX_AUDIO_MB` bilan o'zgartiriladi).

### 3.2. Qo'llab-quvvatlanadigan formatlar

`wav`, `flac`, `ogg`, `mp3`, `m4a`, `aac`, `webm`, `opus`.

Har qanday chastota va kanal soni qabul qilinadi — xizmat o'zi 16 kHz mono
ga o'giradi. Qayta o'girish kerak emas.

### 3.3. Ixtiyoriy parametrlar

| Parametr | Standart | Tavsifi |
|---|---|---|
| `language` | `"uz"` | ISO kod. Ruscha uchun `"ru"`. |
| `beam_size` | `5` | Qidiruv kengligi. `1` — ~20% tez, ~3 punkt yomonroq. |
| `vad` | `true` | Jimlik va shovqinni dekodlashdan oldin olib tashlash. |
| `segments` | `false` | `true` bo'lsa, vaqt belgili bo'laklar ham qaytariladi. |
| `hotwords` | domen lug'ati | Modelga "kutilayotgan" atamalarni bildiradi. Bo'sh satr — o'chiradi. |
| `initial_prompt` | — | Berilsa, `hotwords` o'rniga ishlatiladi. |

**`hotwords` haqida.** Standart qiymat mavjud transkriptlardagi chastota
bo'yicha tuzilgan: `Face ID, davomat dasturi, xodim, apparat, kontrol,
shartnoma, buxgalteriya, oylik to'lov, million so'm, dollar`.

Bu so'zlar umumiy o'zbek nutqida kam uchraydi, shuning uchun model ularni
tovushi o'xshash boshqa so'zlarga almashtirib yuboradi (`Face ID` →
`Besaklik`). Ro'yxat ularni oldindan "kutilayotgan" deb belgilaydi.

Boshqa domenda ishlatsangiz — o'z atamalaringizni bering:

```json
{
  "input": {
    "audio_url": "...",
    "hotwords": "Click, Payme, Uzum, karta raqami, tranzaksiya"
  }
}
```

---

## 4. Javob formati

### Muvaffaqiyatli

```json
{
  "status": "success",
  "text": "Allo, assalomu alaykum. Yaxshimisiz, aka?",
  "duration_sec": 56.96,
  "processing_time_sec": 2.99,
  "realtime_factor": 19.1
}
```

`segments: true` berilgan bo'lsa, qo'shimcha:

```json
{
  "segments": [
    {"start": 0.0,  "end": 3.2,  "text": "Allo, assalomu alaykum."},
    {"start": 3.4,  "end": 6.1,  "text": "Yaxshimisiz, aka?"}
  ]
}
```

### Xato

```json
{
  "status": "error",
  "message": "ValueError: 'audio_base64' yoki 'audio_url' berilishi shart"
}
```

HTTP kodi bunda ham **200** bo'ladi — xatoni `status` maydonidan tekshiring,
HTTP kodidan emas.

---

## 5. Misollar

### 5.1. cURL — sinxron

```bash
curl -X POST https://api.runpod.ai/v2/vwobkifazoyyh1/runsync \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"audio_url":"https://example.com/call.mp3"}}'
```

### 5.2. cURL — asinxron

```bash
# 1) Yuborish
JOB=$(curl -s -X POST https://api.runpod.ai/v2/vwobkifazoyyh1/run \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"audio_url":"https://example.com/call.mp3"}}' | jq -r .id)

# 2) Natijani olish
curl -s https://api.runpod.ai/v2/vwobkifazoyyh1/status/$JOB \
  -H "Authorization: Bearer $RUNPOD_API_KEY" | jq
```

### 5.3. JavaScript / Cloudflare Workers

```js
const BASE = "https://api.runpod.ai/v2/vwobkifazoyyh1";

async function transcribe(env, audioUrl) {
  // 1) Job yuboramiz
  const start = await fetch(`${BASE}/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input: { audio_url: audioUrl } }),
  }).then((r) => r.json());

  // 2) Tayyor bo'lguncha kuzatamiz
  for (let i = 0; i < 120; i++) {
    const res = await fetch(`${BASE}/status/${start.id}`, {
      headers: { Authorization: `Bearer ${env.RUNPOD_API_KEY}` },
    }).then((r) => r.json());

    if (res.status === "COMPLETED") {
      if (res.output?.status !== "success") {
        throw new Error(res.output?.message || "noma'lum xato");
      }
      return res.output.text;
    }
    if (["FAILED", "CANCELLED", "TIMED_OUT"].includes(res.status)) {
      throw new Error(`Job ${res.status}`);
    }
    await new Promise((r) => setTimeout(r, 2000));
  }
  throw new Error("Vaqt tugadi");
}
```

### 5.4. Python

```python
import os, time, requests

BASE = "https://api.runpod.ai/v2/vwobkifazoyyh1"
H = {"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}"}

def transcribe(audio_url: str, timeout: int = 300) -> str:
    job = requests.post(f"{BASE}/run", headers=H,
                        json={"input": {"audio_url": audio_url}}).json()

    deadline = time.time() + timeout
    while time.time() < deadline:
        res = requests.get(f"{BASE}/status/{job['id']}", headers=H).json()
        if res["status"] == "COMPLETED":
            out = res["output"]
            if out.get("status") != "success":
                raise RuntimeError(out.get("message"))
            return out["text"]
        if res["status"] in ("FAILED", "CANCELLED", "TIMED_OUT"):
            raise RuntimeError(f"Job {res['status']}")
        time.sleep(2)
    raise TimeoutError
```

Lokal fayl uchun `audio_url` o'rniga:

```python
import base64
payload = {"input": {"audio_base64": base64.b64encode(open(path, "rb").read()).decode()}}
```

---

## 6. Xarajat

To'lov **GPU ishlagan soniyalar** uchun. Bo'sh turganda to'lov yo'q.

| | |
|---|---|
| GPU narxi | ~$0.00019 / soniya (24 GB) |
| 60 soniyalik qo'ng'iroq | ~$0.0006 |
| 1000 qo'ng'iroq (o'rtacha 2 daq) | ~$1.20 |
| 60 soat audio | ~$2.15 |

Navbatda kutish (`delayTime`) uchun to'lov olinmaydi — faqat `executionTime`.

---

## 7. Cheklovlar va tavsiyalar

**Sovuq start.** Uzoq vaqt so'rov bo'lmasa, birinchi so'rov ~25 soniya
qo'shimcha oladi: konteyner ko'tariladi va model yuklanadi. Doimiy trafik
bo'lsa, RunPod sozlamalarida **active worker** ni 1 ga qo'ying — sovuq start
yo'qoladi, lekin doimiy to'lov boshlanadi.

**Parallellik.** Standart `max workers = 1`, ya'ni so'rovlar navbatda kutadi.
Ko'p bir vaqtda kelsa, RunPod sozlamalaridan oshiring.

**Audio uzunligi.** Cheklov yo'q — xizmat uzun audioni o'zi bo'laklaydi.
Muxlisa'dan farqli o'laroq, 55 soniyalik bo'laklarga bo'lish **kerak emas**.

**Diarizatsiya yo'q.** Xizmat kim gapirganini ajratmaydi — yagona matn
qaytaradi. Ajratish kerak bo'lsa, matnni LLM'ga bering (`cf-call-analyzer`
dagi `diarizeWithClaude` shunday qiladi).

**Sifat.** Real qo'ng'iroqlarda WER ~43% (Muxlisa transkriptiga nisbatan).
Umumiy ma'no ishonchli yetib keladi, ba'zi so'zlar buziladi. Tafsilotlar
muhim bo'lsa, matnni LLM bilan tuzatib oling.

---

## 8. Model boshqaruvi

Model образ ichida emas — ishga tushishda HF'dan yuklab olinadi. Ya'ni
modelni almashtirish uchun образni qayta qurish **shart emas**:

RunPod → Serverless → **Manage → Edit endpoint → Environment Variables**

| O'zgaruvchi | Vazifasi |
|---|---|
| `HF_MODEL_ID` | Model repo nomi (CT2 formatida bo'lishi shart) |
| `HF_TOKEN` | Maxfiy repo uchun (Read huquqi yetarli) |
| `ASR_BEAM_SIZE` | Standart `5` |
| `ASR_HOTWORDS` | Domen lug'ati |
| `ASR_LANGUAGE` | Standart `uz` |
| `MAX_AUDIO_MB` | Standart `50` |

Yangi model tayyorlash:

```bash
ct2-transformers-converter --model <hf-repo> --output_dir out \
  --quantization float16 \
  --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json \
               vocab.json merges.txt special_tokens_map.json \
               added_tokens.json normalizer.json
```

so'ng natijani HF'ga yuklab, `HF_MODEL_ID` ni o'zgartiring.

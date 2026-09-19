# Whisper O'zbek ASR — API hujjati

RunPod Serverless'da joylashtirilgan o'zbek nutqni matnga o'girish xizmati.
Model real telefon qo'ng'iroqlarida fine-tune qilingan.

| | |
|---|---|
| Endpoint ID | `vwobkifazoyyh1` |
| Bazaviy manzil | `https://api.runpod.ai/v2/vwobkifazoyyh1` |
| Model | `Sunnat0091/whisper-large-v3-uz-calls-ct2` (maxfiy) |
| Dvigatel | faster-whisper 1.2.1 (CTranslate2), GPU, float16 |
| Tezlik | **19.1× realtime** — 57 s audio → 2.99 s (o'lchangan) |
| Sovuq start | ~25 s (birinchi so'rov), keyin 3–4 s |
| Sifat | WER ~43% Muxlisa transkriptiga nisbatan (81 namuna) |

---

## 1. Manzil va autentifikatsiya

Xizmatga ikki yo'l bilan murojaat qilish mumkin. **Tashqi foydalanuvchilar
uchun birinchisi.**

### a) Proksi orqali — tavsiya etiladi

```
https://ovoz-konsoli.sellup2.workers.dev/api
```

Sizga shu xizmatdan foydalanish uchun **alohida kalit** beriladi:

```
Authorization: Bearer <sizga berilgan kalit>
```

RunPod hisobining kaliti sizga berilmaydi va kerak ham emas — u server
tomonida qoladi. Kalitingiz yo'qolsa, faqat o'sha bitta kalit bekor
qilinadi; boshqa foydalanuvchilarga ta'sir qilmaydi.

### b) To'g'ridan-to'g'ri RunPod — faqat endpoint egasi uchun

```
https://api.runpod.ai/v2/vwobkifazoyyh1
```

Bunda RunPod hisobining API kaliti (`rpa_...`) kerak. U butun hisobga kirish
huquqini beradi, shuning uchun tarqatilmaydi.

### Ikkalasi uchun umumiy

Yo'llar, so'rov va javob formatlari **aynan bir xil**. Quyidagi misollarda
`$BASE` va `$API_KEY` o'rniga o'zingiznikini qo'ying.

Kalitni brauzer yoki mobil ilova kodida saqlamang — faqat o'z
serveringizdan chaqiring.

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
BASE=https://ovoz-konsoli.sellup2.workers.dev/api

curl -X POST $BASE/runsync \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"audio_url":"https://example.com/call.mp3"}}'
```

### 5.2. cURL — asinxron

```bash
BASE=https://ovoz-konsoli.sellup2.workers.dev/api

# 1) Yuborish
JOB=$(curl -s -X POST $BASE/run \
  -H "Authorization: Bearer $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"input":{"audio_url":"https://example.com/call.mp3"}}' | jq -r .id)

# 2) Natijani olish
curl -s $BASE/status/$JOB -H "Authorization: Bearer $API_KEY" | jq
```

### 5.3. JavaScript / Cloudflare Workers

```js
const BASE = "https://ovoz-konsoli.sellup2.workers.dev/api";

async function transcribe(env, audioUrl) {
  // 1) Job yuboramiz
  const start = await fetch(`${BASE}/run`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${env.ASR_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ input: { audio_url: audioUrl } }),
  }).then((r) => r.json());

  // 2) Tayyor bo'lguncha kuzatamiz
  for (let i = 0; i < 120; i++) {
    const res = await fetch(`${BASE}/status/${start.id}`, {
      headers: { Authorization: `Bearer ${env.ASR_API_KEY}` },
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

BASE = "https://ovoz-konsoli.sellup2.workers.dev/api"
H = {"Authorization": f"Bearer {os.environ['ASR_API_KEY']}"}

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

## 6. Xatolar

| Holat | Javob | Nima qilish |
|---|---|---|
| Kalit yo'q yoki noto'g'ri | `401` + `{"error":"Kalit kerak yoki noto'g'ri..."}` | Sarlavhani tekshiring: `Authorization: Bearer <kalit>` |
| Noma'lum yo'l | `404` + `{"error":"Noma'lum yo'l"}` | Faqat `/run`, `/status/{id}`, `/health` ochiq |
| Audio berilmagan | `200` + `{"status":"error","message":"'audio_base64' yoki 'audio_url' berilishi shart"}` | `input` ichida bittasini bering |
| Fayl juda katta | `200` + `{"status":"error","message":"Fayl juda katta..."}` | 50 MB dan kichik yuboring |
| Audio ochilmadi | `200` + `{"status":"error","message":"..."}` | Format qo'llab-quvvatlanishini tekshiring |

**Diqqat:** transkripsiya xatolarida HTTP kodi **200** bo'ladi. Xatoni
`status` maydonidan tekshiring, HTTP kodidan emas. 401 va 404 esa haqiqiy
HTTP xatolari.

## 7. Xarajat

To'lov **GPU ishlagan soniyalar** uchun. Bo'sh turganda to'lov yo'q.

| | |
|---|---|
| GPU narxi | ~$0.00019 / soniya (24 GB) |
| 60 soniyalik qo'ng'iroq | ~$0.0006 |
| 1000 qo'ng'iroq (o'rtacha 2 daq) | ~$1.20 |
| 60 soat audio | ~$2.15 |

Navbatda kutish (`delayTime`) uchun to'lov olinmaydi — faqat `executionTime`.

---

## 8. Cheklovlar va tavsiyalar

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

## 9. Model boshqaruvi

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

---

## 10. Mijoz kalitlarini boshqarish (endpoint egasi uchun)

Proksi `API_TOKENS` secret'idagi kalitlarni qabul qiladi — vergul bilan
ajratilgan ro'yxat. Har bir mijozga alohida kalit bering, shunda keraksizini
ro'yxatdan chiqarib tashlash kifoya.

```bash
npx wrangler secret put API_TOKENS --name ovoz-konsoli
# so'ralganda: mijoz1_kaliti,mijoz2_kaliti,mijoz3_kaliti
```

`API_TOKENS` **o'rnatilmagan bo'lsa proksi hamma uchun ochiq** — ya'ni
havolani bilgan har kim transkripsiya qildirib, endpoint egasining pulini
sarflashi mumkin. Hujjatni tashqariga berishdan oldin uni albatta
o'rnating.

Kalit yaratish uchun:

```bash
openssl rand -hex 24
```

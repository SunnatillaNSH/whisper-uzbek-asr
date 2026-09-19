# Agent uchun vazifa — production integratsiyasini yakunlash

Bu `AGENT-BRIEF.md` (xato tahlili) dan **alohida** vazifa. U model sifati
bilan shug'ullanadi, bu esa ishlab turgan tizim bilan.

---

## Hozirgi holat

`cf-call-analyzer` (Cloudflare Worker `qongiroq-tahlili`) ikkita STT
provayderini parallel ishlatadi:

| Provayder | Holati | Qayerda |
|---|---|---|
| **SellUp Voice AI** (o'z modelimiz, RunPod) | standart — har doim yuboriladi | `src/stt/voice.js` |
| **Muxlisa AI** (tijorat) | ixtiyoriy — faqat balansi bo'lsa | `src/stt/muxlisa.js` |

Natijalar **alohida jadvallarda** saqlanadi (`transcripts` va
`voice_transcripts`), ya'ni bir-birini bosib ketmaydi. Bu muhim: shu tufayli
bir xil qo'ng'iroqning ikkala versiyasi ham qoladi.

**DIQQAT:** `cf-call-analyzer` **git repo EMAS**. O'zgartirishdan oldin
tegiladigan faylning zaxira nusxasini oling:
`cp src/worker.js src/worker.js.bak-$(date +%Y%m%d-%H%M)`

---

## 1-vazifa: Voice AI so'rovlarini proksi orqali yuborish

### Nega

Hozir `src/stt/voice.js` RunPod'ga to'g'ridan-to'g'ri murojaat qiladi:

```js
const base = (endpoint) => `https://api.runpod.ai/v2/${endpoint || VOICE_DEFAULT_ENDPOINT}`;
```

Buning ikkita kamchiligi bor:

1. **RunPod API kaliti production konfiguratsiyasida saqlanadi** (`voiceKey`).
   Proksi orqali yuborilsa, kalit butunlay chiqib ketadi — u faqat proksi
   Worker'ining secret'ida qoladi.
2. **Jonli oqim production trafigini ko'rmaydi.** Kuzatuv konsoli
   (`ovoz-konsoli.sellup2.workers.dev`) faqat o'zidan o'tgan so'rovlarni
   yozadi. Production to'g'ridan-to'g'ri ketgani uchun u yerda ko'rinmaydi.

### Nima qilish kerak

Proksi manzili: `https://ovoz-konsoli.sellup2.workers.dev/api`

Yo'llar **bir xil**, faqat baza o'zgaradi:

| Hozir | Proksi orqali |
|---|---|
| `https://api.runpod.ai/v2/{endpoint}/run` | `https://ovoz-konsoli.sellup2.workers.dev/api/run` |
| `.../status/{id}` | `.../api/status/{id}` |
| `.../health` | `.../api/health` |

`Authorization` sarlavhasi **kerak emas** — proksi kalitni o'zi qo'shadi.

Amalga oshirish:

* `voice.js` ga yangi sozlama qo'shing (masalan `proxyBase`). Bo'sh bo'lsa —
  hozirgidek to'g'ridan-to'g'ri ishlaydi; to'ldirilsa — proksi orqali.
  **Orqaga moslikni buzmang.**
* Sozlamalar sahifasiga maydon qo'shing (`worker.js` dagi `stt` konfiguratsiyasi
  yonida, `voiceEndpoint` qanday qilingan bo'lsa shunday).
* `checkVoiceKey` ham yangi bazadan foydalansin, aks holda "tekshirish" tugmasi
  eski yo'ldan ketadi.

### Tekshirish

O'zgartirgandan keyin sozlamalardagi "tekshirish" tugmasi ishlashi va konsolning
**Jonli oqim** panelida yangi so'rovlar paydo bo'lishi kerak.

### Ehtiyot

Proksi — qo'shimcha bo'g'in. U ishlamay qolsa, transkripsiya to'xtaydi.
Shuning uchun `voice.js` da xato bo'lganda **eski yo'lga qaytadigan** zaxira
qoldiring yoki hech bo'lmasa xatoni aniq loglang.

---

## 2-vazifa: taqqoslash datasetini eksport qilish

### Nega

`AGENT-BRIEF.md` dagi tahlil uchun "bir xil qo'ng'iroq, ikkita matn" juftliklari
kerak. Ular endi bazada **o'z-o'zidan** to'planyapti, lekin eksport qilinmagan.

### Nima qilish kerak

D1 bazadan quyidagilarni bitta faylga chiqaring:

```sql
SELECT t.call_key, t.full_text AS muxlisa_text, v.full_text AS voice_text,
       t.duration, t.lead_id
FROM transcripts t
JOIN voice_transcripts v ON v.call_key = t.call_key
WHERE t.status = 'ready' AND v.status = 'ready'
  AND t.full_text IS NOT NULL AND v.full_text IS NOT NULL;
```

(Ustun nomlarini avval `PRAGMA table_info(voice_transcripts)` bilan
tasdiqlang — sxema o'zgargan bo'lishi mumkin.)

Buyruq:

```bash
npx wrangler d1 execute qongiroq-tahlili-db --remote --json --command "..."
```

Natijani `whisper-uzbek-asr/data/pairs.json` ga yozing.

### Nima uchun qimmatli

Bu — **bepul o'sib boradigan** taqqoslash to'plami. Har bir yangi qo'ng'iroq
ikkala tizimdan o'tadi, ya'ni ma'lumot qo'shimcha xarajatsiz to'planadi.

Shu juftliklardan `AGENT-BRIEF.md` ning 2-bosqichidagi barcha tahlillarni
qilish mumkin — endpoint'ga qayta so'rov yubormasdan, ya'ni **bepul**.

---

## 3-vazifa: eval to'plamini kengaytirish

### Nega

Hozirgi eval to'plami — 81 namuna, **3757 so'z**. Ishonch oralig'i ±6–7 foiz
punkt. Bu asbob 7 punktdan kichik yaxshilanishni **ko'ra olmaydi**.

Round 4 aynan shuning qurboni bo'ldi: eval loss yaxshilandi, WER'da esa
"farq shovqin doirasida" (p = 0.732) degan javob chiqdi. Ya'ni ma'lumot
qo'shish foyda berdimi yoki yo'qmi — bilib bo'lmadi.

### Nima qilish kerak

2-vazifadagi juftliklardan **150–200 qo'ng'iroq** ajrating (~25 000 so'z).
Shunda xatolik ±2 punktga tushadi.

Qat'iy shartlar:

* **Hozirgi 81 namunani o'zgartirmang.** Ular yangi to'plamga qo'shilsin,
  almashtirilmasin — aks holda oldingi round'lar bilan taqqoslab bo'lmaydi.
* Ajratish **qo'ng'iroq (`call_key` yoki `lead_id`) bo'yicha** bo'lsin, namuna
  bo'yicha emas. Bitta suhbatning bo'laklari train va eval'ga bo'linib ketsa,
  model eval suhbatini treningda ko'radi va natija soxta yaxshi chiqadi.
* Eval'ga tushgan qo'ng'iroqlar trening to'plamidan **chiqarilsin**.

---

## Tartib

1-vazifa mustaqil va tez. 2-vazifa 3-vazifaning shartini yaratadi, shuning
uchun 2 → 3 tartibida.

Har bir vazifadan keyin nima o'zgarganini va nimani tekshirganingizni yozing.

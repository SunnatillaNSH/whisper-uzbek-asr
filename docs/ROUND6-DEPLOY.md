# Round 6 — production'ga joylash rejasi (2026-10-05)

**Egasi qarori (2026-10-05):** Round 6 production'ga qo'yiladi, hozirgi production
model orqaga qaytish (rollback) uchun saqlanadi. Asos: sizib chiqishsiz taqqoslashda
R6 40.4 vs prod 40.7 (durang), Jev A-006 0.84 «deploy candidate».

> Eslatma: `docs/ROUND6-RESULT.md` (eval-120, +5.84 punkt yomon, "joylashtirilmasin")
> bu qaror bilan zid. Qaror egasiniki; sizib chiqishsiz taqqoslash raqamlari
> (40.4 vs 40.7) shu hujjatga kirmagan — ularning manbasi alohida hujjatlansin.

## Qiymatlar

| | Repo (`HF_MODEL_ID`) | Holat |
|---|---|---|
| **Rollback (hozirgi production)** | `Sunnat0091/whisper-large-v3-uz-calls-ct2` | TEGILMAYDI |
| **Yangi (Round 6)** | `Sunnat0091/whisper-large-v3-uz-calls-ct2-r6` | PRIVATE, yuklanmagan |

Endpoint `vwobkifazoyyh1`. Obraz qayta qurilmaydi (model obrazga pishirilmagan,
`HF_MODEL_ID` o'zgaruvchisi yetarli). `handler.py` O'ZGARMAYDI, dekodlash
shartnomasi (beam 5, VAD 2000/400, hotwords yo'q, temperatura zanjiri) bir xil.

## Qurilish (Mac, `.venv-r6`)

1. Baza `Sunnat0091/whisper-large-v3-uz` + `models/round6/adapter` (r=32, q/v_proj,
   checkpoint-3000) -> `merge_and_unload()` -> fp16 -> `models/round6/merged`
   (tokenizer `openai/whisper-large-v3` dan).
2. `ct2-transformers-converter --quantization float16` + `--copy_files`
   (round 5 bilan aynan bir xil buyruq, `models/README.md`) -> `models/round6/ct2`.
3. Versiyalar: transformers 4.57.6, peft 0.21.0, ctranslate2 4.8.2, faster-whisper 1.2.1
   (production bilan bir xil).

## Natija (Mac, 2026-10-05)

- Merge va CT2 muvaffaqiyatli; `models/round6/ct2` fayl ro'yxati va o'lchamlari
  `models/round5/ct2` (production konvensiyasi) bilan bir xil (model.bin 3 087 284 237 bayt).
- `model.bin` SHA256: `d5cf755fc294c7988db7eed65a5b4292f9534b5473edc7c6f1b11f3cfd9e0a6d`
  (yuklangandan keyin repo bilan solishtiriladi).
- Lokal tekshiruv (CPU int8, handler dekodlash sozlamalari, eval-50 dan 3 namuna): model
  ishlaydi, matn o'zbekcha va ma'noli, halqa yo'q. Endpointga hech narsa yuborilmadi ($0).

## Joylash qadamlari

1. Egasi: `/Users/macbookuz/code/whisper-uzbek-asr/.venv-r6/bin/hf auth login`
   (WRITE token; men tokenni ko'rmayman).
2. PM buyruq beradi -> serving agenti yuklaydi (PRIVATE):
   `hf repo create Sunnat0091/whisper-large-v3-uz-calls-ct2-r6 --private`
   `hf upload Sunnat0091/whisper-large-v3-uz-calls-ct2-r6 models/round6/ct2 .`
3. Tekshiruv: repo ro'yxati prod repo bilan bir xil fayllar; `model.bin` SHA256 lokal bilan bir xil.
4. PM RunPod konsolida `HF_MODEL_ID` ni yangi repoga o'zgartiradi (token: endpoint'dagi
   `HF_TOKEN` yangi PRIVATE repoga o'qish huquqiga ega bo'lishi shart — aks holda 401,
   worker yuklay olmaydi; fine-grained token bo'lsa repo qo'shilsin).
5. Workers qayta ishga tushadi (sovuq start ~ model yuklash). Handler javobida model
   maydoni YO'Q, shuning uchun probsiz tasdiq: (a) worker logida yuklangan repo nomi;
   (b) deterministik farq — `scripts/eval_endpoint.py` bilan bir xil audio (greedy,
   temperatura 0) avvalgi production natijasi (`analysis/eval120_prod.json`) bilan
   solishtiriladi: R6 matni baytma-bayt boshqacha, o'zi bilan ikki yurishda bir xil.

## Smoke test (<= 5 qo'ng'iroq, byudjet ~ $0.05, 5 ta eval-50 namunasi ~ 5-10 daq audio)

- 3 ta eval-50 namunasi (`data/eval120/eval50.csv` boshidan): matn o'zbekcha, sensible,
  takrorlanish halqasi yo'q, `status=success`.
- 1 ta jimlik/qisqa shovqin: bo'sh yoki qisqa matn (gallyutsinatsiya yo'q).
- 1 ta SellUp haqiqiy qo'ng'iroq (proksi orqali, tokenli): `/api/stats` da xatosiz o'tdi.
- Muvaffaqiyat: 5/5 `success`, kechikish prod bilan bir xil tartibda (RTF), takroriy halqa 0.
- Muvaffaqiyatsiz -> rollback.

## Rollback (1 daqiqa)

RunPod konsol -> endpoint `vwobkifazoyyh1` -> `HF_MODEL_ID` =
`Sunnat0091/whisper-large-v3-uz-calls-ct2` -> workers qayta ishga tushsin. Obraz va kod
o'zgarmagani uchun boshqa qadam yo'q. Rollback sharti: smoke test muvaffaqiyatsiz yoki
egasi sinovida sezilarli yomonlashuv.

## Kuzatuv

- Birinchi kunlarda `voice_diag` / proksi `/api/stats`: xato ulushi, o'rtacha kechikish.
- Yangi repo ro'yxatda "private" ekani tekshirilsin (mijoz qo'ng'iroqlarida o'qitilgan model).
- Dekodlash sozlamasi o'zgarmagani uchun eski WER raqamlari amal qiladi; R6 uchun yangi
  raqamlar egasining sinovi va leak-free taqqoslash bo'yicha.

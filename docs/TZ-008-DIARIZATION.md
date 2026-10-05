# TZ-008 — Gapiruvchilarni ajratish (diarizatsiya) + rol

Holat: **TZ + Pod sinov rejasi tayyor; hech narsa ishga tushirilmagan.** Pod yo'q, endpointga so'rov yo'q, `handler.py`/production obrazi o'zgartirilmagan.
Sana: 2026-10-05. Repo OCHIQ: bu hujjatda audio, matn, telefon, mijoz nomi yo'q.
Manbalar: `cf-call-analyzer/docs/audit/A-003/README.md` (faktlar), `A-005/README.md` (Jev qarori), `docs/ROUND6-PLAN.md` (Pod).

## 0. Jev qarori (A-005, 2026-10-05)

| Savol | Jev |
|---|---|
| A: alohida diarizator (pyannote/NeMo) + bizning Whisper + sotuvchi enrollment | **2.87/3**, tanlov 1.00 |
| B: Whisper ichida rol tokenlari | 0.01 (yo'q) |
| C: end-to-end ko'p gapiruvchi model | 0.38 |
| D: bulutli diarizatsiya xizmati | 0.83 |
| Avval diarizatsiya, keyin matn? | 0.42 — **AVVAL matn, KEYIN so'zlarni gapiruvchiga biriktirish** (word timestamps) |
| Round 6 trening ichiga kiritish? | **0.10 — yo'q.** Round 6 faqat ASR. |
| Reja (Round 6 hozir + A parallel, Pod'da sinov) | 0.76 — ha |

Bu TZ shu qarorlarga mos: A varianti, matn-birinchi tartib, Round 6 treningiga tegmaydi.

## 1. Maqsad

Voice AI keyin faqat qo'ng'iroq emas, **ISTALGAN audio** (yig'ilish, intervyu, ko'p kishili yozuv) qabul qiladi. Chiqish: gapiruvchi bo'yicha ajratilgan matn `segments: [{start, end, speaker, text, role?}]`.
- **Qo'ng'iroq** (SellUp): `speaker` `0/1`, `role` = `sotuvchi` | `mijoz` | `nomalum` (SellUp `segments_json` allaqachon `speaker`+`role` ni qo'llaydi, sxema o'zgarmaydi).
- **Boshqa audio:** `speaker` = `Speaker 1/2/3…`, `role` yo'q.

## 2. Arxitektura (A varianti, matn birinchi)

```
audio
 ├─► ASR (bizning Whisper, joriy handler, word_timestamps=true) ─► so'zlar [{word,start,end}]
 └─► diarizator (pyannote 3.1; VAD+segmentatsiya+embedding+klasterlash) ─► turlar [{start,end,spk}] + spk embedding
                                   │
        so'z → gapiruvchi (so'z markazi qaysi turga tushsa; tushmasa eng yaqin tur)
                                   ▼
   ketma-ket bir xil gapiruvchi so'zlar → segment {start,end,speaker,text}
                                   ▼
   (faqat qo'ng'iroq) enrollment: spk embedding ↔ sotuvchi izi → role
```

Qarorlar:
1. **Dekodlash shartnomasi o'zgarmaydi.** `handler.py` dagi beam/VAD/hotwords qiymatlariga tegilmaydi. `word_timestamps=True` faqat so'z vaqtini qo'shadi; matn farq qilmasligi **tekshiriladi** (Pod sinovida `text_same_with_word_ts_pct`; <100% bo'lsa eski WER raqamlari diarizatsiyali yo'l uchun yaroqsiz — eval-50 da qayta o'lchanadi).
2. **Gapiruvchi soni.** Qo'ng'iroq: `num_speakers=2` (so'rovda berilsa shu). Boshqa audio: noma'lum, `min_speakers=1`, `max_speakers=8` (so'rov bilan o'zgartirish mumkin).
3. **Rol — akustik, taxminsiz.** Sotuvchi izi (voice enrollment): `seller_id` ma'lum (har qo'ng'iroqda bor, 2 sotuvchi). Har sotuvchining ~12+ qo'ng'irog'idan diarizatsiya markazlari olinadi; qo'ng'iroqlar bo'ylab **takrorlanuvchi** markaz = sotuvchi (mijozlar har safar boshqa). Qo'ng'iroqda sotuvchi izi bilan eng o'xshash klaster `sotuvchi`, qolgani `mijoz`; o'xshashlik chegaradan past bo'lsa — `nomalum` (taxmin QILINMAYDI) va `diarization_confidence` qaytadi. "Birinchi gapirgan = sotuvchi" qoidasi (`assignRoles`) faqat zaxira: chiquvchi qo'ng'iroqda ham mijoz ("allo") birinchi bo'lishi mumkin.
4. **Kodek confound (A-003 F2):** S1 hammasi MP3, S2 hammasi AAC/M4A — kodek sotuvchiga to'liq bog'langan, embedding ovozni emas kodekni o'rganishi mumkin. Shuning uchun (a) "sotuvchini ko'r aniqlash" aniqligi o'z-o'zidan dalil emas (kodek bilan shishishi mumkin), (b) 2-bosqichda enrollment va test audiosi bir xil kanalga keltiriladi (8 kHz + bir xil kodek bilan qayta kodlash) va natija ikkalasida solishtiriladi, (c) asosiy rol metrikasi qo'lda belgilangan to'plamda (§5).
5. **3-shaxs** (rahbar, uchinchi ovoz): `num_speakers=2` da mijozga qo'shilib ketadi; alohida qaror (`num_speakers` ni oshirish yoki `nomalum`).
6. Qo'ng'iroq bo'lmagan audio uchun enrollment yo'q; ixtiyoriy kelajak: `roles` (ism → namuna audio) bilan umumiy enrollment.

## 3. Serving varianti

| | (1) joriy handler ichiga | (2) **alohida endpoint (tavsiya)** |
|---|---|---|
| Obraz | torch + pyannote + og'irliklar: **+4–5 GB** (hozir ~1–2 GB, torch'siz) | joriy obraz o'zgarmaydi; diarizator o'z obrazida (~6–8 GB) |
| Sovuq start | ~25 s → ~60–90 s (taxmin), HAMMA so'rovga ta'sir qiladi | ASR ~25 s o'zgarmaydi; diarizator o'zining sovuq starti, faqat `diarize:true` da |
| Xavf | production'ni buzish (CTranslate2 cuDNN ↔ torch CUDA kutubxona to'qnashuvi), dekodlash shartnomasi xavfi, rollback og'ir | production'ga xavf yo'q; o'zaro mustaqil rollback |
| Narx | ASR so'rovi har doim og'irroq obraz | to'lov faqat ishlatilganda; diarizator GPU soniyalari (taxminan +1–2 s/daq, Pod sinovida o'lchanadi) |
| Murakkablik | bitta chaqiruv | 2 chaqiruv (ASR → diarizator) |

Tavsiya (2): **`voice-diar`** (yangi RunPod serverless endpoint). Oqim: mijoz ASR'ni `segments:true, word_timestamps:true` bilan chaqiradi → keyin diarizatorga `{audio, words, num_speakers?/min/max, seller_id?}` yuboradi → diarizator birlashtirilgan `segments` qaytaradi. Birlashtirish (so'z→gapiruvchi, rol) diarizatorda (Python), shuning uchun SellUp/ovoz-konsoli faqat 2 chaqiruvni tartiblaydi (yoki proksi ularni bitta `diarize:true` so'roviga yig'adi). Audio ikki marta yuboriladi (MAX_AUDIO_MB doirasida). (1) faqat sinov natijalari hajm/start bo'yicha (1) ni oqlasa ko'rib chiqiladi.

## 4. API o'zgarishi (orqaga mos)

Hozirgi so'rov/javob buzilmaydi; yangi maydonlar ixtiyoriy.

ASR (`handler.py`, joriy endpoint) — faqat bitta qo'shimcha:
- `word_timestamps: true` (standart `false`) → `segments[].words: [{word,start,end}]`.

Diarizator (`voice-diar`) — yangi:
- so'rov: `audio_base64|audio_url`, `words` (ASR'dan) yoki `transcribe:true` (diarizator ASR'ni o'zi chaqirmaydi — v1 da `words` majburiy), `num_speakers?`, `min_speakers?`, `max_speakers?`, `seller_id?` (qo'ng'iroq), `roles?` (`"call"` | yo'q).
- javob: `status`, `speakers: n`, `segments:[{start,end,speaker,text,role?}]`, `diarization_confidence?`, `processing_time_sec`.
- Xato/yo'q diarizator: SellUp hozirgidek (`speaker:'?'`, `role:'nomalum'`) ishlashda davom etadi.

`docs/API.md` dagi "Diarizatsiya yo'q" bandi (~324-qator) amalga oshirilgach yangilanadi; `diarize` ASR endpointiga KIRITILMAYDI (tanlov (2)).

## 5. Qabul metrikalari

| Metrika | O'lchash | Taklif chegarasi (PM tasdiqlaydi) |
|---|---|---|
| **DER** (collar 0.25 s, overlap hisobda) | qo'lda belgilangan to'plam: ≥10 qo'ng'iroq (3–5 daq) + 3–5 qo'ng'iroqsiz; RTTM | qo'ng'iroq ≤ 20% (pyannote 3.1 e'lon qilingan 7.8–21.7% oralig'i, o'zbekcha o'lchanmagan); boshqa ≤ 25% |
| **«Sotuvchi kim» aniqligi** | gold: klaster→rol to'g'riligi; qo'shimcha: `seller_id` ma'lum, ko'r aniqlash (kodek tuzatilgach) | ≥ 95%; `nomalum` ulushi ≤ 10%; XATO rol ≤ 3% |
| **cpWER** (concatenated permutation WER) | gold matn bo'lmasa — eval-50 dagi gapiruvchi-bo'yicha referens (qo'lda) | cpWER ≤ joriy WER + 5 punkt (so'zlarni noto'g'ri gapiruvchiga biriktirish xatosi ASR'dan keyin ko'rinadi) |
| Gapiruvchi soni (boshqa audio) | gold bilan | ≥ 85% to'g'ri son |
| Matn o'zgarmasligi | `word_timestamps` yoqilgan/o'chiq matn bir xil | 100% (aks holda shartnoma qayta o'lchanadi) |
| Tezlik/narx | RTF, sovuq start, $/audio soat | o'lchanadi, chegara keyin |

**Gold yo'q.** Hozir akustik gapiruvchi yorlig'i hech qayerda yo'q (Muxlisa belgilari matndan Claude taxmini; 13% qo'ng'iroq — A-003 F5/F6). DER/cpWER uchun ~10–15 qo'ng'iroqni (≈45 daq) qo'lda belgilash kerak (egasi/PM tomondan, ~2–3 soat ish). Shu paytgacha Pod sinovi faqat §6 dagi proksi-ko'rsatkichlarni beradi.

## 6. Pod sinov rejasi (Round 6 Pod'ida, trening TUGAGANDAN KEYIN, Pod o'chirilishidan OLDIN)

Jev: Round 6 treningiga diarizatsiya KIRMAYDI; sinov faqat trening/CT2/eval-120 zanjiri tugagach (`/workspace/round6/PIPELINE_DONE`). Treningning byudjet qo'riqchisi ($3.0) ga ta'sir qilmaydi.

**Ma'lumot.**
- Test: 30 qo'ng'iroq, **eval-50 dan** (sotuvchi bo'yicha muvozanatli), TO'LIQ qo'ng'iroq (`raw55`/`raw28` bo'laklaridan Mac'da yig'iladi — Pod'ga ketayotgan eval chunklari to'liq qo'ng'iroq emas).
- Enrollment: 2 sotuvchi × 12 qo'ng'iroq, **train'dan, eval-120 dan tashqarida** (sizib chiqish yo'q), 60–300 s.
- **Qo'ng'iroqsiz 3–5 yozuv: HOZIR YO'Q** — lokal diskda yig'ilish/intervyu audiosi yo'q (faqat MoyZvonki qo'ng'iroqlari). Egasi bersa `data/round6-diar/other/` ga qo'yiladi (gitignore'da); bo'lmasa sinov faqat qo'ng'iroq bilan, qo'ng'iroqsiz qism "o'lchanmagan" deb yoziladi.
- seller_id/direction: D1 dan faqat SELECT (id, seller_id, direction, duration); telefon/mijoz nomi olinmaydi.

**Qadamlar** (`scripts/round6_diar_test.sh <host> <port>`, ichida `scripts/diar_test.py`):
1. Mac: `diar_test.py prep` → `data/round6-diar/{audio,calls.json}` (gitignore: `data/*`).
2. Yuborish (tar), HF token `~/.config/sellup/hf_token` dan Pod'ning `/root/.hf_token` iga (qiymat chop etilmaydi; oxirida o'chiriladi).
3. `pip install pyannote.audio` (torch Pod'da bor; versiya o'zgarmasligi chiqishda ko'rinadi).
4. Pod: dekodlash `handler.py` bilan mosligini `eval_pod.check_matches_handler` tekshiradi; round6 CT2 (`/workspace/round6/ct2`) bilan ASR (`word_timestamps=True`, so'ng True/False matn solishtirish); pyannote `num_speakers=2` + embedding; enrollment profil; rol; birlashtirilgan `segments/*.json`; `summary.json`.
5. Mac'ga `data/round6-diar/out/` (matn bor — faqat lokal, commit YO'Q).

**Chiqadigan ko'rsatkichlar:** vaqt (yuklash, ASR RTF, diarizatsiya RTF, jami), `role_assigned_pct`, ko'r-sotuvchi aniqligi (kodek ogohlantirishi bilan), klasterlarning o'xshashlik taqsimoti (chegara tanlash uchun), `n_speakers` gistogrammasi, "birinchi gapirgan = sotuvchi klasteri" qoidasi bilan mos kelish (zaxira qoidaning ishonchliligi), matn o'zgarmasligi. **DER/cpWER bu yurishda YO'Q (gold yo'q).**

**Qo'shimcha vaqt va narx** (L40S ≈ $1.10/soat; qismlari taxminiy, o'lchanmagan):

| Bosqich | Vaqt |
|---|---:|
| yig'ish yuborish (~150–250 MB), pip, pyannote modellarini yuklash | 8–10 min |
| enrollment (24 qo'ng'iroq diarizatsiyasi) | ~2 min |
| 30 qo'ng'iroq: ASR ×2 (so'z vaqti bilan/siz) + diarizatsiya | ~8–10 min |
| qo'ng'iroqsiz yozuvlar (bo'lsa), natijani olish, zaxira | ~5 min |
| **Jami** | **~25–30 min ≈ $0.45–0.55** |

Round 6 baholashi $1.8 (zaxira bilan $2.3) → **taxminan $2.3–2.8**, Pod chegarasi $5 ichida; endpointga so'rov yo'q ($1 chegara tegilmaydi). Pod har qo'shimcha daqiqaga $0.018 — sinov PM ruxsatiz $0.6 dan oshsa to'xtatiladi.

## 7. Egasi qiladigan ishlar

1. **HF'da shartlarga rozilik** (login qilingan hisobda, ikkala sahifada "Agree/Access repository" — kompaniya/maqsad so'raladi): `pyannote/speaker-diarization-3.1` va `pyannote/segmentation-3.0`. Agar yuklashda 403 chiqsa, `pyannote/wespeaker-voxceleb-resnet34-LM` sahifasida ham.
2. **HF token** (Read, shu hisobdan) yaratib `~/.config/sellup/hf_token` ga o'zi saqlaydi (`chmod 600`); tokenni chatga/repoga yozmaydi. **Hozir token topilmadi:** `~/.config/sellup/` da faqat `pm_task_token`, `proxy_token`; repoda `.env` yo'q; `HF_TOKEN` muhit o'zgaruvchisi va `~/.cache/huggingface` yo'q.
3. Pod sinovini (≈ $0.5, 25–30 min) Round 6 bilan birga yuritishga **ruxsat**.
4. (Ixtiyoriy, lekin DER uchun shart) 3–5 ta qo'ng'iroqsiz yozuv (yig'ilish/intervyu; roziligi bor, ~3–10 daq) `data/round6-diar/other/` ga.
5. (DER/cpWER uchun) 10–15 qo'ng'iroqni gapiruvchi bo'yicha qo'lda belgilash (~45 daq audio) — kim bajaradi, PM hal qiladi.

## 8. Keyingi bosqichlar (sinovdan keyin)

Natija mos kelsa: `voice-diar` obraz (Dockerfile alohida), `docs/API.md`, kodek tengplashtirish, chegara tanlash, gold bilan DER/cpWER, SellUp `voiceSegments` ni kelgan `speaker/role` ga o'tkazish (cf-call-analyzer, alohida TZ). Production ASR obrazi va modeliga bu TZ tegmaydi.

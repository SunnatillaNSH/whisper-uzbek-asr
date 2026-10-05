# Round 6 — reja: BARCHA Muxlisa transkriptlari (eski + yangi), faqat qo'ng'iroq

Holat: **dataset tayyor (lokal), trening ISHGA TUSHIRILMAGAN.** Pod yo'q, endpointga so'rov yo'q.
Sana: 2026-10-05. Manba: SellUp D1 (`transcripts`, `moizvonki_calls`, faqat SELECT). Raqamlar `data/round6-dataset/report.json` dan.
Mijoz matni/audiosi/telefon raqami bu hujjatda YO'Q (repo ochiq).

## 1. Manba: nechta yangi

| | Qator | Qo'ng'iroq |
|---|---:|---:|
| D1 `transcripts`, muxlisa, `ready` | 2 111 (1 529 bo'lak `:pNofM` + 582 butun) | 752 |
| Takrorsiz matn birliklari (bo'lak yoki bo'laksiz butun) | 1 842 | 752 |
| Oldingi lokal datasetlarda bor (`calls-dataset`, `-28s`, `calls-rejected`) | 1 077 birlik | 466 |
| **YANGI** (oldin yo'q) | **765 birlik** (906 xom qator) | **286** |
| shundan 2026-09-30 dan keyin yozilgan (Dataset eksporti) | 748 | 277 |
| Muxlisa `error` (matni yo'q, tiklanmagan) | 798 qator (458 bo'lak) | — |

Muhim topilmalar:
- **582 "butun" qatorning 269 tasi o'z bo'laklarining birlashmasi** (`mergePartsIfReady`). Eski `build-asr-dataset.mjs` ikkalasini ham chiqarardi (dublikat). Bu raundda bo'laklari bor qo'ng'iroqlarning butun qatori tashlandi.
- **Yangi eksport ham 55 s bo'laklarda** (`MAX_CHUNK_SECONDS=55`, `src/stt/muxlisa.js`): yangi 150 ta ko'p-bo'lakli qo'ng'iroq 55 s ga mos. 28 s partiya faqat eski 62 qo'ng'iroq. Demak ko'pchilik bo'lak >30 s va kesish kerak edi.
- 170 qo'ng'iroqda bo'laklar to'liq emas (qolganlari Muxlisa xatosi) — mavjud bo'laklar ishlatiladi, yo'qlari shunchaki yo'q.

## 2. Dataset (`data/round6-dataset/`, gitignore'da — `git check-ignore -q` exit 0)

Quvur: D1 eksport (`scripts/round6_export_d1.py`) → serverning o'z bo'luvchisi bilan audio (`tools/build-asr-dataset.mjs`, 55 s va 28 s guruhlari alohida, 700+52 yozuv yuklandi, bo'laklar soni mos kelmasligi 1 ta) → `scripts/round6_build_dataset.py`.

Kesish: >30 s bo'lak jimlik joyidan va gap chegarasidan bo'linadi. **Eski `prepare_calls_for_colab.py` dagi xato tuzatildi**: kesish nuqtasi har bo'lak ≤30 s bo'lishini ta'minlovchi feasible oraliqdan qidiriladi, matn↔vaqt xaritasi bir tekis emas, ovozli freymlar bo'yicha. Natija: chetlash 235/843 (28%) → **53/1580 (3.4%)**; chetlangan 0.5 soat. Shu sabab `recover_rejected.py` (GPU) bu raundda KERAK EMAS.

Darvozalar: har bo'lak ≤30 s, ≥1.2 s, cps 6–26; birortasi o'tmasa butun namuna chetlanadi.

| | Namuna | Soat | Qo'ng'iroq |
|---|---:|---:|---:|
| Xom (eval-120 chiqarilmasdan) | 1 827 | 20.49 | 747 |
| eval-120 qo'ng'iroqlari chiqarildi | −247 | | −119 |
| Chetlangan (cps 26, kesish topilmadi 24, qisqa 3) | −53 | −0.50 | |
| **Yakuniy** | **2 763** | **17.47** | **613** |
| **train** | **2 550** | **16.17** | 552 |
| **dev** (qo'ng'iroq bo'yicha 10%, seed 20260919 — `train_calls.py` bilan bir xil algoritm, tengligi tekshirildi) | **213** | **1.30** | 61 |

Taqqoslash: Round 3 — 941 namuna / 6.5 soat; Round 4 — 1 258 / 8.2; Round 5 — 1 654 qo'ng'iroq namunasi. Bu to'plam **~2.7× ko'p**. Shundan **yangi qo'ng'iroqlardan 1 507 namuna / 9.38 soat (276 qo'ng'iroq)**, eskidan 1 256.
So'zlar ≈ 127 000; mediana namuna 25.7 s (57% ≥25 s, 2.3% <5 s); mediana cps 15.2 (butun 14.8, kesilgan 15.2 — mos, kesish nisbatni buzmagan).
Davomiylik: kiruvchi 4.42 soat / chiquvchi 13.05 soat. Oylar: may–okt 2026.

**Sizib chiqish assertlari o'tdi** (build skriptida, treningdan oldin): train ∩ dev = 0; train ∩ eval-120 = 0; dev ∩ eval-120 = 0; eval-50 ⊂ eval-120 va u bilan kesishish 0; bir xil matn (>60 belgi) dev'da train'dagi bilan 0; takroriy yo'l 0. Eval ro'yxati: `eval_calls_120.json` + `eval120/eval.csv`/`eval50.csv` yo'llaridan (120 ta noyob qo'ng'iroq — hujjatdagi eski "117" noto'g'ri; `VOICE-AI-KNOWLEDGE.md` §6 ni PM tuzatsin).

Fayllar: `all.csv` (train+dev, `train_calls.py` o'zi dev kesadi — bir xil natija), `train.csv`, `dev.csv`, `manifest.jsonl` (call, bo'lak, kesilgan-mi, cps, sana, yo'nalish, lead, split, rol), `rejected.jsonl`, `audio/*.flac` (746 MB), `eval_calls_120.json` nusxasi.

## 3. Trening rejasi (TASDIQ KUTILMOQDA)

Domen ulushi: **100% qo'ng'iroq, tashqi korpus 0** (Round 5 saboqi: tashqi toza korpus qo'shilsa qo'ng'iroq ≥70% bo'lsin; bu hajmda tashqi qo'shish ma'nosiz — ≤1 100 namuna). Podkast/UzbekVoice yo'q. Boshlang'ich model: round-2 (`Sunnat0091/whisper-large-v3-uz`) — farq faqat ma'lumotdan chiqsin (Round 4 mantig'i). LoRA r32/α64 q,v; LR 5e-5; batch 8; warmup 100.

**Qadam hisobi.** Formula: `namunalar × 8 / batch = 2 550 × 8 / 8 = 2 550 qadam` (8 epoxa). Taklif: `MAX_STEPS=3000` (9.4 epoxa, yuqori chegara), `EVAL_STEPS=SAVE_STEPS=250`, `PATIENCE=3`, `load_best_model_at_end` (dev loss). Eng yaxshi nuqta ~2 000–2 600 kutiladi.

**Egasining 6000 qadami — KO'P.** 6000×8/2550 = **18.8 epoxa**, formuladan 2.4×. Round 3 da eval loss 8.5-epoxada minimal (0.8134), 25 epoxaga yetganda 0.899 gacha o'sgan (yodlash); Round 4 ham ~8 epoxada. Ma'lumot 2.7× ko'paygani minimumni kechiktirmaydi — u epoxada o'lchanadi, qadamda emas. 6000 ham EarlyStopping tufayli ~3000 da to'xtashi mumkin, lekin qo'riqchi bashorati $3.5 ga chiqib, treningni noto'g'ri to'xtatishi yoki 2× pul sarflashi mumkin. Agar egasi ko'proq natija istasa — qadam emas, **ma'lumot**: 798 xatoli Muxlisa bo'lagini qayta o'girish va yangi qo'ng'iroqlar.

**Pod va narx** (tezlik Round 3-4 da o'lchangan: 1.02 s/qadam L40S, batch 8; bu to'plamda namuna uzunligi shunga teng, 23.7 s vs 22.9 s):
| | L40S 48 GB ($1.10/s, o'lchangan tezlik) — TAVSIYA | A40 48 GB ($0.49/s, tezlik O'LCHANMAGAN, ~1.5–1.8× sekin deb taxmin) |
|---|---|---|
| Sozlash + ma'lumot ko'chirish (≈750 MB) | ~25 min | ~25 min |
| Trening 3000 qadam (+12 dev baholash) | ~53 min + 5 | ~85 min + 6 |
| round6 CT2, eval-50 (129) + eval-120 (310 namuna) | ~25 min | ~30 min |
| **Jami / narx** | **~1.7 soat ≈ $1.9** (zaxira bilan $2.4) | **~2.5 soat ≈ $1.2** |
| 6000 qadam bo'lsa (taqqoslash uchun) | ~2.7 soat ≈ $3.0–3.5 | |

Chegara: ≤$5; `BUDGET_USD=3.0`, `POD_RATE` haqiqiy narxga, `POD_START_EPOCH` Unix epoch. Volume disk 60 GB (Pod bilan o'chadi), container 30 GB. Pod tugagach model Mac'ga `models/round6/` + `SHA256SUMS`, keyin PM o'chiradi.

**Pod skriptlari moslashtirilishi kerak** (hozir ishga tushmaydi): `round6_pod.sh`/`round6_send.sh` `analysis/train_weighted.csv` va `calls-rejected` tiklashga bog'langan. Yangi: `CALLS_DIR=/workspace/round6-dataset TRAIN_CSV=all.csv`, tiklash bosqichi OLIB TASHLANADI, `round6_send.sh` faqat `round6-dataset/{audio,all.csv,eval_calls_120.json}` + `eval120/eval50.csv`+`eval.csv` yo'llarini yuboradi. Eval-120 ham zanjirga qo'shiladi (hozir faqat eval-50) — ±3 punkt uchun.

Ixtiyoriy (alohida PM ruxsati): Pod'da round-2 CT2 bilan train namunalarini bir marta WER'lab, >80% WER (nomoslik) namunalarni chiqarish (~5 min, ~$0.1) — kesilgan bo'laklar moslik tekshiruvi akustik tasdiqlanmagan (hozir faqat cps).

**Muvaffaqiyat mezoni (oldindan):** eval-120 da production'ga nisbatan juftlashgan bootstrap 95% oralig'i BUTUNLAY noldan past. Production eval-50 da 37.84%. Natija ±3 punktdan kichik bo'lsa — "ishonchli emas".

## 4. Rol (SOTUVCHI/MIJOZ) — tekshiruv va speaker-turn varianti

**Manba:** Muxlisa o'zi gapiruvchini QAYTARMAYDI — `normalizeSegments` bitta segment `{speaker:'?', start:0,end:0}` beradi (`src/stt/shared.js`). Rol FAQAT `diarizeWithClaude` (`src/worker.js`) dan: Claude **matnni o'qib** navbatlarga bo'ladi (audioga qaramaydi) va shu bilan birga matnni tuzatadi; `start/end` — navbat indeksi, vaqt EMAS.

**Qamrov (D1, 2 111 qator):**
- segmentlarning **58.4%** (3 452/5 906) rolli, lekin bu **matn belgisining atigi 12.6%** (rolsiz segmentlar bitta uzun blok);
- rolli qatorlar 281/2 111 (13%), **qo'ng'iroqlarning 281/752 (37%)**; hammasi **2026-09-16…19** (+2 qator 10-01). **Yangi (09-30…10-02) eksport matnlarida rol YO'Q** — diarizatsiya ularda ishlamagan;
- 175 ta rolli qo'ng'iroq bo'laksiz (butun) qator, 103 tasi bo'laklari ham bor.

**Ishonchlilik (tekshirilganlar):**
- Rolli butun matn bilan bo'laklar birlashmasi so'zma-so'z deyarli bir xil (103 ta: o'rtacha o'xshashlik 1.000, minimum 0.978) — Claude so'zlarni kam o'zgartirgan, matn Muxlisa'ga yaqin;
- navbatlar qat'iy almashinadi (bir xil rolli ketma-ket juftlik 4 ta) va 8 qo'ng'iroq yagona rolli — Claude ketma-ket gaplarni birlashtirgan; qisqa "aha/ha" kabi tasdiqlar mustaqil navbat bo'lmasligi mumkin; ≤2 so'zli navbatlar 18%;
- birinchi gapiruvchi: chiquvchi qo'ng'iroqda SOTUVCHI 194/211 (92%), kiruvchida 56/70 (80%); sotuvchi ulushi mediana 63% (p10–p90: 44–80%) — mantiqli, lekin bu faqat bilvosita belgi;
- **Haqiqiy aniqlikni o'lchab bo'lmaydi**: audio bo'yicha gold-yorliq yo'q, vaqt belgilari yo'q. Rollar matndan taxmin (LLM), akustik emas. Xato darajasi noma'lum — taxminiy "ishonchli, lekin tasdiqlanmagan".

**Speaker-turn varianti: ixtiyoriy, alohida manifest TAYYORLANMADI** (egasi qarori: diarizatsiya auditi bekor, fokus trening). Faqat statistika: rolli matnga tekislanadigan namunalar asosiy to'plamning **545/2 763 = 19.7%** i (3.4 soat, 205 qo'ng'iroq = 33% qo'ng'iroq); shundan 200 namuna matni Claude tahriridan o'tgan. Yangi (09-30…10-02) matnlarda rol yo'q. Hajm mustaqil trening uchun kichik va yorliq LLM taxmini — tavsiya: hozir qilmaslik. Kerak bo'lsa `round6_build_dataset.py` dagi `role_runs()` shu manifestni qayta hosil qiladi.

## 5. Ma'lum kamchiliklar

- **Muxlisa o'z xatolari yorliqda**: WER Muxlisa'ga nisbatan; model Muxlisa darajasidan o'tolmaydi (strategik cheklov o'zgarmagan). Muxlisa ism/raqamlarni nomuvofiq yozadi; raqamlar so'z bilan.
- **55 s bo'lak chegaralari**: Muxlisa bo'lakni MP3/ADTS kadr chegarasida, nutq chegarasida emas kesadi — har bo'lakning birinchi/oxirgi so'zi kesilgan bo'lishi mumkin (nomuvofiqlik sezilarli emas, lekin sanalmagan).
- **Kesish akustik tasdiqlanmagan**: moslik cps (6–26) va jumla chegarasi bilan tekshirilgan, so'z darajasida emas. Kesilgan bo'lak cps taqsimoti butunlarnikiga mos (mediana 15.2 vs 14.8; p5–p95 10.0–19.2).
- **Nomutanosiblik**: chiquvchi 75% soat; 613 qo'ng'iroqda atigi **208 mijoz (lead)**, top-10 lead 21.5% soat; 145 qo'ng'iroq <30 s; mediana qo'ng'iroq 55 s. Asosan bitta mahsulot (HR/davomat dasturi) taqdimoti — model shu iboralarni yodlashi, WER boshqa domenlarga umumlashmasligi mumkin.
- **Lead kesishuvi**: 61 dev qo'ng'iroqning 54 tasi train bilan bir lead; eval-120 ning 86 leadidan 55 tasi train'da. Bo'lish qo'ng'iroq bo'yicha (qoida shunday), lekin bir mijoz bilan qayta qo'ng'iroqlar ovoz/ismni ulashadi — dev loss va WER biroz optimistik bo'lishi mumkin. (Round 3-5 da ham shunday edi, taqqoslash bir xil sharoitda.)
- **Claude tahrirlangan matn**: 200 namuna (1.18 soat, 6.8%) — butun-qator matni diarizatsiya paytida tuzatilgan; so'zlar deyarli o'zgarmagan (o'xshashlik ≥0.978), lekin tinish/imlo tuzatilgan bo'lishi mumkin. Pseudo-label emas, lekin sof Muxlisa ham emas. `manifest.jsonl` da rol/matn maydonidan ajratib olinadi.
- **798 Muxlisa xatolik qatori** (458 bo'lak) — yo'qolgan matn; 170 qo'ng'iroq bo'laklari to'liq emas.
- **eval-120 chiqarilishi**: 119 qo'ng'iroq (247 namuna) trening to'plamidan chetda — to'g'ri, lekin ma'lumot hajmining 11% ini yeydi.
- Hajm cheklovi saqlanadi: 17.5 soat — Round 3 dagi "qadam emas, MA'LUMOT cheklov" xulosasi 2.7× ko'p ma'lumotda ham tekshirilishi kerak.

## 6. Keyingi qadamlar (PM qarori kerak)

1. Reja (3-bo'lim) tasdig'i: 3000 qadam cap, L40S, ≤$3.0 qo'riqchi chegarasi.
2. Pod skriptlarini moslashtirish (3-bo'lim) — alohida kichik ish.
4. `VOICE-AI-KNOWLEDGE.md` ga qo'shish: eval-120 = 120 qo'ng'iroq; `build-asr-dataset.mjs` butun+bo'lak dublikat xatosi; eski feasible-range xatosi tuzatildi.

Qayta yig'ish: `python3 scripts/round6_export_d1.py` → (cf-call-analyzer'da) `CHUNK_SECONDS=55|28 node tools/build-asr-dataset.mjs <meta> <raw>` → `python3 scripts/round6_build_dataset.py` (numpy, soundfile).

O'zgargan fayllar: `docs/ROUND6-PLAN.md`, `scripts/round6_export_d1.py`, `scripts/round6_build_dataset.py` (yangi, commit qilinmagan).

# Round 6 — reja: BARCHA Muxlisa transkriptlari (eski + yangi), faqat qo'ng'iroq

Holat: **dataset v2 tayyor (lokal; Jev A-004 F1+F2 qo'llandi), Pod skriptlari moslangan, trening ISHGA TUSHIRILMAGAN.** Pod yo'q, endpointga so'rov yo'q. Egasi tasdig'ini kutmoqda.
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

Quvur (v2: oxirida `python3 scripts/round6_resplit.py`): D1 eksport (`scripts/round6_export_d1.py`) → serverning o'z bo'luvchisi bilan audio (`tools/build-asr-dataset.mjs`, 55 s va 28 s guruhlari alohida, 700+52 yozuv yuklandi, bo'laklar soni mos kelmasligi 1 ta) → `scripts/round6_build_dataset.py`.

Kesish: >30 s bo'lak jimlik joyidan va gap chegarasidan bo'linadi. **Eski `prepare_calls_for_colab.py` dagi xato tuzatildi**: kesish nuqtasi har bo'lak ≤30 s bo'lishini ta'minlovchi feasible oraliqdan qidiriladi, matn↔vaqt xaritasi bir tekis emas, ovozli freymlar bo'yicha. Natija: chetlash 235/843 (28%) → **53/1580 (3.4%)**; chetlangan 0.5 soat. Shu sabab `recover_rejected.py` (GPU) bu raundda KERAK EMAS.

Darvozalar: har bo'lak ≤30 s, ≥1.2 s, cps 6–26; birortasi o'tmasa butun namuna chetlanadi.

| | Namuna | Soat | Qo'ng'iroq |
|---|---:|---:|---:|
| Xom (eval-120 chiqarilmasdan) | 1 827 | 20.49 | 747 |
| eval-120 qo'ng'iroqlari chiqarildi | −247 | | −119 |
| Chetlangan (cps 26, kesish topilmadi 24, qisqa 3) | −53 | −0.50 | |
| **Yakuniy** | **2 763** | **17.47** | **613** |
| train (v1) | 2 550 | 16.17 | 552 |
| dev (v1, qo'ng'iroq bo'yicha, seed 20260919) | 213 | 1.30 | 61 |

### Jev A-004 qarori va dataset v2 (2026-10-05)

Jev (TypeSafe) baholadi (`cf-call-analyzer/docs/audit/A-004/`): dataset sifati **«zaif» 1.06/3**; eng yaxshi `max_steps` **3000** (0.90; 2550: 0.08; 6000: 0.01 — 6000 overfit 0.85); tuzatishlar: **F1** mijoz bo'yicha dev (0.89, eng muhim), **F2** Claude tahrirlagan namunalarni olib tashlash (0.67), F4 qisman inson tekshiruvi (0.67, qimmat — hozir emas), F6 xatoli bo'laklarni tiklash (0.54), F3 yo'nalish balansi (0.45), F5 kutish (0.35) — hozir emas. F1+F2 dan keyin trening: 0.62. Qaror: F1+F2, so'ng 3000 qadam, L40S, $3 qo'riqchi.

**F1 — dev MIJOZ (lead_id) bo'yicha.** Eski bo'lish (qo'ng'iroq bo'yicha) da 61 dev qo'ng'irog'ining 54 tasi train bilan bir mijozni ulashgan edi (dev optimistik, checkpoint tanlovi buzilgan bo'lardi). `scripts/round6_resplit.py` (stdlib): mijoz guruhlari (lead_id yo'q 11 qo'ng'iroq — har biri alohida guruh), seed 20261005 bilan aralashtirib dev ≈10% gacha to'ldiriladi; bitta mijoz dev'ga umumiy hajmning 3% idan ko'pini kiritolmaydi (top-mijozlar 61–81 namuna — dev bitta mijoz qo'ng'irog'iga aylanmasin). Eski bo'lish `manifest.jsonl` da `split_v1` sifatida saqlangan.
**F2 — Claude tahrirlagan namunalar:** bo'laksiz (butun-qator) qo'ng'iroq, uning matni `diarizeWithClaude` dan o'tgan = **201 namuna / 1.18 soat / 125 qo'ng'iroq** (eski hujjatdagi «200» aniqlashtirildi; 180 tasi train'da, 21 tasi dev'da edi; `claude_edited` bayrog'i manifestda, ro'yxat `removed_claude_edited.jsonl`). **TRAIN'dan olib tashlandi. DEV'dan HAM olib tashlandi** — sabab: checkpoint dev loss bo'yicha tanlanadi; dev yorlig'i train yorlig'i bilan bir xil taqsimotda (sof Muxlisa) bo'lishi kerak, aks holda tinish/imlo uslubi farqi tanlovni siljitadi; ularni dev'da qoldirish tahrirlangan namunalar sonini tasodifga bog'laydi, qoldirishning foydasi esa yo'q. Natija: ular butunlay chiqarildi (train ham, dev ham emas).

**Yakuniy v2:**

| | Namuna | Soat | Qo'ng'iroq | Mijoz (lead) |
|---|---:|---:|---:|---:|
| v1 (eski) | 2 763 | 17.47 | 613 | 208 |
| F2: Claude-tahrirlangan olib tashlandi | −201 | −1.18 | −125 | |
| Havza (train+dev) | 2 562 | 16.29 | 488 | 183 (+3 lead'siz qo'ng'iroq) |
| **train** | **2 296** | **14.57** | 438 | 162 (+ lead'siz) |
| **dev (mijoz bo'yicha, 10.4%)** | **266** | **1.72** | 50 | 21 (+3 lead'siz qo'ng'iroq) |

Dev yo'nalishi: chiquvchi 220 / kiruvchi 46 (train: 1 678 / 618) — dev chiquvchiga biroz og'ishgan (F3 balans qilinmadi). **Assertlar qayta o'tdi (resplit ichida va train_calls.py da Pod'da yana):** train ∩ dev qo'ng'iroq = 0; **train ∩ dev mijoz (lead) = 0 (eski: 54 qo'ng'iroq ulashgan)**; train/dev ∩ eval-120 ∩ eval-50 = 0; dev matni (>60 belgi) train'da takrorlanmaydi: 0; Claude-tahrirlangan 0; takroriy yo'l 0. `all.csv` o'chirildi (eski bo'lish bilan edi) — endi `train.csv` + `dev.csv`, `train_calls.py` `DEV_CSV=dev.csv` bilan tayyor bo'lishni oladi va manifest bo'yicha lead assertini o'zi ham yuritadi (kesishsa treningni boshlamaydi; salbiy test o'tdi). Eval-120 ning 86 leadidan 55 tasi train'da — bu dev'ga taalluqli emas, lekin eval-120 ham bir mijozli qayta qo'ng'iroqlar tufayli biroz optimistik (Round 3–5 bilan bir xil sharoit).

Pastdagi jadval — v1 (F1/F2 dan oldin), tarix uchun:

Taqqoslash: Round 3 — 941 namuna / 6.5 soat; Round 4 — 1 258 / 8.2; Round 5 — 1 654 qo'ng'iroq namunasi. Bu to'plam **~2.7× ko'p**. Shundan **yangi qo'ng'iroqlardan 1 507 namuna / 9.38 soat (276 qo'ng'iroq)**, eskidan 1 256.
So'zlar ≈ 127 000; mediana namuna 25.7 s (57% ≥25 s, 2.3% <5 s); mediana cps 15.2 (butun 14.8, kesilgan 15.2 — mos, kesish nisbatni buzmagan).
Davomiylik: kiruvchi 4.42 soat / chiquvchi 13.05 soat. Oylar: may–okt 2026.

**Sizib chiqish assertlari o'tdi** (build skriptida, treningdan oldin): train ∩ dev = 0; train ∩ eval-120 = 0; dev ∩ eval-120 = 0; eval-50 ⊂ eval-120 va u bilan kesishish 0; bir xil matn (>60 belgi) dev'da train'dagi bilan 0; takroriy yo'l 0. Eval ro'yxati: `eval_calls_120.json` + `eval120/eval.csv`/`eval50.csv` yo'llaridan (120 ta noyob qo'ng'iroq — hujjatdagi eski "117" noto'g'ri; `VOICE-AI-KNOWLEDGE.md` §6 ni PM tuzatsin).

Fayllar: `train.csv`, `dev.csv` (v2), `manifest.jsonl` (call, bo'lak, kesilgan-mi, cps, sana, yo'nalish, lead, split, split_v1, claude_edited, rol), `rejected.jsonl`, `audio/*.flac` (746 MB), `eval_calls_120.json` nusxasi.

## 3. Trening rejasi (v2; EGASI TASDIG'I KUTILMOQDA)

Domen ulushi: **100% qo'ng'iroq, tashqi korpus 0** (Round 5 saboqi: tashqi toza korpus qo'shilsa qo'ng'iroq ≥70% bo'lsin; bu hajmda tashqi qo'shish ma'nosiz — ≤1 100 namuna). Podkast/UzbekVoice yo'q. Boshlang'ich model: round-2 (`Sunnat0091/whisper-large-v3-uz`) — farq faqat ma'lumotdan chiqsin (Round 4 mantig'i). LoRA r32/α64 q,v; LR 5e-5; batch 8; warmup 100.

**Qadam hisobi (v2).** Formula: `namunalar × 8 / batch = 2 296 × 8 / 8 = 2 296 qadam` (8 epoxa, 287 qadam/epoxa). **`MAX_STEPS=3000` (10.4 epoxa, yuqori chegara; Jev: 3000 — 0.90, aniq tanlov)**, `EVAL_STEPS=SAVE_STEPS=250` (12 baholash), `PATIENCE=3` (750 qadam yaxshilanmasa to'xtaydi), `load_best_model_at_end` (dev loss, dev endi MIJOZ bo'yicha ajratilgan — tanlov ishonchliroq). Eng yaxshi nuqta ~2 000–2 500 kutiladi.

**Egasining 6000 qadami — KO'P (Jev ham tasdiqladi: overfit 0.85).** 6000×8/2296 = **20.9 epoxa**, formuladan 2.6×. Round 3 da eval loss 8.5-epoxada minimal (0.8134), 25 epoxaga yetganda 0.899 gacha o'sgan (yodlash); Round 4 ham ~8 epoxada. Ma'lumot 2.7× ko'paygani minimumni kechiktirmaydi — u epoxada o'lchanadi, qadamda emas. 6000 ham EarlyStopping tufayli ~3000 da to'xtashi mumkin, lekin qo'riqchi bashorati $3.5 ga chiqib, treningni noto'g'ri to'xtatishi yoki 2× pul sarflashi mumkin. Agar egasi ko'proq natija istasa — qadam emas, **ma'lumot**: 798 xatoli Muxlisa bo'lagini qayta o'girish va yangi qo'ng'iroqlar.

**Pod va narx — YAKUNIY BAHO (v2)** (tezlik Round 3-4 da o'lchangan: 1.02 s/qadam L40S, batch 8; shu to'plamda namuna uzunligi o'sha; L40S ≈ $1.10/**soat**):
| Bosqich | Vaqt | $ |
|---|---:|---:|
| Pod ishga tushishi, repo + paketlar, 843 MB ma'lumot ko'chirish (train+dev audio 725 MB + eval-120 audio 114 MB), model yuklash | ~25 min | 0.46 |
| Trening ≤3000 qadam × 1.02 s = 51 min + 12 dev baholash (~5 min) | ~56 min (erta to'xtasa ~45) | 1.03 (0.83) |
| round6 → CT2 + adapter + eval-120 (310 namuna, eval-50 uning ichida; alohida eval-50 yurgizilmaydi) | ~18 min | 0.33 |
| **Jami** | **~1.65 soat** | **≈ $1.8** (zaxira 25% bilan **$2.3**) |

A40 ($0.49/soat, tezlik o'lchanmagan, ~1.5–1.8× sekin deb taxmin): ~2.4 soat ≈ $1.2 — tavsiya EMAS (o'lchanmagan tezlik, qo'riqchi bashorati noaniq). 6000 qadam (taqqoslash uchun; Jev: yo'q) ~2.7 soat ≈ $3.0–3.5.

**Qo'riqchi:** `BUDGET_USD=3.0`, `POD_RATE=1.10` (haqiqiy stavkaga o'zgartiring), `POD_START_EPOCH` Unix epoch (round6_start.sh o'zi beradi). Har 250 qadamda yakuniy narx bashorati = (o'tgan+qolgan trening vaqti)×stavka + SETUP (o'tgan sozlash vaqti×stavka + 0.50 CT2/eval zaxirasi); >$3.0 bo'lsa trening to'xtaydi va eng yaxshi checkpoint saqlanadi. Normal bashorat ≈ $2.0–2.3 (chegaradan ~0.7 past).

Egasining umumiy chegarasi: Pod ≤$5. Volume disk 60 GB (Pod bilan o'chadi), container 30 GB. Pod tugagach model Mac'ga `models/round6/` + `SHA256SUMS`, keyin PM o'chiradi.

**Pod skriptlari MOSLANDI (v2):**
- `round6_pod.sh`: tiklash va round-2 CT2 bosqichlari OLIB TASHLANDI; `CALLS_DIR=data/round6-dataset TRAIN_CSV=train.csv DEV_CSV=dev.csv MANIFEST=manifest.jsonl`; `MAX_STEPS=3000 EVAL/SAVE=250 WARMUP=100 BATCH=8 LR=5e-5 PATIENCE=3` (`load_best_model_at_end` train_calls.py da, metrika: dev loss); `BUDGET_USD=3.0`, `POD_RATE=1.10` standart; oxirida eval-120 (310 namuna; eval-50 uning 129 namunasi — Mac'da per-sample natijadan ajratiladi, juftlashgan bootstrap uchun ham).
- `round6_send.sh`: faqat train+dev audio (2 562) + `train.csv`, `dev.csv`, `manifest.jsonl`, `eval_calls_120.json`, `eval120/eval.csv` + eval-120 audio (310) = 2 879 fayl / 843 MB; hajm 2296/266/310 ga teng bo'lmasa to'xtaydi. `train_calls.py` va `scripts/eval_pod.py` ham yuboriladi (Pod GitHub'dan klonlaydi, lokal o'zgarish commit qilinmagan bo'lishi mumkin).
- `round6_start.sh`: standartlar 1.10 / 3.0 / 3000.
- `train_calls.py`: yangi `DEV_CSV`/`MANIFEST` rejimi (orqaga mos: `DEV_CSV` berilmasa eski xatti-harakat). Assertlar: qo'ng'iroq, **mijoz**, eval-120, Claude-tahrirlangan, yo'l — hammasi treningdan OLDIN; bo'lmasa to'xtaydi.
- Sinov: bash sintaksis, send.sh ro'yxat qismi lokal yuritildi (2879 fayl, 843 MB), `_load_presplit` lokal bajarildi (o'tdi) va dev qatori train'ga qo'shilganda assert ushladi. Pod'da yurgizilmagan (Pod yo'q).

Ixtiyoriy (alohida PM ruxsati): Pod'da round-2 CT2 bilan train namunalarini bir marta WER'lab, >80% WER (nomoslik) namunalarni chiqarish (~5 min, ~$0.1) — kesilgan bo'laklar moslik tekshiruvi akustik tasdiqlanmagan (hozir faqat cps).

**Muvaffaqiyat mezoni (oldindan):** eval-120 (310 namuna) da production'ga nisbatan juftlashgan bootstrap 95% oralig'i BUTUNLAY noldan past. Production eval-50 da 37.84%. Natija ±3 punktdan kichik bo'lsa — "ishonchli emas".

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
- **Lead kesishuvi — dev uchun HAL QILINDI (F1)**: dev endi train bilan bir mijozni ulashmaydi (eski: 54/61). Qolgan: eval-120 ning 86 leadidan 55 tasi train'da — bir mijozli qayta qo'ng'iroqlar ovoz/ismni ulashadi, eval-120 WER biroz optimistik bo'lishi mumkin (Round 3-5 da ham shunday, taqqoslash bir xil sharoitda; eval-120 ni mijoz bo'yicha qayta tuzish alohida qaror).
- **Claude tahrirlangan matn — HAL QILINDI (F2)**: 201 namuna (1.18 soat) train va dev'dan olib tashlandi. Qolgan to'plam sof Muxlisa matni.
- **Jev: dataset «zaif» 1.06/3** — F1+F2 dan keyin ham asosiy zaif tomonlar saqlanadi: Muxlisa yorlig'i (xatolari bilan), mijoz/mahsulot xilma-xilligi kam, inson tasdig'i yo'q (F4), nomutanosiblik (F3). Muvaffaqiyat mezoni shuning uchun oldindan qat'iy.
- **798 Muxlisa xatolik qatori** (458 bo'lak) — yo'qolgan matn; 170 qo'ng'iroq bo'laklari to'liq emas.
- **eval-120 chiqarilishi**: 119 qo'ng'iroq (247 namuna) trening to'plamidan chetda — to'g'ri, lekin ma'lumot hajmining 11% ini yeydi. F2 qo'shimcha 7% ni oldi: v2 train 14.57 soat (v1: 16.17).
- Hajm cheklovi saqlanadi: 14.6 soat train (16.3 havza) — Round 3 dagi "qadam emas, MA'LUMOT cheklov" xulosasi 2.7× ko'p ma'lumotda ham tekshirilishi kerak.

## 6. Keyingi qadamlar (PM qarori kerak)

1. **Egasi tasdig'i:** 3000 qadam cap, L40S (~$1.10/soat), taxminan 1.65 soat ≈ $1.8 (zaxira bilan $2.3), qo'riqchi $3.0. Tasdiqdan keyin PM Pod yaratadi (L40S, container 30 GB, volume 60 GB) va:
   `./scripts/round6_send.sh <host> <port> && POD_START_EPOCH=<Pod boshlangan Unix epoch> POD_RATE=<haqiqiy $/soat> ./scripts/round6_start.sh <host> <port>`
2. Pod tugagach (`/workspace/round6/PIPELINE_DONE`): model Mac'ga `models/round6/` + `SHA256SUMS`, so'ng PM Pod'ni o'chiradi; natija `docs/ROUND6-RESULT.md` (eval-120 va ichidagi eval-50, juftlashgan bootstrap).
3. `VOICE-AI-KNOWLEDGE.md` ga qo'shish: eval-120 = 120 qo'ng'iroq; `build-asr-dataset.mjs` butun+bo'lak dublikat xatosi; eski feasible-range xatosi tuzatildi; dev mijoz (lead) bo'yicha ajratilishi shart (Jev A-004 F1) va Claude tahrirlagan matnlar yorliqqa kirmasligi (F2).
4. Keyingi raundlar uchun (hozir emas, Jev): F4 qisman inson tekshiruvi, F6 798 xatoli bo'lakni qayta o'girish, F3 yo'nalish balansi.

Qayta yig'ish: `python3 scripts/round6_export_d1.py` → (cf-call-analyzer'da) `CHUNK_SECONDS=55|28 node tools/build-asr-dataset.mjs <meta> <raw>` → `python3 scripts/round6_build_dataset.py` (numpy, soundfile) → `python3 scripts/round6_resplit.py` (v2 bo'lish; stdlib).

O'zgargan fayllar (commit qilinmagan): `docs/ROUND6-PLAN.md`, `scripts/round6_resplit.py` (yangi), `scripts/round6_pod.sh`, `scripts/round6_send.sh`, `scripts/round6_start.sh`, `train_calls.py`. Dataset (`data/round6-dataset/`) gitignore'da.

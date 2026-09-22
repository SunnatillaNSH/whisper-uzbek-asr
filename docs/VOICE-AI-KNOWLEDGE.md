# SellUp Voice AI — bilim bazasi

Bu hujjat `whisper-uzbek-asr` loyihasining butun tarixini, texnik yechimlarini
va tuzoqlarini bitta joyga jamlaydi. Manba: loyiha xotirasi
(`whisper-uzbek-asr-project.md`, `uzbek-voice-datasets.md`) va repo ichidagi
kod/hujjatlar (2026-09-20 holatiga). Raqamlar faqat manbada aniq yozilgan
joylardan olingan; noaniq joylar "tasdiqlanmagan" deb belgilangan.

**Mijoz matni, telefon raqami va kalit qiymatlari bu hujjatda YO'Q** — repo
ochiq, faqat raqamli o'lchovlar keltirilgan.

---

## 1. Maqsad va o'lchov falsafasi

Loyiha maqsadi — SellUp'ning qo'ng'iroq tahlil tizimi (`cf-call-analyzer`)
uchun o'zbek tilida ishlaydigan, arzon va tez ishlaydigan o'z nutqni matnga
o'girish (ASR) xizmatini qurish. Bozordagi tayyor yechim — **Muxlisa AI**
(Uzinfocom mahsuloti, https://muxlisa.uz) — call-markazlar uchun mo'ljallangan
tijorat STT/TTS xizmati va bu loyihaning bilvosita raqobatchisi. Muxlisa'ning
ochiq korpusi yo'q, faqat API sifatida mavjud.

**O'lchov etaloni — Muxlisa AI matni.** Har bir real qo'ng'iroq ikkala tizim
bilan ham matnga o'giriladi (SellUp Voice AI va Muxlisa AI, alohida
jadvallarda: `voice_transcripts` va `transcripts`), so'ng ular orasidagi
farq WER (Word Error Rate) sifatida o'lchanadi. Bu WER **"Muxlisa bilan
qancha farq"** degani, **"qanchalik noto'g'ri"** degani EMAS — Muxlisa ham
audioni eshitgan mustaqil tizim, mutlaq haqiqat emas. WER = 0% Muxlisa bilan
aynan bir xil chiqish demakdir, undan yaxshiroq ekanini bildirmaydi.

**Strategik cheklov.** Shu o'lchov usuli bilan model **Muxlisa'dan o'tib keta
olmaydi** — faqat unga yaqinlasha oladi. Agar maqsad Muxlisa'dan sifatli
bo'lish bo'lsa, bu yondashuv yetarli emas; agar maqsad Muxlisa'ga yaqin
sifatni **arzonroq va o'z infratuzilmasida** olish bo'lsa (so'rov uchun
to'lovsiz, Muxlisa esa balans talab qiladi), unda mantiqiy. Bu ikkilanish
ochiq savol sifatida qoladi (10-bo'limga qarang).

Modelning nazariy yuqori chegarasi — **Muxlisa darajasi**. Undan
yaxshiroq bo'lish bu pipeline bilan o'lchanmaydi va tasdiqlanmagan.

---

## 2. Model zanjiri

```
openai/whisper-large-v3  (1.5 mlrd parametr, bazaviy)
        │
        ▼  LoRA fine-tune (PEFT, faqat q_proj/v_proj, r=32/alpha=64/dropout=0.05, ~0.3% parametr)
        │
        ▼  merge_and_unload() — adapter asosiy modelga birlashtiriladi, fp16
        │
        ▼  CTranslate2 konvertatsiya (--quantization float16)
        │
        ▼  faster-whisper (CT2 dvigateli) — RunPod Serverless'da serving
```

To'liq fine-tuning T4 GPU'ga sig'maydi (optimizator holati ~25 GB talab
qiladi), shuning uchun LoRA ishlatiladi. Trening tugagach adapter asosiy
modelga birlashtiriladi va oddiy `WhisperForConditionalGeneration` sifatida
saqlanadi — serving kodi PEFT'ga bog'liq bo'lib qolmaydi.

CT2'ga o'tish **qayta trening emas** — faqat format konvertatsiyasi,
og'irliklar o'zgarmaydi:

```bash
ct2-transformers-converter --model <hf-repo> --output_dir out \
  --quantization float16 \
  --copy_files tokenizer.json tokenizer_config.json preprocessor_config.json \
               vocab.json merges.txt special_tokens_map.json \
               added_tokens.json normalizer.json
```

### Hugging Face repolar

| Repo | Format | Ochiq/maxfiy | Vazifasi |
|---|---|---|---|
| `Sunnat0091/whisper-large-v3-uz` | transformers | round 1-2 natijasi | Round 3+ uchun boshlang'ich model |
| `Sunnat0091/whisper-large-v3-uz-calls` | transformers | **maxfiy** | trening uchun (real mijoz suhbatida o'qigan) |
| `Sunnat0091/whisper-large-v3-uz-calls-ct2` | CTranslate2 float16 | **maxfiy** | **production serving** (joriy endpoint) |
| `Sunnat0091/whisper-large-v3-uz-calls-r5-ct2` | CTranslate2 float16 | maxfiy, **yuklanmagan** | Round 5 — reja bosqichida qoldi, joylashtirilmadi |

`upload_to_hf.py` Round 3 dan keyin (commit `7c31d07`) standart bo'yicha
repo'ni **maxfiy** yaratadigan qilib o'zgartirildi — sabab: model real mijoz
suhbatlarida ko'p epoxa aylangan, iboralarni yodlab olgan bo'lishi mumkin.

---

## 3. Raundlar jadvali

| Raund | Ma'lumot | Boshlang'ich model | Natija | Xulosa |
|---|---|---|---|---|
| **1** | ~23k namuna (4 ta ochiq HF dataset, niyat ~47k edi, bug tufayli kamroq yozildi) | `whisper-large-v3` | WER **34.01%** (aralash ochiq eval) | Aniq/formal nutqda yaxshi, real qo'ng'iroqda ism nomuvofiqligi kabi xatolar bor |
| **2** | ~98k namuna, 7 ta ochiq manba | `whisper-large-v3` | WER **27.82%** (ochiq eval), real qo'ng'iroqda **61.46%** | Eval'da 18% nisbiy yaxshilanish, lekin real qo'ng'iroqda sezilarli farq yo'q — **til bilimi yetarli, domen farqi hal qilinmagan** |
| **3** | 941 namuna (o'z qo'ng'iroqlari, ≤30s, telefon augmentatsiyasi bilan) | round-2 modeli | WER **48.02%** (real qo'ng'iroq) | 61.46% → 48.02%, **−13.44 punkt** — birinchi haqiqiy domen yaxshilanishi |
| **4** | 1258 namuna (+308 yangi 28s partiyadan, +9 forced-alignment) | round-2 modeli (ATAYLAB, round-3 emas) | WER **49.06%**, dev loss 2.8% yaxshi | Juftlashgan bootstrap: farq +1.04 punkt, **p=0.732** — statistik jihatdan **TENG** round-3 bilan |
| faster-whisper o'tishi | — | round-3/4 modeli | transformers pipeline 47.54% → CT2 **43.47%**, 5.9x → **14.7x** tezlik | Bir xil og'irlik, faqat dekodlash (VAD, `condition_on_previous_text=False`, temperatura zaxirasi) sifatni ham, tezlikni ham oshirdi |
| Endpoint tuzatishlari | — | — | VAD `min_silence_duration_ms` 500→2000: 48.27%→**44.38%** (bir vaqtda hotwords o'chirilgach) | Ikkita alohida tuzatish birga o'lchandi (4-bo'limga qarang) |
| **5** | 1654 namuna (989 + 665 tiklangan) + UzbekVoice 25 000, qo'ng'iroq ulushi batch'da 36% | round-2 modeli | eval-50: production **37.84%** vs Round 5 **46.01%**, **+8.17 punkt YOMON**, p=0.000 | Tashqi toza korpus (UzbekVoice) batch'ni bosib ketdi, model telefon domenidan uzoqlashdi — Round 1-2 saboqining takrori. **Production almashtirilmadi.** |
| **6** | rejalashtirilgan: FAQAT qo'ng'iroq (round6_calls.csv + tiklangan), UzbekVoice YO'Q | round-2 modeli | **hali yurilmagan** (2026-09-20 holatiga: kod tayyor, `scripts/round6_pod.sh` yozilgan, ishga tushirilmagan) | Round 5 saboqidan kelib chiqqan reja — B qo'li (faqat qo'ng'iroq)ni sinash |

---

## 4. Endpoint sozlash tarixi (ablatsiya)

Bitta 60 namunali barqaror to'plamda o'lchangan (`analysis/endpoint_ablation.md`):

| Sozlama | WER | So'z qamrovi |
|---|---|---|
| `beam_size=1` | 50.18% | 89.2% |
| VAD 2000 (hozirgi standart) | 48.27% | 88.2% |
| VAD o'chiq | 47.98% | 89.7% |
| **hotwords o'chiq** | **44.38%** | **95.1%** |
| baseline (dastlabki hujjat) | 43.47% | — |

**Hotwords — asosiy aybdor edi (−3.9 punkt).** Domen lug'atini (`Face ID,
davomat dasturi`, va h.k.) standart sifatida berish dekoderni ro'yxat tomon
og'dirar, mos kelmasa haqiqiy so'zni tashlab yuborardi. Xulosa: `handler.py`
da `ASR_HOTWORDS` standart **bo'sh**, kerak bo'lganda so'rovda alohida
beriladi (commit `58b3122`).

**VAD — muammo boshqa joyda hal bo'lgan edi.** `min_silence_duration_ms`ni
500 → 2000 ga o'zgartirilgach (commit `060e327`), VAD yoqiq/o'chiq deyarli
farq qilmaydi (48.27% vs 47.98%, 60 namunada shovqin doirasida). Tuzatishdan
oldin (254 qo'ng'iroqli o'lchov, eski hujjat): tushib qolish xatolarining
**44.3%** i, so'z qamrovi **80.7%**. Tuzatishdan keyin: tushib qolish
**31.5%**, qamrov **90%** atrofida.

**beam_size — tegilmasin.** `beam_size=1` standart 5'dan 2 punkt yomon —
faster-whisper stack'ida beam 5 haqiqatan foydali.

Ikkala tuzatish (VAD ostonasi + hotwords) birga, **bir xil 74 qo'ng'iroq**
kesishmasida o'lchandi: **50.6% → 36.0%**, ya'ni **+14.6 punkt yaxshilanish**
(`analysis/FINAL_COMPARISON.md`).

---

## 5. Datasetlar

### Ochiq HF korpuslari (Round 1-2, 7 ta manba)

| Manba | Cheklov | Tavsif |
|---|---:|---|
| `yakhyo/mozilla-common-voice-uzbek` | 8 000 | Common Voice, validatsiya qilingan |
| `DavronSherbaev/uzbekvoice-filtered` (mirror `ai4uz/uzbekvoice-filtered`) | 20 000 (Round 5 da 25 000) | 503 000 qator / ~583 soat, Apache 2.0, UzbekVoiceBot Telegram orqali yig'ilgan, 20 mintaqa aksenti |
| `mrmuminov/uzbek_voice` | 30 000 | 861k qatorlik pool |
| `shunyalabs/uzbek-speech-dataset` | hammasi (~2 943–4 168) | matn ustuni `transcript` (boshqa nom emas) |
| `islomov/news_youtube_uzbek_speech_dataset` | 15 000 | YouTube yangiliklari, aralash dialekt |
| `islomov/it_youtube_uzbek_speech_dataset` | 10 000 | ba'zi ingliz aralash |
| `BoburAmirov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset` | hammasi (~14 547) | tabiiy suhbat, Toshkent dialekti — qo'ng'iroq domeniga eng yaqin ochiq manba |

Boshqa tekshirilgan, lekin ishlatilmagan/muammoli: **USC**
(`issai/Uzbek_Speech_Corpus`, 105 soat, MIT — HF standart yuklovchisida
WebDataset xatosi, tasdiqlanmagan holda qoldirilgan), **FeruzaSpeech**
(60 soat, gated, HF_TOKEN + shartlarga rozilik kerak — ishlatilmagan).
**O'zbekcha raqamlar/sonlar uchun alohida dataset yo'q** — bu tasdiqlangan
(2026-09-17 tekshirilgan).

### O'z qo'ng'iroq datasetlari (Round 3+)

| Dataset | Namuna | Soat | ≤30s ulushi | Izoh |
|---|---:|---:|---:|---|
| `calls-dataset` (55s bo'lak) | 843 (405 unikal qo'ng'iroq) | 11.74 | 28% (238) | Muxlisa transkriptiga MoyZvonki audiosi moslashtirildi; server bo'luvchilari bilan kesildi (`MAX_CHUNK_SECONDS=55`) |
| `calls-colab` (jimlik/gap chegarasidan kesilgan) | 1022 (239 butun + 783 bo'lak) | 6.51 | 100% | `prepare_calls_for_colab.py`: energiya asosidagi jimlikdan kesish + gap chegarasidan matn bo'lish |
| `calls-rejected` | 235 bo'lak | 5.19 | 0% (shu sabab chetlangan) | `calls-colab` kesa olmagan uzun bo'laklar (mediana 55s) |
| `calls-dataset-28s` | 308 | 2.20 | **100%** | `MAX_CHUNK_SECONDS` 55→28 ga tushirilgach yig'ilgan yangi partiya, qayta bo'lishsiz to'liq yaroqli |
| `sellup-train-ready.csv` (birlashgan) | 546 | 3.32 | 100% | eski 238 + yangi 308 |
| tiklangan (`recover_rejected.py`, Round 5) | 665 | 1.54 | 100% | 183 bo'lakdan (eval-120'ning 52 bo'lagi chiqarilgach) so'z-vaqt langarlash bilan tiklandi |

**Sifat darvozalari** (`prepare_calls_for_colab.py`, `recover_rejected.py`,
`build_weighted.py` da bir xil): har bo'lak **≤30 s**, **≥1.2 s**,
belgi/soniya (cps) **6–26** oralig'ida; birortasi o'tmasa **butun namuna
chetlatiladi** — noto'g'ri moslashtirilgan namuna yo'q namunadan yomonroq
degan tamoyil bilan.

**`recover_rejected.py` nima qiladi:** vaqtni modelning o'zidan, matnni
yorliqdan oladi. faster-whisper `word_timestamps=True` bilan gipoteza
so'zlari va vaqtlarini chiqaradi, gipoteza so'zlari yorliq so'zlariga
Levenshtein bilan tekislanadi, **faqat aynan mos tushgan so'zlar** "langar"
bo'ladi. Kesim faqat yorliqning jumla chegarasida (`.!?`) va ikkala uchi ham
langarlangan joyda qilinadi — ya'ni **yorliq matni o'zgarmaydi**, faqat unga
vaqt biriktiriladi. Forced-alignment'dan farqi: bu segment emas, **so'z**
vaqtlarini ishlatadi (segment vaqtlari bu fayllarda buzuq edi — 5-band,
8-bo'lim).

---

## 6. Eval to'plamlari

| To'plam | Qo'ng'iroq | Namuna | So'z | Holati |
|---|---:|---:|---:|---|
| eval-81 (eski) | 31 | 81 | 3 757 | **eskirgan** — ishonch oralig'i ±6-7 punkt, 7 punktdan kichik farqni ko'rmaydi. Round 2→3 sakrashini (13.4 punkt) to'g'ri ko'rsatdi, lekin Round 3→4 (1.04 punkt)ni ajrata olmadi |
| **eval-120** | 117 (asl 31 ta "locked" + 89 yangi) | 310 | 14 015 (hujjatlarda 14 016/13 616 talqinlari ham uchraydi — aniq raqam `eval_calls_120.json`) | Joriy asosiy eval. Qatlamlangan-proporsional tanlov: partiya (eski/yangi) × yo'nalish (kiruvchi/chiquvchi) bo'yicha populyatsiya nisbatida, davomiylik kvartillari bo'yicha teng. Ishonch oralig'i **~±3 punkt** |
| **eval-50** | 50 (31 "locked" + 19 qatlamli) | 129 | 6 049 | eval-120'ning qat'iy belgilangan qismi to'plami (foydalanuvchi "ko'pi bilan 50 qo'ng'iroq" cheklovi tufayli). eval-120'ning 42% so'zi. Ishonch oralig'i **~±5 punkt** (120'nikidan ~1.54 barobar keng, `sqrt(14015/5900)` taxminiga asosan) |

**Nega qatlamlash muhim:** ajratish **qo'ng'iroq** bo'yicha, namuna bo'yicha
emas — bitta suhbatning bo'laklari train va eval'ga bo'linib ketsa, model
eval suhbatini treningda ko'radi va natija soxta yaxshi chiqadi. Yo'nalishni
(kiruvchi/chiquvchi) teng ulushda tanlash ham XATO bo'lardi — populyatsiyada
chiquvchi 76%, kiruvchi 24%; teng tanlansa eval haqiqiy trafikni aks
ettirmaydi.

**Juftlashgan bootstrap** (`scripts/compare_runs.py`, 10 000 qayta tanlash) —
ikki model/sozlamani **bir xil namunalarda** solishtirish usuli. Har namuna
juftini (eski WER, yangi WER) qayta-qayta tasodifiy tanlab, farqning 95%
ishonch oralig'i va p-qiymati hisoblanadi. Natija "yaxshilandi/yomonlashdi/
teng" namuna soni bilan birga chiqadi. Round 4 (p=0.732, statistik teng) va
Round 5 (p=0.000, ishonchli yomon) ikkalasi ham shu usul bilan ajratildi.

**S/D/I tahlili** — har WER'ni substitution (almashtirish) / deletion
(tushib qolish) / insertion (qo'shib yuborish) ga ajratib ko'rish. Round 5'da
aynan shu tahlil model xulqi o'zgarganini isbotladi: tushib qolish kamaydi
(25%→14%), lekin almashtirish (58%→64%) va qo'shish (16%→20%) oshdi — sof
hisobda zarar, chunki apparat farqi (turli GPU) xato TURINI bunday
yo'naltirilgan holda o'zgartirmaydi.

**Normallashtirish — apostrof birlashtirish.** O'zbek lotin yozuvida
apostrof harfning bir qismi (`o'`, `g'`). Eval to'plamida uch xil unikod
variant aralash uchraydi: U+2018 (`'`), ASCII (`'`), U+2019 (`'`).
Normallashtirmasak bir xil so'z boshqa-boshqa hisoblanadi; so'zlarning
**18.7% ida apostrof bor**, ya'ni bu WER'ni ~19 foiz punktgacha sun'iy
oshirib yuborishi mumkin edi. Yechim (`norm()` funksiyasi, commit
`4570470`): barcha variantlarni bitta belgiga keltirish, lekin apostrofning
o'zini **harf sifatida saqlash** (olib tashlansa `ozim`/`o'zim` qo'shilib
ketadi). Tuzatilgach qoldiq — 0.09 punkt.

---

## 7. Serving

### handler.py sozlamalari (faster-whisper, production)

| Parametr | Qiymat | Sabab |
|---|---|---|
| `beam_size` | 5 | beam 1 — 2 punkt yomon (4-bo'lim) |
| `vad_filter` | `true`, `min_silence_duration_ms=2000`, `speech_pad_ms=400` | standart 500'dan 4x yumshatilgan — qisqa tasdiq so'zlarini (`ha`, `xo'p`) kesib tashlamaslik uchun |
| `hotwords` | standart **bo'sh** (`ASR_HOTWORDS=""`) | to'ldirilsa WER yomonlashadi (4-bo'lim) |
| `condition_on_previous_text` | `False` | standart `True` Whisper'ning eng mashhur nuqsoniga olib keladi — shovqindan keyin matnni takrorlash halqasi |
| temperatura zaxirasi | `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` | dekodlash chalkashsa (siqilish nisbati yoki log-ehtimollik chegaradan chiqsa) yuqoriroq temperatura bilan qayta uriniladi |
| gallyutsinatsiya darvozalari | `compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`, `no_speech_threshold=0.6` | + regex asosidagi tozalash (`_clean()`): "obuna bo'ling", "subscribe", "продолжение следует" kabi trening qoldiqlarini va ketma-ket takrorlangan bir xil jumlani olib tashlaydi |

### RunPod Serverless

| | |
|---|---|
| Endpoint ID | `vwobkifazoyyh1` |
| GPU | 24 GB (1-chi) + 16 GB (2-chi), max workers 1, idle timeout 5s, execution timeout 600s |
| Model manzili | Docker образ ichida EMAS — konteyner ishga tushganda HF'dan `HF_MODEL_ID`/`HF_TOKEN` orqali yuklanadi (maxfiy repo + build-arg maydoni yo'qligi sababli) |
| Sovuq start | ~155 s (dastlabki, transformers pipeline bilan) → **~25 s** (faster-whisper'ga o'tgach, образ 8 GB → 1-2 GB) |
| Tezlik | **19.1× realtime** (57 s audio → 2.99 s, o'lchangan) |
| Narx | ~$0.00019/GPU-soniya, ~**$0.037/audio-soat** ekvivalenti (60 soat audio ≈ $2.15 hisobidan) |

### Ovoz-konsoli proksisi (Cloudflare Worker)

`https://ovoz-konsoli.sellup2.workers.dev` — statik konsol sahifasi +
`/api/*` proksi. RunPod hisobining API kaliti Worker secret'ida
(`RUNPOD_API_KEY`), brauzerga hech qachon yuborilmaydi. Faqat
`health`/`run`/`status` ochiq — pod yaratish/o'chirish proksi orqali
mumkin emas. Tashqi mijozlar `API_TOKENS` secret'idagi (vergul bilan
ajratilgan) alohida kalitlar bilan kiradi. Uch rejim (`console/worker.js`):
`API_TOKENS` yo'q — ochiq, hech narsa tekshirilmaydi; `API_TOKENS` bor,
`API_ENFORCE≠1` — **kuzatuv**: hamma o'tadi, tokensiz so'rovlar statistikada
alohida sanaladi; `API_ENFORCE=1` — **majburiy**: tokensiz so'rov rad
etiladi. Majburiy rejimga faqat SellUp trafigi token bilan kelayotgani
kuzatuvda tasdiqlangach o'tiladi — aks holda production transkripsiyasi
to'xtaydi. Jonli oqim KV'da (`OVOZ_FEED`, TTL 7200 s = 2 soat, o'zi
o'chadi), `FEED_TOKEN` bilan himoyalangan.

---

## 8. Tuzoqlar (takrorlanmasin)

| # | Tuzoq | Ta'siri / yechimi |
|---|---|---|
| 1 | `predict_with_generate=True` + PEFT/LoRA | `RuntimeError: Input type (float) and bias type (Half)` — Trainer'ning ichki `generate()` chaqiruvi autocast tashqarisida ishlaydi. Yechim: `predict_with_generate` ishlatmaslik, loss-only eval, WER'ni treningdan keyin qo'lda hisoblash. |
| 2 | `dataset.map(num_proc=...)` model GPU'da bo'lganda | Har qanday `num_proc` (hatto `1`) CUDA context to'qnashuvi sabab **deadlock** (0% da abadiy osilib qoladi). Argumentni umuman bermaslik kerak. |
| 3 | transformers 5.x vs 4.x tokenizer formati | v5: `return_timestamps` + timestamp logits processor `TypeError`; `processor_config.json` (`preprocessor_config.json` o'rniga); `extra_special_tokens` RO'YXAT (4.x lug'at kutadi) → `AttributeError`. Yechim: `pip install "transformers>=4.46,<5"`, yoki tokenizer fayllarini `openai/whisper-large-v3`dan nusxalash (LoRA faqat q/v ga tegadi, tokenizerga emas). |
| 4 | `datasets<4` va torchcodec | `datasets` 4+ audio dekodlash uchun torchcodec talab qiladi (obrazda yo'q) → `datasets<4` pin qilinadi. |
| 5 | HF Audio ustunini NOM emas, TUR bo'yicha topish | `DavronSherbaev/uzbekvoice-filtered`da audio `path` ustunida, lekin uning HF feature turi `Audio` — nom bo'yicha qidirish barcha 20 000 qatorni jim tashlab yuborgan. Yechim: `isinstance(feat, Audio)` bilan tur bo'yicha topib, keyin nomini `audio`ga o'zgartirish. |
| 6 | PEP 668 | Yangi Ubuntu obrazida pip bloklanadi, `--break-system-packages` shart. |
| 7 | RunPod natijalari ~30 daqiqada o'chishi | `/status/{id}` faqat cheklangan vaqt saqlanadi. Yuborish va yig'ishni **aralashtirish** kerak (masalan 50 tadan yuborib, har paketdan keyin yig'ish) — aks holda ketma-ket 1249 ta yuborilib, 45 daqiqadan keyin so'ralganda 869 tasi 404 qaytargan (haqiqiy voqea, `analysis/FINAL_COMPARISON.md`). |
| 8 | SSH fon jarayoni SIGHUP | Uzoq trening jarayoni oddiy `&` bilan ishga tushirilsa, SSH ulanish uzilganda stdin ushlab turgani sabab SIGHUP olib o'ladi (Round 5'da shunday bo'lgan). Yechim: `setsid nohup ... < /dev/null > /dev/null 2>&1 & disown`. |
| 9 | UTC va mahalliy vaqt farqi byudjet hisobida | Pod soati UTC, Mac mahalliy vaqtda — byudjet bazasi (`elapsed`) MANFIY chiqib, byudjet qo'riqchisi ishlamay qolgan. Yechim: Pod boshlanish vaqtini Unix epoch'da uzatish (vaqt mintaqasi muammosi yo'q). |
| 10 | Maxfiy HF repo tokensiz yuklanmaydi | `snapshot_download`/`from_pretrained` maxfiy repo uchun `HF_TOKEN` talab qiladi; token yo'q bo'lsa (masalan production CT2 repo'siga kirish huquqi yo'q sessiyada) muqobil ochiq/boshqa model bilan ishlash kerak bo'ladi (Round 5'da `calls-rejected` tiklash shu sabab round-2 modelida qilingan, production'da emas). |
| 11 | Wikimedia/urllib UA bloklari | `commons.wikimedia.org` `requests`ning standart `python-requests/...` User-Agent'ini blokladi (403); descriptive UA qo'yilgach ham baribir 403 — ehtimol RunPod datacenter IP'lari ham bloklangan. Endpoint sinovlari uchun Wikimedia URL ishlatilmasin. |

---

## 9. Nima ishlamadi (takrorlamaslik uchun)

- **Forced alignment (segment darajasida).** Maqsad: 235 ta chetlangan uzun
  bo'lakni (5.2 soat) qutqarish. Natija: 40 namunadan atigi 4 tasi yaroqli
  chiqdi. Sabab: `return_timestamps=True` (segment) ko'p fayllarda BUZUQ edi
  — bitta 55 s fayl bitta "segment" qaytarardi, ichida 97-314 so'z, boshlanish
  vaqti ma'nosiz. `return_timestamps="word"` aniq ishlagan, lekin
  `batch_size>1`da CUDA OOM (cross-attention og'irliklari 42 GB yeyapti) va
  batch=1da namuna boshiga ~50 s (235 namuna = 3.3 soat = ~$3.60) — byudjetga
  arzimadi. **To'g'ri yechim manbada topildi**: `MAX_CHUNK_SECONDS`ni 55dan
  28ga tushirish — bu 100% natija berdi, eski uzun ma'lumotni qutqarishga
  urinish shart emas edi.
- **Pseudo-label (Claude bilan tuzatilgan eski matnni yorliq sifatida
  ishlatish).** O'lchandi — modeldan yomonroq chiqdi (tasdiqlangan, batafsili
  jarayon xotira faylida yo'q, faqat xulosa qayd etilgan).
- **Toza tashqi korpusning katta ulushi.** Round 1-2 (ochiq datasetlar 100%)
  va Round 5 (UzbekVoice batch'ning 64% i) — ikkalasida ham model ochiq/toza
  nutqda yaxshilandi, lekin real qo'ng'iroq domenidan uzoqlashdi. Saboq:
  tashqi toza korpus qo'shilsa, qo'ng'iroq ulushi batch'da **≥70%** bo'lsin
  (35-36% emas) va telefon augmentatsiyasi **majburiy** bo'lsin.
- **"Imlo" keyin-ishlovi umumiy yechim sifatida.** Dastlab almashtirish
  xatolarining ~31% i imlo/normallashtirish bilan tuzatiladi degan xulosa
  chiqarilgan edi (`char_dist ≤ 2` chelagi asosida), lekin bu chelak haqiqiy
  xatolarni ham (`olib`→`topib`, masofa 2) yutib yuborar edi. To'g'ri
  ajratilganda keyin-ishlovning **tavan darajasi ~2.7 punkt**, xulosa
  rasman **bekor qilindi** (commit `0fb5c48`).

---

## 10. Byudjet va xavfsizlik qoidalari

- **Xarajat chegaralari:** endpoint (production, real trafik) jami **$1**;
  har bir Pod yurishi uchun alohida chegara — odatda **$5** (Round 3'da
  jami ~$1.60, Round 5'da $4.89/$5.00 bashorat bilan deyarli chegarada).
  **Har yurishdan oldin PM'ga hisob yuboriladi.**
- `train_calls.py` ichida byudjet qo'riqchisi bor: birinchi `EVAL_STEPS`dan
  keyin haqiqiy s/qadam ma'lum bo'lib, yakuniy narx bashorat qilinadi;
  chegaradan oshsa trening **to'xtaydi** va eng yaxshi checkpoint saqlanadi.
- **Repo ochiq** (https://github.com/SunnatillaNSH/whisper-uzbek-asr) — audio
  va transkript **hech qachon commit qilinmaydi**. `.gitignore`da
  `data/calls-*`, `*.wav/*.mp3/*.flac/*.tar`, `analysis/*` (faqat `*.py`
  ochiq) qat'iy yopilgan. Repo tarixida bir marta (`git add -A` tufayli)
  boshqa agent tomonidan hujjat fayllari tasodifan o'chirilgan holat bo'lgan
  — shu loyihada **`git add -A` ishlatilmasin**, faqat aniq fayl yo'llari
  qo'shilsin.
- **Kalitlar/tokenlar** faqat `~/.config/sellup/` da saqlanadi — **qiymati
  bu yoki boshqa hech qanday hujjatga yozilmaydi**. RunPod API kaliti va
  HF token Worker/environment secret sifatida saqlanadi, koddan tashqarida.

---

## 11. Ochiq savollar

1. **Tiklangan 665 namuna (Round 5, `recover_rejected.py`) foyda beradimi?**
   Round 5'da UzbekVoice bilan birga ishlatilgani uchun uning alohida
   ta'siri ajratib bo'linmadi ("C" — qo'ng'iroq+UzbekVoice bor, "B" — faqat
   qo'ng'iroq yo'q edi). Round 6 (rejalashtirilgan, faqat qo'ng'iroq + shu
   665 namuna) bu savolga javob berishi kutilmoqda.
2. **Muxlisa balansi bilan yorliqni ko'paytirish kerakmi?** Muxlisa faqat
   qo'lda va balans bor paytda chaqiriladi (avtomatik emas — birinchi
   "Insufficient balance" javobidan keyin 1 soat butunlay chaqirilmaydi).
   Ko'proq yorliqli ma'lumot uchun bu jarayonni tezlashtirish yoki
   kengaytirish masalasi hal qilinmagan.
3. **Imlo/keyin-ishlov qo'shilsinmi?** Tavan darajasi ~2.7 punkt bilan
   cheklangan (9-bo'lim) va uning kattasi ham (qo'shimcha farqi — `to'lov`
   vs `to'lovi`) matndan emas, faqat audio orqali hal qilinadigan turdan.
   Qo'shish qiymatga arziydimi — hal qilinmagan.
4. **Maqsad Muxlisa'ga yaqinlashishmi yoki undan arzonroq bo'lishmi?**
   1-bo'limdagi strategik ikkilanish hali foydalanuvchi bilan rasman
   aniqlashtirilmagan.

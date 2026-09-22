# Baholash (eval) protokoli — Whisper Uzbek ASR

Manba: `scripts/make_eval_120.py`, `scripts/make_eval_50.py`,
`scripts/eval_pod.py`, `scripts/eval_endpoint.py`, `scripts/compare_runs.py`,
`docs/ROUND5-PLAN.md`, `docs/ROUND5-RESULT.md`. Skript bilan bu hujjat
orasida farq bo'lsa, skript haqiqiy manba hisoblanadi.

## 1. Eval to'plamlari: eval-120 va eval-50

### eval-120

`scripts/make_eval_120.py` eval to'plamini 31 dan **120** qo'ng'iroqqa
kengaytiradi. Sabab — eski 31 qo'ng'iroq / 81 namuna / 3757 so'zlik
to'plamning ishonch oralig'i ±7 foiz punkt edi (Round 4 shu cheklovning
qurboni bo'lgan: eval loss yaxshilangan, lekin WER'da p=0.732 chiqqan,
xulosa chiqarib bo'lmagan). 120 qo'ng'iroqda oraliq ~±3 punktga tushadi.

**Qatlamlash va qo'ng'iroq bo'yicha ajratish:**

- Eski 31 qo'ng'iroq (`locked_from_previous`) **shartsiz** ichida qoladi —
  aks holda oldingi raundlar bilan tarixiy taqqoslash uziladi.
- Ajratish **qo'ng'iroq** darajasida, namuna/bo'lak darajasida emas: bitta
  suhbatning bo'laklari train va eval'ga bo'linib ketsa, model eval
  suhbatini treningda ko'rgan bo'lib chiqadi va natija soxta yaxshi
  chiqadi.
- Tanlov qatlamlangan va **proporsional**: partiya (`eski`/`yangi` —
  `calls-dataset` vs `calls-dataset-28s`) va yo'nalish (kiruvchi/chiquvchi)
  bo'yicha populyatsiyadagi nisbatda, har katakda davomiylik **kvartili**
  bo'yicha teng taqsimlangan (`by_quartile()`). Yo'nalishni teng (50/50)
  qilish ATAYLAB rad etilgan: populyatsiyada chiquvchi 76%, kiruvchi 24% —
  teng tanlansa kiruvchi eval'da 44%ga chiqib ketib, WER haqiqiy trafikni
  aks ettirmay qoladi.
- Bitta mijozdan (`lead_id`) ko'p namuna olinmasligiga harakat qilinadi
  (`used_leads` to'plami).
- Urug' (`SEED=20260919`) qat'iy — qayta ishga tushirilsa aynan o'sha
  to'plam chiqishi kerak.

**Fayllar**: `data/eval_calls_120.json` (metama'lumot: `call_id`, `batch`,
`direction`, `duration`, `samples`, `words`; **transkript matni, sana, CRM
identifikatori YOZILMAYDI** — repo ochiq), `data/eval120/eval.csv`
(`path,sentence`, treningda/o'lchashda ishlatiladigan haqiqiy fayl).

### eval-50

`scripts/make_eval_50.py` — foydalanuvchi qarori bilan Round 5'dan keyin
**faqat 50** qo'ng'iroq o'giriladi (Muxlisa yorlig'i qimmatga tushadi).
Bu eval-120'ni **almashtirmaydi** — u o'zgarmay qoladi; eval-50 uning
qat'iy belgilangan qismi. Tanlash: asl 31 "locked" qo'ng'iroq shartsiz,
qolgan 19 tasi `(batch, direction)` bo'yicha populyatsiyaga mutanosib
qatlamlab, har qatlamda davomiylik bo'yicha teng oraliqlardan olinadi.
Urug' `SEED=20260920`, natija `data/eval_calls_50.json` va
`data/eval120/eval50.csv`.

### O'zgartirilmaslik qoidasi

Ikkala to'plam ham **bir marta yaratiladi va commit qilinadi** — keyingi
raundlar AYNAN shu ro'yxatni ishlatadi. Sabab shu hujjatning har joyida
takrorlanadi: agar to'plam har safar qayta tanlansa, ikki model turli
namunada o'lchanadi va taqqoslash ma'nosini yo'qotadi. `train_calls.py`
`eval_calls_120.json`ni o'qib topilmasa trening BOSHLANMAYDI — bu fayl
shunchaki hisobot emas, treningni to'xtatuvchi darvoza.

## 2. Metrikalar

### WER va uning tarkibi (S/D/I)

`eval_pod.py`/`eval_endpoint.py`/`compare_runs.py`dagi `wer_counts()` —
so'z darajasidagi Levenshtein, ikki qatorli jadval bilan (xotira uchun,
to'liq matritsa emas):

```python
S, D, I = wer_counts(ref_words, hyp_words)   # Almashtirish, Tushish, Qo'shish
wer = 100 * (S + D + I) / N                  # N = yorliq so'zlari
```

Umumiy WER **korpus darajasida** hisoblanadi — har namuna WER'ining
o'rtachasi EMAS, so'zlar yig'indisi ustidan (`corpus_wer()` izohi:
"uzun namuna ko'proq vazn olishi kerak, aks holda bir so'zlik 'Rahmat.'
butun suhbat bilan teng bo'lib qoladi").

Xato tarkibi hisobotda foizda beriladi (Round 5 misoli):

| | S | D | I |
|---|---|---|---|
| Production | 1342 (58%) | 575 (25%) | 372 (16%) |
| Round 5 | 1798 (64%) | 412 (14%) | 573 (20%) |

Bu taqsimot modelning **xulq o'zgarishini** ko'rsatadi: tushish kamayib,
almashtirish/qo'shish oshsa — model "ko'proq gapiradigan, lekin ko'proq
xato" bo'lgan (ROUND5-RESULT.md, 2-bo'lim).

### So'z qamrovi

`ROUND5-PLAN.md`dagi hisobotda alohida ko'rsatiladi ("So'z qamrovi
96.2%") — gipotezaning yorliq uzunligiga nisbatan qanchalik to'liqligi
(qisqa/bo'sh chiqish yoki haddan tashqari uzun gallyutsinatsiya
belgisi). `handler.py`dagi hotwords sinovida ham ishlatilgan:
"hotwords yoqiq WER 48.27% so'z qamrovi 88.2%" vs "hotwords o'chiq WER
44.38% so'z qamrovi 95.1%" — past qamrov ba'zan yashirin WER yomonlashuvi
belgisi bo'ladi.

### Imlo-normallashtirilgan WER

`compare_runs.py`dagi ikkinchi o'lchov: almashtirilgan so'z juftligi
belgi-Levenshtein masofasi ≤2 bo'lsa (`char_dist(a, b, cap=3) <= 2`), bu
almashtirish **imlo tebranishi** deb hisoblanadi va xato hisobidan olib
tashlanadi ("allo"/"alo", "o'ttiz"/"o'tiz"):

```python
spelling = sum(1 for a, b in subs if char_dist(a, b) <= 2)
err_norm = len(subs) + dels + ins - spelling
```

`ROUND5-PLAN.md`da bu chegara qayta tekshirilgan: dastlab "~31%
almashtirish imloviy" deyilgan, lekin `char_dist<=2` chegarasi haqiqiy
xatolarni ham yutib yuboradi ("olib"→"topib" ham masofa 2). To'g'ri
ajratilganda matn darajasidagi keyin-ishlov eng ko'pi bilan **2.7
punktga** tegadi — WER'ning asosiy qismi (72%, 26.2 punkt Round 5
tahlilida) akustik almashtirish va tushib qolishda, ya'ni faqat trening
tuzata oladi, keyin-ishlov emas.

### Normallashtirish qoidalari (apostrof)

Barcha eval/o'lchov skriptlarida bir xil `norm()`:

```python
APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}

def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)
    return re.sub(r"\s+", " ", s).strip()
```

O'zbek lotin yozuvida apostrof **harfning bir qismi** (`o'`, `g'`) — olib
tashlanmaydi, faqat besh xil unicode varianti (`‘’ʻʼ`´`) bitta ASCII
`'` ga keltiriladi. Bu tuzatilmaguncha (commit `4570470`) eval
to'plamining o'zida uch xil variant aralash edi va so'zlarning 18.7%ida
apostrof borligi uchun WER ~19 foiz punktgacha sun'iy yuqori chiqardi.
Apostrofni olib tashlash ham noto'g'ri — "ozim" va "o'zim" qo'shilib
ketadi.

## 3. Statistik qoida

### Juftlashgan bootstrap

`compare_runs.py` ikki yurishni (masalan production vs nomzod) **bir xil
namunalarda** solishtiradi — ikkala model AYNAN bir xil `path`larda
o'lchangan bo'lishi shart (`keys = sorted(set(pa) & set(pb))`; faqat
bittasida bor namunalar chiqarib tashlanadi). Sabab: ikki modelning
WER'ini alohida hisoblab ayirmasini olish kam ma'lumot beradi — namunalar
qiyinligi har xil, farqning katta qismi qaysi namunalar tushganidan kelib
chiqishi mumkin.

Bootstrap: 10 000 marta (standart `--rounds`) namunalar **qaytarib
qo'yib** tasodifiy tanlanadi (`pick = [keys[rng.randrange(k)] for _ in
range(k)]`), har safar ikkala model o'sha tanlovda solishtiriladi va
farq yig'iladi.

### 95% oraliq, p qiymati

```python
diffs.sort()
lo = diffs[int(0.025 * rounds)]
hi = diffs[int(0.975 * rounds)]
le = sum(1 for d in diffs if d <= 0) / rounds
ge = sum(1 for d in diffs if d >= 0) / rounds
p = min(1.0, 2 * min(le, ge))
```

`p` hisoblash ataylab `2 * min(le, ge)` shaklida — eski versiya
`2 * min(neg, 1-neg)` teng holatda buzilardi: barcha farq aynan 0 bo'lsa
`P(<=0)=1` bo'lib, `p=0` chiqib "farq yo'q" holati "juda ishonchli farq"
bo'lib ko'rinardi.

### "Yaxshilanish" sharti

```python
better = hi < 0    # nomzod xatosi asosiydan kam — oraliq BUTUNLAY noldan past
worse  = lo > 0
```

Faqat 95% oraliq **butunlay** noldan past bo'lsa "YAXSHILANDI — ishonchli"
deyiladi; oraliq nolni kesib o'tsa — "farq ISHONCHLI EMAS (oraliq nolni
kesib o'tadi)", hatto nuqta baho ijobiy bo'lsa ham. Round 4 aynan shu
tarzda rad etilgan: nuqta farq +1.04 punkt, lekin oraliq [−6.14, +7.13],
p=0.732 — statistik jihatdan TENG.

Muvaffaqiyat mezoni oldindan (trening boshlanmasdan) yoziladi
(`ROUND5-PLAN.md`, 5-bo'lim): "Round N muvaffaqiyatli deb sanaladi, agar
eval-120da juftlashgan bootstrap 95% oralig'i butunlay noldan past
bo'lsa." Bu tartib teskari qilinmaydi — natija ko'rilgach mezon
moslashtirilmaydi.

### eval-50 dagi ~±5 punkt cheklovi

`make_eval_50.py`ning o'z hisobi: 50 qo'ng'iroq ~5900 yorliq so'zi
beradi, 120dagi 14 015ning 42%i. Bootstrap oralig'i taxminan
`sqrt(14015/5900) ≈ 1.54` barobar kengayadi — eval-120da ~±3 punkt
bo'lgani eval-50da ~**±5 punkt** bo'ladi. **3 punktdan kichik
yaxshilanishni eval-50 ISHONCHLI ko'rsata olmaydi.** `ROUND5-RESULT.md`
7-bo'limida shu cheklov qayta tasdiqlangan: "Bu safar farq katta
bo'lgani uchun muhim bo'lmadi (+8.17 punkt), kichik farqda bo'ladi" —
shuning uchun keyingi taqqoslashlar eval-120da (14 015 so'z) bo'lishi
tavsiya etiladi, faqat tezkor tekshiruv uchun eval-50 ishlatiladi.

## 4. Bir xil stack qoidasi

### `eval_pod.py` dekodlashni `handler.py` bilan majburlab solishtiradi

`eval_pod.py`dagi `DECODE` lug'ati (`beam_size=5`,
`vad_parameters={"min_silence_duration_ms":2000,"speech_pad_ms":400}`,
`condition_on_previous_text=False`, `temperature=[0.0,0.2,0.4,0.6,0.8,1.0]`,
`compression_ratio_threshold=2.4`, `log_prob_threshold=-1.0`,
`no_speech_threshold=0.6`) qo'lda yozilgan, lekin skript ishga
tushishidan OLDIN `check_matches_handler()` `handler.py` manbasini `ast`
bilan o'qib, `model.transcribe(...)` chaqiruvidagi haqiqiy qiymatlar bilan
solishtiradi. Mos kelmasa `sys.exit(1)`. Bu tekshiruv uchinchi yo'l sifatida
tanlangan: `handler.py`dan to'g'ridan-to'g'ri import qilib bo'lmaydi
(Dockerfile `COPY handler.py .` — yangi modulga ajratish uni obrazga
kiritishni talab qiladi va ishlab turgan serverless'ga tegadi), lekin
sozlamalarni qo'lda ikki joyda qo'lda saqlash ajralib ketishga olib
keladi. `HOTWORDS` standarti ham tekshiriladi — bo'sh bo'lishi shart
(o'lchangan: yoqiq bo'lsa +3.9 punkt WER).

### Endpoint va Pod o'lchovlarini solishtirishda e'tibor

- **Ikkalasi ham bir xil `DECODE`/`handler.py` sozlamasida** bo'lishi
  shart — `eval_endpoint.py` production endpointining o'ziga so'rov
  yuboradi (haqiqiy `handler.py` ishlaydi), `eval_pod.py` esa CT2 modelni
  Pod ustida to'g'ridan chaqiradi va `check_matches_handler()` bilan
  moslikni tasdiqlaydi — ikkalasi ham amalda bir xil parametrlarga tayanadi.
- **Turli GPU** (masalan endpoint A4000-sinf, Pod L40S) fp16'da "eng
  ko'pi bilan o'ndan bir punktlik tebranish" beradi (`ROUND5-RESULT.md`).
  Agar farq bir necha punktdan katta bo'lsa va S/D/I taqsimoti
  **yo'nalishli siljigan** bo'lsa (masalan tushish kamayib, qo'shish
  oshsa) — bu apparat farqi emas, model xulqining o'zgargani (Round 5
  misoli: 8.17 punkt farq, apparat buni tushuntirmaydi).
- **CT2 konvertatsiyasi bir xil bo'lishi shart**: `--quantization float16`
  va aynan bir xil `--copy_files` tokenizer ro'yxati — aks holda farq
  modeldan emas, konvertatsiyadan chiqishi mumkin.
- **Endpoint xarajati.** Har `eval_endpoint.py` yurishi haqiqiy pulga
  tushadi (production endpointga so'rov); shuning uchun modelni
  taqqoslashda birinchi tanlov Pod ustida (`eval_pod.py`, endpoint
  byudjetiga $0) — production'ning o'zi (`analysis/eval120_prod.json`)
  faqat bir marta `eval_endpoint.py` bilan o'lchanadi va keyingi
  taqqoslashlarning "asosiy" (`base`) qatori sifatida qayta ishlatiladi.

## 5. Natijani yozish shakli

**Saqlanadigan JSON maydonlari** (`eval_pod.py`/`eval_endpoint.py`
chiqishi): `csv` (qaysi eval fayl), `model`/`opts` (qaysi model, qaysi
sozlama), `wer`, `sub`/`del`/`ins`, `ref_words`, `samples`, `failed`, va
**har bir namuna uchun** `results` ro'yxati (`path`, `reference`,
`hypothesis`, `duration`, `proc`, `error`). Namuna darajasidagi natija
saqlanishi shart — `compare_runs.py` shu ro'yxatdan juftlashgan bootstrap
qiladi, agregat WER'dan emas.

**Qaysi raqamlar hisobotga tushadi** (`ROUND5-RESULT.md` namunasi):

1. Asosiy taqqoslash jadvali — WER va imlo-normallashtirilgan WER,
   ikkalasi uchun ham nuqta farq, 95% oraliq, p, hukm.
2. Namuna darajasida: yaxshilandi/yomonlashdi/teng soni.
3. Xato tarkibi — S/D/I foizda, ikkala yurish uchun yonma-yon.
4. O'lchov sharoiti — qaysi GPU/stack'da, `eval_pod.py`ning moslik
   tekshiruvidan o'tgani eslatiladi.
5. Trening jadvali — boshlang'ich model, to'xtash qadam raqami va sababi,
   qo'ng'iroq/tashqi namuna soni, LoRA sozlamalari, **eval-120 sizishi = 0
   (qanday tasdiqlangani)**.
6. Narx — Pod turi, soat, byudjet qo'riqchisi ishga tushdimi.
7. Model qayerda — katalog yo'li, SHA256SUMS bilan.
8. Tavsiya — joylashtirilsinmi, va agar yo'q bo'lsa nega (rad etilgan
   joylashtirish qadamlari alohida bo'limda, ochiq "BAJARILMASIN" belgisi
   bilan qoldiriladi — `ROUND5-RESULT.md` 8-bo'limi kabi).

**Fayllar**: `analysis/eval120_prod.json` (production, asosiy taqqoslash
nuqtasi), `analysis/eval120_r<N>.json` yoki `models/round<N>/eval50_r<N>.json`
(nomzod), `docs/ROUND<N>-RESULT.md` (yakuniy hisobot matni).

## 6. Xatolar

### Eval sizib chiqishi (train'da eval qo'ng'irog'i bo'lishi)

Eng jiddiy xato turi — model o'z eval suhbatini ko'rib o'sgan bo'lsa,
WER soxta yaxshi chiqadi. Uch qatlamli himoya (1-bo'limga qarang):
`train_calls.py`dagi qat'iy `sys.exit`, `recover_rejected.py`dagi
oldindan chiqarib tashlash, va uchta `assert`. Bundan tashqari eval-120
qo'ng'iroqlari **qo'ng'iroq** darajasida saqlanadi — namuna/bo'lak
darajasida ajratish (masalan bitta suhbatning ba'zi bo'laklari train'da,
ba'zilari eval'da) ham amalda sizib chiqishning yashirin shakli.

### Checkpoint tanlashda eval to'plamiga qarash

`train_calls.py`dagi izoh (`TRAIN_CSV` yonida) buni aniq tushuntiradi:
agar `load_best_model_at_end=True` bo'lganda checkpoint tanlovi uchun
ishlatiladigan **dev** bo'lagi aynan eval-120 qo'ng'iroqlaridan iborat
bo'lsa — gradientga tushmasa ham, checkpoint TANLOVI o'sha qo'ng'iroqlar
bo'yicha qilingan bo'ladi va bu sizish. Shuning uchun dev bo'lagi HAR
DOIM trening CSV'sining o'zidan, qo'ng'iroq bo'yicha, alohida kesiladi
(`DEV_FRAC=0.10`) — dataset paketidagi tayyor `eval.csv`dan EMAS.

### Kichik to'plamda xulosa chiqarish

3-bo'limdagi ±5 punkt (eval-50) va ±3 punkt (eval-120) cheklovlaridan
oshib xulosa chiqarish — masalan "1-2 punkt yaxshilandi" deb e'lon
qilish, bootstrap oralig'i nolni kesib o'tsa ham. Qoida qat'iy: faqat
95% oraliq butunlay noldan past/yuqori bo'lganda "yaxshilandi"/
"yomonlashdi" deyiladi, aks holda "farq ISHONCHLI EMAS" deb yoziladi —
hatto nuqta baho ijobiy ko'rinsa ham (Round 4 dars sifatida qayd
etilgan). Eval-50 bilan olingan har qanday "yaxshilanish" xulosasi
keyinroq eval-120da tasdiqlanishi kerak, aks holda vaqtinchalik deb
belgilanadi.

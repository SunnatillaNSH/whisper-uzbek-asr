# Round 5 — natija

**Xulosa: Round 5 production'dan ishonchli darajada YOMON. Joylashtirilmasin.**

O'lchov sanasi 2026-09-20. Transkript va mijoz ma'lumoti yo'q — faqat raqamlar.

## 1. Asosiy taqqoslash — eval-50

Bir xil 50 qo'ng'iroq, 129 namuna, 6049 yorliq so'zi. Juftlashgan bootstrap
(`scripts/compare_runs.py`, 10 000 qayta tanlash).

| | WER | Imlo normallashtirilgan |
|---|---|---|
| Production | **37.84%** | 31.44% |
| Round 5 | **46.01%** | 37.94% |
| Farq | **+8.17 punkt** | +6.50 punkt |
| 95% oraliq | [+6.08, +10.33] | [+4.43, +8.63] |
| p | 0.000 | 0.000 |
| Hukm | **YOMONLASHDI — ishonchli** | **YOMONLASHDI — ishonchli** |

Namuna darajasida: yaxshilandi 26 | yomonlashdi 95 | teng 8.

**Bu o'lchov aniqligi masalasi emas.** 50 qo'ng'iroqli to'plam ~±5 punktdan
kichik farqni ajrata olmaydi — bu oldindan aytilgan va `data/eval_calls_50.json`
da yozilgan. Bu yerdagi farq 8.17 punkt va oraliq nolga yaqin ham kelmaydi.

### O'lchov sharoiti

Ikki raqam TURLI GPU da olingan, lekin bir xil kod va bir xil dekodlash
sozlamalarida:

* production — RunPod Serverless endpoint (A4000-sinf), `eval_endpoint.py`
* Round 5 — Pod (L40S), `eval_pod.py`

`eval_pod.py` ishga tushishdan oldin `handler.py` ni `ast` bilan o'qib,
dekodlash production bilan aynan bir xil ekanini tasdiqlaydi (beam 5,
VAD 2000/400, hotwords bo'sh, temperatura zaxirasi,
`condition_on_previous_text=False`) va mos kelmasa to'xtaydi. CT2
konvertatsiyasi ham bir xil: `--quantization float16` va o'sha
`--copy_files` tokenizer ro'yxati.

Stack farqi bu natijani TUSHUNTIRMAYDI. Turli GPU fp16 da eng ko'pi bilan
o'ndan bir punktlik tebranish beradi, bu yerda esa 8 punkt farq va S/D/I
taqsimotining yo'nalishli siljishi bor — tushish 25% dan 14% ga, qo'shish
16% dan 20% ga. Apparat farqi xato TURINI bunday o'zgartirmaydi; buni
model xulqining o'zgargani tushuntiradi.

## 2. Xato tarkibi — nima o'zgargani

| | S (almashtirish) | D (tushish) | I (qo'shish) |
|---|---|---|---|
| Production | 1342 (58%) | 575 (25%) | 372 (16%) |
| Round 5 | 1798 (64%) | 412 (14%) | 573 (20%) |

Tushib qolish kamaydi, almashtirish va qo'shib yuborish esa oshdi. Ya'ni model
ko'proq gapiradigan bo'ldi, lekin ko'proq noto'g'ri so'z chiqardi. Sof hisobda
zarar.

## 3. Sababi

UzbekVoice — toza, mikrofonga yaqin, O'QILGAN nutq. Batch'ning 64% i shundan
iborat edi (qo'ng'iroqlar 36%). Model o'zbek tilida yaxshilandi, lekin telefon
domenidan uzoqlashdi.

Bu Round 1-2 nosozligining aynan takrori: o'shanda ochiq datasetlarda eval WER
34.01% → 27.82% tushgan, real qo'ng'iroqda esa foyda bermagan. Sabab bir xil —
til bilimi emas, domen farqi.

Dev loss buni trening davomida ham ko'rsatgan: u 3500-qadamda 0.6922 da tubiga
yetib, keyin qimirlamagan (0.6988 / 0.6970 / 0.6996 / 0.6960). Model
qo'ng'iroqlarda erta to'yingan; keyingi qadamlar faqat UzbekVoice'ga
moslashtirgan.

## 4. Trening

| | |
|---|---|
| Boshlang'ich model | `Sunnat0091/whisper-large-v3-uz` (round-2) |
| To'xtash | **5500-qadam, ERTA TO'XTASH** (patience 4; eng yaxshisi 3500) |
| To'xtash sababi | dev loss 4 ta ketma-ket evalda yaxshilanmadi — **byudjet emas** |
| Eng yaxshi dev loss | 0.6922 (3500-qadam) |
| Tezlik | 0.96 s/qadam, L40S |
| Qo'ng'iroq namunalari | 989 + 665 tiklangan = 1654 (1560 train / 94 dev) |
| Qo'ng'iroqlar | 337 (303 train / 34 dev) |
| Tashqi korpus | UzbekVoice 25 000 namuna / 26.54 soat |
| Aralashma | qo'ng'iroq takrori ×9, jami 39 040, qo'ng'iroq ulushi 36% |
| LoRA | r 32, alpha 64, 15.7M parametr (1.01%) |
| eval-120 sizishi | 0 — treningdan oldin assert bilan tasdiqlangan |

### `calls-rejected` tiklash

183 bo'lakdan (eval-120 ning 52 bo'lagi chiqarilgandan keyin) **665 namuna /
1.54 soat** tiklandi. Bashoratim ~2.9 soat edi — past chiqdi, chunki tiklash
production modelida emas, **round-2 modelida** qilindi (production CT2 repo'si
maxfiy, HF tokeni yo'q edi). Round-2 qo'ng'iroqlarda zaifroq, ya'ni langar
zichligi past: 277 bo'lak "uchi langarlanmagan" deb tashlandi.

## 5. Narx

| | |
|---|---|
| Pod | L40S, $1.11/soat |
| Trening | ~1.5 soat |
| Tayyorgarlik | o'rnatish, ma'lumot ko'chirish, tiklash, UzbekVoice, CT2, eval |
| Byudjet qo'riqchisi | ishga tushmadi (bashorat $4.89 / $5.00 da qolgan) |
| Endpoint xarajati | **$0** — production endpointiga bitta so'rov ham yuborilmadi |

## 6. Model qayerda

`models/round5/` (gitignore'da — repo ochiq):

* `ct2/` — CTranslate2 float16 model
* `adapter/` — LoRA adapteri (3500-checkpointdan)
* `SHA256SUMS` — `shasum -a 256 -c SHA256SUMS` bilan tekshiriladi

Merged fp16 ko'chirilmadi: u adapterdan qayta yig'iladi, yo'riqnoma
`models/README.md` da.

## 7. Tavsiya

**1. Production almashtirilmasin.** Joylash sharti — "Round 5 statistik jihatdan
yaxshiroq chiqsa" — bajarilmadi. 8-bo'limdagi qadamlar bajarilmaydi.

**2. Keyingi raund UzbekVoice'siz bo'lsin.** Faqat qo'ng'iroq: 1654 namuna,
~8.1 soat, round-2 dan boshlab, 8 epoxa ≈ 1550 qadam (batch 8).

Bu aynan dastlabki rejadagi **B qo'li** edi va uni o'tkazib yuborish endi
qimmatga tushdi: bizda C (qo'ng'iroq + UzbekVoice) bor, B (faqat qo'ng'iroq)
yo'q. Shuning uchun "tiklangan 665 namuna foyda berdimi" degan savol javobsiz
qoldi — C ning yomonligi UzbekVoice'danmi yoki tiklangan ma'lumotdanmi, ajratib
bo'lmaydi.

**3. Agar tashqi korpus baribir kerak bo'lsa:** qo'ng'iroq ulushi ≥70% bo'lsin
(35% emas) va telefon augmentatsiyasi majburiy bo'lsin.

**4. O'lchov to'plami.** Keyingi taqqoslashlar eval-120 da (14 015 so'z)
bo'lsin, 50 qo'ng'iroqda emas — 42% so'z oralig'ni 1.5 barobar kengaytiradi.
Bu safar farq katta bo'lgani uchun muhim bo'lmadi, kichik farqda bo'ladi.

## 8. Joylash qadamlari (HALI BAJARILMAGAN — VA BAJARILMASIN)

Bu bo'lim tayyorlab qo'yilgan. **Hech narsa joylanmadi**: HF write tokeni yo'q,
va joylash faqat yuqoridagi taqqoslash Round 5 ni statistik jihatdan yaxshiroq
ko'rsatgan taqdirdagina qilinadi. Oraliq nolni kesib o'tsa — joylanmaydi.

### 1. Modelni maxfiy HF repo'siga yuklash

Yangi nom bilan, production repo'si USTIGA YOZILMAYDI — rollback shunga tayanadi.

```bash
export HF_TOKEN=<write huquqli yangi token>
python3 - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
repo = "Sunnat0091/whisper-large-v3-uz-calls-r5-ct2"
api.create_repo(repo, private=True, exist_ok=True)
api.upload_folder(folder_path="models/round5/ct2", repo_id=repo)
PY
```

### 2. Endpoint'da modelni almashtirish

RunPod Serverless → endpoint sozlamalari → Environment Variables:

| O'zgaruvchi | Yangi qiymat |
|---|---|
| `HF_MODEL_ID` | `Sunnat0091/whisper-large-v3-uz-calls-r5-ct2` |

Obrazni qayta qurish SHART EMAS — model образga pishirilmagan, konteyner uni
ishga tushganda yuklab oladi. O'zgarishdan keyin workerlar qayta ishga
tushiriladi va birinchi so'rov sovuq start bo'ladi (~30 s).

### 3. Rollback

Bitta qadam: `HF_MODEL_ID` ni eski qiymatga qaytarish.

```
HF_MODEL_ID = Sunnat0091/whisper-large-v3-uz-calls-ct2
```

Eski model o'chirilmaydi va ustiga yozilmaydi, ya'ni rollback har doim mumkin.

### 4. Joylashdan keyingi smoke test (≤5 qo'ng'iroq)

```bash
python3 scripts/eval_endpoint.py --csv data/eval120/eval50.csv --limit 5 \
    --out analysis/smoke_r5.json --workers 2
```

Nimaga qaraladi:

* `failed` = 0 — model yuklandi va javob berdi
* WER eval-50 dagi Pod o'lchovidan keskin farq qilmasin (5 namunada shovqin
  katta, shuning uchun bu sifat o'lchovi emas, **tiriklik** tekshiruvi)
* `duration_sec` va `processing_time_sec` nisbati oldingidek (~10x realtime)

Smoke test 5 qo'ng'iroqdan oshmaydi — foydalanuvchining "ko'pi bilan 50
qo'ng'iroq" cheklovi va endpoint byudjeti kuchda qoladi.

### 5. Joylashdan KEYIN kuzatish

Konsol (`ovoz-konsoli`) da `/api/stats` bo'yicha bir kun kuzatilsin: xato
soni, o'rtacha `proc_sec`, navbat vaqti. Keskin o'sish bo'lsa — 3-banddagi
rollback.

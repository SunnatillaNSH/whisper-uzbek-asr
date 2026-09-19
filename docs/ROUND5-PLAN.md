# Round 5 — reja va asoslash

O'lchov sanasi 2026-09-19. Barcha raqamlar **bitta stack'da** (production
endpoint: faster-whisper, beam 5, VAD standart, hotwords bo'sh).

## 1. Hozirgi holat — eval-120

| | |
|---|---|
| Namuna | 310 (117 qo'ng'iroq), xatosiz |
| Yorliq so'zlari | 14 015 |
| **WER** | **36.39%** |
| So'z qamrovi | 96.2% |
| Audio | 2.00 soat, handler 0.185 soat, RTF 10.8× |
| Xarajat | ~$0.073 |

Yorliq — Muxlisa AI. Ya'ni WER "Muxlisa'dan qanchalik uzoq" degani, "qanchalik
noto'g'ri" degani emas. WER = 0 Muxlisa bilan aynan bir xil chiqish demak,
undan yaxshiroq degani emas.

## 2. Xato anatomiyasi — qayerga sarflash kerak

5100 xato / 14 015 so'z = 36.39%.

| Turi | Xatodan | WER'ga hissa |
|---|---|---|
| Akustik almashtirish (boshqa so'z) | 46.0% | **16.74 punkt** |
| Tushib qolish | 26.1% | **9.51 punkt** |
| Qo'shib yuborish | 15.7% | 5.72 punkt |
| Qo'shimcha farqi (`to'lov`→`to'lovi`) | 5.9% | 2.14 punkt |
| Bir belgi farqi (`u`→`bu`) | 4.7% | 1.71 punkt |
| Tasdiqlash so'zi (`hm`→`ha`) | 1.3% | 0.49 punkt |
| Apostrof | 0.2% | 0.09 punkt |

**Avvalgi xulosam noto'g'ri edi.** `patterns.md` 3-bandida imloviy
keyin-ishlov ~31% almashtirishga tegadi deb yozilgandi. U raqam
`char_dist <= 2` chelagidan olingan, lekin bu chelak haqiqiy xatolarni ham
yutib yuboradi: `olib`→`topib` ham masofa 2. To'g'ri ajratilganda matn
darajasidagi keyin-ishlov eng ko'pi bilan **2.7 punktga** tegadi, uning
kattasi esa baribir tuzatib bo'lmaydigan turdan — `to'lov` mi yoki `to'lovi`
mi ekanini matndan bilib bo'lmaydi, buni faqat audio hal qiladi.

Apostrof masalasi ham yopilgan: `norm()` uni allaqachon hal qilgan, qoldig'i
0.09 punkt.

**Xulosa: WER ning 72% i (26.2 punkt) akustik almashtirish va tushib
qolishda. Ularga faqat trening tegadi, keyin-ishlov emas.**

## 3. Round 5 arziydimi

Round 4 o'sha ma'lumotdan yana bir marta o'tdi va **o'lchanadigan foyda
bermadi** (farq 1.04 punkt, 95% oraliq [−6.14, +7.13], p = 0.732).

Round 5 ning dataseti tozalandi — takroriy `calls-dataset` chiqarildi,
eval-120 chetlandi — lekin **kattalashmadi**: 975 namuna / 312 qo'ng'iroq /
6.49 soat, ya'ni Round 3/4 dagi hajm. Shu holicha takrorlash Round 4 ning
taqdirini takrorlashi ehtimoli katta.

### Topilgan zaxira: `data/calls-rejected` — 5.19 soat yorliqli audio

235 ta bo'lak, mediana 55 s. Ular **yorliqli**, lekin bo'luvchi skript
ishonchli kesim topa olmagani uchun chetda qolgan (30 soniyalik Whisper
oynasiga sig'maydi, jumla chegarasi esa jimlikka to'g'ri kelmagan).

Tiklansa trening audiosi 6.49 → ~11.7 soatga chiqadi (+80%). Round 4 da
yetishmagan narsa aynan shu edi — yangi ma'lumot.

Tiklash yo'li: so'z vaqtlari bo'yicha kesish — `scripts/recover_rejected.py`.
Vaqt modelning o'zidan, matn yorliqdan olinadi; kesim faqat yorliqning jumla
chegarasida va faqat ikkala uchi ham AYNAN mos tushgan so'z bilan
langarlangan joyda bo'ladi. Ya'ni yorliq matni o'zgarmaydi.

**ENDPOINT ORQALI EMAS, POD USTIDA.** Production `handler.py` so'z vaqtlarini
umuman qaytarmaydi — unda `word_timestamps` yo'q. Endpoint yo'li obrazni
qayta qurib, ishlab turgan serverless'ni yangilashni talab qilardi. Pod'da
faster-whisper to'g'ridan chaqiriladi: o'sha narx (~30 GPU-daqiqa), endpoint
byudjetiga **$0**, production'ga xavf yo'q. Round 5 Pod'i baribir kerak —
tiklash o'sha seansning boshiga qo'shiladi.

**Kutilayotgan hosil (lokal hisoblandi, GPU'siz).**

| Holat | Namuna | Soat | Kirishdan |
|---|---|---|---|
| Nazariy shift (to'liq langar) | 719 | 4.97 | 96% |
| So'z aniqligi 80% | 1135 | 4.40 | 85% |
| **So'z aniqligi 64% (o'lchangan)** | **~1250** | **~3.5** | **69%** |
| So'z aniqligi 50% (yomon holat) | 1136 | 2.61 | 50% |

O'lchangan holatda trening audiosi **6.49 → ~10.0 soat (+55%)**. Yomon
holatda ham +40%. Uch xil tasodifiy urug'da natija barqaror.

5643 jumladan faqat **1 tasi** o'zi 30 s dan uzun — ya'ni jumla tuzilishi
to'siq emas, hamma narsa langar zichligiga bog'liq.

### Tavsiya — shu tartibda

1. **Avval `calls-rejected` ni tiklash** — o'sha Pod seansida, birinchi ~30
   daqiqa. Endpoint byudjetiga tegmaydi.
2. Keyin Round 5 ni ~10 soatda o'qitish.
3. Agar 1-band rad etilsa, Round 5 ni 6.49 soatda ham qilish mumkin, lekin
   kutilayotgan foyda kichik — buni oldindan aytib qo'yaman, keyin
   "yaxshilanmadi" degan natija kutilmagan bo'lmasin.

## 4. Round 5 — texnik shartlar

**Tuzatilishi SHART bo'lgan xato.** `train_calls.py` `eval.csv` ni dataset
paketidan o'qiydi, u esa `calls-colab/eval.csv` — 31 qo'ng'iroq, hammasi
eval-120 ichida. `load_best_model_at_end=True` bo'lgani uchun eng yaxshi
nazorat nuqtasi aynan eval-120 qo'ng'iroqlari bo'yicha TANLANADI. Gradientga
tushmasa ham, bu tanlov sizishi: yakuniy 36.39% bilan taqqoslanadigan raqam
xolis bo'lmay qoladi.

Tuzatish: dev bo'lagi 312 ta trening qo'ng'irog'idan kesiladi (~10% = ~31
qo'ng'iroq, ~95 namuna). eval-120 ga trening davomida UMUMAN tegilmaydi.

| Sozlama | Qiymat | Sabab |
|---|---|---|
| Boshlang'ich model | `Sunnat0091/whisper-large-v3-uz` (round-2) | o'zbek tili unda bor, qadamlar domenga ketadi |
| Dataset | 975 namuna, **og'irliksiz** | og'irlashning foydasi hech qachon alohida o'lchanmagan; namuna darajasidagi ballar yo'qolgan |
| Dev bo'lagi | trening qo'ng'iroqlaridan ~31 ta | eval-120 tegilmasin |
| `MAX_STEPS` | 900 | ~880 namuna / batch 8 = 110 qadam/epoxa × 8 epoxa |
| `WARMUP_STEPS` | 100 | |
| `EVAL_STEPS` / `SAVE_STEPS` | 100 | 9 ta nazorat nuqtasi |
| `load_best_model_at_end` | True | eval loss ko'tarilsa eng yaxshisi qoladi |
| LoRA | r 32, alpha 64, dropout 0.05 | o'zgarishsiz |
| LR | 5e-5 | o'zgarishsiz |
| `USE_PODCAST` | 0 | faqat qo'ng'iroq |

Eslatma: oldingi raund 3000 qadam edi, bu ~1022 namunada 23 epoxa degani —
ortiqcha o'qish hududi. 8 epoxa qoidasi shundan kelib chiqqan.

**Intizom**
- Nazorat nuqtalari `/workspace` da (tarmoq hajmi) — Pod o'lsa yo'qolmasin.
- Model MAXFIY HF repo'siga YANGI nom bilan yuklanadi (`...-r5`), production
  almashtirilmaydi.
- Pod ish tugagach o'chiriladi va o'chirilgani tasdiqlanadi.
- Yakuniy o'lchov eval-120 da. **Endpoint budjetiga tegmasligi uchun o'lchov
  Pod ustida qilinadi**: model CT2 ga o'giriladi va 310 namuna o'sha yerda
  yuriladi. Endpoint xarajati $0.
- O'lchov `scripts/eval_pod.py` bilan. U ishga tushishdan oldin `handler.py`
  ni `ast` bilan o'qib, dekodlash production bilan AYNAN bir xil ekanini
  tasdiqlaydi va mos kelmasa to'xtaydi — aks holda taqqoslash modelni emas,
  sozlamani o'lchaydi.
- Taqqoslash `scripts/compare_runs.py` bilan — juftlashgan bootstrap,
  `analysis/eval120_prod.json` asosiy yurish sifatida.

**Narx.** A40 ~$0.39/soat:

| Bosqich | Vaqt |
|---|---|
| O'rnatish, model yuklash | ~20 daq |
| `calls-rejected` tiklash (5.19 soat audio) | ~30 daq |
| Trening, 900 qadam | ~25–30 daq |
| CT2 o'girish + eval-120 (310 namuna) | ~15 daq |
| **Jami** | **~1.6 soat ≈ $0.62** |

$5 chegarasidan ancha past. Endpoint xarajati $0.

## 5. Muvaffaqiyat mezoni — oldindan belgilanadi

Round 5 **muvaffaqiyatli** deb sanaladi, agar eval-120 da juftlashgan
bootstrap 95% oralig'i butunlay noldan past bo'lsa. 36.39% dan 1–2 punkt
"yaxshilanish" oraliq nolni kesib o'tsa — bu yaxshilanish emas, shovqin.
Round 4 aynan shu tarzda ajratilgan edi.

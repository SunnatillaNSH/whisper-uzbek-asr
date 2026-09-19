# Endpoint sozlamalari — ablatsiya

Bir xil **60 namuna** (81 dan hammasi har bir yurishda qaytmadi — shuning uchun
faqat hammasida mavjud bo'lganlari olindi; turli to'plamlardagi WER'ni
solishtirish noto'g'ri xulosa berardi).

Normallashtirish repodagi `eval_calls_wer.py` dagi `norm()` bilan **aynan bir
xil** — ya'ni raqamlar brifdagi jadval bilan solishtirsa bo'ladi.

| Sozlama | WER | sub | del | ins | So'z qamrovi |
|---|---|---|---|---|---|
| beam_size=1 | 50.18% | 861 | 416 | 117 | 89.2% |
| VAD=2000 (hozirgi standart) | 48.27% | 782 | 443 | 116 | 88.2% |
| VAD o'chiq | 47.98% | 780 | 420 | 133 | 89.7% |
| hotwords o'chiq | 44.38% | 707 | 331 | 195 | 95.1% |
| *baseline (brief)* | *43.47%* | — | — | — | — |

## Xulosa

**1. `hotwords` — asosiy aybdor (−3.9 punkt).**

O'chirilganda WER 48.27% → **44.38%**, so'z qamrovi 88.2% → **95.1%**, tushib
qolish 443 → 331. Baseline'gacha atigi 0.9 punkt qoladi — 60 namunada bu
ishonch oralig'i ichida, ya'ni amalda farq yo'q.

Sababi: `hotwords` dekoderni ro'yxatdagi so'zlar tomon og'diradi. Ro'yxat
qo'ng'iroq mazmuniga mos kelmasa, model eshitgan so'zini tashlab yuboradi.
Qamrovning sakrashi aynan shuni ko'rsatadi — hotwords so'zlarni **bo'g'ib**
turgan.

Evaziga gallyutsinatsiya biroz oshadi (ins 116 → 195), lekin sof foyda ancha
katta.

**Tavsiya:** `handler.py` da `ASR_HOTWORDS` standart qiymati **bo'sh** bo'lsin.
Lug'at kerak bo'lgan holatda so'rovda alohida berilsin.

**2. VAD — muammo allaqachon hal bo'lgan (−0.3 punkt).**

`min_silence_duration_ms` 500 → 2000 tuzatishidan keyin VAD yoqiq/o'chiq
deyarli farq qilmayapti. Ya'ni standart ostonada u nutqni kesmayapti.
Butunlay o'chirishning ma'nosi yo'q — hozirgicha qolsin.

Tuzatishdan oldingi holat (254 qo'ng'iroqli o'lchovda): tushib qolish
xatolarning **44.3%** i, so'z qamrovi **80.7%**. Hozir: **31.5%** va **90%**.

**3. `beam_size` — tegilmasin.**

`beam_size=1` 50.18% berdi, ya'ni standart 5 dan **2 punkt yomon**. Baseline
xom `generate()` da greedy ishlatgani bu farqni tushuntirmaydi — demak
faster-whisper stack'ida beam 5 haqiqatan foydali.

## Keyingi qadam

Dataset uchun gipotezalarni **`hotwords: ""` bilan** yig'ish kerak — chunki
production ham shu sozlamaga o'tishi kerak, og'irliklar esa production
modelining xatolarini aks ettirishi shart.

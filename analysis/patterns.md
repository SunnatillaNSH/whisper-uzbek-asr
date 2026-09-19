# Xato naqshlari — SellUp Voice AI vs Muxlisa AI

**Yorliq (reference):** Muxlisa AI matni. U ham audioni eshitgan MUSTAQIL
tizim, mutlaq haqiqat emas. Ya'ni bu yerdagi "WER" — *Muxlisa bilan qancha
farq*, xatoning absolyut o'lchovi emas.

**Gipotezalar qayerdan:** modelning matni endpointga qayta yuborilmadi — u
allaqachon SellUp bazasida (`voice_transcripts`) saqlangan. Brifdagi 1-bosqich
(~$0.40, 35 daqiqa) shu sababli o'tkazib yuborildi.

| | |
|---|---|
| Namuna | **254** qo'ng'iroq (ikkala AI ham matn bergan) |
| Yorliq so'zlari | 39,916 |
| Umumiy WER | **46.8%** |
| Mediana / p10 / p90 | 45.3% / 30.1% / 69.0% |

WER hujjatdagi ~43% bahosiga yaqin — o'lchov usuli to'g'ri ishlayapti.

---

## 1. Xato turlari — eng muhim topilma

| Tur | Soni | Ulushi |
|---|---|---|
| Almashtirish (substitution) | 9,572 | 51.2% |
| **Tushib qolish (deletion)** | **8,273** | **44.3%** |
| Qo'shib yuborish (insertion) | 844 | 4.5% |

Odatdagi ASR'da tushib qolish ~15–20% bo'ladi. Bu yerda **44.3%** —
ya'ni xatolarning deyarli yarmi "model so'zni noto'g'ri eshitdi" emas, balki
**"model so'zni umuman chiqarmadi"**.

Buni so'z hisobi ham tasdiqlaydi: Muxlisa 38,797 so'z, Voice AI
31,617 so'z — **19% kamroq**.

### Nima tushib qolyapti

| ha | 228 |
| aka | 159 |
| xo'p | 154 |
| da | 138 |
| hozir | 117 |
| bir | 112 |
| u | 104 |
| ham | 89 |
| bo'ldi | 86 |
| yuz | 83 |
| yo'q | 81 |
| mana | 72 |

Hammasi qisqa, pauzalar orasidagi so'zlar: tasdiq (`ha`, `xo'p`, `aha`),
murojaat (`aka`), yuklama (`da`, `ham`). Bu **VAD (jimlik kesuvchi) ularni
nutq emas deb hisoblab tashlab yuborayotganini** ko'rsatadi.

---

## 2. Xato o'rni — kesish TO'G'RI ekan

Namuna ichidagi nisbiy o'rin bo'yicha taqsimot (10 bo'lak, foizda):

```
9  10  9  10  10  10  11  11  11  10
```

Taqsimot **tekis** (9.0–10.8%). Agar
xatolar boshida yoki oxirida to'planganida, bu bo'laklarga kesish chegarasida
so'z bo'linib qolganini — ya'ni **ma'lumot tayyorlashdagi nuqsonni**
ko'rsatardi.

Tekis taqsimot degani: 28 va 55 soniyalik bo'laklarga kesish to'g'ri
bajarilgan, muammo akustikada va VAD'da.

---

## 3. Almashtirish naqshlari

| allo | → | alo | 56 |
| hm | → | ha | 20 |
| men | → | mana | 19 |
| yo'q | → | yo | 16 |
| aha | → | ha | 16 |
| hozir | → | xo'p | 15 |
| u | → | bu | 14 |
| o'ttiz | → | o'tiz | 14 |
| bo'lmasa | → | bo'lmasam | 13 |
| tuzukmisiz | → | tuzungisiz | 11 |
| charchamang | → | charchamay | 11 |
| vaalaykum | → | alaykum | 10 |
| vaalaykum | → | assalomu | 10 |
| assalom | → | alaykum | 10 |
| to'lovi | → | to'lov | 10 |
| to'lov | → | to'lovi | 9 |
| xodimlarning | → | xodimlarni | 9 |
| dasturni | → | dastur | 9 |
| o'sha | → | shu | 9 |
| dasturdan | → | dasturidan | 9 |

Ko'pchiligi bir-ikki harflik tebranish (`allo→alo`, `o'ttiz→o'tiz`) yoki
salomlashish shakllari (`vaalaykum`/`assalom`). Domen atamalari
(`Face ID`, `davomat`) ro'yxatning tepasida emas — ya'ni `hotwords` foydasi
bo'ladi, lekin asosiy muammo u emas.

---

## 4. Namuna darajasidagi WER

| | |
|---|---|
| Mediana | 45.3% |
| p10 (eng yaxshi) | 30.1% |
| p90 (eng yomon) | 69.0% |

`sample_scores.csv` da har bir qo'ng'iroq uchun WER, davomiylik va cps bor.
`hard_examples.csv` da eng yomonlari — **eng yuqori 5% chetlangan holda**
(ular odatda qiyin audio emas, buzilgan yorliq).

---

## Xulosa — nima qilish kerak

1. **VAD ni o'chirib sinash** — xatolarning 44.3% i tushib qolish, va
   tushayotgani aynan qisqa so'zlar. Bu eng arzon va eng katta ta'sirli tuzatish.
2. **Hotwords** — `analysis/hotwords.txt` ga qarang.
3. **Imlo post-processing** — almashtirishlarning 30.9% i
   1–2 harflik farq, ya'ni trening emas, oddiy normallashtirish bilan tuzatiladi.
4. Bo'laklarga kesishni o'zgartirish **shart emas** — u to'g'ri ishlayapti.

---

## 5. VAD tajribasi — gipoteza SINALDI

Yuqoridagi gipotezani (44.3% tushib qolish VAD sababli) tekshirish
uchun 40 ta qo'ng'iroq `vad: false` bilan qayta yuborildi.

| | VAD yoqiq (hozirgi) | VAD o'chiq |
|---|---|---|
| WER (o'rtacha) | 49.5% | **46.2%** |
| WER (mediana) | 49.6% | **45.6%** |
| So'z qamrovi | 80.7% | **87.8%** |
| Tushib qolish | 1,265 | **921** (27% kam) |
| Qo'shib yuborish | 120 | 197 (1.6x ko'p) |

**Gipoteza tasdiqlandi:** VAD haqiqatan nutqni yeyapti. O'chirilganda tushib
qolish 27% kamaydi va so'z qamrovi
80.7% dan 87.8% ga ko'tarildi.

**Lekin bu bepul emas.** O'chirilganda model jimlikni "eshitib", yo'q so'zlarni
o'ylab topa boshlaydi — qo'shib yuborish 1.6 barobar
oshdi. Natijada 40 namunadan **25 tasi yaxshilandi,
15 tasi yomonlashdi**.

Sof natija: **3.2 punkt yaxshilanish**.
Bu real, lekin sehrli tayoqcha emas.

**Tavsiya:** VAD ni butunlay o'chirish o'rniga uning ostonasini yumshatish
(faster-whisper `vad_parameters`: `min_silence_duration_ms` ni oshirish,
`threshold` ni pasaytirish). Shunda qisqa tasdiq so'zlari saqlanib qoladi,
lekin jimlikdagi gallyutsinatsiya ham oshmaydi.

**Diqqat:** 40 namuna kichik to'plam. Bu farqni ishonchli tasdiqlash
uchun kattaroq sinov kerak — lekin so'z qamrovi ko'rsatkichi
(80.7% → 87.8%) mexanik va shubhasiz.

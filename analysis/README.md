# Xato tahlili va og'irlangan dataset

Voice AI (o'z modelimiz) matnini Muxlisa AI matni bilan solishtirish natijasi.
Hammasi SellUp bazasidagi mavjud transkriptlardan qurilgan — modelga qayta
so'rov yuborilmadi (VAD tajribasidan tashqari).

## Fayllar

| Fayl | Mazmuni |
|---|---|
| `patterns.md` | **Asosiy hisobot** — xato turlari, o'rni, naqshlar, VAD tajribasi |
| `punctuation.md` | Tinish belgilari, apostrof, imlo (WER ko'rsatmaydigan qism) |
| `hotwords.txt` | Domen atamalari — endpoint sozlamasiga |
| `hypotheses.json` | 254 juft: reference + hypothesis + WER |
| `sample_scores.csv` | Har bir qo'ng'iroq: WER, S/D/I, davomiylik, cps |
| `hard_examples.csv` | Eng yomonlari (eng yuqori 5% chetlangan) |
| `vad_experiment.json` | VAD yoqiq/o'chiq taqqoslash, 40 namuna |
| `train_weighted.csv` | **Trening dataseti** — `path,sentence`, qiyinlik bo'yicha takrorlangan |
| `train_scores.csv` | Shaffoflik: har bir namunaning WER/cps/takrorlash soni |
| `excluded.csv` | Chetlangan namunalar va sababi |

## Dataset

| | |
|---|---|
| Kirgan namuna | 1,394 |
| Chetlangan | 698 |
| Yakuniy qatorlar (takror bilan) | **2,198** |
| Audio (takror bilan) | **14.09 soat** (8.89 soat asl) |

Takrorlash: WER ≥60% → 3 marta (96 ta), 30–60% → 2 marta
(612 ta), <30% → 1 marta (686 ta).
Eng yuqori 5% (WER ≥ 0.76) **chetlangan** — ular odatda qiyin
audio emas, buzilgan yorliq.

Chetlash sabablari: uzun — 563, cps chegaradan tashqari — 36, eval qo'ng'irog'i — 61, WER eng yuqori 5% — 34, juda qisqa — 4.

Eval to'plamining 61 ta
qo'ng'irog'i ataylab chiqarildi — aks holda model o'lchov suhbatini treningda
ko'rib, WER soxta yaxshi chiqardi.

## Muhim cheklov

Bu dataset **yangi yorliq yaratmaydi** — faqat mavjud Muxlisa yorliqlarini
qayta tartiblaydi. Ya'ni model Muxlisa darajasidan **o'tib keta olmaydi**,
faqat unga tezroq yaqinlashadi. Undan oshish uchun odam tuzatgan yorliq kerak.

## Qayta ishga tushirish

```bash
python3 analysis/analyze_errors.py /tmp/an/pairs.json   # xato tahlili
python3 analysis/build_weighted.py                      # dataset
```

`pairs.json` SellUp D1 bazasidan olinadi: `transcripts` (Muxlisa) va
`voice_transcripts` (Voice AI) jadvallarini `call_key` bo'yicha birlashtirish.

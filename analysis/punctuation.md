# Tinish belgilari va imlo tahlili

**Nega alohida:** WER bu xatolarni **ko'rsatmaydi** — normallashtirishda tinish
belgilari olib tashlanadi. Lekin transkript keyin LLM tahliliga beriladi va u
yerda gap chegaralari ma'noni tashiydi.

## Zichlik (100 so'zga)

| Belgi | Muxlisa | Voice AI | Farq |
|---|---|---|---|
| `.` | 13.54 | 13.63 | +0.09 |
| `,` | 17.94 | 23.55 | +5.61 |
| `?` | 3.61 | 3.13 | -0.48 |
| `!` | 0.2 | 0.08 | -0.12 |

Voice AI **nuqtani xuddi shunday**, **vergulni ko'proq**, **savol belgisini
kamroq** qo'yadi. Ya'ni model uzluksiz matn chiqarmayapti — gap chegaralarini
ko'radi. Bu yaxshi xabar.

## Gap chegaralari

| | Muxlisa | Voice AI |
|---|---|---|
| Gaplar soni | 6,725 | 5,321 |
| O'rtacha gap uzunligi (so'z) | 5.8 | 5.9 |

Deyarli bir xil (5.8 va 5.9 so'z) —
model gaplarni birlashtirib ham, mayda-mayda qilib ham yubormayapti.

## Savol belgisi

Muxlisa 1,399 ta, Voice AI 989 ta —
**29% kamroq**.
Qo'ng'iroqda savol ko'p (`Yaxshimisiz?`, `Bo'ladimi?`), shuning uchun bu
sezilarli kamchilik: LLM tahlilida savol-javob tuzilishi yo'qoladi.

## Apostrof

| Variant | Muxlisa | Voice AI |
|---|---|---|
| `'‘'` | 2421 | 0 |
| `"'"` | 4517 | 5982 |
| `'’'` | 123 | 0 |

**Voice AI izchilroq** — faqat bitta variant (ASCII) ishlatadi, Muxlisa esa
uchtasini aralashtiradi. O'lchovga ta'sir qilmaydi (normallashtiramiz), lekin
chiqish sifati uchun Voice AI'niki afzal.

## Bosh harf

Muxlisa 20.2%, Voice AI 19.2% — deyarli teng.

## Imlo tebranishi

Almashtirishlarning **30.9%** i (2,955 tadan
9,572 ta) 1–2 harf bilan farq qiladi:

| allo | → | alo | 56 |
| hm | → | ha | 20 |
| men | → | mana | 19 |
| yo'q | → | yo | 16 |
| aha | → | ha | 16 |
| u | → | bu | 14 |
| o'ttiz | → | o'tiz | 14 |
| bo'lmasa | → | bo'lmasam | 13 |
| tuzukmisiz | → | tuzungisiz | 11 |
| charchamang | → | charchamay | 11 |
| vaalaykum | → | alaykum | 10 |
| to'lovi | → | to'lov | 10 |

Bular akustik xato **emas** — post-processing lug'ati bilan tuzatiladi,
qayta trening talab qilmaydi. Qolgan 6,617 tasi haqiqiy
akustik almashtirish.

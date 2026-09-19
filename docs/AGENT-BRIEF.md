# Agent uchun topshiriq — xato tahlili va dataset tayyorlash

Bu hujjat AI agentga beriladigan to'liq topshiriq. Agent kod yoza oladi va
ishga tushira oladi deb faraz qilinadi.

---

## Vazifa

O'zbek nutqni matnga o'giruvchi model ishlab turibdi, lekin real telefon
qo'ng'iroqlarida WER **~43%**. Nima uchun adashayotganini aniqlash va keyingi
trening uchun ma'lumotni tayyorlash kerak.

Uchta natija kutiladi:

1. **Xato naqshlari** — model qaysi so'zlarni qaysiga almashtiradi
2. **Diagnostika** — xatolar qayerda to'planadi va bu ma'lumot tayyorlashdagi
   muammoni ko'rsatadimi
3. **Qiyin namunalar ro'yxati** — keyingi treningda ko'proq o'rgatish uchun

---

## Kontekst

**Tizim.** Whisper large-v3 real qo'ng'iroqlarda fine-tune qilingan, RunPod
Serverless'da faster-whisper (CTranslate2) bilan ishlaydi. API hujjati:
`docs/API.md`. Endpoint: `https://api.runpod.ai/v2/vwobkifazoyyh1`.

**Yorliqlar qayerdan.** Barcha "to'g'ri" matnlar **Muxlisa AI** degan tijorat
xizmatidan olingan — u audioni eshitgan mustaqil tizim. Ya'ni WER "Muxlisa
bilan qancha farq" degani, mutlaq haqiqat emas.

**Ma'lumot.**

| Yo'l | Nima |
|---|---|
| `data/calls-colab/train.csv` | 941 namuna, `path,sentence` |
| `data/calls-colab/eval.csv` | 81 namuna — treningda ko'rilmagan |
| `data/calls-colab/audio/*.flac` | 16 kHz mono |
| `data/calls-dataset-28s/` | yangi partiya, 308 namuna, 28 s bo'laklar |
| `data/calls-dataset/manifest.jsonl` | to'liq metadata: duration, call_id, part |

Eval to'plami **qo'ng'iroq bo'yicha** ajratilgan — bitta suhbatning bo'laklari
train va eval'ga bo'linib ketmagan.

**Hozirgi raqamlar** (81 namunali eval, 3757 so'z):

| Model | WER |
|---|---|
| Round 2 (ochiq datasetlar) | 61.46% |
| Round 3 (qo'ng'iroqlar) | 48.02% |
| Round 4 (+34% ma'lumot) | 49.06% |
| Round 3 + faster-whisper | **43.47%** |

Round 3 va 4 farqi statistik ahamiyatsiz (p = 0.732).

---

## 1-bosqich: gipotezalarni yig'ish

Har bir eval namunasini endpoint'ga yuborib, modelning matnini saqlang.

```
POST /run   {"input": {"audio_base64": "...", "language": "uz", "beam_size": 5, "vad": true}}
GET  /status/{id}   → {"output": {"text": "...", "duration_sec": ..., "processing_time_sec": ...}}
```

Autentifikatsiya: `Authorization: Bearer rpa_...`. Batafsil `docs/API.md` da.

Natijani JSON'ga yozing: har bir namuna uchun `path`, `reference` (CSV dagi
matn), `hypothesis` (model matni), `duration`.

Sovuq start ~25 soniya, keyingilari ~3 soniya. 81 namuna ≈ 4 daqiqa, ≈ $0.05.

---

## 2-bosqich: xato tahlili

### a) Almashtirish naqshlari

Har bir namunada mos matnni so'zma-so'z tekislang (`difflib.SequenceMatcher`
yoki `jiwer.process_words` ning `alignments` maydoni) va **almashtirish
juftliklarini** sanang: `(haqiqiy so'z → model so'zi)`.

Eng ko'p uchraydigan 40 tasini chiqaring. Domen atamalari (`Face ID`,
`davomat`, `xodim`, `apparat`) alohida ajratilsin — ular `hotwords` ro'yxatiga
ketadi.

### b) Xato turlari

Umumiy bo'yicha ajrating: **almashtirish / tushib qolish / qo'shib yuborish**
(substitution / deletion / insertion). Nisbat muhim:

| Ustun tur | Nimani anglatadi |
|---|---|
| Tushib qolish ko'p | VAD nutqni yeyapti yoki audio kesilgan |
| Qo'shib yuborish ko'p | model gallyutsinatsiya qilyapti |
| Almashtirish ko'p | akustik muammo — normal holat |

### c) Xato o'rni — ENG MUHIM QISM

Har bir xatoning namuna ichidagi **nisbiy o'rnini** hisoblang (0 = boshi,
1 = oxiri) va taqsimotni chiqaring (masalan 10 ta bo'lakda).

Agar xatolar **boshida yoki oxirida** to'planayotgan bo'lsa, bu akustik muammo
emas — **ma'lumot tayyorlashdagi nuqson**. Namunalar jimlik joyidan kesilgan
va chegarada so'z bo'linib qolgan bo'lishi mumkin. Bunday nuqson **butun
datasetga** ta'sir qiladi, shuning uchun uni topish eng qimmatli natija.

Tekis taqsimot = kesish to'g'ri, muammo akustikada.

### d) Namuna darajasidagi WER

Har bir namuna uchun WER hisoblang va taqsimotni bering: mediana, p10, p90.

Davomiylik, cps (belgi/soniya) va WER o'rtasida bog'liqlik bormi — tekshiring.
Masalan uzun namunalarda WER yuqori bo'lsa, bu 30 soniyalik oyna bilan bog'liq
muammoni ko'rsatadi.

---

## 3-bosqich: chiqish fayllari

| Fayl | Mazmuni |
|---|---|
| `analysis/hypotheses.json` | 1-bosqich natijasi |
| `analysis/patterns.md` | Xato naqshlari, turlari, o'rni — odam o'qiydigan hisobot |
| `analysis/hotwords.txt` | Domen atamalari, vergul bilan ajratilgan |
| `analysis/hard_examples.csv` | Eng yomon namunalar, `path,wer` |

`hard_examples.csv` uchun: WER bo'yicha saralang, lekin **eng yuqori 5% ni
chetlating** — ular odatda qiyin audio emas, balki buzilgan yorliq.

---

## MAJBURIY QOIDALAR

Quyidagilar oldingi bosqichlarda qimmatga tushgan xatolardan kelib chiqqan.
Ularni buzmang.

### 1. Apostrofni normallashtiring

O'zbek lotin yozuvida apostrof **harfning bir qismi** (`o'`, `g'`). Ma'lumotda
uch xil variant aralash uchraydi: U+2018 (425 marta), ASCII (279), U+2019 (7).
Whisper yana boshqasini chiqarishi mumkin.

Normallashtirmasangiz, bir xil jumla **WER 0.50** beradi. To'plamdagi
so'zlarning **18.7%** ida apostrof bor — ya'ni o'lchov ~19 foiz punktgacha
buziladi va butun tahlil ma'nosiz bo'lib qoladi.

```python
APOS = {ord(c): "'" for c in "‘’ʻʼ`´"}
def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)   # tinish belgilari → bo'shliq
    return re.sub(r"\s+", " ", s).strip()
```

Apostrofning **o'zini olib tashlamang** — `ozim` va `o'zim` qo'shilib ketadi.

### 2. jiwer transform argumentlariga bog'lanmang

`jiwer.wer(truth_transform=...)` yangi versiyalarda **yo'q**
(`TypeError: unexpected keyword argument`). Normallashtirishni qo'lda qiling
va tayyor satrlarni bering.

### 3. LLM tuzatgan matnni YORLIQ sifatida ishlatmang

Agar model matnini LLM'ga berib tuzattirsangiz, natija **trening uchun
yaroqsiz**. LLM audioni eshitmaydi — u kontekstdan taxmin qiladi. Bunday
yorliqda o'qitilgan model "audioga mos" emas, "ishonchli eshitiladigan" matn
chiqarishni o'rganadi, ya'ni gallyutsinatsiya kuchayadi.

LLM tuzatishi **production'da** foydali (matn odamga tushunarli bo'ladi),
trening yorlig'i sifatida emas.

### 4. Yuqori WER har doim "qiyin audio" degani emas

U ikki narsani bildirishi mumkin: audio qiyin **yoki yorliq noto'g'ri**.
Ikkinchisini ko'proq o'rgatish zarar qiladi. Shuning uchun eng yomon 5% ni
qo'lda ko'rib chiqing yoki chetlating.

### 5. Whisper oynasi — qat'iy 30 soniya

Undan uzun namunani berish audioni kesadi, matn esa to'liq qoladi va model
"eshitilmagan" so'zlarni o'ylab topishga o'rganadi. Har qanday yangi namuna
**≤30 soniya** bo'lishi shart.

### 6. Eval to'plamini o'zgartirmang

81 namuna o'z holicha qolsin — aks holda oldingi round'lar bilan taqqoslab
bo'lmaydi. Yangi ma'lumotning hammasi train'ga ketadi.

---

## Bilib qo'yish kerak bo'lgan cheklov

Eval to'plami kichik: 81 namuna, **3757 so'z**. Ishonch oralig'i ±6–7 foiz
punkt. Ya'ni bu o'lchov asbobi **7 punktdan kichik yaxshilanishni ko'ra
olmaydi**.

Agar tahlil natijasida "farq bor, lekin kichik" degan xulosa chiqsa, uni
tasdiqlash uchun avval eval to'plamini kengaytirish kerak (150–200 qo'ng'iroq).
Kichik to'plamda olingan kichik farqlarga ishonmang.

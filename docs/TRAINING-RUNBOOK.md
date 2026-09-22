# Trening runbook — Whisper Uzbek ASR (qo'ng'iroq domeniga moslash)

Manba: `train_calls.py` va `scripts/round6_*.sh` zanjiri. Skript bilan bu
hujjat orasida farq bo'lsa, skript haqiqiy manba hisoblanadi. `round6_pod.sh`
shu runbook'ning ishlaydigan namunasi — keyingi raundda shu tartibni
takrorlang.

## 1. Oldindan tekshiruv (preflight)

`round6_pod.sh` trening boshlanishidan OLDIN, bitta joyda tekshiradi va
yetishmagan narsa bo'lsa darhol `die` bilan to'xtaydi:

```bash
[ -f "$DATA/round6_calls.csv" ]    || die "$DATA/round6_calls.csv yo'q"
[ -f "$DATA/eval120/eval50.csv" ]  || die "eval50.csv yo'q"
[ -f "$DATA/eval_calls_120.json" ] || die "eval_calls_120.json yo'q (dev-split assertlari usiz ishlamaydi)"
[ -d "$DATA/calls-rejected" ]      || die "calls-rejected yo'q"
python3 scripts/eval_pod.py --check-only || die "dekodlash handler.py bilan mos emas"
```

- `round6_calls.csv` — asosiy trening CSV, `round6_send.sh` uni
  `analysis/train_weighted.csv`dan nusxalab yaratadi.
- `eval120/eval50.csv` — zanjir oxiridagi `eval_pod.py` shuni o'lchaydi.
- `eval_calls_120.json` — `train_calls.py`dagi `load_eval120_calls()` shuni
  o'qiydi; topilmasa trening **umuman boshlanmaydi** ("Bu ro'yxatsiz dev
  bo'lagi eval-120 bilan kesishmasligini tekshirib bo'lmaydi").
- `calls-rejected` — `recover_rejected.py` shu katalogdagi
  `rejected.jsonl`ni tiklaydi.
- `eval_pod.py --check-only` — GPU ishlamasdan, `handler.py`dagi
  `model.transcribe(...)` chaqiruvini `ast` bilan o'qib Pod dekodlash
  sozlamalarining production bilan mosligini tasdiqlaydi (beam 5, VAD
  2000/400 ms, `condition_on_previous_text=False`, temperatura zaxirasi,
  bo'sh `hotwords`). Mos kelmasa: "DEKODLASH MOS EMAS — o'lchov modelni
  emas, sozlamani o'lchagan bo'lardi."

### Assertlar: train/dev/eval-120 kesishuvi 0

`train_calls.py` `load_calls()`: avval qat'iy chiqish —
`set(df["call"]) & eval120` bo'sh bo'lmasa `sys.exit`. So'ng train/dev
bo'lish **qo'ng'iroq** darajasida (`DEV_FRAC=0.10`, `DEV_SEED=20260919`) —
fayl darajasida emas, chunki bitta suhbatning qo'shni bo'laklari bir xil
ovoz va iboralarni ulashadi. Oxirida uchta `assert`:

```python
assert not (set(tr["call"]) & set(ev["call"])), "train va dev kesishdi"
assert not (set(tr["call"]) & eval120), "train eval-120 bilan kesishdi"
assert not (set(ev["call"]) & eval120), "dev eval-120 bilan kesishdi"
```

`recover_rejected.py` xuddi shu tekshiruvni tiklashda takrorlaydi:
`eval_calls_120.json`dagi qo'ng'iroqlarni `rejected.jsonl`dan chiqarib
tashlaydi va sonini bosadi ("eval-120 chetlandi: N bo'lak"). Bu
`train_calls.py`dagi assert **oxirgi** to'siq, birinchisi emas — sizib
o'tgan eval qo'ng'irog'i shu yergacha yetsa, butun trening vaqti behuda
ketgan bo'ladi.

### Byudjet chegarasi

`round6_pod.sh`: `BUDGET_USD=${BUDGET_USD:-2.5}`. `train_calls.py`ning o'z
standarti `BUDGET_USD=5.0`, `SETUP_USD=1.5`. To'liq mexanizm 6-bo'limda.

### GPU tanlash — A40 vs L40S

| GPU | Narx | Xususiyat |
|---|---|---|
| A40 48 GB | ~$0.49/soat (`round6_pod.sh` standarti, `POD_RATE`) | sekinroq |
| L40S 48 GB | ~$1.09–1.11/soat (`train_calls.py` standarti, `POD_RATE=1.10`) | tezroq |

**A40**: kichik dataset (bir necha yuz-ming namuna, ~1000-2500 qadam) va tor
byudjet bo'lganda — Round 6 shu holat (`BUDGET_USD=2.5`). **L40S**: trening
uzoqroq (5000+ qadam) yoki natijani tezroq olish muhim bo'lganda; Round
3-4-5 L40S bilan qilingan (~$1.10/soat, jami ~$1.6-2.0). Ikkala holatda ham
`POD_RATE` haqiqiy Pod narxiga moslab berilishi shart — byudjet
qo'riqchisi shu qiymatdan hisoblaydi.

## 2. Pod yaratish

RunPod → Pods → Deploy:

- **GPU**: A40 yoki L40S, 48 GB (yuqoriga qarang).
- **Template**: PyTorch obrazi (CUDA, Python 3 oldindan o'rnatilgan).
- **Container disk**: OS + paketlar uchun — Pod o'chirilganda **yo'qoladi**,
  shuning uchun bu yerga doimiy hech narsa saqlanmaydi.
- **Volume disk**: `/workspace` ga ulanadi. Amalda Round 5 da **80 GB**
  ishlatilgan va yetgan (2026-09-19, L40S Pod, container disk 30 GB);
  README'dagi eski tavsiya 100 GB, Round 3 da 60 GB yetgan.
  **Turi muhim:** "Volume disk" Pod bilan birga o'chadi — ataylab shu
  tanlanadi, aks holda unutilgan volume har oy pul yechib turadi
  ($0.10/GB/oy ishlayotganda, $0.20 to'xtatilganda). Kerakli joy: baza/round-2 model (~3 GB) + CT2
  nusxasi (~1.6 GB) + dataset (audio) + checkpointlar (`save_total_limit=3`,
  har biri LoRA-adapter hajmida) + yakuniy merged model (~3 GB) + yakuniy
  CT2 (~1.6 GB).
- **TCP 22**: SSH uchun ochiq portlar ro'yxatiga qo'shiladi; RunPod buni
  tashqi portga map qiladi — `round6_send.sh`/`round6_start.sh` shu portni
  `$PORT` sifatida oladi (`./scripts/round6_send.sh <host> <port>`).
- **SSH kalit oldindan qo'shilishi shart**: ulanish skriptlari
  `BatchMode=yes` bilan ishlaydi:
  ```bash
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -i $KEY -p $PORT root@$HOST
  ```
  `BatchMode=yes` parol so'ramaydi — Mac'ning ochiq kaliti RunPod
  hisobida Pod yaratishdan OLDIN qo'shilmagan bo'lsa, ulanish shovqinsiz
  muvaffaqiyatsiz tugaydi (interaktiv prompt chiqmaydi).

### Pod O'CHIRILISHI majburiyligi

Ikki sabab: (1) Pod ishlab yoki hatto to'xtatilgan holatda ham soatiga
to'lov davom etadi — byudjet qo'riqchisi faqat trening jarayonini
to'xtatadi, Pod'ni emas; (2) **volume disk Pod bilan birga o'chadi** —
`/workspace`dagi checkpoint, CT2 model, eval natijalari Pod o'chirilganda
butunlay yo'qoladi. Shuning uchun 7-bo'limdagi "Mac'ga scp" qadami Pod'ni
o'chirishdan OLDIN, `PIPELINE_DONE` chiqqandan keyin bajarilishi shart.

## 3. Ulanish va ma'lumot ko'chirish

`scripts/round6_send.sh <host> <port>` Mac'dan yuriladi:

1. **Ulanish tekshiruvi** — GPU, Python versiyasi, `df -h /`.
2. **Repo + paketlar fonda** — `git clone --depth 1`, so'ng
   `pip install --break-system-packages --no-cache-dir -r
   requirements-train.txt faster-whisper ctranslate2` fon jarayonida
   (`/root/pip.log`ga), `torchao` ATAYLAB o'chiriladi (sabab 8-bo'limda).
3. **Fayllar ro'yxati** — lokal Python `analysis/train_weighted.csv`,
   `data/eval120/eval50.csv` va `data/calls-rejected/rejected.jsonl`dagi
   barcha audio yo'llarini yig'adi, mavjudligini tekshiradi va
   **fayllar soni + hajmni MB da** bosadi:
   ```python
   print(f"fayllar: {len(files)} | hajm: {mb:.0f} MB")
   ```
   Round 5'da butun `data/` katalogi yuborilib, `calls-colab.tar` (audioning
   ikkinchi nusxasi) sabab ~600 MB ortiqcha trafik ketgan edi — endi faqat
   kerakli fayllar.
4. **Ko'chirish** — `tar cf - -T /tmp/round6_files.txt ... | ssh ... 'tar
   xf -'`, oraliq fayl yaratmasdan. Pod tomonida `analysis/train_weighted.csv`
   → `data/round6_calls.csv` deb nusxalanadi. Davomiylik soniyada bosiladi.

**sha256 tekshiruvi**: uzatish paytida checksum solishtirish yo'q — Mac
tomonidagi "fayllar: N | hajm: M MB" bilan Pod tomonidagi "fayllar soni"
(`find data -type f | wc -l`) qo'l bilan solishtiriladi. Yakuniy model
uchun sha256 majburiy (7-bo'lim).

## 4. Zanjir: `round6_pod.sh` bosqichlari

`scripts/round6_start.sh <host> <port>` `round6_pod.sh`ni `scp` bilan
ko'chiradi va ajratilgan jarayon sifatida ishga tushiradi:

```bash
setsid nohup /root/round6_pod.sh < /dev/null > /dev/null 2>&1 & disown
```

`setsid` + `< /dev/null` **majburiy**: Round 5'da trening SSH sessiyasi
yopilishi bilan o'lgan — fon jarayoni stdin'ni ushlab turgan va SSH
uzilganda SIGHUP olgan. `POD_START_EPOCH` Unix epoch sifatida Mac'dan
beriladi (sababi 6-bo'limda).

Zanjir o'zi (hammasi `$OUT/pipeline.log`ga, `exec >> "$OUT/pipeline.log"
2>&1`):

1. **round-2 → CT2.** `Sunnat0091/whisper-large-v3-uz` yuklanadi, tokenizer
   fayllari `openai/whisper-large-v3`dan ustiga nusxalanadi (sabab
   8-bo'limda), `ct2-transformers-converter --quantization float16
   --copy_files ...` bilan `$OUT/round2-ct2`ga o'giriladi. Tiklash uchun
   kerak; allaqachon bo'lsa o'tkazib yuboriladi.
2. **`calls-rejected` tiklash.** `recover_rejected.py --model
   "$OUT/round2-ct2" --src data/calls-rejected --out data/calls-recovered
   --csv data/recovered.csv` — faster-whisper so'z vaqtlaridan jumla
   chegarasida kesadi (yorliq matni o'zgarmaydi). Bor bo'lsa o'tkaziladi.
3. **Trening — FAQAT qo'ng'iroq.** `EXTRA_DIR` berilmaydi, `USE_PODCAST=0`
   — Round 5'da tashqi korpus batch'ning aksariyatini egallab, modelni
   telefon domenidan uzoqlashtirgan (production'dan +8.17 punkt yomon,
   statistik ishonchli). `python3 -u train_calls.py` 5-bo'limdagi env
   bilan.
4. **Adapter ajratish.** Barcha `checkpoint-*/trainer_state.json`dan
   oxirgisi o'qiladi, `best_model_checkpoint` va `best_metric` (dev loss)
   bosiladi, o'sha checkpointdan `adapter*` + `README.md` `$OUT/adapter`ga
   nusxalanadi, `$OUT/best.txt` ga yoziladi.
5. **Yakuniy model → CT2.** `$OUT/out-final` (LoRA allaqachon birlashtirilgan
   holda saqlangan) production bilan **aynan bir xil** buyruq bilan
   `$OUT/ct2`ga o'giriladi.
6. **eval-50.** `eval_pod.py --model "$OUT/ct2" --csv data/eval120/eval50.csv
   --out "$OUT/eval50_r6.json"`.
7. **DONE.** `touch "$OUT/PIPELINE_DONE"`.

Kuzatish: `ssh ... 'tail -f /workspace/round6/pipeline.log'`. Har bosqich
`step "..."` bilan vaqt tamg'ali sarlavha chiqaradi.

## 5. Giperparametrlar va ularning ASOSI

`round6_pod.sh`: `MAX_STEPS=${MAX_STEPS:-2000} EVAL_STEPS=200 SAVE_STEPS=200
WARMUP_STEPS=100 BATCH_SIZE=8 GRAD_ACCUM=1 LR=5e-5 PATIENCE=3`.
`train_calls.py`ning o'z standarti: `MAX_STEPS=900 EVAL_STEPS=100
SAVE_STEPS=100 WARMUP_STEPS=100 PATIENCE=4 LORA_R=32 LORA_ALPHA=64
LORA_DROPOUT=0.05`.

**8 epoxa qoidasi.** Skript oldindan hisoblab bosadi:
```python
epochs = MAX_STEPS * BATCH_SIZE * GRAD_ACCUM / len(train_ds)
if epochs > 15:
    print("  ⚠️  Epoxa soni yuqori — yodlab olish (overfitting) xavfi bor.")
```
Qoida: **`MAX_STEPS = namunalar × 8 / batch`**. Round 3 (1000/941=8.5
epoxa) va Round 4 (1250/1258=7.9 epoxa)da dev loss ~8 epoxa atrofida
minimal bo'lib, keyin tekis/yomonlashgan. Avvalgi standart (3000 qadam,
~1022 namunada ~23 epoxa) ancha oshiq edi. Real qadam sonini shu formula
bilan hisoblang, taxmin qilmang.

**Eval/save oralig'i** odatda teng (Round 6: 200/200; standart: 100/100)
— har nazorat nuqtasida baholanadi VA saqlanadi, shu bilan
`load_best_model_at_end` istalgan nazorat nuqtasidan tanlay oladi.

**EarlyStopping patience.** `PATIENCE` marta ketma-ket dev loss
yaxshilanmasa to'xtaydi. Round 6: 3 (standart 4dan qattiqroq — tor byudjet).
Round 5'da mexanizm 5500-qadamda (eng yaxshisi 3500) to'xtatgan — bu
**natija, nosozlik emas**: ma'lumot yetishmasligi belgisi.

**LoRA r32/alpha64 q/v.** `target_modules=["q_proj","v_proj"]`, barcha
raundlarda o'zgarmagan (~15.7M parametr, ~1.01%). Kengaytirish kichik
datasetda overfitting xavfini oshiradi, shuning uchun ataylab qoldirilgan.

**LR 5e-5, batch 8.** `LR=5e-5` aralash datasetdagi `1e-4`dan past —
faqat-qo'ng'iroq bilan kichikroq LR barqarorroq. `BATCH_SIZE=8
GRAD_ACCUM=1` A40/L40S 48 GB'ga sig'adi.

**`load_best_model_at_end`.** `metric_for_best_model="loss"`,
`greater_is_better=False`. `predict_with_generate` ISHLATILMAYDI — PEFT +
Whisper + Seq2SeqTrainer'da `generate()` autocast tashqarisida dtype
xatosi beradi. Trening ichidagi baholash faqat loss orqali, WER trening
tugagach alohida (`eval_pod.py`/`eval_endpoint.py`) hisoblanadi.

## 6. Byudjet qo'riqchisi

`ProgressCallback` har `EVAL_STEPS`da:
```python
per = el / s
left = per * (MAX_STEPS - s)
train_usd = (el + left) / 3600 * POD_RATE
total = train_usd + SETUP_USD
if total > BUDGET_USD:
    control.should_training_stop = True
```
— haqiqiy o'tgan vaqtdan s/qadam hisoblaydi, qolganini shu tezlikda
bashorat qiladi. `total > BUDGET_USD` bo'lsa trening **darhol** to'xtaydi
(eng yaxshi checkpoint allaqachon diskda). `0.8×BUDGET_USD`da bir marta
ogohlantirish.

`round6_pod.sh` `SETUP_USD`ni real o'tgan vaqtdan hisoblaydi:
```bash
SETUP_USD=$(python3 -c "
import time
el = (time.time() - $START_EPOCH)/3600 if $START_EPOCH else 0.5
print(f'{el*$RATE + 0.35:.2f}')")
```
— Pod boshlangandan trening boshlanguncha o'tgan vaqt × soat narxi + $0.35
zaxira (CT2/eval uchun).

**Nega Unix epoch.** `round6_start.sh` izohi: "Round 5'da Pod soati UTC,
Mac'niki mahalliy bo'lgani uchun byudjet bazasi MANFIY chiqqandi va
qo'riqchi ishlamay qolgandi." Unix epoch (`date +%s`) vaqt mintaqasidan
mustaqil bo'lgani uchun bu muammoni yo'q qiladi.

## 7. Yakunlash

1. **Model Mac'ga scp.** Minimal to'plam: **CT2 model** (`$OUT/ct2/`) va
   **LoRA adapteri** (`$OUT/adapter/`, ~100 MB). Merged fp16 (~3 GB) odatda
   ko'chirilMAYDI — adapter + baza modeldan qayta yig'iladi
   (`WhisperForConditionalGeneration.from_pretrained(baza)` →
   `PeftModel.from_pretrained(..., adapter)` → `merge_and_unload().half()`,
   `models/README.md`da to'liq retsept).
2. **SHA256SUMS.** Har katalog uchun `shasum -a 256` bilan yozib qo'yiladi,
   Mac'da `shasum -a 256 -c SHA256SUMS` bilan tasdiqlanadi (namuna:
   `models/round5/SHA256SUMS`).
3. **DONE belgisi.** `$OUT/PIPELINE_DONE` — zanjir to'liq tugaganining
   mashina-tekshiriladigan belgisi.
4. **Natija hujjati.** `docs/ROUND{N}-RESULT.md`, `ROUND5-RESULT.md`
   tuzilishida: asosiy taqqoslash (juftlashgan bootstrap), xato tarkibi
   (S/D/I), trening jadvali, narx, model qayerda (SHA256SUMS bilan),
   tavsiya (joylashtirilsinmi).
5. **Pod o'chirish** — FAQAT 1-4 tugagandan keyin (volume disk shu bilan
   birga yo'qoladi).
6. **Tekshirish.** `shasum -c` muvaffaqiyatli, `models/round<N>/ct2/model.bin`
   va `adapter/adapter_model.*` mavjud, RunPod panelida Pod endi
   ishlamayotgani tasdiqlanadi.

## 8. Nosozliklar jadvali

| Belgi | Sabab | Yechim |
|---|---|---|
| `pip install` "externally-managed-environment" | PEP 668 — yangi Ubuntu obrazida tizim pip'i himoyalangan | `--break-system-packages` |
| `AttributeError: 'list' object has no attribute 'keys'` | Model repo transformers **5.x** formatida (`extra_special_tokens` ro'yxat), Pod'dagi transformers **4.x** kutmaydi | Tokenizer fayllarini `openai/whisper-large-v3`dan nusxalang — LoRA faqat q/v'ga tegadi (`load_processor()` avtomatik shunday qiladi) |
| `datasets.map`/audio yuklashda torchcodec xatosi | `datasets` **4+** audio dekodlash uchun torchcodec talab qiladi, obrazda yo'q | `datasets<4` pin |
| `torchao` import xatosi (`ScalingType`) | Pod'dagi torchao versiyasi torch 2.8 bilan mos emas | `pip uninstall -y --break-system-packages torchao` (Colab'da muammo TESKARI — u yerda torchao eski edi) |
| Trening SSH uzilishi bilan to'xtaydi | Fon jarayon stdin'ni ushlab turgan, SIGHUP olgan | `setsid nohup ... < /dev/null > /dev/null 2>&1 & disown` |
| `pkill` o'z buyrug'ini ham o'ldiradi | `pkill -f <nom>` terminaldagi buyruqning o'zidagi matnni ham topadi | `pkill -f '[a]lign_long_calls'` kabi kvadrat qavs bilan |
| CUDA OOM | `batch_size` katta yoki `word_timestamps=True` + `batch_size>1` (cross-attention xotira yeydi) | `BATCH_SIZE` kamaytiring; so'z-vaqti kerak bo'lsa `batch_size=1` |
| "No space left on device" | Volume disk to'ldi (model nusxalari + checkpointlar + dataset) | `save_total_limit=3` shu uchun bor; katta Volume bilan qayta yarating |
| Pod o'chirilgach natijaga murojaat qilib bo'lmaydi | `/workspace` Pod bilan birga yo'qoladi | `PIPELINE_DONE` chiqishi bilan darhol 7-bo'limdagi scp'ni bajaring, Pod'ni o'chirishdan OLDIN |
| `eval_pod.py --check-only` to'xtaydi | `DECODE` lug'ati yoki `HANDLER` yo'li production `handler.py`dan farq qiladi | Qaysi kalit (`beam_size`, `vad_parameters`, `temperature`, `HOTWORDS`) mos emasligini o'qing va tenglashtiring |

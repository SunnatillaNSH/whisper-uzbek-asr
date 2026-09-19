#!/usr/bin/env python3
"""Whisper large-v3 ni REAL QO'NG'IROQLARGA moslash (RunPod Pod).

`train.py` dan farqi: u ochiq HF datasetlarida o'zbek TILINI o'rgatadi, bu esa
o'z qo'ng'iroqlaringizda telefon DOMENIGA moslaydi.

Round 1-2 eval WER'ni 34.01% → 27.82% ga tushirdi, lekin real qo'ng'iroqda
sezilarli yaxshilanish bermadi — sabab til bilimi emas, domen farqi: ochiq
datasetlar toza va mikrofonga yaqin, qo'ng'iroq esa 8 kHz, siqilgan, shovqinli.

Ma'lumot: bitta CSV (`TRAIN_CSV`) — Round 5 uchun `analysis/build_weighted.py`
bergan 975 namuna / 312 qo'ng'iroq / 6.49 soat. Dev bo'lagi shu CSV ichidan
QO'NG'IROQ bo'yicha kesiladi (884 train / 91 dev), dataset paketidagi
`eval.csv` dan EMAS — sababi `load_calls()` ustidagi izohda.

Dataset paketida `eval_calls_120.json` bo'lishi SHART: usiz bo'lish eval-120
bilan kesishmasligini tekshirib bo'lmaydi va skript to'xtaydi.

Ishga tushirish:
    pip install -r requirements-train.txt
    python train_calls.py

Sozlash (muhit o'zgaruvchilari, standart qiymatlar qavsda):
    CALLS_DIR=/workspace/calls    MAX_STEPS=900    BATCH_SIZE=8   LR=5e-5
    TRAIN_CSV=train.csv           DEV_FRAC=0.10    DEV_SEED=20260919
    USE_PODCAST=0                 OUTPUT_DIR=/workspace/whisper-uz-calls
"""

import json
import os
import random
import sys
import tarfile

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Audio, Dataset, Features, Value, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from scipy.signal import butter, lfilter, resample_poly
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ──────────────────────────── Sozlamalar ────────────────────────────

def env(name, default, cast=str):
    return cast(os.environ.get(name, default))

MODEL_NAME = env("MODEL_NAME", "Sunnat0091/whisper-large-v3-uz")
# Round-2 modelidan boshlaymiz, bazadan emas: o'zbek tili unda allaqachon bor,
# shuning uchun 3000 qadamning hammasi domenga ketadi.

CALLS_DIR  = env("CALLS_DIR", "/workspace/calls")
CALLS_TAR  = env("CALLS_TAR", "/workspace/calls-colab.tar")
OUTPUT_DIR = env("OUTPUT_DIR", "/workspace/whisper-uz-calls")

# Dev to'plami TRENING qo'ng'iroqlaridan kesiladi, dataset paketidagi
# `eval.csv` dan emas.
#
# Nega. Paketdagi `eval.csv` — `calls-colab/eval.csv`, 31 qo'ng'iroq, va
# ularning hammasi eval-120 ichida. `load_best_model_at_end=True` bo'lgani
# uchun eng yaxshi nazorat nuqtasi aynan o'sha qo'ng'iroqlar bo'yicha
# TANLANADI. Gradientga tushmasa ham bu tanlov sizishi: yakuniy eval-120
# raqami o'z nazorat nuqtasini tanlagan to'plamda o'lchanadi va xolis
# bo'lmaydi.
# Vergul bilan bir nechta CSV berilishi mumkin: asosiy to'plam + Pod'da
# tiklangan `calls-rejected`. Ular birlashtiriladi, so'ng dev bo'lagi
# BIRLASHGAN to'plamdan qo'ng'iroq bo'yicha kesiladi — aks holda tiklangan
# qo'ng'iroq train'da, uning boshqa bo'lagi dev'da qolib ketishi mumkin.
TRAIN_CSV  = env("TRAIN_CSV", "train.csv")      # CALLS_DIR ichida
DEV_FRAC   = env("DEV_FRAC", "0.10", float)     # qo'ng'iroq bo'yicha
DEV_SEED   = env("DEV_SEED", "20260919", int)
EVAL_CALLS = env("EVAL_CALLS", "eval_calls_120.json")   # CALLS_DIR ichida

LANGUAGE, TASK, SR = "uzbek", "transcribe", 16000
MAX_LABEL = 448

# 884 namuna / batch 8 = 111 qadam/epoxa, 8 epoxa = 888. Avvalgi standart
# 3000 edi — o'sha hajmda 27 epoxa degani, ya'ni ortiqcha o'qish hududi.
MAX_STEPS   = env("MAX_STEPS", "900", int)
EVAL_STEPS  = env("EVAL_STEPS", "100", int)
SAVE_STEPS  = env("SAVE_STEPS", "100", int)
WARMUP      = env("WARMUP_STEPS", "100", int)
BATCH_SIZE  = env("BATCH_SIZE", "8", int)      # A40/L40S 48 GB → 16 ham bo'ladi
GRAD_ACCUM  = env("GRAD_ACCUM", "1", int)
LR          = env("LR", "5e-5", float)
WORKERS     = env("DATALOADER_WORKERS", "6", int)

# Erta to'xtash. 10 000 qadam qo'ng'iroqlarda ~15-20 epoxa degani, Round 3/4
# da esa optimum ~8 epoxa edi. Dev loss PATIENCE marta ketma-ket
# yaxshilanmasa to'xtaymiz va `load_best_model_at_end` eng yaxshisini oladi.
# To'xtash — natija, nosozlik emas.
PATIENCE    = env("PATIENCE", "4", int)

# Byudjet qo'riqchisi. Pod soatiga pul turadi; birinchi EVAL_STEPS dan keyin
# haqiqiy s/qadam ma'lum bo'ladi va yakuniy narx bashorat qilinadi. Chegaradan
# oshsa trening TO'XTAYDI — eng yaxshi nazorat nuqtasi saqlanib qoladi.
POD_RATE    = env("POD_RATE", "1.10", float)     # $/soat (L40S)
BUDGET_USD  = env("BUDGET_USD", "5.0", float)
SETUP_USD   = env("SETUP_USD", "1.0", float)     # tiklash, yuklash, eval, CT2

LORA_R       = env("LORA_R", "32", int)
LORA_ALPHA   = env("LORA_ALPHA", "64", int)
LORA_DROPOUT = env("LORA_DROPOUT", "0.05", float)

# Podkast qo'shish (0 = faqat qo'ng'iroq). Qo'shilsa overfitting xavfi kamayadi,
# lekin domen diqqati susayadi va yuklab olish vaqt oladi.
USE_PODCAST = env("USE_PODCAST", "0", int)
POD_ID = "BoburAmirov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset"
CALL_RATIO = env("CALL_RATIO", "0.35", float)

# Tashqi korpus (Round 5 da UzbekVoice). HF'dan to'g'ridan yuklanmaydi —
# `scripts/prep_extra.py` uni oqim bilan olib, faqat kerakli qismini diskka
# yozadi va CSV beradi. Sabab: to'liq dataset o'nlab GB, bizga esa ~25k
# namuna kerak, va Pod soatiga pul to'lanadi.
EXTRA_DIR = env("EXTRA_DIR", "")                 # bo'sh = ishlatilmaydi
EXTRA_CSV = env("EXTRA_CSV", "extra.csv")        # EXTRA_DIR ichida

AUG_PROB      = env("AUG_PROB", "0.75", float)       # podkast uchun
CALL_AUG_PROB = env("CALL_AUG_PROB", "0.6", float)   # qo'ng'iroq uchun


BASE_PROCESSOR = "openai/whisper-large-v3"


def load_processor(model_id):
    """Processor'ni yuklaydi, kerak bo'lsa baza modeldan.

    LoRA faqat q_proj/v_proj matritsalarini o'zgartiradi — tokenizer va
    feature extractor baza model bilan AYNAN bir xil qoladi. Shuning uchun
    model repo'sidagi tokenizer o'qilmasa (masalan transformers v5 formatida
    saqlangan bo'lsa, 4.x uni o'qiy olmaydi: "'list' object has no attribute
    'keys'"), baza modeldan olish mutlaqo xavfsiz.
    """
    try:
        return WhisperProcessor.from_pretrained(model_id, language=LANGUAGE, task=TASK)
    except Exception as e:
        print(f"⚠️  {model_id} processor o'qilmadi ({type(e).__name__}), "
              f"{BASE_PROCESSOR} dan olinadi", flush=True)
        return WhisperProcessor.from_pretrained(BASE_PROCESSOR, language=LANGUAGE, task=TASK)


# ──────────────────────────── Augmentatsiya ────────────────────────────

_B, _A = butter(4, [300 / (SR / 2), 3400 / (SR / 2)], btype="band")
_SPEED = [(19, 20), (21, 20), (39, 40), (41, 40)]     # ±5% va ±2.5%
MAX_AUDIO_SAMPLES = 30 * SR      # Whisper encoder oynasi


def mulaw(x, mu=255.0):
    """G.711 μ-law: siqish → 8 bitga kvantlash → yozish. Telefon kodegi."""
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    y = np.round(y * 127.0) / 127.0
    return np.sign(y) * ((1 + mu) ** np.abs(y) - 1) / mu


def phone_augment(x, rng):
    """Toza (podkast) audioni telefon kanalidan o'tkazilgandek qiladi.

    TARTIB MUHIM: shovqin ham, kodek ham 8 kHz oqimning ICHIDA ishlaydi. Shovqin
    16 kHz'da, qayta namunalashdan keyin qo'shilsa, u 4 kHz'dan yuqorida ham
    energiya qoldiradi — haqiqiy telefon audiosida u yerda hech narsa yo'q, va
    model shu soxta belgini "telefon audiosi" deb o'rganib oladi.
    """
    x = lfilter(_B, _A, x).astype(np.float32)      # 300-3400 Hz polosa
    x8 = resample_poly(x, 1, 2)                    # 16 kHz → 8 kHz (tarmoq)
    if rng.random() < 0.7:                         # liniya shovqini
        snr = rng.uniform(12, 30)
        p = np.mean(x8 ** 2) + 1e-12
        x8 = x8 + rng.normal(0, np.sqrt(p / (10 ** (snr / 10))), len(x8))
    if rng.random() < 0.5:                         # G.711
        x8 = mulaw(np.clip(x8, -1, 1))
    x = resample_poly(x8, 2, 1).astype(np.float32)  # 8 kHz → 16 kHz
    return np.clip(x * rng.uniform(0.6, 1.2), -1, 1).astype(np.float32)


def light_augment(x, rng):
    """Qo'ng'iroq uchun yengil augmentatsiya.

    Polosa cheklovi va kodek QO'LLANMAYDI — qo'ng'iroq allaqachon telefon
    audiosi. Maqsad boshqa: 941 namunada ~25 epoxa aylanayotganda modelning
    aynan shu yozuvlarni yodlab olishiga to'sqinlik qilish. Tezlikni
    o'zgartirish — ASR'da klassik usul.
    """
    if rng.random() < 0.5:
        num, den = _SPEED[rng.integers(len(_SPEED))]
        # Sekinlatish audioni UZAYTIRADI. 30 soniyaga yaqin namuna oshib ketsa,
        # Whisper uni kesadi, matn esa to'liq qoladi — model "eshitilmagan"
        # so'zlarni o'ylab topishga o'rganadi. Shuning uchun faqat sig'sa.
        if len(x) * num / den <= MAX_AUDIO_SAMPLES:
            x = resample_poly(x, num, den).astype(np.float32)
    if rng.random() < 0.5:
        snr = rng.uniform(18, 35)
        p = np.mean(x ** 2) + 1e-12
        x = x + rng.normal(0, np.sqrt(p / (10 ** (snr / 10))), len(x))
    return np.clip(x * rng.uniform(0.7, 1.15), -1, 1).astype(np.float32)


# ──────────────────────────── Dataset ────────────────────────────

def call_id(path):
    """Fayl yo'lidan qo'ng'iroq raqami: `15286_p1_s2.flac` -> `15286`.

    Bo'lish qo'ng'iroq darajasida bo'lishi shart. Bitta suhbatning qo'shni
    bo'laklarida bir xil ovoz, bir xil mavzu va ko'pincha bir xil iboralar
    bor — faylni ajratish bo'lakni ajratadi, suhbatni emas.
    """
    return os.path.basename(str(path)).split(".")[0].split("_")[0]


def load_eval120_calls():
    """eval-120 qo'ng'iroq raqamlari. Topilmasa TRENING BOSHLANMAYDI."""
    path = os.path.join(CALLS_DIR, EVAL_CALLS)
    if not os.path.exists(path):
        sys.exit(f"{path} topilmadi.\n"
                 f"Bu ro'yxatsiz dev bo'lagi eval-120 bilan kesishmasligini "
                 f"tekshirib bo'lmaydi. Repodagi data/eval_calls_120.json ni "
                 f"dataset paketiga qo'shing.")
    with open(path) as f:
        meta = json.load(f)
    ids = {str(c["call_id"]) for c in meta.get("calls", [])}
    ids |= {str(x) for x in meta.get("locked_from_previous", [])}
    if not ids:
        sys.exit(f"{path} bo'sh — qo'ng'iroq raqamlari o'qilmadi.")
    return ids


def load_calls():
    """CSV(lar)dan train va dev to'plamlarini QO'NG'IROQ bo'yicha kesadi."""
    names = [x.strip() for x in TRAIN_CSV.split(",") if x.strip()]
    if not os.path.exists(os.path.join(CALLS_DIR, names[0])):
        if not os.path.exists(CALLS_TAR):
            sys.exit(f"Topilmadi: {CALLS_DIR}/{names[0]} va {CALLS_TAR}.\n"
                     f"Qo'ng'iroq datasetini Pod'ga ko'chiring (README'ga qarang).")
        print(f"📦 {CALLS_TAR} ochilmoqda → {CALLS_DIR}", flush=True)
        os.makedirs(CALLS_DIR, exist_ok=True)
        with tarfile.open(CALLS_TAR) as t:
            t.extractall(CALLS_DIR)

    parts = []
    for name in names:
        fp = os.path.join(CALLS_DIR, name)
        if not os.path.exists(fp):
            sys.exit(f"{fp} topilmadi (TRAIN_CSV: {TRAIN_CSV})")
        d = pd.read_csv(fp)
        print(f"  {name}: {len(d)} namuna")
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    before = len(df)
    df = df.drop_duplicates(subset=["path"]).reset_index(drop=True)
    if len(df) != before:
        print(f"  takroriy yo'l olib tashlandi: {before - len(df)}")
    df["audio"] = df["path"].apply(lambda p: os.path.join(CALLS_DIR, p))
    missing = [p for p in df["audio"] if not os.path.exists(p)]
    if missing:
        sys.exit(f"{TRAIN_CSV}: {len(missing)} ta audio fayl yo'q, masalan {missing[0]}")

    df["call"] = df["path"].apply(call_id)
    eval120 = load_eval120_calls()

    # Qat'iy darvoza: eval-120 qo'ng'irog'i trening CSV'sida bo'lmasligi kerak.
    # Bu `build_weighted.py` da hal qilingan, lekin CSV qo'lda ham yasalishi
    # mumkin — tekshiruv treningga eng yaqin nuqtada turishi kerak.
    leaked = sorted(set(df["call"]) & eval120)
    if leaked:
        sys.exit(f"TRENING TO'XTATILDI: {TRAIN_CSV} da eval-120 ning "
                 f"{len(leaked)} qo'ng'irog'i bor, masalan {leaked[:5]}.\n"
                 f"analysis/build_weighted.py ni qaytadan yurgizing.")

    calls = sorted(set(df["call"]))
    rng = random.Random(DEV_SEED)
    rng.shuffle(calls)
    n_dev = max(1, round(len(calls) * DEV_FRAC))
    dev_calls = set(calls[:n_dev])

    tr = df[~df["call"].isin(dev_calls)].reset_index(drop=True)
    ev = df[df["call"].isin(dev_calls)].reset_index(drop=True)
    if len(tr) == 0 or len(ev) == 0:
        sys.exit(f"Bo'lish bo'sh to'plam berdi: train {len(tr)}, dev {len(ev)}.")

    # Uch shart ham bajarilishi shart — biri buzilsa raqam ma'nosini yo'qotadi.
    assert not (set(tr["call"]) & set(ev["call"])), "train va dev kesishdi"
    assert not (set(tr["call"]) & eval120), "train eval-120 bilan kesishdi"
    assert not (set(ev["call"]) & eval120), "dev eval-120 bilan kesishdi"

    print(f"Bo'lish    : {len(calls)} qo'ng'iroq → train {len(tr)} namuna / "
          f"{len(calls) - n_dev} qo'ng'iroq | dev {len(ev)} / {n_dev} "
          f"(seed {DEV_SEED})")
    print(f"eval-120   : {len(eval120)} qo'ng'iroq chetda, kesishuv 0 — tasdiqlandi")
    return [tr[["audio", "sentence"]], ev[["audio", "sentence"]]]


def to_ds(df, is_call):
    """DataFrame → HF Dataset, audio ustuni bilan.

    `from_pandas` + `cast_column` ISHLATILMAYDI: pandas matn ustunini arrow'da
    `large_string` qilib yaratadi, pyarrow esa uni Audio struct'iga o'gira
    olmaydi ("Unsupported cast from large_string to struct"). Buning o'rniga
    Audio turini qurilish paytida e'lon qilamiz — hech qanday cast bo'lmaydi.
    """
    feats = Features({
        "audio": Audio(sampling_rate=SR),
        "sentence": Value("string"),
        "is_call": Value("int64"),
    })
    return Dataset.from_dict({
        "audio": [str(p) for p in df["audio"]],
        "sentence": [str(s) for s in df["sentence"]],
        "is_call": [is_call] * len(df),
    }, features=feats)


def load_extra():
    """Tashqi korpusni CSV'dan o'qiydi (UzbekVoice). is_call=0 → telefon aug.

    HF'dan to'g'ridan `load_dataset` QILINMAYDI. UzbekVoice o'nlab GB, bizga
    esa ~25k namuna kerak, va Pod soatiga pul turadi — to'liq yuklab olish
    treningdan ko'proq vaqt olishi mumkin. `scripts/prep_extra.py` uni oqim
    bilan olib, faqat kerakli qismini flac qilib yozadi va CSV beradi.

    Audio yo'llari EXTRA_DIR ga nisbatan. Fayl yetishmasa to'xtaymiz:
    yarim yuklangan korpus bilan o'qitish natijani tushuntirib bo'lmaydigan
    qiladi.
    """
    path = os.path.join(EXTRA_DIR, EXTRA_CSV)
    if not os.path.exists(path):
        sys.exit(f"{path} topilmadi. Avval: python3 scripts/prep_extra.py "
                 f"--out {EXTRA_DIR}")
    df = pd.read_csv(path)
    df["audio"] = df["path"].apply(lambda x: os.path.join(EXTRA_DIR, x))
    missing = [x for x in df["audio"] if not os.path.exists(x)]
    if missing:
        sys.exit(f"{EXTRA_CSV}: {len(missing)} ta audio yo'q, masalan {missing[0]}")
    print(f"Tashqi      : {len(df)} namuna ({EXTRA_DIR})")
    return df[["audio", "sentence"]].reset_index(drop=True)


def load_podcast():
    ds = load_dataset(POD_ID, split="train")
    acol = next(n for n, f in ds.features.items() if isinstance(f, Audio))
    if acol != "audio":
        ds = ds.rename_column(acol, "audio")
    tcol = next(c for c in ("text", "sentence", "transcript") if c in ds.column_names)
    ds = ds.map(lambda x: {"sentence": str(x[tcol]).strip()},
                remove_columns=[c for c in ds.column_names if c != "audio"])
    ds = ds.filter(lambda x: len(x["sentence"]) >= 5)
    ds = ds.cast_column("audio", Audio(sampling_rate=SR))
    return ds.add_column("is_call", [0] * len(ds))


# ──────────────────────────── Collator ────────────────────────────

@dataclass
class CallCollator:
    """Mel-spektrogrammani joyida hisoblaydi va augmentatsiyani qo'llaydi.

    Trainer train va eval uchun BITTA collator ishlatadi, shuning uchun farq
    `is_call` belgisi orqali qilinadi:
        0 = podkast      → to'liq telefon augmentatsiyasi
        1 = qo'ng'iroq   → yengil augmentatsiya
        2 = eval         → augmentatsiya YO'Q (o'lchov toza bo'lishi kerak)
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng()
        audios = []
        for f in features:
            a = np.asarray(f["audio"]["array"], dtype=np.float32)
            if f["is_call"] == 0 and rng.random() < AUG_PROB:
                a = phone_augment(a, rng)
            elif f["is_call"] == 1 and rng.random() < CALL_AUG_PROB:
                a = light_augment(a, rng)
            audios.append(a)

        batch = self.processor.feature_extractor(audios, sampling_rate=SR, return_tensors="pt")
        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


class ProgressCallback(TrainerCallback):
    """Tezlik, tugash vaqti va NARXni chiqaradi; byudjetdan oshsa to'xtatadi.

    Pod soatiga to'lanadi, shuning uchun "necha qadam qoldi" emas, "qancha
    pul qoldi" muhim. Haqiqiy s/qadam faqat yurish boshlangach ma'lum
    bo'ladi — oldindan qilingan taxmin GPU bandligi va ma'lumot yuklashga
    qarab ikki barobar chalg'itishi mumkin.
    """

    def on_train_begin(self, args, state, control, **kw):
        import time
        self.time = time
        self.t0 = time.time()
        self.warned = False

    def on_step_end(self, args, state, control, **kw):
        s = state.global_step
        if s % EVAL_STEPS or s == 0:
            return
        el = self.time.time() - self.t0
        per = el / s
        left = per * (MAX_STEPS - s)
        train_usd = (el + left) / 3600 * POD_RATE
        total = train_usd + SETUP_USD
        print(f"  ⏱  {s}/{MAX_STEPS} | {per:.2f} s/qadam | o'tdi {el/60:.0f} daq"
              f" | qoldi ~{left/60:.0f} daq", flush=True)
        print(f"  💰 trening ~${train_usd:.2f} + tayyorgarlik ~${SETUP_USD:.2f}"
              f" = ~${total:.2f} / ${BUDGET_USD:.2f}", flush=True)
        if total > BUDGET_USD:
            # Oshirib yuborishdan ko'ra erta to'xtash yaxshi: eng yaxshi
            # nazorat nuqtasi allaqachon diskda va u yuklanadi.
            print(f"  ⛔ BYUDJET: bashorat ${total:.2f} > ${BUDGET_USD:.2f} — "
                  f"trening {s}-qadamda to'xtatilmoqda", flush=True)
            control.should_training_stop = True
        elif total > 0.8 * BUDGET_USD and not self.warned:
            self.warned = True
            print(f"  ⚠️  byudjetning 80% iga yaqinlashildi", flush=True)


# ──────────────────────────── Asosiy ────────────────────────────

def main():
    resume = "--resume" in sys.argv
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "YO'Q"
    print(f"GPU        : {gpu}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Chiqish    : {OUTPUT_DIR}")
    print(f"Qadamlar   : {MAX_STEPS} | Batch {BATCH_SIZE}x{GRAD_ACCUM} | LR {LR}")

    processor = load_processor(MODEL_NAME)

    # --- Ma'lumot (modelni yuklashdan OLDIN: GPU band bo'lsa .map qotib qoladi) ---
    train_df, eval_df = load_calls()
    calls_tr, calls_ev = to_ds(train_df, 1), to_ds(eval_df, 2)
    print(f"Qo'ng'iroq : {len(calls_tr)} train | {len(calls_ev)} eval")

    if EXTRA_DIR:
        # Qo'ng'iroqlar batch'ning CALL_RATIO ulushini egallashi uchun ular
        # R marta takrorlanadi. Aks holda 25k tashqi namuna 2k qo'ng'iroqni
        # bosib ketadi va model yana toza mikrofon audiosiga moslanadi —
        # Round 1-2 da aynan shu bo'lgan (eval WER tushdi, real qo'ng'iroqda
        # foyda bermadi).
        ext = to_ds(load_extra(), 0)
        R = max(1, round(CALL_RATIO * len(ext) / ((1 - CALL_RATIO) * len(calls_tr))))
        train_ds = concatenate_datasets([calls_tr] * R + [ext]).shuffle(seed=42)
        print(f"Tashqi     : {len(ext)} | qo'ng'iroq takrori x{R} | "
              f"jami {len(train_ds)} ({R*len(calls_tr)/len(train_ds):.0%} qo'ng'iroq)")
    elif USE_PODCAST:
        pod = load_podcast()
        R = max(1, round(CALL_RATIO * len(pod) / ((1 - CALL_RATIO) * len(calls_tr))))
        train_ds = concatenate_datasets([calls_tr] * R + [pod]).shuffle(seed=42)
        print(f"Podkast    : {len(pod)} | takrorlash x{R} | "
              f"jami {len(train_ds)} ({R*len(calls_tr)/len(train_ds):.0%} qo'ng'iroq)")
    else:
        train_ds = calls_tr.shuffle(seed=42)
        print(f"Jami train : {len(train_ds)} (100% qo'ng'iroq)")

    def tok(b):
        ids = processor.tokenizer(b["sentence"]).input_ids
        return {"labels": ids, "llen": len(ids)}

    train_ds = train_ds.map(tok, remove_columns=["sentence"], desc="Tokenlashtirish")
    calls_ev = calls_ev.map(tok, remove_columns=["sentence"], desc="Tokenlashtirish (eval)")
    before = len(train_ds)
    train_ds = train_ds.filter(lambda n: n <= MAX_LABEL, input_columns=["llen"]).remove_columns(["llen"])
    calls_ev = calls_ev.filter(lambda n: n <= MAX_LABEL, input_columns=["llen"]).remove_columns(["llen"])
    if len(train_ds) < before:
        print(f"Uzun matn chiqarildi: {before} → {len(train_ds)}")

    epochs = MAX_STEPS * BATCH_SIZE * GRAD_ACCUM / len(train_ds)
    print(f"Epoxa      : ~{epochs:.1f}")
    if epochs > 15:
        print("  ⚠️  Epoxa soni yuqori — yodlab olish (overfitting) xavfi bor.")
        print("     Eval loss ko'tarilsa, load_best_model_at_end eng yaxshisini tanlaydi.")

    # --- Model ---
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME).to("cuda")
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False
    model.model.encoder.conv1.register_forward_hook(lambda m, i, o: o.requires_grad_(True))
    model = get_peft_model(model, LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", target_modules=["q_proj", "v_proj"]))
    model.print_trainable_parameters()

    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        learning_rate=LR,
        warmup_steps=WARMUP,
        max_steps=MAX_STEPS,
        lr_scheduler_type="linear",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        # predict_with_generate ISHLATILMAYDI — PEFT bilan mos kelmaydi
        # (generate() autocast'dan tashqarida ishga tushib dtype xatosi beradi).
        # WER trening tugagach eval_wer.py bilan hisoblanadi.
        label_names=["labels"],
        remove_unused_columns=False,
        logging_steps=25,
        report_to="none",
        dataloader_num_workers=WORKERS,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model, args=args,
        train_dataset=train_ds, eval_dataset=calls_ev,
        data_collator=CallCollator(processor, model.config.decoder_start_token_id),
        processing_class=processor,
        callbacks=[ProgressCallback(),
                   EarlyStoppingCallback(early_stopping_patience=PATIENCE)],
    )

    print(trainer.train(resume_from_checkpoint=resume or None))

    final = OUTPUT_DIR + "-final"
    trainer.model.merge_and_unload().half().save_pretrained(final)
    processor.save_pretrained(final)
    print(f"\n✅ Saqlandi: {final}")
    print(f"   WER      : python eval_wer.py {final}")
    print(f"   HF'ga    : python upload_to_hf.py")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Whisper large-v3 ni o'zbek tili uchun LoRA bilan fine-tune qilish (RunPod Pod).

Colab notebook'idan farqlari:
  * Colab'ga bog'liq narsalar yo'q (drive.mount, userdata, ngrok)
  * Audio fayllar diskka YOZILMAYDI — HF datasets keshidan to'g'ridan-to'g'ri
    o'qiladi, mel-spektrogramma esa collator ichida joyida hisoblanadi.
    Bu ~100 GB disk va juda ko'p RAM tejaydi.
  * Checkpoint'lar doimiy volume'ga yoziladi, shuning uchun Pod o'chsa ham
    `--resume` bilan davom ettirish mumkin.

Ishga tushirish (RunPod Pod ichida):
    pip install -r requirements-train.txt
    python train.py

Sozlash — muhit o'zgaruvchilari orqali (barchasi ixtiyoriy, pastda standart
qiymatlari ko'rsatilgan). Masalan:
    MAX_STEPS=20000 OUTPUT_DIR=/workspace/whisper-uz python train.py
"""

import os
import sys

import numpy as np
import torch
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import Audio, concatenate_datasets, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ──────────────────────────── Sozlamalar ────────────────────────────

def env(name, default, cast=str):
    return cast(os.environ.get(name, default))

MODEL_NAME = env("MODEL_NAME", "openai/whisper-large-v3")
OUTPUT_DIR = env("OUTPUT_DIR", "/workspace/whisper-large-v3-uz")
LANGUAGE = env("LANGUAGE", "uzbek")
TASK = "transcribe"
SAMPLING_RATE = 16000

MAX_STEPS = env("MAX_STEPS", "10000", int)
EVAL_STEPS = env("EVAL_STEPS", "1000", int)
SAVE_STEPS = env("SAVE_STEPS", "1000", int)
WARMUP_STEPS = env("WARMUP_STEPS", "500", int)
BATCH_SIZE = env("BATCH_SIZE", "16", int)      # L40S/A100 uchun; kichikroq GPU'da 8 yoki 4
GRAD_ACCUM = env("GRAD_ACCUM", "1", int)       # effektiv batch = BATCH_SIZE * GRAD_ACCUM
LR = env("LR", "1e-4", float)
EVAL_MAX_SAMPLES = env("EVAL_MAX_SAMPLES", "500", int)
DATALOADER_WORKERS = env("DATALOADER_WORKERS", "8", int)

LORA_R = env("LORA_R", "32", int)
LORA_ALPHA = env("LORA_ALPHA", "64", int)
LORA_DROPOUT = env("LORA_DROPOUT", "0.05", float)
LORA_TARGET_MODULES = ["q_proj", "v_proj"]

# --- Manbalar: (HF dataset id, split, matn ustuni nomzodlari, cheklov) ---
# TABIIY SUHBAT nutqi — qo'ng'iroq tahlili uchun eng mos tur. O'qib yozdirilgan
# (Common Voice, UzbekVoice) datasetlar bu yerda ATAYLAB ishlatilmaydi: ular toza
# va mikrofonga yaqin, qo'ng'iroq esa shovqinli va erkin suhbat.
SOURCES = [
    ("BoburAmirov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset",
     "train", ["text", "sentence"], env("POD_CAP", "0", int) or None),        # ~14 500, hammasi
    ("islomov/news_youtube_uzbek_speech_dataset",
     "train", ["text", "sentence"], env("YT_CAP", "0", int) or None),         # ~20 800, hammasi

    # IT mavzusidagi YouTube (~21 000) — ichida inglizcha so'zlar aralash.
    # Yoqsangiz pool ~56 000 ga chiqadi va overfitting xavfi kamayadi:
    # ("islomov/it_youtube_uzbek_speech_dataset", "train", ["text", "sentence"], None),

    # O'qib yozdirilgan manbalar (kerak bo'lsa):
    # ("DavronSherbaev/uzbekvoice-filtered", "train", ["text", "sentence"], 150000),
    # ("mrmuminov/uzbek_voice", "train", ["original_sentence", "text"], 150000),
    # ("yakhyo/mozilla-common-voice-uzbek", "validated", ["sentence", "text"], 30000),
    # ("shunyalabs/uzbek-speech-dataset", "train", ["transcript", "text"], None),
]

# ──────────────────────────── Dataset ────────────────────────────

def get_text(example, candidates):
    for col in candidates:
        val = example.get(col)
        if val:
            return str(val).strip()
    return ""


def load_source(hf_id, split, text_cols, cap):
    """Bitta manbani yuklab, ["audio", "sentence"] ustunlariga keltiradi."""
    print(f"\n📥 {hf_id} ({split})", flush=True)
    ds = load_dataset(hf_id, split=split)
    print(f"   Ustunlar: {ds.column_names}", flush=True)

    # Audio ustunini NOMI bo'yicha emas, TURI bo'yicha topamiz — ba'zi
    # datasetlarda u "audio" emas, masalan "path" deb nomlangan.
    audio_col = next((n for n, f in ds.features.items() if isinstance(f, Audio)), None)
    if audio_col is None:
        raise ValueError(f"Audio ustuni topilmadi: {ds.column_names}")
    if audio_col != "audio":
        ds = ds.rename_column(audio_col, "audio")
        print(f"   Audio ustuni: '{audio_col}' → 'audio'", flush=True)

    # Bo'sh matnli qatorlarni CHEKLOVDAN OLDIN chiqaramiz, aks holda so'ralgan
    # miqdor amalda kamayib qoladi.
    before = len(ds)
    ds = ds.filter(lambda x: any(x.get(c) for c in text_cols), num_proc=4)
    if len(ds) < before:
        print(f"   Bo'sh matn chiqarildi: {before} → {len(ds)}", flush=True)

    # Sifat filtri — ustun nomlari manbalar orasida farq qiladi
    up = next((c for c in ("up_votes", "upvotes", "upvotes_count") if c in ds.column_names), None)
    down = next((c for c in ("down_votes", "downvotes", "downvotes_count") if c in ds.column_names), None)
    if up and down:
        before = len(ds)
        ds = ds.filter(lambda x: x[up] >= 1 and x[down] == 0, num_proc=4)
        print(f"   Sifat filtri ({up}/{down}): {before} → {len(ds)}", flush=True)

    ds = ds.shuffle(seed=42)
    if cap:
        ds = ds.select(range(min(cap, len(ds))))

    # Matnni yagona "sentence" ustuniga keltiramiz, qolgan metadata kerak emas
    ds = ds.map(
        lambda x: {"sentence": get_text(x, text_cols)},
        remove_columns=[c for c in ds.column_names if c != "audio"],
        num_proc=4,
        desc="   matnni normallashtirish",
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=SAMPLING_RATE))
    print(f"   Tayyor: {len(ds)} namuna", flush=True)
    return ds


def build_dataset():
    parts = []
    for hf_id, split, text_cols, cap in SOURCES:
        try:
            parts.append(load_source(hf_id, split, text_cols, cap))
        except Exception as e:
            print(f"   ⚠️  O'tkazib yuborildi: {e}", flush=True)
    if not parts:
        sys.exit("Hech qanday manba yuklanmadi.")
    full = concatenate_datasets(parts).shuffle(seed=42)
    print(f"\n✅ Jami: {len(full)} namuna ({len(parts)} manbadan)", flush=True)
    return full


# ──────────────────────────── Collator ────────────────────────────

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Mel-spektrogrammani JOYIDA hisoblaydi.

    Oldindan hisoblab saqlash large-v3 uchun namunasiga ~1.5 MB (128x3000
    float32) — yuz minglab namunada bu yuzlab GB va RAM tugashi degani.
    CPU buni GPU'dan tezroq ulguradi, shuning uchun tezlik pasaymaydi."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [f["audio"]["array"] for f in features]
        batch = self.processor.feature_extractor(
            audios, sampling_rate=SAMPLING_RATE, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ──────────────────────────── Asosiy ────────────────────────────

def main():
    resume = "--resume" in sys.argv

    print(f"Model      : {MODEL_NAME}")
    print(f"Chiqish    : {OUTPUT_DIR}")
    print(f"Qadamlar   : {MAX_STEPS}")
    print(f"Batch      : {BATCH_SIZE} x {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "topilmadi"
    print(f"GPU        : {gpu_name}")

    processor = WhisperProcessor.from_pretrained(MODEL_NAME, language=LANGUAGE, task=TASK)

    # Asosiy model fp32'da — aralash aniqlik training_args'dagi fp16=True
    # (autocast) orqali boshqariladi. Modelni to'g'ridan-to'g'ri fp16'da
    # yuklash PEFT bilan dtype nomuvofiqligiga olib keladi.
    model = WhisperForConditionalGeneration.from_pretrained(MODEL_NAME)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.config.use_cache = False

    # Muzlatilgan model + gradient checkpointing kombinatsiyasida kirish
    # gradientni uzatishi uchun (rasmiy PEFT+Whisper tavsiyasi)
    model.model.encoder.conv1.register_forward_hook(
        lambda module, inp, out: out.requires_grad_(True)
    )

    model = get_peft_model(
        model,
        LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
        ),
    )
    model.print_trainable_parameters()

    # --- Dataset ---
    full = build_dataset()
    max_label_len = model.config.max_target_positions

    def tokenize(batch):
        ids = processor.tokenizer(batch["sentence"]).input_ids
        return {"labels": ids, "labels_length": len(ids)}

    full = full.map(tokenize, remove_columns=["sentence"], num_proc=4, desc="Tokenlashtirish")
    before = len(full)
    full = full.filter(lambda n: n <= max_label_len, input_columns=["labels_length"], num_proc=4)
    full = full.remove_columns(["labels_length"])
    if len(full) < before:
        print(f"Uzun matnlar chiqarildi: {before} → {len(full)}")
    # Audio 30 soniyadan uzun bo'lsa, feature extractor uni o'zi kesadi —
    # shuning uchun alohida davomiylik filtri kerak emas.

    split = full.train_test_split(test_size=0.02, seed=42)
    eval_ds = split["test"]
    if len(eval_ds) > EVAL_MAX_SAMPLES:
        eval_ds = eval_ds.shuffle(seed=42).select(range(EVAL_MAX_SAMPLES))
    print(f"Train: {len(split['train'])} | Eval: {len(eval_ds)}")

    # --- Trening ---
    args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        fp16=True,
        learning_rate=LR,
        warmup_steps=WARMUP_STEPS,
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
        # WER trening tugagach alohida hisoblanadi (eval_wer.py).
        label_names=["labels"],
        remove_unused_columns=False,
        logging_steps=50,
        report_to="none",
        dataloader_num_workers=DATALOADER_WORKERS,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=split["train"],
        eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        ),
        processing_class=processor,
    )

    result = trainer.train(resume_from_checkpoint=resume or None)
    print(result)

    # --- Saqlash: LoRA adapterlarni birlashtirib, oddiy Whisper model sifatida ---
    merged = trainer.model.merge_and_unload().half()
    merged.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"\n✅ Saqlandi: {OUTPUT_DIR}")
    print("   Hugging Face'ga yuklash uchun: python upload_to_hf.py")


if __name__ == "__main__":
    main()

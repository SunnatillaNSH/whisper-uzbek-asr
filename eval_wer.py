#!/usr/bin/env python3
"""Fine-tune qilingan modelning WER ko'rsatkichini hisoblash.

Trening skripti predict_with_generate ishlatmaydi (PEFT bilan mos kelmaydi),
shuning uchun WER shu alohida skript orqali, Trainer'dan tashqarida hisoblanadi.

    MODEL_DIR=/workspace/whisper-large-v3-uz python eval_wer.py
"""

import os

import evaluate
import torch
from datasets import Audio, load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

MODEL_DIR = os.environ.get("MODEL_DIR", "/workspace/whisper-large-v3-uz")
DATASET = os.environ.get("EVAL_DATASET", "BoburAmirov/podcasts_tashkent_dialect_youtube_uzbek_speech_dataset")
SPLIT = os.environ.get("EVAL_SPLIT", "train")
N = int(os.environ.get("EVAL_N", "300"))
BATCH = int(os.environ.get("EVAL_BATCH", "8"))
SR = 16000

processor = WhisperProcessor.from_pretrained(MODEL_DIR, language="uzbek", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_DIR, torch_dtype=torch.float16).cuda().eval()

ds = load_dataset(DATASET, split=SPLIT).shuffle(seed=1234).select(range(N))
audio_col = next(n for n, f in ds.features.items() if isinstance(f, Audio))
if audio_col != "audio":
    ds = ds.rename_column(audio_col, "audio")
ds = ds.cast_column("audio", Audio(sampling_rate=SR))

text_col = next(c for c in ("text", "sentence", "transcript", "original_sentence") if c in ds.column_names)

wer_metric = evaluate.load("wer")
preds, refs = [], []

with torch.no_grad():
    for i in range(0, len(ds), BATCH):
        batch = ds[i : i + BATCH]
        feats = processor.feature_extractor(
            [a["array"] for a in batch["audio"]], sampling_rate=SR, return_tensors="pt"
        ).input_features.to("cuda").half()
        ids = model.generate(feats, language="uzbek", task="transcribe", max_new_tokens=225)
        preds.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
        refs.extend(str(t).strip() for t in batch[text_col])
        print(f"  {min(i+BATCH, len(ds))}/{len(ds)}", flush=True)

wer = 100 * wer_metric.compute(predictions=preds, references=refs)
print(f"\nWER: {wer:.2f}%  ({DATASET}, {N} namuna)")
print("\nNamunalar:")
for p, r in list(zip(preds, refs))[:5]:
    print(f"  ASL      : {r}")
    print(f"  BASHORAT : {p}\n")

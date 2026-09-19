#!/usr/bin/env python3
"""WER'ni REAL QO'NG'IROQLARNING eval to'plamida o'lchaydi va modellarni solishtiradi.

`eval_wer.py` ochiq HF datasetlarida ishlaydi — u o'zbek tili bilimini o'lchaydi.
Bu esa `calls-colab` ning eval qismida (31 ta alohida qo'ng'iroq, 81 namuna)
o'lchaydi, ya'ni bizga amalda kerak bo'lgan narsani.

Eval qo'ng'iroqlari train'dagilardan BUTUN QO'NG'IROQ bo'yicha ajratilgan —
bitta suhbatning bo'laklari ikkala to'plamga bo'linib ketmagan.

    python eval_calls_wer.py                                  # yangi vs eski
    python eval_calls_wer.py /workspace/whisper-uz-calls-final
    CALLS_DIR=/workspace/calls python eval_calls_wer.py A B    # ixtiyoriy ikkita
"""

import gc
import os
import re
import sys

import jiwer
import pandas as pd
import soundfile as sf
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

CALLS_DIR = os.environ.get("CALLS_DIR", "/workspace/calls")
BATCH = int(os.environ.get("EVAL_BATCH", "8"))
SR = 16000

DEFAULT_MODELS = [
    os.environ.get("NEW_MODEL", "/workspace/whisper-uz-calls-final"),
    os.environ.get("OLD_MODEL", "Sunnat0091/whisper-large-v3-uz"),
]

# Normallashtirish qo'lda qilinadi — jiwer'ning transform argumentlari
# versiyalar orasida nomini o'zgartirgan (truth_transform → reference_transform),
# bu esa versiyaga bog'liq bo'lib qolishni anglatadi.
#
# ENG MUHIMI — APOSTROF. O'zbek lotin yozuvida u harfning bir qismi (o', g').
# Muxlisa U+2018 (') ishlatadi, Whisper esa U+2019 (') yoki oddiy ASCII (')
# chiqarishi mumkin. Normallashtirmasak, "to'g'ri" va "to'g'ri" turli so'z
# hisoblanadi va WER sun'iy ravishda yuqori chiqadi. Shuning uchun barcha
# apostrof variantlarini bittaga keltiramiz va uni HARF sifatida saqlaymiz.
APOS = {ord(c): "'" for c in "\u2018\u2019\u02bb\u02bc\u0060\u00b4"}


def norm(s):
    s = str(s).translate(APOS).lower()
    s = re.sub(r"[^\w\s']", " ", s, flags=re.UNICODE)   # tinish belgilari → bo'shliq
    return re.sub(r"\s+", " ", s).strip()


def load_eval():
    path = os.path.join(CALLS_DIR, "eval.csv")
    if not os.path.exists(path):
        sys.exit(f"Topilmadi: {path}  (CALLS_DIR ni to'g'ri ko'rsating)")
    df = pd.read_csv(path)
    df["audio"] = df["path"].apply(lambda p: os.path.join(CALLS_DIR, p))
    return df


def transcribe_all(model_id, df):
    processor = WhisperProcessor.from_pretrained(model_id, language="uzbek", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16).cuda().eval()

    hyps = []
    with torch.no_grad():
        for i in range(0, len(df), BATCH):
            chunk = df.iloc[i:i + BATCH]
            audios = [sf.read(p, dtype="float32")[0] for p in chunk["audio"]]
            feats = processor.feature_extractor(
                audios, sampling_rate=SR, return_tensors="pt"
            ).input_features.cuda().half()
            ids = model.generate(feats, language="uzbek", task="transcribe",
                                 max_new_tokens=400)
            hyps.extend(processor.tokenizer.batch_decode(ids, skip_special_tokens=True))
            print(f"    {min(i+BATCH, len(df))}/{len(df)}", flush=True)

    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    return hyps


def main():
    models = sys.argv[1:] or DEFAULT_MODELS
    df = load_eval()
    refs = [str(t).strip() for t in df["sentence"]]
    print(f"Eval to'plami: {len(df)} namuna ({CALLS_DIR}/eval.csv)\n")

    results = {}
    for m in models:
        print(f"▶ {m}")
        hyps = transcribe_all(m, df)
        results[m] = (100 * jiwer.wer([norm(r) for r in refs],
                                      [norm(h) for h in hyps]), hyps)
        print(f"  WER: {results[m][0]:.2f}%\n")

    print("=" * 60)
    for m, (w, _) in results.items():
        print(f"  {w:6.2f}%   {m}")
    if len(models) == 2:
        a, b = (results[m][0] for m in models)
        print(f"\n  Farq: {b - a:+.2f} foiz punkt "
              f"({'yaxshilandi ✅' if a < b else 'yomonlashdi ❌'})")
    print("=" * 60)

    first = results[models[0]][1]
    print("\nNamunalar (birinchi model):")
    for i in range(min(3, len(df))):
        print(f"\n  HAQIQIY : {refs[i][:220]}")
        print(f"  MODEL   : {first[i][:220]}")


if __name__ == "__main__":
    main()
